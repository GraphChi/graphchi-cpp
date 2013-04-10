/**
 * @file
 * @author  Danny Bickson
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright [2012] [Carnegie Mellon University]
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.

 *
 * @section DESCRIPTION
 *
 * Implementation of the libfm algorithm.
 * Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th IEEE International Conference on Data Mining (ICDM 2010), Sydney, Australia.
 * Original implementation by Qiang Yan, Chinese Academy of Science.
 * note: this code version implements the SGD version of libfm. In the original library there are also ALS and MCMC methods.
 * Also the treatment of features is richer in libfm. The code here can serve for a quick evaluation but the user
 * is encouraged to try libfm as well.
 */



#include "common.hpp"
#include "eigen_wrapper.hpp"

double libfm_rate = 1e-02;
double libfm_mult_dec = 0.9;
double libfm_regw = 1e-3;
double libfm_regv = 1e-3;
double reg0 = 0.1;
bool debug = false;
int time_offset = 1; //time bin starts from 1?

bool is_user(vid_t id){ return id < M; }
bool is_item(vid_t id){ return id >= M && id < N; }
bool is_time(vid_t id){ return id >= M+N; }
#define BIAS_POS -1

struct vertex_data {
  vec pvec;
  double bias;
  int last_item;

  vertex_data() {
    bias = 0;
    last_item = 0;
  }
  void set_val(int index, float val){
    if (index == BIAS_POS)
      bias = val;
    else pvec[index] = val;
  }
  float get_val(int index){
    if (index== BIAS_POS)
      return bias;
    else return pvec[index];
  }


};

struct edge_data {
  double weight;
  double time;

  edge_data() { weight = time = 0; }

  edge_data(double weight, double time) : weight(weight), time(time) { }
};


struct vertex_data_libfm{
  double * bias;
  double * v;
  int *last_item; 

  vertex_data_libfm(const vertex_data & vdata){
    v = (double*)&vdata.pvec[0];
    bias = (double*)&vdata.bias;
    last_item = (int*)&vdata.last_item;
  }

  vertex_data_libfm & operator=(vertex_data & data){
    v = (double*)&data.pvec[0];
    bias = (double*)&data.bias;
    last_item = (int*)&data.last_item;
    return * this;
  }   
};

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef edge_data EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
graphchi_engine<VertexDataType, EdgeDataType> * pvalidation_engine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

#include "io.hpp"
#include "rmse.hpp"
#include "rmse_engine4.hpp"

float libfm_predict(const vertex_data_libfm& user, 
    const vertex_data_libfm& movie, 
    const vertex_data_libfm& time,
    const float rating, 
    double& prediction, vec * sum){

  vertex_data & last_item = latent_factors_inmem[M+N+K+(*user.last_item)]; //TODO, when no ratings, last item is 0
  vec sum_sqr = zeros(D);
  *sum = zeros(D);
  prediction = globalMean + *user.bias + *movie.bias + *time.bias + last_item.bias;
  for (int j=0; j< D; j++){
    sum->operator[](j) += user.v[j] + movie.v[j] + time.v[j] + last_item.pvec[j];    
    sum_sqr[j] = pow(user.v[j],2) + pow(movie.v[j],2) + pow(time.v[j],2) + pow(last_item.pvec[j],2); 
    prediction += 0.5 * (pow(sum->operator[](j),2) - sum_sqr[j]);
  }
  //truncate prediction to allowed values
  prediction = std::min((double)prediction, maxval);
  prediction = std::max((double)prediction, minval);
  //return the squared error
  float err = rating - prediction;
  assert(!std::isnan(err));
  return err*err; 

}
float libfm_predict(const vertex_data& user, 
    const vertex_data& movie, 
    const float rating, 
    double & prediction, 
    void * extra){
  vec sum; 
  return libfm_predict(vertex_data_libfm((vertex_data&)user), vertex_data_libfm((vertex_data&)movie), vertex_data_libfm(*(vertex_data*)extra), rating, prediction, &sum);
}


void init_libfm(){

  srand(time(NULL));
  latent_factors_inmem.resize(M+N+K+M);

  assert(D > 0);
  double factor = 0.1/sqrt(D);
#pragma omp parallel for
  for (int i=0; i< (int)(M+N+K+M); i++){
      latent_factors_inmem[i].pvec = (debug ? 0.1*ones(D) : (::randu(D)*factor));
  }
}



/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct LIBFMVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

  /*
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {


    if (gcontext.iteration == 0){
    if (is_user(vertex.id())) { //user node. find the last rated item and store it
      vertex_data_libfm user = latent_factors_inmem[vertex.id()]; 
      int max_time = 0;
      for(int e=0; e < vertex.num_outedges(); e++) {
        const edge_data & edge = vertex.edge(e)->get_data();
        if (edge.time >= max_time){
          max_time = (int)(edge.time - time_offset);
          *user.last_item = vertex.edge(e)->vertex_id() - M;
        }
      }
    }
if (is_user(vertex.id()) && vertex.num_outedges() == 0)
      logstream(LOG_WARNING)<<"Vertex: " << vertex.id() << " with no edges: " << std::endl;
    return;
    return;
  } 
 
    //go over all user nodes
    if (is_user(vertex.id())){
      vertex_data_libfm user = latent_factors_inmem[vertex.id()]; 
      assert(*user.last_item >= 0 && *user.last_item < (int)N);
      vertex_data & last_item = latent_factors_inmem[M+N+K+(*user.last_item)]; 

      for(int e=0; e < vertex.num_outedges(); e++) {
        vertex_data_libfm movie(latent_factors_inmem[vertex.edge(e)->vertex_id()]);

        float rui = vertex.edge(e)->get_data().weight;
        double pui;
        vec sum;
        vertex_data & time = latent_factors_inmem[(int)vertex.edge(e)->get_data().time - time_offset];
        float sqErr = libfm_predict(user, movie, time, rui, pui, &sum);
        float eui = pui - rui;

        globalMean -= libfm_rate * (eui + reg0 * globalMean);
        *user.bias -= libfm_rate * (eui + libfm_regw * *user.bias);
        *movie.bias -= libfm_rate * (eui + libfm_regw * *movie.bias);
        time.bias -= libfm_rate * (eui + libfm_regw * time.bias);
        assert(!std::isnan(time.bias));
        last_item.bias -= libfm_rate * (eui + libfm_regw * last_item.bias);

        for(int f = 0; f < D; f++){
          // user
          float grad = sum[f] - user.v[f];
          user.v[f] -= libfm_rate * (eui * grad + libfm_regv * user.v[f]);
          // item
          grad = sum[f] - movie.v[f];
          movie.v[f] -= libfm_rate * (eui * grad + libfm_regv * movie.v[f]);
          // time
          grad = sum[f] - time.pvec[f];
          time.pvec[f] -= libfm_rate * (eui * grad + libfm_regv * time.pvec[f]);
          // last item
          grad = sum[f] - last_item.pvec[f];
          last_item.pvec[f] -= libfm_rate * (eui * grad + libfm_regv * last_item.pvec[f]);

        }

        rmse_vec[omp_get_thread_num()] += sqErr;
      }

    }

  };

 /**
   * Called before an iteration is started.
   */
  void before_iteration(int iteration, graphchi_context &gcontext) {
    reset_rmse(gcontext.execthreads);
  }


  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    libfm_rate *= libfm_mult_dec;
    training_rmse(iteration, gcontext);
    run_validation4(pvalidation_engine, gcontext);
  };


};


void output_libfm_result(std::string filename) {
  MMOutputter_mat<vertex_data> mmoutput_left(filename + "_U.mm", 0, M, "This file contains LIBFM output matrix U. In each row D factors of a single user node.", latent_factors_inmem);
  MMOutputter_mat<vertex_data> mmoutput_right(filename + "_V.mm", M ,M+N, "This file contains -LIBFM  output matrix V. In each row D factors of a single item node.", latent_factors_inmem);
  MMOutputter_mat<vertex_data> mmoutput_time(filename + "_T.mm", M+N ,M+N+K, "This file contains -LIBFM  output matrix T. In each row D factors of a single time node.", latent_factors_inmem);
  MMOutputter_mat<vertex_data> mmoutput_last_item(filename + "_L.mm", M+N+K ,M+N+K+M, "This file contains -LIBFM  output matrix L. In each row D factors of a single last item node.", latent_factors_inmem);
  MMOutputter_vec<vertex_data> mmoutput_bias_left(filename + "_U_bias.mm", 0, M, BIAS_POS, "This file contains LIBFM output bias vector. In each row a single user bias.", latent_factors_inmem);
  MMOutputter_vec<vertex_data> mmoutput_bias_right(filename + "_V_bias.mm",M ,M+N, BIAS_POS,  "This file contains LIBFM output bias vector. In each row a single item bias.", latent_factors_inmem);
  MMOutputter_vec<vertex_data> mmoutput_bias_time(filename + "_T_bias.mm",M+N ,M+N+K , BIAS_POS, "This file contains LIBFM output bias vector. In each row a single time bias.", latent_factors_inmem);
  MMOutputter_vec<vertex_data> mmoutput_bias_last_item(filename + "_L_bias.mm",M+N+K ,M+N+K+M , BIAS_POS, "This file contains LIBFM output bias vector. In each row a single last item bias.", latent_factors_inmem);
  MMOutputter_scalar gmean(filename + "_global_mean.mm", "This file contains LIBFM global mean which is required for computing predictions.", globalMean);

  logstream(LOG_INFO) << " LIBFM output files (in matrix market format): " << filename << "_U.mm" << ", " << filename + "_V.mm " << filename + "_T.mm, " << filename << "_L.mm, " << filename <<  "_global_mean.mm, " << filename << "_U_bias.mm " << filename << "_V_bias.mm, " << filename << "_T_bias.mm, " << filename << "_L_bias.mm " <<std::endl;
}

int main(int argc, const char ** argv) {


  print_copyright();  

  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("libfm");

  //specific command line parameters for libfm
  libfm_rate = get_option_float("libfm_rate", libfm_rate);
  libfm_regw = get_option_float("libfm_regw", libfm_regw);
  libfm_regv = get_option_float("libfm_regv", libfm_regv);
  libfm_mult_dec = get_option_float("libfm_mult_dec", libfm_mult_dec);
  D = get_option_int("D", D);

  parse_command_line_args();
  parse_implicit_command_line();

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket4<edge_data>(training, false);
  init_libfm();
  if (validation != ""){
    int vshards = convert_matrixmarket4<EdgeDataType>(validation, true, M==N, VALIDATION);
    init_validation_rmse_engine<VertexDataType, EdgeDataType>(pvalidation_engine, vshards, &libfm_predict, false, true, 1);
   }


  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, D);
    load_matrix_market_matrix(training + "_V.mm", M, D);
    load_matrix_market_matrix(training + "_T.mm", M+N, D);
    load_matrix_market_matrix(training + "_L.mm", M+N+K, D);
    vec user_bias =      load_matrix_market_vector(training +"_U_bias.mm", false, true);
    vec item_bias =      load_matrix_market_vector(training +"_V_bias.mm", false, true);
    vec time_bias =      load_matrix_market_vector(training+ "_T_bias.mm", false, true);
    vec last_item_bias = load_matrix_market_vector(training+"_L_bias.m", false, true);
    for (uint i=0; i<M+N+K+M; i++){
      if (i < M)
        latent_factors_inmem[i].bias = user_bias[i];
      else if (i <M+N)
        latent_factors_inmem[i].bias = item_bias[i-M];
      else if (i <M+N+K)
        latent_factors_inmem[i].bias = time_bias[i-M-N];
      else 
        latent_factors_inmem[i].bias = last_item_bias[i-M-N-K];
    }
    vec gm = load_matrix_market_vector(training + "_global_mean.mm", false, true);
    globalMean = gm[0];
}


  /* Run */
  LIBFMVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output test predictions in matrix-market format */
  output_libfm_result(training);
  test_predictions3(&libfm_predict, 1);    

  /* Report execution metrics */
  if (!quiet) 
    metrics_report(m);
  return 0;
}

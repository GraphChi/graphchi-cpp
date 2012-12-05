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



#include "graphchi_basic_includes.hpp"
#include "common.hpp"
#include "eigen_wrapper.hpp"

double libfm_rate = 1e-02;
double libfm_mult_dec = 0.9;
double libfm_regw = 1e-3;
double libfm_regv = 1e-3;
double reg0 = 0.1;
int D = 20; //feature vector width, can be changed on runtime using --D=XX flag
bool debug = false;
int time_offset = 1; //time bin starts from 1?

bool is_user(vid_t id){ return id < M; }
bool is_item(vid_t id){ return id >= M && id < N; }
bool is_time(vid_t id){ return id >= M+N; }

inline double sum(double * pvec){
  double tsum = 0;
  for (int j=0; j< D; j++)
    tsum += pvec[j];
  return tsum;
}


struct vertex_data {
  vec pvec;
  double rmse;
  double bias;
  int last_item;

  vertex_data() {
    rmse = 0;
    bias = 0;
    last_item = 0;
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
  double * rmse;

  vertex_data_libfm(const vertex_data & vdata){
    v = (double*)&vdata.pvec[0];
    bias = (double*)&vdata.bias;
    last_item = (int*)&vdata.last_item;
    rmse = (double*)& vdata.rmse;
  }

  vertex_data_libfm & operator=(vertex_data & data){
    v = (double*)&data.pvec[0];
    bias = (double*)&data.bias;
    last_item = (int*)&data.last_item;
    rmse = (double*)&data.rmse;
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
std::vector<vertex_data> latent_factors_inmem;

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
    const vertex_data& time,
    const float rating, 
    double & prediction){
  vec sum; 
  return libfm_predict(vertex_data_libfm((vertex_data&)user), vertex_data_libfm((vertex_data&)movie), vertex_data_libfm((vertex_data&)time), rating, prediction, &sum);
}


void init_libfm(){

  srand(time(NULL));
  latent_factors_inmem.resize(M+N+K+M);

  assert(D > 0);
  double factor = 0.1/sqrt(D);
#pragma omp parallel for
  for (uint i=0; i< M+N+K+M; i++){
      latent_factors_inmem[i].pvec = (debug ? 0.1*ones(D) : (::randu(D)*factor));
  }
}


#include "io.hpp"
#include "rmse.hpp"


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
          max_time = edge.time - time_offset;
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
      *user.rmse = 0; 
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
        time.bias -= libfm_regw * (eui + libfm_regw * time.bias);
        assert(!std::isnan(time.bias));
        last_item.bias -= libfm_regw * (eui + libfm_regw * last_item.bias);

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

        *user.rmse += sqErr;
      }

    }

  };

  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    libfm_rate *= libfm_mult_dec;
    training_rmse(iteration, gcontext);
    validation_rmse3(&libfm_predict, gcontext, 4, 1);
  };


};

struct  MMOutputter_bias{
  FILE * outf;
  MMOutputter_bias(std::string fname, uint start, uint end, std::string comment)  {
    MM_typecode matcode;
    set_matcode(matcode);
    outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, end-start, 1); 
    for (uint i=start; i< end; i++)
      fprintf(outf, "%1.12e\n", latent_factors_inmem[i].bias);
  }


  ~MMOutputter_bias() {
    if (outf != NULL) fclose(outf);
  }

};

struct  MMOutputter_global_mean {
  FILE * outf;
  MMOutputter_global_mean(std::string fname, std::string comment)  {
    MM_typecode matcode;
    set_matcode(matcode);
    outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, 1, 1); 
    fprintf(outf, "%1.12e\n", globalMean);
  }

  ~MMOutputter_global_mean() {
    if (outf != NULL) fclose(outf);
  }

};


struct  MMOutputter{
  FILE * outf;
  MMOutputter(std::string fname, uint start, uint end, std::string comment)  {
    assert(start < end);
    MM_typecode matcode;
    set_matcode(matcode);     
    outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, end-start, latent_factors_inmem[start].pvec.size()); 
    for (uint i=start; i < end; i++)
      for(int j=0; j < latent_factors_inmem[i].pvec.size(); j++) {
        fprintf(outf, "%1.12e\n", latent_factors_inmem[i].pvec[j]);
      }
  }

  ~MMOutputter() {
    if (outf != NULL) fclose(outf);
  }

};


void output_libfm_result(std::string filename) {
  MMOutputter mmoutput_left(filename + "_U.mm", 0, M, "This file contains LIBFM output matrix U. In each row D factors of a single user node.");
  MMOutputter mmoutput_right(filename + "_V.mm", M ,M+N, "This file contains -LIBFM  output matrix V. In each row D factors of a single item node.");
  MMOutputter mmoutput_time(filename + "_T.mm", M+N ,M+N+K, "This file contains -LIBFM  output matrix T. In each row D factors of a single time node.");
  MMOutputter mmoutput_last_item(filename + "_L.mm", M+N+K ,M+N+K+M, "This file contains -LIBFM  output matrix L. In each row D factors of a single last item node.");
   MMOutputter_bias mmoutput_bias_left(filename + "_U_bias.mm", 0, M, "This file contains LIBFM output bias vector. In each row a single user bias.");
  MMOutputter_bias mmoutput_bias_right(filename + "_V_bias.mm",M ,M+N , "This file contains LIBFM output bias vector. In each row a single item bias.");
  MMOutputter_bias mmoutput_bias_time(filename + "_T_bias.mm",M+N ,M+N+K , "This file contains LIBFM output bias vector. In each row a single time bias.");
  MMOutputter_bias mmoutput_bias_last_item(filename + "_L_bias.mm",M+N+K ,M+N+K+M , "This file contains LIBFM output bias vector. In each row a single last item bias.");
  MMOutputter_global_mean gmean(filename + "_global_mean.mm", "This file contains LIBFM global mean which is required for computing predictions.");

  logstream(LOG_INFO) << " time-svd++ output files (in matrix market format): " << filename << "_U.mm" << ", " << filename + "_V.mm " << filename + "_T.mm, " << filename << "_L.mm, " << filename <<  "_global_mean.mm, " << filename << "_U_bias.mm " << filename << "_V_bias.mm, " << filename << "_T_bias.mm, " << filename << "_L_bias.mm " <<std::endl;
}

int main(int argc, const char ** argv) {


  print_copyright();  

  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("als-tensor-inmemory-factors");

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

  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, D);
    load_matrix_market_matrix(training + "_V.mm", M, D);
    load_matrix_market_matrix(training + "_T.mm", M+N, D);
    load_matrix_market_matrix(training + "_L.mm", M+N+K, D);
     vec user_bias = load_matrix_market_vector(training +"_U_bias.mm", false, true);
    vec item_bias = load_matrix_market_vector(training +"_V_bias.mm", false, true);
    vec time_bias = load_matrix_market_vector(training+ "_T_bias.mm", false, true);
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
  metrics_report(m);
  return 0;
}

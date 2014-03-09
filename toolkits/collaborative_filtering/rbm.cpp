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
 * Matrix factorization using RBM (Restricted Bolzman Machines) algorithm.
 * Algorithm is described in the paper:
 * G. Hinton. A Practical Guide to Training Restricted Boltzmann Machines. University of Toronto Tech report UTML TR 2010-003
 * 
 */

#include "common.hpp"
#include "eigen_wrapper.hpp"

double rbm_alpha        = 0.1;
double rbm_beta         = 0.06;
int    rbm_bins         = 6;
double rbm_scaling      = 1;
double rbm_mult_step_dec= 0.9;

bool is_user(vid_t id){ return id < M; }
bool is_item(vid_t id){ return id >= M && id < N; }
bool is_time(vid_t id){ return id >= M+N; }

void setRand2(double * a, int d, float c){
  for(int i = 0; i < d; ++i)
    a[i] = ((drand48() - 0.5) * c);
}

float dot(double * a, double * b){
  float ret = 0;
  for(int i = 0; i < D; ++i)
    ret += a[i] * b[i];
  return ret;
}

#define BIAS_POS -1
struct vertex_data {
  vec pvec; //storing the feature vector
  double bias;

  vertex_data() {
    bias = 0;
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


/*
 * h = pvec = D * DOUBLE
 * h0 = weight = D * DOUBLE
 * h1 = weight+D = D * DOUBLE
 */
struct rbm_user{
  double * h;
  double * h0;
  double * h1;

  rbm_user(const vertex_data & vdata){
    h = (double*)&vdata.pvec[0];
    h0 = h + D;
    h1 = h0 + D;
  }

  rbm_user & operator=(vertex_data & data){
    h = &data.pvec[0];
    h0 = h + D;
    h1 = h0 + D;
    return * this;
  }   
};


/**
 * bi = pvec = rbm_bins * DOUBLE 
 * w = weight = rbm_bins * D * Double
 */
struct rbm_movie{
  double * bi;
  double * w;

  rbm_movie(const vertex_data& vdata){
    bi = (double*)&vdata.pvec[0];
    w = bi + rbm_bins;
  }

  rbm_movie & operator=(vertex_data & data){
    bi = (double*)&data.pvec[0];
    w = bi + rbm_bins;
    return * this;
  }
};




float rbm_predict(const rbm_user & usr, 
    const rbm_movie & mov, 
    const float rating, 
    double & prediction, 
    void * extra){

  float ret = 0;
  double nn = 0;
  for(int r = 0; r < rbm_bins; ++r){               
    double zz = exp(mov.bi[r] + dot(usr.h, &mov.w[r*D]));
    if (std::isinf(zz))
      std::cout<<" mov.bi[r] " << mov.bi[r] << " dot: " << dot(usr.h, &mov.w[r*D]) << std::endl;
    ret += zz * (float)(r);
    assert(!std::isnan(ret));
    nn += zz;
  }
  assert(!std::isnan(ret));
  assert(std::fabs(nn) > 1e-32);
  ret /= nn;
  if(ret < minval) ret = minval;
  else if(ret > maxval) ret = maxval;
  assert(!std::isnan(ret));
  prediction = ret * rbm_scaling;
  assert(!std::isnan(prediction));
  return pow(prediction - rating,2);
}

float rbm_predict(const vertex_data & usr, 
    const vertex_data & mov, 
    const float rating, 
    double & prediction, 
    void * extra){
  return rbm_predict(rbm_user((vertex_data&)usr), rbm_movie((vertex_data&)mov), rating, prediction, NULL);
}   

float predict1(const rbm_user & usr, 
    const rbm_movie & mov, 
    const float rating, 
    double & prediction){

  vec zz = zeros(rbm_bins);
  float szz = 0;
  for(int r = 0; r < rbm_bins; ++r){
    zz[r] = exp(mov.bi[r] + dot(usr.h0, &mov.w[r*D]));
    szz += zz[r];
  }
  float rd = drand48() * szz;
  szz = 0;
  int ret = 0;
  for(int r = 0; r < rbm_bins; ++r){
    szz += zz[r];
    if(rd < szz){ 
      ret = r;
      break;
    }
  }
  prediction = ret * rbm_scaling;
  assert(!std::isnan(prediction));
  return pow(prediction - rating, 2);
}

inline float sigmoid(float x){
  return 1 / (1 + exp(-1 * x));
}

#include "util.hpp"

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef float EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
graphchi_engine<VertexDataType, EdgeDataType> * pvalidation_engine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

#include "rmse.hpp"
#include "rmse_engine.hpp"
#include "io.hpp"



/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct RBMVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {
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
    rbm_alpha *= rbm_mult_step_dec;
    training_rmse(iteration, gcontext);
    if (iteration >= 2)
      run_validation(pvalidation_engine, gcontext);
    else std::cout<<std::endl;
  }

  /**
   *  Vertex update function.
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
        

    if (gcontext.iteration == 0){
      if (is_user(vertex.id()) && vertex.num_outedges() > 0){
        vertex_data& user = latent_factors_inmem[vertex.id()];
        user.pvec = zeros(D*3);
        for(int e=0; e < vertex.num_outedges(); e++) {
          rbm_movie mov = latent_factors_inmem[vertex.edge(e)->vertex_id()];
          float observation = vertex.edge(e)->get_data();                
          int r = (int)(observation/rbm_scaling);
          assert(r < rbm_bins);
          mov.bi[r]++;
        }
      }
      return;
    }
    else if (gcontext.iteration == 1){
      if (vertex.num_inedges() > 0){
        rbm_movie mov = latent_factors_inmem[vertex.id()]; 
        setRand2(mov.w, D*rbm_bins, 0.001);
        for(int r = 0; r < rbm_bins; ++r){
          mov.bi[r] /= (double)vertex.num_inedges();
          mov.bi[r] = log(1E-9 + mov.bi[r]);
         
          if (mov.bi[r] > 1000){
            assert(false);
            logstream(LOG_FATAL)<<"Numerical overflow" <<std::endl;
          }
        }
      }

      return; //done with initialization
    }
    //go over all user nodes
    if (is_user(vertex.id()) && vertex.num_outedges()){
      vertex_data & user = latent_factors_inmem[vertex.id()]; 
      user.pvec = zeros(3*D);
      rbm_user usr(user);

      vec v1 = zeros(vertex.num_outedges()); 
      //go over all ratings
      for(int e=0; e < vertex.num_outedges(); e++) {
        float observation = vertex.edge(e)->get_data();                
        rbm_movie mov = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        int r = (int)(observation / rbm_scaling);
        assert(r < rbm_bins);  
        for(int k=0; k < D; k++){
          usr.h[k] += mov.w[D*r + k];
          assert(!std::isnan(usr.h[k]));
        }
      }

      for(int k=0; k < D; k++){
        usr.h[k] = sigmoid(usr.h[k]);
        if (drand48() < usr.h[k]) 
          usr.h0[k] = 1;
        else usr.h0[k] = 0;
      }


      int i = 0;
      double prediction;
      for(int e=0; e < vertex.num_outedges(); e++) {
        rbm_movie mov = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        float observation = vertex.edge(e)->get_data();
        predict1(usr, mov, observation, prediction);    
        int vi = (int)(prediction / rbm_scaling);
        v1[i] = vi;
        i++;
      }

      i = 0;
      for(int e=0; e < vertex.num_outedges(); e++) {
        rbm_movie mov = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        int r = (int)v1[i];
        for (int k=0; k< D;k++){
          usr.h1[k] += mov.w[r*D+k];
        }
        i++;
      }

      for (int k=0; k < D; k++){
        usr.h1[k] = sigmoid(usr.h1[k]);
        if (drand48() < usr.h1[k]) 
          usr.h1[k] = 1;
        else usr.h1[k] = 0;
      }

      i = 0;
      for(int e=0; e < vertex.num_outedges(); e++) {
        rbm_movie mov = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        float observation = vertex.edge(e)->get_data();
        double prediction;
        rbm_predict(user, mov, observation, prediction, NULL);
        double pui = prediction / rbm_scaling;
        double rui = observation / rbm_scaling;
        rmse_vec[omp_get_thread_num()] += (pui - rui) * (pui - rui);
        //nn += 1.0;
        int vi0 = (int)(rui);
        int vi1 = (int)v1[i];
        for (int k = 0; k < D; k++){
          mov.w[D*vi0+k] += rbm_alpha * (usr.h0[k] - rbm_beta * mov.w[vi0*D+k]);
          assert(!std::isnan(mov.w[D*vi0+k]));
          mov.w[D*vi1+k] -= rbm_alpha * (usr.h1[k] + rbm_beta * mov.w[vi1*D+k]);
          assert(!std::isnan(mov.w[D*vi1+k]));
        }
        i++; 
      }
    }
  }    
};


//dump output to file
void output_rbm_result(std::string filename) {
  MMOutputter_mat<vertex_data> user_mat(filename + "_U.mm", 0, M, "This file contains RBM output matrix U. In each row D factors of a single user node.", latent_factors_inmem);
  MMOutputter_mat<vertex_data> mmoutput_right(filename + "_V.mm", M ,M+N,  "This file contains RBM  output matrix V. In each row D factors of a single item node.", latent_factors_inmem);
  logstream(LOG_INFO) << "RBM output files (in matrix market format): " << filename << "_U.mm" <<
                                                                           ", " << filename + "_V.mm " << std::endl;
}

void rbm_init(){
  srand48(time(NULL));

  latent_factors_inmem.resize(M+N);

#pragma omp parallel for
  for(int i = 0; i < (int)N; ++i){
    vertex_data & movie = latent_factors_inmem[M+i];
    movie.pvec = zeros(rbm_bins + D * rbm_bins);
    movie.bias = 0;
  }

  logstream(LOG_INFO) << "RBM initialization ok" << std::endl;

}
int main(int argc, const char ** argv) {

  print_copyright();

  //* GraphChi initialization will read the command line arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("rbm-inmemory-factors");

  /* Basic arguments for RBM algorithm */
  rbm_bins      = get_option_int("rbm_bins", rbm_bins);
  rbm_alpha     = get_option_float("rbm_alpha", rbm_alpha);
  rbm_beta      = get_option_float("rbm_beta", rbm_beta);
  rbm_mult_step_dec  = get_option_float("rbm_mult_step_dec", rbm_mult_step_dec);
  rbm_scaling   = get_option_float("rbm_scaling", rbm_scaling);

  parse_command_line_args();
  parse_implicit_command_line();

  mytimer.start();

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket<float>(training);

  rbm_init();

  if (validation != ""){
    int vshards = convert_matrixmarket<EdgeDataType>(validation, 0, 0, 3, VALIDATION);
    init_validation_rmse_engine<VertexDataType, EdgeDataType>(pvalidation_engine, vshards, &rbm_predict);
  }

  /* load initial state from disk (optional) */
  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, 3*D);
    load_matrix_market_matrix(training + "_V.mm", M, rbm_bins*(D+1));
   }

  print_config();

  /* Run */
  RBMVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output latent factor matrices in matrix-market format */
  output_rbm_result(training);
  test_predictions(&rbm_predict);    


  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  return 0;
}

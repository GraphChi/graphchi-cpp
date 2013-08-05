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
 * Matrix factorization with the Bias Stochastic Gradient Descent (BIASSGD) algorithm.
 * Algorithm is described in the paper:
 * Y. Koren. Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model. ACM SIGKDD 2008. Equation (5).
 * Thanks to Zeno Gantner, MyMediaLight for teaching me how to compute the derivative in case of logistic and absolute loss.
 * http://mymedialite.net/
 */


#include "common.hpp"
#include "eigen_wrapper.hpp"

double biassgd_lambda = 1e-3; //sgd step size
double biassgd_gamma = 1e-3;  //sgd regularization
double biassgd_step_dec = 0.9; //sgd step decrement
#define BIAS_POS -1

struct vertex_data {
  vec pvec; //storing the feature vector
  double bias;

  vertex_data() {
    pvec = zeros(D);
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

/** compute a missing value based on bias-SGD algorithm */
float bias_sgd_predict(const vertex_data& user, 
    const vertex_data& movie, 
    const float rating, 
    double & prediction, 
    void * extra = NULL){


  prediction = globalMean/maxval + user.bias + movie.bias + dot_prod(user.pvec, movie.pvec); 

  double exp_prediction = 1.0 / (1.0 + exp(-prediction));
  //truncate prediction to allowed values
  prediction = minval + exp_prediction *(maxval-minval);
  //return the squared error
  float err = rating - prediction;
  if (std::isnan(err))
    logstream(LOG_FATAL)<<"Got into numerical errors. Try to decrease step size using bias-SGD command line arugments)" << std::endl;

  if (extra != NULL)
    *(double*)extra = exp_prediction;

  return calc_loss(exp_prediction, err); 

}



/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct BIASSGDVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {


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
    biassgd_gamma *= biassgd_step_dec;
    training_rmse(iteration, gcontext);
    run_validation(pvalidation_engine, gcontext);
  }

  /**
   *  Vertex update function.
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
    //user node
    if ( vertex.num_outedges() > 0){
      vertex_data & user = latent_factors_inmem[vertex.id()]; 
      for(int e=0; e < vertex.num_edges(); e++) {
        float observation = vertex.edge(e)->get_data();                
        vertex_data & movie = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        double prediction;
        double exp_prediction;
        rmse_vec[omp_get_thread_num()] += bias_sgd_predict(user, movie, observation, prediction, &exp_prediction);
        double err = observation - prediction;
        err = calc_error_f(exp_prediction, err);

        if (std::isnan(err) || std::isinf(err))
          logstream(LOG_FATAL)<<"BIASSGD got into numerical error. Please tune step size using --biassgd_gamma and biassgd_lambda" << std::endl;

        user.bias += biassgd_gamma*(err - biassgd_lambda* user.bias);
        movie.bias += biassgd_gamma*(err - biassgd_lambda* movie.bias); 
        //NOTE: the following code is not thread safe, since potentially several
        //user nodes may update this item gradient vector concurrently. However in practice it
        //did not matter in terms of accuracy on a multicore machine.
        //if you like to defend the code, you can define a global variable
        //mutex mymutex;
        //
        //and then do: mymutex.lock()
        movie.pvec += biassgd_gamma*(err*user.pvec - biassgd_lambda*movie.pvec);
        //here add: mymutex.unlock();
        user.pvec += biassgd_gamma*(err*movie.pvec - biassgd_lambda*user.pvec);
      }
    }
  }
};

void output_biassgd_result(std::string filename){
  MMOutputter_mat<vertex_data> user_mat(filename + "_U.mm", 0, M, "This file contains bias-SGD output matrix U. In each row D factors of a single user node.", latent_factors_inmem);
  MMOutputter_mat<vertex_data> item_mat(filename + "_V.mm", M, M+N , "This file contains bias-SGD  output matrix V. In each row D factors of a single item node.", latent_factors_inmem);
  MMOutputter_vec<vertex_data> user_bias_vec(filename + "_U_bias.mm", 0, M, BIAS_POS, "This file contains bias-SGD output bias vector. In each row a single user bias.",latent_factors_inmem);
  MMOutputter_vec<vertex_data> item_bias_vec(filename + "_V_bias.mm",M ,M+N, BIAS_POS, "This file contains bias-SGD output bias vector. In each row a single item bias.", latent_factors_inmem);
  MMOutputter_scalar gmean(filename + "_global_mean.mm", "This file contains SVD++ global mean which is required for computing predictions.", globalMean);

  logstream(LOG_INFO) << "SVDPP output files (in matrix market format): " << filename << "_U.mm" <<
                                                                             ", " << filename + "_V.mm, " << filename << "_U_bias.mm, " << filename << "_V_bias.mm, " << filename << "_global_mean.mm" << std::endl;
}


int main(int argc, const char ** argv) {
  print_copyright();

  //* GraphChi initialization will read the command line arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("biassgd2");

  biassgd_lambda    = get_option_float("biassgd_lambda", 1e-3);
  biassgd_gamma     = get_option_float("biassgd_gamma", 1e-3);
  biassgd_step_dec  = get_option_float("biassgd_step_dec", 0.9);
  parse_command_line_args();
  parse_implicit_command_line();

  if (maxval == 1e100 || minval == -1e100)
    logstream(LOG_FATAL)<<"You must set min allowed rating and max allowed rating using the --minval and --maval flags" << std::endl;

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket<EdgeDataType>(training, 0, 0, 3, TRAINING, false);
  init_feature_vectors<std::vector<vertex_data> >(M+N, latent_factors_inmem, !load_factors_from_file);
  if (validation != ""){
    int vshards = convert_matrixmarket<EdgeDataType>(validation, 0, 0, 3, VALIDATION, false);
    init_validation_rmse_engine<VertexDataType, EdgeDataType>(pvalidation_engine, vshards, &bias_sgd_predict);
  }

  /* load initial state from disk (optional) */
  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, D);
    load_matrix_market_matrix(training + "_V.mm", M, D);
    vec user_bias = load_matrix_market_vector(training +"_U_bias.mm", false, true);
    vec item_bias = load_matrix_market_vector(training +"_V_bias.mm", false, true);
    for (uint i=0; i<M+N; i++){
      latent_factors_inmem[i].bias = ((i<M)?user_bias[i] : item_bias[i-M]);
    }
    vec gm = load_matrix_market_vector(training + "_global_mean.mm", false, true);
    globalMean = gm[0];
  }


  /* Run */
  BIASSGDVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output latent factor matrices in matrix-market format */
  output_biassgd_result(training);
  test_predictions(&bias_sgd_predict);    


  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  return 0;
}

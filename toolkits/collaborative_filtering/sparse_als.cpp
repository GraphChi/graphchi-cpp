/* 
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
 * Matrix factorizatino with the Alternative Least Squares (ALS) algorithm
 * using sparse factors. Sparsity is obtained using the CoSaMP algorithm.
 *
 * 
 */


#include "cosamp.hpp"
#include "eigen_wrapper.hpp"
#include "common.hpp"

double lambda = 0.065;

struct vertex_data {
  vec pvec;

  vertex_data() {
   pvec = zeros(D); 
  }
  void set_val(int index, float val){
    pvec[index] = val;
  }
  float get_val(int index){
    return pvec[index];
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

#include "io.hpp"

//algorithm run mode
enum {
  SPARSE_USR_FACTOR = 1, SPARSE_ITM_FACTOR = 2, SPARSE_BOTH_FACTORS = 3
};

int algorithm;
double user_sparsity;
double movie_sparsity;

#include "rmse.hpp"
#include "rmse_engine.hpp"

/** compute a missing value based on ALS algorithm */
float sparse_als_predict(const vertex_data& user, 
    const vertex_data& movie, 
    const float rating, 
    double & prediction, 
    void * extra = NULL){


  prediction = user.pvec.dot(movie.pvec);
  //truncate prediction to allowed values
  prediction = std::min((double)prediction, maxval);
  prediction = std::max((double)prediction, minval);
  //return the squared error
  float err = rating - prediction;
  assert(!std::isnan(err));
  return err*err; 

}




/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct ALSVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {


  /**
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
    vertex_data & vdata = latent_factors_inmem[vertex.id()];
    mat XtX = mat::Zero(D, D); 
    vec Xty = vec::Zero(D);

    bool compute_rmse = (vertex.num_outedges() > 0);
    // Compute XtX and Xty (NOTE: unweighted)
    for(int e=0; e < vertex.num_edges(); e++) {
      float observation = vertex.edge(e)->get_data();                
      vertex_data & nbr_latent = latent_factors_inmem[vertex.edge(e)->vertex_id()];
      Xty += nbr_latent.pvec * observation;
      XtX += nbr_latent.pvec * nbr_latent.pvec.transpose();
      if (compute_rmse) {
        double prediction;
        rmse_vec[omp_get_thread_num()] += sparse_als_predict(vdata, nbr_latent, observation, prediction);
      }
    }

    double regularization = lambda;
    if (regnormal)
      lambda *= vertex.num_edges();
    for(int i=0; i < D; i++) XtX(i,i) += regularization;


    bool isuser = vertex.id() < (uint)M;
    if (algorithm == SPARSE_BOTH_FACTORS || (algorithm == SPARSE_USR_FACTOR && isuser) || 
        (algorithm == SPARSE_ITM_FACTOR && !isuser)){ 
      double sparsity_level = 1.0;
      if (isuser)
        sparsity_level -= user_sparsity;
      else sparsity_level -= movie_sparsity;
      vdata.pvec = CoSaMP(XtX, Xty, (int)ceil(sparsity_level*(double)D), 10, 1e-4, D); 
    }
    else vdata.pvec = XtX.selfadjointView<Eigen::Upper>().ldlt().solve(Xty);
  }

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
    training_rmse(iteration, gcontext);
    run_validation(pvalidation_engine, gcontext);
  }


};

void output_als_result(std::string filename) {
  MMOutputter_mat<vertex_data> user_mat(filename + "_U.mm", 0, M, "This file contains ALS output matrix U. In each row D factors of a single user node.", latent_factors_inmem);
  MMOutputter_mat<vertex_data>  item_mat(filename + "_V.mm", M, M+N, "This file contains ALS  output matrix V. In each row D factors of a single item node.", latent_factors_inmem);
  logstream(LOG_INFO) << "ALS output files (in matrix market format): " << filename << "_U.mm" <<
                                                                           ", " << filename + "_V.mm " << std::endl;
}

int main(int argc, const char ** argv) {

  print_copyright();
 
  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("als-inmemory-factors");

  lambda        = get_option_float("lambda", 0.065);
  user_sparsity = get_option_float("user_sparsity", 0.9);
  movie_sparsity = get_option_float("movie_sparsity", 0.9);
  algorithm      = get_option_int("algorithm", SPARSE_USR_FACTOR);

  parse_command_line_args();
  parse_implicit_command_line(); 

  if (user_sparsity < 0.5 || user_sparsity >= 1)
    logstream(LOG_FATAL)<<"Sparsity level should be [0.5,1). Please run again using --user_sparsity=XX in this range" << std::endl;

  if (movie_sparsity < 0.5 || movie_sparsity >= 1)
    logstream(LOG_FATAL)<<"Sparsity level should be [0.5,1). Please run again using --movie_sparsity=XX in this range" << std::endl;

if (algorithm != SPARSE_USR_FACTOR && algorithm != SPARSE_BOTH_FACTORS && algorithm != SPARSE_ITM_FACTOR)
    logstream(LOG_FATAL)<<"Algorithm should be 1 for SPARSE_USR_FACTOR, 2 for SPARSE_ITM_FACTOR and 3 for SPARSE_BOTH_FACTORS" << std::endl;

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket<EdgeDataType>(training);
  init_feature_vectors<std::vector<vertex_data> >(M+N, latent_factors_inmem, !load_factors_from_file);
  if (validation != ""){
    int vshards = convert_matrixmarket<EdgeDataType>(validation, 0, 0, 3, VALIDATION);
    init_validation_rmse_engine<VertexDataType, EdgeDataType>(pvalidation_engine, vshards, &sparse_als_predict);
  }
 

  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, D);
    load_matrix_market_matrix(training + "_V.mm", M, D);
  }

  /* Run */
  ALSVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output latent factor matrices in matrix-market format */
  output_als_result(training);
  test_predictions(&sparse_als_predict);    

  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  return 0;
}

/**
 * @file
 * @author  Danny Bickson, based on code by Aapo Kyrola
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
 * Matrix factorizatino with the Alternative Least Squares (ALS) algorithm.
 * This code is based on GraphLab's implementation of ALS by Joey Gonzalez
 * and Danny Bickson (CMU). A good explanation of the algorithm is 
 * given in the following paper:
 *    Large-Scale Parallel Collaborative Filtering for the Netflix Prize
 *    Yunhong Zhou, Dennis Wilkinson, Robert Schreiber and Rong Pan
 *    http://www.springerlink.com/content/j1076u0h14586183/
 *
 * Faster version of ALS, which stores latent factors of vertices in-memory.
 * Thus, this version requires more memory. See the version "als_edgefactors"
 * for a low-memory implementation.
 *
 *
 * In the code, we use movie-rating terminology for clarity. This code has been
 * tested with the Netflix movie rating challenge, where the task is to predict
 * how user rates movies in range from 1 to 5.
 *
 * This code is has integrated preprocessing, 'sharding', so it is not necessary
 * to run sharder prior to running the matrix factorization algorithm. Input
 * data must be provided in the Matrix Market format (http://math.nist.gov/MatrixMarket/formats.html).
 *
 * ALS uses free linear algebra library 'Eigen'. See Readme_Eigen.txt for instructions
 * how to obtain it.
 *
 * At the end of the processing, the two latent factor matrices are written into files in 
 * the matrix market format. 
 *
 * @section USAGE
 *
 * bin/example_apps/matrix_factorization/als_edgefactors file <matrix-market-input> niters 5
 *
 * 
 */



#include <string>
#include <algorithm>

#include "graphchi_basic_includes.hpp"

#include <assert.h>
#include <cmath>
#include <errno.h>
#include <string>
#include <stdint.h>

#include "../../example_apps/matrix_factorization/matrixmarket/mmio.h"
#include "../../example_apps/matrix_factorization/matrixmarket/mmio.c"

#include "api/chifilenames.hpp"
#include "api/vertex_aggregator.hpp"
#include "preprocessing/sharder.hpp"

#include "eigen_wrapper.hpp"
using namespace graphchi;

#ifndef NLATENT
#define NLATENT 20   // Dimension of the latent factors. You can specify this in compile time as well (in make).
#endif

double lambda = 0.065;
double minval = -1e100;
double maxval = 1e100;
std::string training;
std::string validation;
std::string test;
uint M, N, K;
size_t L;
uint Me, Ne, Le;
double globalMean = 0;
/// RMSE computation
double rmse=0.0;
bool load_factors_from_file = false;

bool is_user(vid_t id){ return id < M; }
bool is_item(vid_t id){ return id >= M && id < N; }
bool is_time(vid_t id){ return id >= M+N; }

struct vertex_data {
  double pvec[NLATENT];
  double rmse;

  vertex_data() {
    for(int k=0; k < NLATENT; k++) pvec[k] =  drand48(); 
    rmse = 0;
  }

  double dot(const vertex_data &oth, const vertex_data time) const {
    double x=0;
    for(int i=0; i<NLATENT; i++) x+= oth.pvec[i]*pvec[i]*time.pvec[i];
    return x;
  }

};

struct edge_data {
  double weight;
  double time;

  edge_data() { weight = time = 0; }

  edge_data(double weight, double time) : weight(weight), time(time) { }
};


/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef edge_data EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

#include "io.hpp"
#include "rmse.hpp"

/** compute a missing value based on ALS algorithm */
float als_tensor_predict(const vertex_data& user, 
    const vertex_data& movie, 
    const vertex_data& time_node,
    const float rating, 
    double & prediction){


  prediction = user.dot(movie, time_node);
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
   * Called before an iteration starts.
   */
  void before_iteration(int iteration, graphchi_context &gcontext) {
  }

  /**
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
    vertex_data & vdata = latent_factors_inmem[vertex.id()];
    vdata.rmse = 0;
    mat XtX = mat::Zero(NLATENT, NLATENT); 
    vec Xty = vec::Zero(NLATENT);

    bool compute_rmse = is_user(vertex.id()); 
    // Compute XtX and Xty (NOTE: unweighted)
    for(int e=0; e < vertex.num_edges(); e++) {
      float observation = vertex.edge(e)->get_data().weight;                
      uint time = (uint)vertex.edge(e)->get_data().time;
      vertex_data & nbr_latent = latent_factors_inmem[vertex.edge(e)->vertex_id()];
      vertex_data & time_node = latent_factors_inmem[time];
      assert(time != vertex.id() && time != vertex.edge(e)->vertex_id());
      Map<vec> X(nbr_latent.pvec, NLATENT);
      Map<vec> Y(time_node.pvec, NLATENT);
      vec XY = X.cwiseProduct(Y);
      Xty += XY * observation;
      XtX.triangularView<Eigen::Upper>() += XY * XY.transpose();
      if (compute_rmse) {
        double prediction;
        vdata.rmse += als_tensor_predict(vdata, nbr_latent, time_node, observation, prediction);
      }
    }

    for(int i=0; i < NLATENT; i++) XtX(i,i) += (lambda); // * vertex.num_edges();

    // Solve the least squares problem with eigen using Cholesky decomposition
    Map<vec> vdata_vec(vdata.pvec, NLATENT);
    vdata_vec = XtX.selfadjointView<Eigen::Upper>().ldlt().solve(Xty);
  }



  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    training_rmse(iteration, gcontext);
    validation_rmse3(&als_tensor_predict, gcontext);
  }

  /**
   * Called before an execution interval is started.
   */
  void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
  }

  /**
   * Called after an execution interval has finished.
   */
  void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
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
    mm_write_mtx_array_size(outf, end-start, NLATENT); 
    for (uint i=start; i < end; i++)
      for(int j=0; j < NLATENT; j++) {
        fprintf(outf, "%1.12e\n", latent_factors_inmem[i].pvec[j]);
      }
  }

  ~MMOutputter() {
    if (outf != NULL) fclose(outf);
  }

};


void output_als_result(std::string filename) {
  MMOutputter mmoutput_left(filename + "_U.mm", 0, M, "This file contains tensor-ALS output matrix U. In each row NLATENT factors of a single user node.");
  MMOutputter mmoutput_right(filename + "_V.mm", M ,M+N, "This file contains tensor-ALS  output matrix V. In each row NLATENT factors of a single item node.");
  MMOutputter mmoutput_time(filename + "_T.mm", M+N ,M+N+K, "This file contains tensor-ALS  output matrix T. In each row NLATENT factors of a single time node.");
  logstream(LOG_INFO) << "tensor - ALS output files (in matrix market format): " << filename << "_U.mm" <<
                                                                           ", " << filename + "_V.mm " << filename + "_T.mm" << std::endl;
}

int main(int argc, const char ** argv) {


  logstream(LOG_WARNING)<<"GraphChi Collaborative filtering library is written by Danny Bickson (c). Send any "
    " comments or bug reports to danny.bickson@gmail.com " << std::endl;
  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("als-inmemory-factors");

  /* Basic arguments for application. NOTE: File will be automatically 'sharded'. */
  int unittest = get_option_int("unittest", 0);
  int niters    = get_option_int("max_iter", 6);  // Number of iterations
  if (unittest > 0)
    training = get_option_string("training", "");    // Base filename
  else training = get_option_string("training");
  if (unittest == 1){
    if (training == "") training = "test_als"; 
    niters = 100;
  }
  validation = get_option_string("validation", "");
  test = get_option_string("test", "");

  if (validation == "")
    validation += training + "e";  
  if (test == "")
    test += training + "t";

  maxval        = get_option_float("maxval", 1e100);
  minval        = get_option_float("minval", -1e100);
  lambda        = get_option_float("lambda", 0.065);
  bool quiet    = get_option_int("quiet", 0);
  if (quiet)
    global_logger().set_log_level(LOG_ERROR);
  halt_on_rmse_increase = get_option_int("halt_on_rmse_increase", 0);
  load_factors_from_file = get_option_int("load_factors_from_file", 0);

  parse_implicit_command_line();

  bool scheduler       = false;                        // Selective scheduling not supported for now.


  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket4<edge_data>(training, true);
  latent_factors_inmem.resize(M+N+K); // Initialize in-memory vertices.
  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, NLATENT);
    load_matrix_market_matrix(training + "_V.mm", M, NLATENT);
    load_matrix_market_matrix(training + "_T.mm", M+N, NLATENT);
  }


  /* Run */
  ALSVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, scheduler, m); 
  engine.set_disable_vertexdata_storage();  
  engine.set_enable_deterministic_parallelism(false);
  engine.set_modifies_inedges(false);
  engine.set_modifies_outedges(false);
  pengine = &engine;
  engine.run(program, niters);

  m.set("train_rmse", rmse);
  m.set("latent_dimension", NLATENT);

  /* Output test predictions in matrix-market format */
  output_als_result(training);
  test_predictions3(&als_tensor_predict);    

  if (unittest == 1){
    if (dtraining_rmse > 0.03)
      logstream(LOG_FATAL)<<"Unit test 1 failed. Training RMSE is: " << training_rmse << std::endl;
    if (dvalidation_rmse > 1.03)
      logstream(LOG_FATAL)<<"Unit test 1 failed. Validation RMSE is: " << dvalidation_rmse << std::endl;

  }
  
  /* Report execution metrics */
  metrics_report(m);
  return 0;
}

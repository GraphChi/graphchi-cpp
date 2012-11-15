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
 * using sparse factors.
 *
 * 
 */



#include <string>
#include <algorithm>

#include "graphchi_basic_includes.hpp"

/* ALS-related classes are contained in als.hpp */
#include "als.hpp"
#include "cosamp.hpp"

using namespace graphchi;


/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef float EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
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

/** compute a missing value based on ALS algorithm */
float sparse_als_predict(const vertex_data& user, 
    const vertex_data& movie, 
    const float rating, 
    double & prediction){


  prediction = user.dot(movie);
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
    vdata.rmse = 0;
    mat XtX = mat::Zero(NLATENT, NLATENT); 
    vec Xty = vec::Zero(NLATENT);

    bool compute_rmse = (vertex.num_outedges() > 0);
    // Compute XtX and Xty (NOTE: unweighted)
    for(int e=0; e < vertex.num_edges(); e++) {
      float observation = vertex.edge(e)->get_data();                
      vertex_data & nbr_latent = latent_factors_inmem[vertex.edge(e)->vertex_id()];
      Map<vec> X(nbr_latent.pvec, NLATENT);
      Xty += X * observation;
      XtX += X * X.transpose();
      if (compute_rmse) {
        double prediction;
        vdata.rmse += sparse_als_predict(vdata, nbr_latent, observation, prediction);
      }
    }

    for(int i=0; i < NLATENT; i++) XtX(i,i) += (lambda); // * vertex.num_edges();

    bool isuser = vertex.id() < (uint)M;
    Map<vec> vdata_vec(vdata.pvec, NLATENT);
    if (algorithm == SPARSE_BOTH_FACTORS || (algorithm == SPARSE_USR_FACTOR && isuser) || 
        (algorithm == SPARSE_ITM_FACTOR && !isuser)){ 
      double sparsity_level = 1.0;
      if (isuser)
        sparsity_level -= user_sparsity;
      else sparsity_level -= movie_sparsity;
      vdata_vec = CoSaMP(XtX, Xty, (int)ceil(sparsity_level*(double)NLATENT), 10, 1e-4, NLATENT); 
    }
    else vdata_vec = XtX.selfadjointView<Eigen::Upper>().ldlt().solve(Xty);
  }



  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    training_rmse(iteration, gcontext);
    validation_rmse(&sparse_als_predict, gcontext);
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


void output_als_result(std::string filename, vid_t numvertices, vid_t max_left_vertex) {
  MMOutputter mmoutput_left(filename + "_U.mm", 0, max_left_vertex, "This file contains ALS output matrix U. In each row NLATENT factors of a single user node.");
  MMOutputter mmoutput_right(filename + "_V.mm", max_left_vertex ,numvertices, "This file contains ALS  output matrix V. In each row NLATENT factors of a single item node.");
  logstream(LOG_INFO) << "ALS output files (in matrix market format): " << filename << "_U.mm" <<
                                                                           ", " << filename + "_V.mm " << std::endl;
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
  training = get_option_string("training");    // Base filename
  validation = get_option_string("validation", "");
  test = get_option_string("test", "");

  if (validation == "")
    validation += training + "e";  
  if (test == "")
    test += training + "t";

  int niters    = get_option_int("max_iter", 6);  // Number of iterations
  maxval        = get_option_float("maxval", 1e100);
  minval        = get_option_float("minval", -1e100);
  lambda        = get_option_float("lambda", 0.065);
  user_sparsity = get_option_float("user_sparsity", 0.9);
  movie_sparsity = get_option_float("movie_sparsity", 0.9);
  algorithm      = get_option_int("algorithm", SPARSE_USR_FACTOR);

  bool quiet    = get_option_int("quiet", 0);
  if (quiet)
    global_logger().set_log_level(LOG_ERROR);
  halt_on_rmse_increase = get_option_int("halt_on_rmse_increase", 0);
  load_factors_from_file = get_option_int("load_factors_from_file", 0);

  parse_implicit_command_line(); 

  if (user_sparsity < 0.5 || user_sparsity >= 1)
    logstream(LOG_FATAL)<<"Sparsity level should be [0.5,1). Please run again using --user_sparsity=XX in this range" << std::endl;

  if (movie_sparsity < 0.5 || movie_sparsity >= 1)
    logstream(LOG_FATAL)<<"Sparsity level should be [0.5,1). Please run again using --movie_sparsity=XX in this range" << std::endl;

if (algorithm != SPARSE_USR_FACTOR && algorithm != SPARSE_BOTH_FACTORS && algorithm != SPARSE_ITM_FACTOR)
    logstream(LOG_FATAL)<<"Algorithm should be 1 for SPARSE_USR_FACTOR, 2 for SPARSE_ITM_FACTOR and 3 for SPARSE_BOTH_FACTORS" << std::endl;

  bool scheduler       = false;                        // Selective scheduling not supported for now.

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket<float>(training);
 latent_factors_inmem.resize(M+N); // Initialize in-memory vertices.
  assert(M > 0 && N > 0);

  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, NLATENT);
    load_matrix_market_matrix(training + "_V.mm", M, NLATENT);
  }


  /* Run */
  ALSVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, scheduler, m); 
  engine.set_enable_deterministic_parallelism(false);
  engine.set_modifies_inedges(false);
  engine.set_modifies_outedges(false);
  engine.set_disable_vertexdata_storage();  
  pengine = &engine;
  engine.run(program, niters);

  m.set("train_rmse", rmse);
  m.set("latent_dimension", NLATENT);

  /* Output latent factor matrices in matrix-market format */
  output_als_result(training, M+N, M);
  test_predictions(&sparse_als_predict);    

  /* Report execution metrics */
  metrics_report(m);
  return 0;
}

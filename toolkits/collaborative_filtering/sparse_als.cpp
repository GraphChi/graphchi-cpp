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

#include "cosamp.hpp"
#include "eigen_wrapper.hpp"
#include "common.hpp"

using namespace graphchi;

double lambda = 0.065;

struct vertex_data {
  double pvec[NLATENT];
  double rmse;

  vertex_data() {
    for(int k=0; k < NLATENT; k++) pvec[k] =  drand48(); 
    rmse = 0;
  }

  double dot(const vertex_data &oth) const {
    double x=0;
    for(int i=0; i<NLATENT; i++) x+= oth.pvec[i]*pvec[i];
    return x;
  }

};



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
  MMOutputter mmoutput_left(filename + "_U.mm", 0, M, "This file contains ALS output matrix U. In each row NLATENT factors of a single user node.");
  MMOutputter mmoutput_right(filename + "_V.mm", M, M+N, "This file contains ALS  output matrix V. In each row NLATENT factors of a single item node.");
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
  int nshards = convert_matrixmarket<float>(training);
 latent_factors_inmem.resize(M+N); // Initialize in-memory vertices.
  assert(M > 0 && N > 0);

  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, NLATENT);
    load_matrix_market_matrix(training + "_V.mm", M, NLATENT);
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
  metrics_report(m);
  return 0;
}

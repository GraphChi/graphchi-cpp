/**
 * @file
 * @author  Danny Bickson, CMU
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

double minval = -1e100;
double maxval = 1e100;
std::string training;
std::string validation;
std::string test;
uint M, N, K;
size_t L;
uint Me, Ne, Le;
double globalMean = 0;
const double epsilon = 1e-16;

vid_t max_left_vertex =0 ;
vid_t max_right_vertex = 0;

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
vec x1, x2;
int iter;

#include "rmse.hpp"
#include "io.hpp"

/** compute a missing value based on NMF algorithm */
float nmf_predict(const vertex_data& user, 
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

void pre_user_iter(){
  x1 = zeros(NLATENT);
  for (uint i=M; i<M+N; i++){
    vertex_data & data = latent_factors_inmem[i];
    Map<vec> pvec(data.pvec, NLATENT);
    x1 += pvec;
  }
}
void pre_movie_iter(){

  x2 = zeros(NLATENT);
  for (uint i=0; i<M; i++){
    vertex_data & data = latent_factors_inmem[i];
    Map<vec> pvec(data.pvec, NLATENT);
    x2 += pvec;
  }
}




/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct NMFVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {



  /**
   * Called before an iteration starts.
   */
  void before_iteration(int iteration, graphchi_context &gcontext) {
    iter = iteration;
    if (iteration == 0) {
      latent_factors_inmem.resize(gcontext.nvertices); // Initialize in-memory vertices.
      assert(M > 0 && N > 0);
      max_left_vertex = M-1;
      max_right_vertex = M+N-1;
    }
    else {
      if (iteration % 2 == 1)
        pre_user_iter();
      else pre_movie_iter();
    }
  }

  /**
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {

    if (gcontext.iteration == 0){
      if (vertex.num_outedges() == 0 && vertex.id() < M)
        logstream(LOG_FATAL)<<"NMF algorithm can not work when the row " << vertex.id() << " of the matrix contains all zeros" << std::endl;
      for(int e=0; e < vertex.num_edges(); e++) {
        float observation = vertex.edge(e)->get_data();                
        if (observation < 0 ){
          logstream(LOG_FATAL)<<"Found a negative entry in matirx row " << vertex.id() << " with value: " << observation << std::endl;
        }
      }
      return;   
    }

    bool isuser = (vertex.id() < M);
    if ((iter % 2 == 1 && !isuser) ||
        (iter % 2 == 0 && isuser))
      return;
    
    vec ret = zeros(NLATENT);

    vertex_data & vdata = latent_factors_inmem[vertex.id()];
    Map<vec> pvec(vdata.pvec, NLATENT);
    vdata.rmse = 0;
    mat XtX = mat::Zero(NLATENT, NLATENT); 
    vec Xty = vec::Zero(NLATENT);

    bool compute_rmse = true;
    
    for(int e=0; e < vertex.num_edges(); e++) {
      float observation = vertex.edge(e)->get_data();                
      vertex_data & nbr_latent = latent_factors_inmem[vertex.edge(e)->vertex_id()];
      double prediction;
      if (compute_rmse)
        vdata.rmse += nmf_predict(vdata, nbr_latent, observation, prediction);
      if (prediction == 0)
        logstream(LOG_FATAL)<<"Got into numerical error! Please submit a bug report." << std::endl;
      Map<vec> nbr_pvec(nbr_latent.pvec, NLATENT);
      ret += nbr_pvec * (observation / prediction);
    }
    
    vec px;
    if (isuser)
      px = x1;
    else 
      px = x2;
    for (int i=0; i<NLATENT; i++){
      assert(px[i] != 0);
      pvec[i] *= ret[i] / px[i];
      if (pvec[i] < epsilon)
        pvec[i] = epsilon;
    }
  }



  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    //print rmse every other iteration, since 2 iterations are considered one NMF round
    int now = iteration % 2;
    if (now == 0){
      training_rmse(iteration/2);
      validation_rmse(&nmf_predict);
    }
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


void output_nmf_result(std::string filename, vid_t numvertices, vid_t max_left_vertex) {
  MMOutputter mmoutput_left(filename + "_U.mm", 0, max_left_vertex + 1, "This file contains NMF output matrix U. In each row NLATENT factors of a single user node.");
  MMOutputter mmoutput_right(filename + "_V.mm", max_left_vertex +1 ,numvertices, "This file contains NMF  output matrix V. In each row NLATENT factors of a single item node.");
  logstream(LOG_INFO) << "NMF output files (in matrix market format): " << filename << "_U.mm" <<
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
  metrics m("nmf-inmemory-factors");

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
  bool quiet    = get_option_int("quiet", 0);
  if (quiet)
    global_logger().set_log_level(LOG_ERROR);

  parse_implicit_command_line();

  niters *= 2; //each NMF iteration is composed of two sub iters

  bool scheduler       = false;                        // Selective scheduling not supported for now.

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket<float>(training);

  x1 = zeros(NLATENT);
  x2 = zeros(NLATENT);

  srand(time(NULL));

  /* Run */
  NMFVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, scheduler, m); 
  engine.set_modifies_inedges(false);
  engine.set_modifies_outedges(false);
  engine.set_disable_vertexdata_storage();
  pengine = &engine;
  engine.run(program, niters);

  m.set("latent_dimension", NLATENT);

  /* Output latent factor matrices in matrix-market format */
  vid_t numvertices = engine.num_vertices();
  assert(numvertices == max_right_vertex + 1); // Sanity check
  output_nmf_result(training, numvertices, max_left_vertex);
  test_predictions(&nmf_predict);    

  /* Report execution metrics */
  metrics_report(m);
  return 0;
}

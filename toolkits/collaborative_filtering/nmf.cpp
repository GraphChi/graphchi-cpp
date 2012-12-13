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

#include "common.hpp"
#include "api/chifilenames.hpp"
#include "api/vertex_aggregator.hpp"
#include "preprocessing/sharder.hpp"

#include "eigen_wrapper.hpp"
using namespace graphchi;

const double epsilon = 1e-16;
struct vertex_data {
  vec pvec;
  double rmse;

  vertex_data() {
   pvec = zeros(D); 
   rmse = 0;
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
    double & prediction, 
    void * extra = NULL){

  prediction = dot_prod(user.pvec, movie.pvec);
  //truncate prediction to allowed values
  prediction = std::min((double)prediction, maxval);
  prediction = std::max((double)prediction, minval);
  //return the squared error
  float err = rating - prediction;
  assert(!std::isnan(err));
  return err*err; 

}

void pre_user_iter(){
  x1 = zeros(D);
  for (uint i=M; i<M+N; i++){
    vertex_data & data = latent_factors_inmem[i];
    x1 += data.pvec;
  }
}
void pre_movie_iter(){

  x2 = zeros(D);
  for (uint i=0; i<M; i++){
    vertex_data & data = latent_factors_inmem[i];
    x2 += data.pvec;
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
    if (iteration > 0) {
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
    
    vec ret = zeros(D);

    vertex_data & vdata = latent_factors_inmem[vertex.id()];
    vdata.rmse = 0;
    mat XtX = mat::Zero(D, D); 
    vec Xty = vec::Zero(D);

    bool compute_rmse = true;
    
    for(int e=0; e < vertex.num_edges(); e++) {
      float observation = vertex.edge(e)->get_data();                
      vertex_data & nbr_latent = latent_factors_inmem[vertex.edge(e)->vertex_id()];
      double prediction;
      if (compute_rmse)
        vdata.rmse += nmf_predict(vdata, nbr_latent, observation, prediction);
      if (prediction == 0)
        logstream(LOG_FATAL)<<"Got into numerical error! Please submit a bug report." << std::endl;
      ret += nbr_latent.pvec * (observation / prediction);
    }
    
    vec px;
    if (isuser)
      px = x1;
    else 
      px = x2;
    for (int i=0; i<D; i++){
      assert(px[i] != 0);
      vdata.pvec[i] *= ret[i] / px[i];
      if (vdata.pvec[i] < epsilon)
        vdata.pvec[i] = epsilon;
    }
  }



  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    //print rmse every other iteration, since 2 iterations are considered one NMF round
    int now = iteration % 2;
    if (now == 0){
      training_rmse(iteration/2, gcontext);
      validation_rmse(&nmf_predict, gcontext);
    }
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
    mm_write_mtx_array_size(outf, end-start, D); 
    for (uint i=start; i < end; i++)
      for(int j=0; j < D; j++) {
        fprintf(outf, "%1.12e\n", latent_factors_inmem[i].pvec[j]);
      }
  }

  ~MMOutputter() {
    if (outf != NULL) fclose(outf);
  }

};



void output_nmf_result(std::string filename){
  MMOutputter mmoutput_left(filename + "_U.mm", 0, M, "This file contains NMF output matrix U. In each row D factors of a single user node.");
  MMOutputter mmoutput_right(filename + "_V.mm", M, M+N, "This file contains NMF  output matrix V. In each row D factors of a single item node.");
  logstream(LOG_INFO) << "NMF output files (in matrix market format): " << filename << "_U.mm" <<
                                                                           ", " << filename + "_V.mm " << std::endl;
}

int main(int argc, const char ** argv) {


  print_copyright(); 
 
  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("nmf-inmemory-factors");

  parse_command_line_args();
  parse_implicit_command_line();

  niters *= 2; //each NMF iteration is composed of two sub iters

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket<float>(training);
  init_feature_vectors<std::vector<vertex_data> >(M+N, latent_factors_inmem, !load_factors_from_file);

  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, D);
    load_matrix_market_matrix(training + "_V.mm", M, D);
  }

  x1 = zeros(D);
  x2 = zeros(D);


  /* Run */
  NMFVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output latent factor matrices in matrix-market format */
  output_nmf_result(training);
  test_predictions(&nmf_predict);    

  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  return 0;
}

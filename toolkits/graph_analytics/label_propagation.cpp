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
 * Implementation of the label propagation algorithm 
 */


#include "../collaborative_filtering/common.hpp"
#include "../collaborative_filtering/eigen_wrapper.hpp"

double alpha = 0.15;
int debug = 0;

struct vertex_data {
  vec pvec;
  bool seed;

  vertex_data() {
    pvec = zeros(D);
    seed = false;
  }
  //this function is only called for seed nodes
  void set_val(int index, float val){
    pvec[index] = val;
    seed = true;
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
std::vector<vertex_data> latent_factors_inmem;

#include "../collaborative_filtering/io.hpp"




/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct LPVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {



  /**
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
    vertex_data & vdata = latent_factors_inmem[vertex.id()];
    if (debug)
      logstream(LOG_DEBUG)<<"Entering node: " << vertex.id() << " seed? " << vdata.seed << " in vector: " << vdata.pvec << std::endl;
    if (vdata.seed || vertex.num_outedges() == 0) //if this is a seed node, don't do anything
      return;
    vec ret = zeros(D);

    for(int e=0; e < vertex.num_outedges(); e++) {
      float weight = vertex.edge(e)->get_data();                
      assert(weight != 0);
      vertex_data & nbr_latent = latent_factors_inmem[vertex.edge(e)->vertex_id()];
      ret += weight * nbr_latent.pvec;
    }

    //normalize probabilities
    assert(sum(ret) != 0);
    ret = ret / sum(ret);
    vdata.pvec = alpha * vdata.pvec + (1-alpha)*ret;
    vdata.pvec/= sum(vdata.pvec);
  }


};



void output_lp_result(std::string filename) {
  MMOutputter_mat<vertex_data> user_mat(filename + "_U.mm", 0, M , "This file contains LP output matrix U. In each row D probabilities for the Y labels", latent_factors_inmem);
  logstream(LOG_INFO) << "LP output files (in matrix market format): " << filename << "_U.mm" << std::endl;
}

int main(int argc, const char ** argv) {

  print_copyright();
 
  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("label_propagation");

  alpha        = get_option_float("alpha", alpha);
  debug        = get_option_int("debug", debug);
  
  parse_command_line_args();


  //load graph (adj matrix) from file
  int nshards = convert_matrixmarket<EdgeDataType>(training, 0, 0, 3, TRAINING, true);
  if (M != N)
    logstream(LOG_FATAL)<<"Label propagation supports only square matrices" << std::endl;

  init_feature_vectors<std::vector<vertex_data> >(M, latent_factors_inmem, false);
  
  //load seed initialization from file
  load_matrix_market_matrix(training + ".seeds", 0, D);

  #pragma omp parallel for
  for (int i=0; i< (int)M; i++){

    //normalize seed probabilities to sum up to one
    if (latent_factors_inmem[i].seed){
      assert(sum(latent_factors_inmem[i].pvec) != 0);
      latent_factors_inmem[i].pvec /= sum(latent_factors_inmem[i].pvec);
      continue;
    }
    //other nodes get random label probabilities
    for (int j=0; j< D; j++)
       latent_factors_inmem[i].pvec[j] = drand48();
  }

  /* load initial state from disk (optional) */
  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, D);
  }

  /* Run */
  LPVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output latent factor matrices in matrix-market format */
  output_lp_result(training);

  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  
  return 0;
}

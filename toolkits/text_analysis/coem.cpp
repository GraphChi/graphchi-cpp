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
#include "../parsers/common.hpp"
#include "../collaborative_filtering/eigen_wrapper.hpp"

double alpha = 0.15;

#define TEXT_LENGTH 64

std::string contexts_file, nouns_file, pos_seeds, neg_seeds;
double_map nouns;
double_map contexts;

struct vertex_data {
  vec pvec;
  bool seed;
  double normalizer;
  int nb_count;
  char text[TEXT_LENGTH];

  vertex_data() {
    pvec = zeros(D);
    seed = false;
    normalizer = 0;
    nb_count = 0;
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

struct edge_data{
  int cooccurence_count;
  edge_data(double val, double nothing){
     cooccurence_count = (int)val;
  } 
  edge_data(double val){
     cooccurence_count = (int)val;
  }
  edge_data() : cooccurence_count(0) { }

};
/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef edge_data EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

#include "../collaborative_filtering/io.hpp"


// tfidc is a modified weight formula for Co-EM (see Justin
// Betteridge's "CoEM results" page)
#define TFIDF(coocc, num_neighbors, vtype_total) (log(1+coocc)*log(vtype_total*1.0/num_neighbors))   

/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct COEMVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {



  /**
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
    vertex_data & vdata = latent_factors_inmem[vertex.id()];
    if (vertex.num_edges() == 0 || vdata.seed) //no edges, nothing to do here
      return;
    
    vec ret = zeros(D);
    double normalization = 0;
    for(int e=0; e < vertex.num_edges(); e++) {
      edge_data edge = vertex.edge(e)->get_data();                
      vertex_data & nbr_latent = latent_factors_inmem[vertex.edge(e)->vertex_id()];
      ret += edge.cooccurence_count * nbr_latent.pvec;
      normalization += edge.cooccurence_count;
    }

    ret /= normalization;
    vdata.pvec = alpha * vdata.pvec + (1-alpha)*ret;
  }


};


void load_seeds_from_txt_file(std::map<std::string,uint> & map, const std::string filename, bool negative){
  logstream(LOG_INFO)<<"loading " << (negative ? "negative" : "positive" ) << " seeds from txt file: " << filename << std::endl;
  FILE * f = fopen(filename.c_str(), "r");
  if (f == NULL)
    logstream(LOG_FATAL)<<"Failed to open file: " << filename << std::endl;

  char * linebuf = NULL;
  size_t linesize;
  int line = 0;
  while (true){
    int rc = getline(&linebuf, &linesize, f);
    if (rc == -1)
      break;

    char *pch = strtok(linebuf,"\r\n\t_^$");
    if (!pch){
      logstream(LOG_FATAL) << "Error when parsing file: " << filename << ":" << line <<std::endl;
    }
    uint pos = map[pch];
    if (pos <= 0)
      logstream(LOG_FATAL)<<"Failed to find " << pch << " in map. Aborting" << std::endl;

    assert(pos <= M);
    latent_factors_inmem[pos-1].seed = true;
    latent_factors_inmem[pos-1].pvec[0] = negative ? 0 : 1;
    line++;
    //free(to_free);
  }
  logstream(LOG_INFO)<<"Seed list size is: " << line << std::endl;
  fclose(f);
}



void output_coem_result(std::string filename) {
  MMOutputter_mat<vertex_data> user_mat(filename + "_U.mm", 0, M , "This file contains COEM output matrix U. In each row D probabilities for the Y labels", latent_factors_inmem);
  logstream(LOG_INFO) << "COEM output files (in matrix market format): " << filename << "_U.mm" << std::endl;
}

int main(int argc, const char ** argv) {

  print_copyright();
 
  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("label_propagation");

  contexts_file = get_option_string("contexts");
  nouns_file = get_option_string("nouns"); 
  pos_seeds = get_option_string("pos_seeds");
  neg_seeds = get_option_string("neg_seeds");
  parse_command_line_args();

  load_map_from_txt_file(contexts.string2nodeid, contexts_file, 1);
  load_map_from_txt_file(nouns.string2nodeid, nouns_file, 1);
    //load graph (adj matrix) from file
  int nshards = convert_matrixmarket<EdgeDataType>(training, 0, 0, 3, TRAINING, true);

  init_feature_vectors<std::vector<vertex_data> >(M+N, latent_factors_inmem);

  load_seeds_from_txt_file(nouns.string2nodeid, pos_seeds, false);
  load_seeds_from_txt_file(nouns.string2nodeid, neg_seeds, true); 

#pragma omp parallel for
  for (int i=0; i< (int)M; i++){

    //normalize seed probabilities to sum up to one
    if (latent_factors_inmem[i].seed){
      if (sum(latent_factors_inmem[i].pvec) != 0)
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
  COEMVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output latent factor matrices in matrix-market format */
  output_coem_result(training);

  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  
  return 0;
}

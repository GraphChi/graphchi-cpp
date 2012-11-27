/**  
 * Copyright (c) 2009 Carnegie Mellon University. 
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://www.graphlab.ml.cmu.edu
 

   Written by Danny Bickson, CMU
   File for extracting a subgraph out of the input graph, starting with a given set of seeds.
 *
 */



#include <cmath>
#include <cstdio>
#include <limits>
#include <iostream>
#include "graphchi_basic_includes.hpp"
#include "../../example_apps/matrix_factorization/matrixmarket/mmio.h"
#include "../../example_apps/matrix_factorization/matrixmarket/mmio.c"
#include "api/chifilenames.hpp"
#include "api/vertex_aggregator.hpp"
#include "preprocessing/sharder.hpp"
#include "../collaborative_filtering/util.hpp"
#include "../collaborative_filtering/eigen_wrapper.hpp"
#include "../collaborative_filtering/timer.hpp"

using namespace graphchi;

std::string training;
std::string validation;
std::string test;
uint M, N, Me, Ne, Le, K;
size_t L;
double globalMean = 0;
int square = 0;
int tokens_per_row = 3;

bool debug = false;
int max_iter = 50;
int iiter = 0; //current iteration
uint num_active = 0;
uint links = 0;
mutex mymutex;
timer mytimer;
//out_file * pout = NULL;
FILE * pfile = NULL;
size_t edges = 1000; //number of edges to cut from graph
size_t nodes = 0; //number of nodes in original file (optional)
size_t orig_edges = 0; // number of edges in original file (optional)

struct vertex_data {
  bool active;
  bool done;
  bool next_active;
  vec pvec; //to remove
  vertex_data() : active(false), done(false), next_active(false)  {}

}; // end of vertex_data

//edges in kcore algorithm are binary
struct edge_data {
  edge_data()  { }
  //compatible with parser which have edge value (we don't need it)
  edge_data(double val)  { }
  edge_data(double val, double time) { }
};



typedef vertex_data VertexDataType;
typedef edge_data EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

#include "../collaborative_filtering/io.hpp"

struct KcoresProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

  /**
   *  Vertex update function.
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
 
   vertex_data & vdata = latent_factors_inmem[vertex.id()];
    if (debug && iiter > 99 && vertex.id() % 1000 == 0)
      std::cout<<"Entering node: " << vertex.id() << std::endl;

    if (!vdata.active)
      return;
    num_active++;

    mymutex.lock();
    for(int e=0; e < vertex.num_edges(); e++) {
      vertex_data & other = latent_factors_inmem[vertex.edge(e)->vertex_id()];
      if (links >= edges)
        break;
      if (other.done)
        continue;
      fprintf(pfile, "%u %u %u\n", vertex.id()+1, vertex.edge(e)->vertex_id()+1,iiter+1);
      links++;
      if (!other.done){
        other.next_active = true;
      }
    }
    vdata.active=false;
    vdata.done = true;
    mymutex.unlock();
  }

  
}; // end of  aggregator




int main(int argc,  const char *argv[]) {

  logstream(LOG_WARNING)<<"GraphChi graph analytics library is written by Danny Bickson (c). Send any "
    " comments or bug reports to danny.bickson@gmail.com " << std::endl;

  //* GraphChi initialization will read the command line arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("subgraph-inmemory-factors");

  std::string datafile;
  int max_iter    = get_option_int("hops", 3);  // Number of iterations
  bool quiet    = get_option_int("quiet", 0);
  if (quiet)
    global_logger().set_log_level(LOG_ERROR);
  debug         = get_option_int("debug", 0);
  datafile      = get_option_string("training");
  square        = get_option_int("square", 0);
  tokens_per_row = get_option_int("tokens_per_row", tokens_per_row);
  edges         = get_option_int("edges", 2460000000);
  nodes         = get_option_int("nodes", nodes);
  orig_edges         = get_option_int("orig_edges", orig_edges);

  std::string seeds   = get_option_string("seeds");

 mytimer.start();


  /* Preprocess data if needed, or discover preprocess files */

  int nshards = 0;
  if (tokens_per_row == 4 )
    convert_matrixmarket4<edge_data>(datafile, false, square);
  else if (tokens_per_row == 3 || tokens_per_row == 2) 
    convert_matrixmarket<edge_data>(datafile, NULL, nodes, orig_edges, tokens_per_row);
  else logstream(LOG_FATAL)<<"Please use --tokens_per_row=3 or --tokens_per_row=4" << std::endl;

  latent_factors_inmem.resize(square? std::max(M,N) : M+N);
  char * pseeds = strdup(seeds.c_str());
  char * pch = strtok(pseeds, ",\n\r\t ");
  int node = atoi(pch);
  latent_factors_inmem[node-1].active = true;
  while ((pch = strtok(NULL, ",\n\r\t "))!= NULL){
    node = atoi(pch);
    latent_factors_inmem[node-1].active = true;
  }
 
  unlink((datafile +".out").c_str());
  pfile = fopen((datafile +".out").c_str(), "w");
  std::cout<<"Writing output to: " << datafile +".out" << std::endl;

  num_active = 0;
  for (iiter=0; iiter< max_iter; iiter++){
      //std::cout<<mytimer.current_time() << ") Going to run subgraph iteration " << iiter << std::endl;
     /* Run */
      //while(true){
      KcoresProgram program;
      //num_active = 0;
      graphchi_engine<VertexDataType, EdgeDataType> engine(datafile, nshards, false, m); 
      engine.set_disable_vertexdata_storage();  
      engine.set_modifies_inedges(false);
      engine.set_modifies_outedges(false);
      engine.run(program, 1);
      std::cout<< iiter << ") " << mytimer.current_time() << " Number of active nodes: " << num_active <<" Number of links: " << links << std::endl;
      for (uint i=0; i< M+N; i++){
        if (latent_factors_inmem[i].next_active && !latent_factors_inmem[i].done){
          latent_factors_inmem[i].next_active = false;
          latent_factors_inmem[i].active = true;
        }
      }
      if (links >= edges){
        std::cout<<"Grabbed enough edges!" << std::endl;
        break;
      }
  }
      
  std::cout<< iiter << ") Number of active nodes: " << num_active <<"Number of links: " << links << std::endl;
 
  std::cout << "subgraph finished in " << mytimer.current_time() << std::endl;
  std::cout << "Number of passes: " << iiter<< std::endl;
  std::cout << "Total active nodes: " << num_active << " edges: " << links << std::endl;
  fflush(pfile);
   fclose(pfile);
   //delete pout;
   return EXIT_SUCCESS;
}



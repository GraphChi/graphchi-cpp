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
 1) File for extracting a subgraph out of the input graph, starting with a given set of seeds, for X hops.
 2) This program also prints the degree distribution of a graph (using --degrees=1 command line argument)
 3) This program also counts the number of edges for each connected compoentns (using the --cc=filename command line)
 *
 */



#include <cmath>
#include <cstdio>
#include <limits>
#include <iostream>
#include "graphchi_basic_includes.hpp"
//#include "../../example_apps/matrix_factorization/matrixmarket/mmio.h"
//#include "../../example_apps/matrix_factorization/matrixmarket/mmio.c"
#include "api/chifilenames.hpp"
#include "api/vertex_aggregator.hpp"
#include "preprocessing/sharder.hpp"
#include "../collaborative_filtering/util.hpp"
#include "../collaborative_filtering/eigen_wrapper.hpp"
#include "../collaborative_filtering/timer.hpp"
#include "../collaborative_filtering/common.hpp"

using namespace graphchi;

int square = 0;
int tokens_per_row = 3;
int _degree = 0;
std::string cc;

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
  int component;
  vec pvec; //to remove
  vertex_data() : active(false), done(false), next_active(false), component(0)  {}

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

vec component_edges;
vec component_nodes;
vec component_seeds;
#include "../collaborative_filtering/io.hpp"

struct KcoresProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

  /**
   *  Vertex update function.
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {

    vertex_data & vdata = latent_factors_inmem[vertex.id()];

    /* printout degree distribution and finish */
    if (_degree){
      if (vertex.num_edges() > 0)
        fprintf(pfile, "%u %u\n", vertex.id()+1, vertex.num_edges());
      return;
    }
    /* calc component number of nodes and edges and finish */
    else if (cc != ""){
      assert(vdata.component>= 0 && vdata.component < component_nodes.size());
      //if (vertex.id() == 1104 || vertex.id() == 1103 || vertex.id() == 1105)
      //  logstream(LOG_DEBUG)<<"Node 1104 has " << vertex.num_edges() << std::endl;
      component_nodes[vdata.component]++;
      for(int e=0; e < vertex.num_edges(); e++) {
        vertex_data & other = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        //if (vertex.id() ==1103 && vertex.edge(e)->vertex_id() < 100000)
        //logstream(LOG_DEBUG)<<"Going over edge: " << vertex.id() << "=>" << vertex.edge(e)->vertex_id() << " component: " << vdata.component <<" : "<<other.component<< " seed? " << vdata.active << std::endl;
        if (vdata.component == other.component)
          //logstream(LOG_INFO)<<"Added an edge for component: " << other.component << std::endl;
          component_edges[vdata.component]++;
        }
      return; 
      }

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
  _degree =  get_option_int("degree", _degree);
  cc = get_option_string("cc", cc);

  if (_degree || cc != "")
    max_iter = 1;

  std::string seeds   = get_option_string("seeds","");
  std::string seed_file = get_option_string("seed_file", "");

  mytimer.start();


  /* Preprocess data if needed, or discover preprocess files */

  int nshards = 0;
  if (tokens_per_row == 4 )
    convert_matrixmarket4<edge_data>(datafile, false, square);
  else if (tokens_per_row == 3 || tokens_per_row == 2) 
    convert_matrixmarket<edge_data>(datafile, NULL, nodes, orig_edges, tokens_per_row);
  else logstream(LOG_FATAL)<<"Please use --tokens_per_row=2 or --tokens_per_row=3 or --tokens_per_row=4" << std::endl;

  latent_factors_inmem.resize(square? std::max(M,N) : M+N);

  vec vseed;
  if (seed_file == ""){
    if (seeds == "")
      logstream(LOG_FATAL)<<"Must specify either seeds or seed_file"<<std::endl;
    char * pseeds = strdup(seeds.c_str());
    char * pch = strtok(pseeds, ",\n\r\t ");
    int node = atoi(pch);
    latent_factors_inmem[node-1].active = true;
    while ((pch = strtok(NULL, ",\n\r\t "))!= NULL){
      node = atoi(pch);
      latent_factors_inmem[node-1].active = true;
    }
  }
  else {
    vseed = load_matrix_market_vector(seed_file, false, false);
    for (int i=0; i< vseed.size(); i++){
      assert(vseed[i] < latent_factors_inmem.size());
      latent_factors_inmem[vseed[i]].active = true;
    }
  }

  vec components;

  if (cc != ""){
    components = load_matrix_market_vector(cc, false,true);
    assert((int)components.size() <= (int) latent_factors_inmem.size());
    for (uint i=0; i< components.size(); i++){
      assert(i+1 < latent_factors_inmem.size());
      //if (components[i] == 1104 || i == 1104 || i == 1103 || i == 1105)
      //logstream(LOG_DEBUG)<<"Setting node : " <<i<<" component : " << components[i] << std::endl;
      latent_factors_inmem[i].component = components[i];
    }
    component_edges = zeros(nodes);
    component_nodes = zeros(nodes);
    component_seeds = zeros(nodes);
    for (uint i=0; i< vseed.size(); i++){
      assert(vseed[i] >= 1 && vseed[i] <= latent_factors_inmem.size());
      component_seeds[latent_factors_inmem[vseed[i]-1].component]++;
    }
  }

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
    set_engine_flags(engine);
    engine.run(program, 1);
    std::cout<< iiter << ") " << mytimer.current_time() << " Number of active nodes: " << num_active <<" Number of links: " << links << std::endl;
    for (uint i=0; i< (M==N?M:M+N); i++){
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

  if (cc != ""){
    logstream(LOG_INFO)<<"component nodes sum: " << sum(component_nodes) << std::endl;
    logstream(LOG_INFO)<<"component edges sum: " << sum(component_edges) << std::endl;

    int total_written = 0;
    for (uint i=0; i< nodes; i++)
      if (component_nodes[i] > 0 && component_edges[i] > 0){
        fprintf(pfile, "%d %d %d %d\n", i, (int)component_nodes[i], (int)component_edges[i], (int)component_seeds[i]);
        total_written++;
      }
    logstream(LOG_INFO)<<"total written components: " << total_written << " sum : " << sum(component_nodes) << std::endl;
  }
  else { 
    std::cout<< iiter << ") Number of active nodes: " << num_active <<" Number of links: " << links << std::endl;

    std::cout << "subgraph finished in " << mytimer.current_time() << std::endl;
    std::cout << "Number of passes: " << iiter<< std::endl;
    std::cout << "Total active nodes: " << num_active << " edges: " << links << std::endl;
  }
  fflush(pfile);
  fclose(pfile);
  //delete pout;
  return EXIT_SUCCESS;
  }



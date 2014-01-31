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
#include <set>
#include "graphchi_basic_includes.hpp"
#include "api/chifilenames.hpp"
#include "api/vertex_aggregator.hpp"
#include "preprocessing/sharder.hpp"
#include "../collaborative_filtering/util.hpp"
#include "../collaborative_filtering/eigen_wrapper.hpp"
#include "../collaborative_filtering/timer.hpp"
#include "../collaborative_filtering/common.hpp"

using namespace graphchi;
using namespace std;

int square = 0;
int _degree = 0;
int seed_edges_only = 0;
int undirected = 1;
std::string cc;
size_t singleton_nodes = 0;
bool debug = false;
int max_iter = 50;
int iiter = 0; //current iteration
uint num_active = 0;
uint links = 0;
mutex mymutex;
timer mytimer;
FILE * pfile = NULL;
size_t edges = 1000; //number of edges to cut from graph
size_t nodes = 0; //number of nodes in original file (optional)
size_t orig_edges = 0; // number of edges in original file (optional)
int min_range = 0;
int max_range = 2400000000;

struct vertex_data {
  bool active;
  bool done;
  bool next_active;
  int component;
  vec pvec; //to remove
  vertex_data() : active(false), done(false), next_active(false), component(0)  {}
  void set_val(int index, double val){};
  float get_val(int index){ return 0; }

}; // end of vertex_data

//edges in kcore algorithm are binary
struct edge_data {
  edge_data()  { }
  //compatible with parser which have edge value (we don't need it)
  edge_data(double val)  { }
  edge_data(double val, double time) { }
};



typedef vertex_data VertexDataType;
typedef edge_data EdgeDataType;  

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

vec component_edges;
vec component_nodes;
vec component_seeds;
size_t changes = 0;
#include "../collaborative_filtering/io.hpp"

struct SubgraphsProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

   /**
   *  Vertex update function.
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {

    vertex_data & vdata = latent_factors_inmem[vertex.id()];

    /* printout degree distribution and finish */
    if (_degree){
      if (vertex.num_edges() > 0 || max_range != 2400000000)
        if (vertex.id() >= (uint)min_range && vertex.id() < (uint)max_range)
          fprintf(pfile, "%u %u\n", vertex.id()+1, vertex.num_edges());
      return;
    }
    /* calc component number of nodes and edges and finish */
    else if (cc != ""){
      assert(vdata.component>= 0 && vdata.component < component_nodes.size());
      if (debug && vdata.component == 9322220)
      logstream(LOG_DEBUG)<<"Node " << vertex.id() << " has " << vertex.num_edges() << std::endl;

      if (vdata.component == 0)
         return;

      mymutex.lock();
      component_nodes[vdata.component]++;
      mymutex.unlock();      

      if (vertex.num_edges() == 0){
        mymutex.lock();
        singleton_nodes++;
        mymutex.unlock();
      }

      for(int e=0; e < vertex.num_edges(); e++) {
        vertex_data & other = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        if (debug && vdata.component == 9322220)
        logstream(LOG_DEBUG)<<"Going over edge: " << vertex.id() << "=>" << vertex.edge(e)->vertex_id() << " component: " << vdata.component <<" : "<<other.component<< " seed? " << vdata.active << std::endl;
        
        if (vdata.component != other.component)
           logstream(LOG_FATAL)<<"BUG Going over edge: " << vertex.id() << "=>" << vertex.edge(e)->vertex_id() << " component: " << vdata.component <<" : "<<other.component<< " seed? " << vdata.active << std::endl;
        if (vertex.id() < vertex.edge(e)->vertex_id()){

          if (debug && other.component == 9322220)
          logstream(LOG_INFO)<<"Added an edge for component: " << other.component << std::endl;
          mymutex.lock();
          component_edges[vdata.component]++;
          mymutex.unlock();
        }
        }
      return; 
      }

    if (!vdata.active)
      return;

    mymutex.lock();
    num_active++;
  
    std::set<uint> myset;
    std::set<uint>::iterator it;

    for(int e=0; e < vertex.num_edges(); e++) {
      vertex_data & other = latent_factors_inmem[vertex.edge(e)->vertex_id()];
       if (links >= edges)
        break;
      if (other.done)
        continue;
      if (seed_edges_only && !other.active)
        continue;
      //solve a bug where an edge appear twice if A->B and B->A in the data
      if (undirected){
      it = myset.find(vertex.edge(e)->vertex_id());
      if (it != myset.end())
        continue;
      }
      fprintf(pfile, "%u %u %u\n", vertex.id()+1, vertex.edge(e)->vertex_id()+1,iiter+1);
      if (undirected)
        myset.insert(vertex.edge(e)->vertex_id());
      if (debug && (vertex.id()+1 == 9322220 || vertex.edge(e)->vertex_id()+1 == 9322220))
        cout<<"Found edge: $$$$ " << vertex.id() << " => "  << vertex.edge(e)->vertex_id()+1 << " other.done " << other.done << endl;
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
  undirected    = get_option_int("undirected", undirected);
  if (quiet)
    global_logger().set_log_level(LOG_ERROR);
  debug         = get_option_int("debug", 0);
  datafile      = get_option_string("training");
  square        = get_option_int("square", 0);
  edges         = get_option_int("edges", 2460000000);
  nodes         = get_option_int("nodes", nodes);
  orig_edges         = get_option_int("orig_edges", orig_edges);
  _degree =  get_option_int("degree", _degree);
  cc = get_option_string("cc", cc);
  seed_edges_only = get_option_int("seed_edges_only", seed_edges_only);

  if (_degree || cc != "" || seed_edges_only)
    max_iter = 1;

  std::string seeds   = get_option_string("seeds","");
  std::string seed_file = get_option_string("seed_file", "");
  min_range = get_option_int("min_range", min_range);
  max_range = get_option_int("max_range", max_range);

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
  if (!_degree){
  if (seed_file == ""){/* read list of seeds from the --seeds=XX command line argument */
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
  else { /* load initial set of seeds from file */
    vseed = load_matrix_market_vector(seed_file, false, false);
    for (int i=0; i< vseed.size(); i++){
      assert(vseed[i] > 0 && vseed[i] <= latent_factors_inmem.size());
      latent_factors_inmem[vseed[i]-1].active = true;
    }
  }
  }
  vec components;

  /* read a vector of connected components for each node */
  if (cc != ""){
    components = load_matrix_market_vector(cc, false,true);
    assert((int)components.size() <= (int) latent_factors_inmem.size());
    for (uint i=0; i< components.size(); i++){
      assert(i+1 < latent_factors_inmem.size());
      assert(components[i] >= 1 && components[i] <= nodes);
      if (debug && components[i] == 9322220)
      logstream(LOG_DEBUG)<<"Setting node : " <<i<<" component : " << components[i] << std::endl;
      latent_factors_inmem[i].component = components[i];
    }
    component_edges = zeros(nodes);
    component_nodes = zeros(nodes);
    component_seeds = zeros(nodes);
    for (uint i=0; i< vseed.size(); i++){
      assert(vseed[i] >= 1 && vseed[i] <= latent_factors_inmem.size());
      component_seeds[latent_factors_inmem[vseed[i]-1].component]++;
    }
    assert(sum(component_seeds) == vseed.size());
  }
  else if (seed_edges_only){
    for (uint i=0; i< latent_factors_inmem.size(); i++){
      vertex_data & vdata = latent_factors_inmem[i];
      if (!vdata.active)
        vdata.done = true;
    }
  }

  std::string suffix;
  if (cc != "")
    suffix = "-cc.txt";
  else if (seed_edges_only)
    suffix = "-subset.txt";
  else if (_degree)
    suffix = "-degree.txt";
  else suffix = "-subgraph.txt";

  pfile = open_file((datafile + suffix).c_str(), "w", false);
  std::cout<<"Writing output to: " << datafile << suffix << std::endl;

  num_active = 0;
  graphchi_engine<VertexDataType, EdgeDataType> engine(datafile, nshards, false, m); 
  set_engine_flags(engine);
  engine.set_maxwindow(nodes+1);
  SubgraphsProgram program;
  for (iiter=0; iiter< max_iter; iiter++){
    //std::cout<<mytimer.current_time() << ") Going to run subgraph iteration " << iiter << std::endl;
    /* Run */
    //while(true){
     engine.run(program, 1);
      std::cout<< iiter << ") " << mytimer.current_time() << " Number of active nodes: " << num_active <<" Number of links: " << links << std::endl;
      for (uint i=0; i< latent_factors_inmem.size(); i++){
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
    assert(sum(component_nodes) == components.size());
    assert(pfile != NULL);
    int total_seeds = 0;
    for (uint i=0; i< component_nodes.size(); i++){
        // DANNY: What's this? Some debug? The condition never fires as 2400000000 is not in int range
      if ((max_range != 2400000000 && i >= (uint)min_range && i < (uint)max_range) || (max_range == 2400000000 && (component_nodes[i] > 1 || component_edges[i] > 0))){
        fprintf(pfile, "%d %d %d %d\n", i, (int)component_nodes[i], (int)component_edges[i], (int)component_seeds[i]);
        total_written++;
        total_seeds+= component_seeds[i];
      }
      if (component_nodes[i] > 1 && component_edges[i] == 0)
         logstream(LOG_FATAL)<<"Bug: component " << i << " has " << component_nodes[i] << " but not edges!" <<std::endl;
      if (component_nodes[i] == 0 && component_edges[i] > 0)
         logstream(LOG_FATAL)<<"Bug: component " << i << " has " << component_edges[i] << " but not nodes!" <<std::endl;
      if (component_seeds[i] == 0 && component_edges[i] > 0)
         logstream(LOG_FATAL)<<"Bug: component " << i << " has " << component_edges[i] << " but not seeds!" << std::endl;
      if (component_edges[i] > 0 && component_edges[i]+2 < component_nodes[i] )
        logstream(LOG_FATAL)<<"Bug: component " << i << " has missing edges: " << component_edges[i] << " nodes: " << component_nodes[i] << std::endl;
      if (component_nodes[i] == 2 && component_edges[i] == 2)
        logstream(LOG_FATAL)<<"Bug: component " << i << " 2 nodes +2 edges: " << component_edges[i] << " nodes: " << component_nodes[i] << std::endl;
    }

    logstream(LOG_INFO)<<"total written components: " << total_written << " sum : " << sum(component_nodes) << std::endl;
    logstream(LOG_INFO)<<"Singleton nodes (no edges): " << singleton_nodes << std::endl;
    logstream(LOG_INFO)<<"Total seeds: " << total_seeds << std::endl;
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



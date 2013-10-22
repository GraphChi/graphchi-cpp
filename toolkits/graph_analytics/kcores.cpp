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
 *
 */



#include <cmath>
#include <cstdio>
#include <limits>
#include <iostream>
#include "graphchi_basic_includes.hpp"
#include "api/chifilenames.hpp"
#include "api/vertex_aggregator.hpp"
#include "preprocessing/sharder.hpp"
#include "../collaborative_filtering/eigen_wrapper.hpp"
#include "../collaborative_filtering/timer.hpp"
#include "../collaborative_filtering/common.hpp"

using namespace graphchi;

int square = 0;

bool debug = false;
int max_iter = 50;
ivec active_nodes_num;
ivec active_links_num;
int iiter = 0; //current iteration
uint nodes = 0;
uint orig_edges = 0;
uint num_active = 0;
uint links = 0;
mutex mymutex;
timer mytimer;


struct vertex_data {
  bool active;
  int kcore, degree;
  vec pvec; //to remove
  vertex_data() : active(true), kcore(-1), degree(0)  {}
  void set_val(int index, double val){}
  float get_val(int index){ return 0;}
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

    int cur_iter = iiter;
    int cur_links = 0;
    int increasing_links = 0;
    
    for(int e=0; e < vertex.num_edges(); e++) {
      const vertex_data & other = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        if (other.active){
          if (debug && iiter > 99)
             std::cout<<"neighbor: "<<vertex.edge(e)->vertex_id()<<" active"<<std::endl;
      	  cur_links++;
          if (vertex.edge(e)->vertex_id() > vertex.id())
          increasing_links++;
        }
    }

    if (cur_links <= cur_iter){
        vdata.active = false;
        vdata.kcore = cur_iter;
    }
    else {
      if (debug && iiter > 99)
        std::cout<<vertex.id()<<": cur links: " << cur_links << std::endl;
      mymutex.lock();
      links += increasing_links;
      mymutex.unlock();
    }

    if (vdata.active){
      mymutex.lock();
      num_active++;
      mymutex.unlock();
    }
  }

  void after_iteration(int iteration, graphchi_context &gcontext) {
   active_nodes_num[iiter] = num_active;
   if (num_active == 0)
	links = 0;
   printf("Number of active nodes in round %d is %di, links: %d\n", iiter, num_active, links);
   active_links_num[iiter] = links;

  }
  
  void before_iteration(int iteration, graphchi_context &gcontext) {
    num_active = 0;
    links = 0;
  }
}; // end of  aggregator



vec fill_output(){
  vec ret = vec::Zero(latent_factors_inmem.size());
  for (uint i=0; i < latent_factors_inmem.size(); i++)
    ret[i] = latent_factors_inmem[i].kcore;
    return ret;
}

int main(int argc,  const char *argv[]) {

  logstream(LOG_WARNING)<<"GraphChi graph analytics library is written by Danny Bickson (c). Send any "
    " comments or bug reports to danny.bickson@gmail.com " << std::endl;

  //* GraphChi initialization will read the command line arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("kcores-inmemory-factors");

  std::string datafile;
  int unittest = 0;

  int max_iter    = get_option_int("max_iter", 15000);  // Number of iterations
  maxval        = get_option_float("maxval", 1e100);
  minval        = get_option_float("minval", -1e100);
  bool quiet    = get_option_int("quiet", 0);
  if (quiet)
    global_logger().set_log_level(LOG_ERROR);
  debug         = get_option_int("debug", 0);
  unittest      = get_option_int("unittest", 0); 
  datafile      = get_option_string("training");
  square        = get_option_int("square", 0);
  nodes = get_option_int("nodes", nodes);
  orig_edges = get_option_int("orig_edges", orig_edges);

  active_nodes_num = ivec(max_iter+1);
  active_links_num = ivec(max_iter+1);



  //unit testing
  if (unittest == 1){
     datafile = "kcores_unittest1";
  }
  mytimer.start();

  /* Preprocess data if needed, or discover preprocess files */

  int nshards = 0;
  if (tokens_per_row == 4 )
    convert_matrixmarket4<edge_data>(datafile, false, square);
  else if (tokens_per_row == 3 || tokens_per_row == 2) 
    convert_matrixmarket<edge_data>(datafile, NULL, nodes, orig_edges, tokens_per_row);
  else logstream(LOG_FATAL)<<"Please use --tokens_per_row=3 or --tokens_per_row=4" << std::endl;

  latent_factors_inmem.resize(square? std::max(M,N) : M+N);

  KcoresProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(datafile, nshards, false, m); 
 set_engine_flags(engine);
  engine.set_maxwindow(nodes+1);
  int pass = 0;
  for (iiter=1; iiter< max_iter+1; iiter++){
    logstream(LOG_INFO)<<mytimer.current_time() << ") Going to run k-cores iteration " << iiter << std::endl;
    while(true){
      int prev_nodes = active_nodes_num[iiter];
     /* Run */
           engine.run(program, 1);
      pass++;
      int cur_nodes = active_nodes_num[iiter];
      if (prev_nodes == cur_nodes)
        break; 
    }
    if (active_nodes_num[iiter] == 0){
      max_iter = iiter;
      break;
    }
  }
 
  std::cout << "KCORES finished in " << mytimer.current_time() << std::endl;
  std::cout << "Number of updates: " << pass*(M+N) << " pass: " << pass << std::endl;
  imat retmat = imat(max_iter+1, 4);
  memset((int*)data(retmat),0,sizeof(int)*retmat.size());


  active_nodes_num[0] = (square? std::max(M,N) : M+N);
  active_links_num[0] = L;
  assert(L>0);

  std::cout<<"     Core Removed Total    Removed"<<std::endl;
  std::cout<<"     Num  Nodes   Removed  Links" <<std::endl;
  for (int i=0; i <= max_iter; i++){
    set_val(retmat, i, 0, i);
    if (i >= 1){
      set_val(retmat, i, 1, active_nodes_num[i-1]-active_nodes_num[i]);
      set_val(retmat, i, 2, active_nodes_num[0]-active_nodes_num[i]);
      assert(active_nodes_num[i] >= 0);
      set_val(retmat, i, 3, L - active_links_num[i]);
    }
  } 
  //write_output_matrix(datafile + ".kcores.out", format, retmat);
  std::cout<<retmat<<std::endl;

  vec ret = fill_output();
  write_output_vector(datafile + "x.out", ret,false, "This vector holds for each node its kcore degree");

  if (unittest == 1){
    imat sol = init_imat("0 0 0 0; 1 1 1 1; 2 4 5 7; 3 4 9 13", 4, 4);
    assert(sumsum(sol - retmat) == 0);
  }

   return EXIT_SUCCESS;
}



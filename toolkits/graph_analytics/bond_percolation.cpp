/**
 * @file
 * @author  Aapo Kyrola <akyrola@cs.cmu.edu>
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright [2012] [Aapo Kyrola, Guy Blelloch, Carlos Guestrin / Carnegie Mellon University]
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
 * Application for computing the connected components of a graph.
 * The algorithm is simple: on first iteration each vertex sends its
 * id to neighboring vertices. On subsequent iterations, each vertex chooses
 * the smallest id of its neighbors and broadcasts its (new) label to
 * its neighbors. The algorithm terminates when no vertex changes label.
 *
 * @section REMARKS
 *
 * Version of connected components that keeps the vertex values
 * in memory.
 * @author Aapo Kyrola
 * 
 * Danny B: added output of each vertex label
 */

#define GRAPHCHI_DISABLE_COMPRESSION

#include <cmath>
#include <string>
#include <map>
#include "graphchi_basic_includes.hpp"
#include "label_analysis.hpp"
#include "../collaborative_filtering/eigen_wrapper.hpp"
#include "../collaborative_filtering/timer.hpp"
using namespace graphchi;


bool edge_count = false;
std::map<uint,uint> state;
mutex mymutex;
/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program.
 */
typedef vid_t VertexDataType;       // vid_t is the vertex id type
typedef vid_t EdgeDataType;
VertexDataType * vertex_values;
size_t changes = 0;
timer mytimer;
int actual_vertices = 0;
bool * active_nodes;
int iter = 0;

/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type>
 * class. The main logic is usually in the update function.
 */
struct ConnectedComponentsProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {


  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &ginfo) {
    logstream(LOG_DEBUG)<<mytimer.current_time() << "iteration: " << iteration << " changes: " << changes << std::endl;
    if (changes == 0)
      ginfo.set_last_iteration(iteration);
    changes = 0;
    iter++;
  }


  vid_t neighbor_value(graphchi_edge<EdgeDataType> * edge) {
    return vertex_values[edge->vertex_id()];
  }

  void set_data(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, vid_t value) {
    vertex_values[vertex.id()] = value;
  }

  /**
   *  Vertex update function.
   *  On first iteration ,each vertex chooses a label = the vertex id.
   *  On subsequent iterations, each vertex chooses the minimum of the neighbor's
   *  label (and itself).
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {

    /* On subsequent iterations, find the minimum label of my neighbors */
    if (!edge_count){
      vid_t curmin = vertex_values[vertex.id()];
      //first time, count the number of nodes which actually have edges
      if (gcontext.iteration == 0 && vertex.num_edges() > 0){
        mymutex.lock(); actual_vertices++; mymutex.unlock();
      }
      for(int i=0; i < vertex.num_edges(); i++) {
        vid_t nblabel = neighbor_value(vertex.edge(i));
        curmin = std::min(nblabel, curmin);
      }

      //in case of a new min reschedule neighbors
      if (vertex_values[vertex.id()] > curmin) {
        changes++;
        set_data(vertex, curmin);
        for (int i=0; i< vertex.num_edges(); i++){
          active_nodes[vertex.edge(i)->vertex_id()] = true;
        }
      }
      else active_nodes[vertex.id()] = false;
    }
    else {
      vid_t curmin = vertex_values[vertex.id()];
      for(int i=0; i < vertex.num_edges(); i++) {
        vid_t nblabel = neighbor_value(vertex.edge(i));
        curmin = std::min(nblabel, curmin);
        if (vertex.edge(i)->vertex_id() > vertex.id()){
        mymutex.lock();
        state[curmin]++;
        mymutex.unlock();
        }
      }
    }
  }

  /**
   * Called before an iteration starts.
   */
  void before_iteration(int iteration, graphchi_context &ctx) {
    changes = 0;
    ctx.scheduler->remove_tasks(0, (int) ctx.nvertices - 1);
    if (iteration == 0 && !edge_count) {
      /* initialize  each vertex with its own lable */
      vertex_values = new VertexDataType[ctx.nvertices];
      for(int i=0; i < (int)ctx.nvertices; i++) {
        vertex_values[i] = i;
      }
    }
    for (int i=0; i< (int)ctx.nvertices; i++)
      if (active_nodes[i])
        ctx.scheduler->add_task(i);
  }


};

int main(int argc, const char ** argv) {
  /* GraphChi initialization will read the command line
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("connected-components-inmem");

  /* Basic arguments for application */
  std::string filename = get_option_string("file");  // Base filename
  int niters           = get_option_int("niters", 100); // Number of iterations (max)
  int output_labels    = get_option_int("output_labels", 0); //output node labels to file?
  bool scheduler       = true;    // Always run with scheduler

  /* Process input file - if not already preprocessed */
  float p                 = get_option_float("p", -1);
  int n                 = get_option_int("n", -1);
  int quiet = get_option_int("quiet", 0);
  if (quiet)
    global_logger().set_log_level(LOG_ERROR);
  int nshards             = (int) convert_if_notexists<EdgeDataType>(filename, get_option_string("nshards", "auto"));
  mytimer.start();

  /* Run */
  ConnectedComponentsProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, scheduler, m);
  engine.set_disable_vertexdata_storage();  
  engine.set_enable_deterministic_parallelism(false);
  engine.set_modifies_inedges(false);
  engine.set_modifies_outedges(false);
  engine.set_maxwindow(engine.num_vertices());

  mytimer.start();

  active_nodes = new bool[engine.num_vertices()];
  for (int i=0; i< engine.num_vertices(); i++)
    active_nodes[i] = true;
  engine.run(program, niters);


  /* Run analysis of the connected components  (output is written to a file) */
  if (output_labels){
    FILE * pfile = fopen((filename + "-components").c_str(), "w");
    if (!pfile)
      logstream(LOG_FATAL)<<"Failed to open file: " << filename << std::endl;
    fprintf(pfile, "%%%%MatrixMarket matrix array real general\n");
    fprintf(pfile, "%lu %u\n", engine.num_vertices()-1, 1);
    for (uint i=1; i< engine.num_vertices(); i++){
      fprintf(pfile, "%u\n", vertex_values[i]);
      assert(vertex_values[i] >= 0 && vertex_values[i] < engine.num_vertices());
    }
    fclose(pfile); 
    logstream(LOG_INFO)<<"Saved succesfully to out file: " << filename << "-components" << " time for saving: " << mytimer.current_time() << std::endl;
  } 

  std::cout<<"Total runtime: " << mytimer.current_time() << std::endl;
  if (p > 0)
    std::cout << "site fraction p= " << p << std::endl;
  if (n > 0){
    std::cout << "n=" << n*p << std::endl;
    std::cout << "isolated sites: " << p*(double)n-actual_vertices << std::endl;
  }
  std::cout << "Number of sites: " << actual_vertices << std::endl;
  std::cout << "Number of bonds: " << engine.num_edges() << std::endl;
  if (n){
    std::cout << "Percentage of sites: " << (double)actual_vertices / (double)n << std::endl;
    std::cout << "Percentage of bonds: " << (double)engine.num_edges() / (2.0*n) << std::endl;
  }
  std::cout  << "Number of iterations: " << iter << std::endl;
  std::cout << "SITES RESULT:\nsize\tcount\n";
  std::map<uint,uint> final_countsv;
  std::map<uint,uint> final_countse;
  std::map<uint,uint> statv;
  for (int i=0; i< engine.num_vertices(); i++)
    statv[vertex_values[i]]++;


  uint total_sites = 0;
  for (std::map<uint, uint>::const_iterator iter = statv.begin();
      iter != statv.end(); iter++) {
    //std::cout << iter->first << "\t" << iter->second << "\n";
    final_countsv[iter->second] += 1;
    total_sites += iter->second;
  }
  for (std::map<uint, uint>::const_iterator iter = final_countsv.begin();
      iter != final_countsv.end(); iter++) {
    std::cout << iter->first << "\t" << iter->second << "\n";
  }
  edge_count = 1;
  engine.run(program, 1);
  std::cout << "BONDS RESULT:\nsize\tcount\n";
  uint total_bonds = 0;
  for (std::map<uint, uint>::const_iterator iter = state.begin();
      iter != state.end(); iter++) {
    //std::cout << iter->first << "\t" << iter->second << "\n";
    final_countse[iter->second] += 1;
    total_bonds += iter->second;
  }
  for (std::map<uint, uint>::const_iterator iter = final_countse.begin();
      iter != final_countse.end(); iter++) {
    std::cout << iter->first << "\t" << iter->second << "\n";
  }
  assert(total_sites == graph.num_vertices());
  assert(total_bonds == graph.num_edges());

  return 0;
}


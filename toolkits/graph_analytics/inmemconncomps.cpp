
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


#include <cmath>
#include <string>

#include "graphchi_basic_includes.hpp"
#include "label_analysis.hpp"
#include "../collaborative_filtering/eigen_wrapper.hpp"
#include "../collaborative_filtering/timer.hpp"
using namespace graphchi;



/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program.
 */
typedef vid_t VertexDataType;       // vid_t is the vertex id type
typedef vid_t EdgeDataType;
vid_t * vertex_values;
vid_t * edge_count;
vid_t * out_degree;
mutex mymutex;

size_t changes = 0;
timer mytimer;

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
    vid_t curmin = vertex_values[vertex.id()];
    for(int i=0; i < vertex.num_edges(); i++) {
      vid_t nblabel = neighbor_value(vertex.edge(i));
      curmin = std::min(nblabel, curmin);
    }

    if (vertex_values[vertex.id()] > curmin) {
      changes++;
      set_data(vertex, curmin);
    }
  }

  /**
   * Called before an iteration starts.
   */
  void before_iteration(int iteration, graphchi_context &ctx) {
    changes = 0;
    if (iteration == 0) {
      /* initialize  each vertex with its own lable */
      vertex_values = new VertexDataType[ctx.nvertices];
      for(int i=0; i < (int)ctx.nvertices; i++) {
        vertex_values[i] = i;
      }
    }
  }


};

/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type>
 * class. The main logic is usually in the update function.
 */
struct EdgeCountProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

    
  /**
   *  Vertex update function.
   *  On first iteration ,each vertex chooses a label = the vertex id.
   *  On subsequent iterations, each vertex chooses the minimum of the neighbor's
   *  label (and itself).
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {

    vid_t curmin = vertex_values[vertex.id()];
    for(int i=0; i < vertex.num_edges(); i++) {
      vid_t other = vertex_values[vertex.edge(i)->vertex_id()];
      assert(other != vertex.id());
      if (curmin == other && vertex.id() < vertex.edge(i)->vertex_id()){
          mymutex.lock();
          edge_count[curmin]++;
          out_degree[vertex.id()]++;
          mymutex.unlock();
      } 
    }
  }

  void before_iteration(int iteration, graphchi_context &ctx) {
     ctx.set_last_iteration(1);
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
  bool scheduler       = false;    // Always run with scheduler

  /* Process input file - if not already preprocessed */
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

  engine.run(program, niters);

  mytimer.start();

  /* Run analysis of the connected components  (output is written to a file) */
  if (output_labels){
    /* compute edge count for each component */
    edge_count = new vid_t[engine.num_vertices()];
    out_degree = new vid_t[engine.num_vertices()];
    memset(edge_count, 0, sizeof(vid_t)*engine.num_vertices());
    memset(out_degree, 0, sizeof(vid_t)*engine.num_vertices());
    EdgeCountProgram program2;
    engine.run(program2, 1);

    FILE * pfile = fopen((filename + "-components").c_str(), "w");
    if (!pfile)
      logstream(LOG_FATAL)<<"Failed to open file: " << filename << std::endl;
    for (uint i=1; i< engine.num_vertices(); i++){
      fprintf(pfile, "%u %u %u\n", i, vertex_values[i], out_degree[i]);
      assert(vertex_values[i] >= 0 && vertex_values[i] < engine.num_vertices());
    }
    fclose(pfile); 
    logstream(LOG_INFO)<<"Saved succesfully to out file: " << filename << "-components" << " time for saving: " << mytimer.current_time() << std::endl;

   pfile = fopen((filename + "-edges").c_str(), "w");
    if (!pfile)
      logstream(LOG_FATAL)<<"Failed to open file: " << filename << std::endl;
    for (uint i=1; i< engine.num_vertices(); i++){
      if (edge_count[i] > 0)
         fprintf(pfile, "%u %u\n", i, edge_count[i]);
    }
    fclose(pfile); 
    logstream(LOG_INFO)<<"Saved succesfully to out file: " << filename << "-edges" << " time for saving: " << mytimer.current_time() << std::endl;
  } 


  return 0;
}


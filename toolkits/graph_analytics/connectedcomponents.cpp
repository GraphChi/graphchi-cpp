
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
 * This application is interesting demonstration of the asyncronous capabilities
 * of GraphChi, improving the convergence considerably. Consider
 * a chain graph 0->1->2->...->n. First, vertex 0 will write its value to its edges,
 * which will be observed by vertex 1 immediatelly, changing its label to 0. Nexgt,
 * vertex 2 changes its value to 0, and so on. This all happens in one iteration.
 * A subtle issue is that as any pair of vertices a<->b share an edge, they will
 * overwrite each others value. However, because they will be never run in parallel
 * (due to deterministic parallellism of graphchi), this does not compromise correctness.
 *
 * @author Aapo Kyrola
 */


#include <cmath>
#include <string>

#include "graphchi_basic_includes.hpp"
#include "label_analysis.hpp"
#include "../collaborative_filtering/eigen_wrapper.hpp"

using namespace graphchi;

FILE * pfile = NULL;

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vid_t VertexDataType;       // vid_t is the vertex id type
typedef vid_t EdgeDataType;

vec unique_labels;
int niters;
/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct ConnectedComponentsProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    /**
     *  Vertex update function.
     *  On first iteration ,each vertex chooses a label = the vertex id.
     *  On subsequent iterations, each vertex chooses the minimum of the neighbor's
     *  label (and itself). 
     */
    void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
        /* This program requires selective scheduling. */
        assert(gcontext.scheduler != NULL);
        
        if (gcontext.iteration == 0) {
            vertex.set_data(vertex.id());
            gcontext.scheduler->add_task(vertex.id()); 
        }
    
        /* On subsequent iterations, find the minimum label of my neighbors */
        vid_t curmin = vertex.get_data();
        for(int i=0; i < vertex.num_edges(); i++) {
            vid_t nblabel = vertex.edge(i)->get_data();
            if (gcontext.iteration == 0) nblabel = vertex.edge(i)->vertex_id();  // Note!
            curmin = std::min(nblabel, curmin); 
        }
        
        /* Check if label changed */
        vertex.set_data(curmin);
        
        /** 
         * Broadcast new label to neighbors by writing the value
         * to the incident edges.
         * Note: on first iteration, write only to out-edges to avoid
         * overwriting data (this is kind of a subtle point)
         */
        vid_t label = vertex.get_data();
        
        if (gcontext.iteration > 0) {
            for(int i=0; i < vertex.num_edges(); i++) {
                if (label < vertex.edge(i)->get_data()) {
                    vertex.edge(i)->set_data(label);
                    /* Schedule neighbor for update */
                    gcontext.scheduler->add_task(vertex.edge(i)->vertex_id()); 
                }
            }
        } else if (gcontext.iteration == 0) {
            for(int i=0; i < vertex.num_outedges(); i++) {
                vertex.outedge(i)->set_data(label);
            }
        }
    }    
};


/* class for output the label number for each node (optional) */
class OutputVertexCallback : public VCallback<VertexDataType> {
  public:
    /* print node id and then the label id */
    virtual void callback(vid_t vertex_id, VertexDataType &value) {
        fprintf(pfile, "%u 1 %u\n", vertex_id+1, value); //graphchi offsets start from zero, while matlab from 1
        unique_labels[value] = 1;
    }
};


int main(int argc, const char ** argv) {
    /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
    graphchi_init(argc, argv);
    
    /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
    metrics m("connected-components");
    
    /* Basic arguments for application */
    std::string filename = get_option_string("file");  // Base filename
    niters           = get_option_int("niters", 10); // Number of iterations (max)
    int output_labels    = get_option_int("output_labels", 0); //output node labels to file?
    bool scheduler       = true;    // Always run with scheduler
    
    /* Process input file - if not already preprocessed */
    int nshards             = convert_if_notexists<EdgeDataType>(filename, get_option_string("nshards", "auto"));
    
    if (get_option_int("onlyresult", 0) == 0) {
        /* Run */
        ConnectedComponentsProgram program;
        graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, scheduler, m); 
        engine.run(program, niters);

         /* optional: output labels for each node to file */
        if (output_labels){
          pfile = fopen((filename + "-components").c_str(), "w");
          if (!pfile)
            logstream(LOG_FATAL)<<"Failed to open file: " << filename << std::endl;
          fprintf(pfile, "%%%%MatrixMarket matrix coordinate real general\n");
          fprintf(pfile, "%lu %u %lu\n", engine.num_vertices()-1, 1, engine.num_vertices()-1);
          OutputVertexCallback callback;
          //unique_labels = zeros(engine.num_vertices());
          //foreach_vertices<VertexDataType>(filename, 0, engine.num_vertices(), callback);
          //fclose(pfile);
          logstream(LOG_INFO)<<"Found: " << sum(unique_labels) << " unique labels " << std::endl;
   /* Run analysis of the connected components  (output is written to a file) */
    m.start_time("label-analysis");
    analyze_labels2<vid_t>(filename, pfile);
    m.stop_time("label-analysis");
    fclose(pfile); 
     }

    }
    
    return 0;
}


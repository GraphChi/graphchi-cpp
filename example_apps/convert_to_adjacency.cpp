
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
 * Simple program that writes a graph into adjacency list
 */

#include <iostream>

#include "graphchi_basic_includes.hpp"

using namespace graphchi;

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef bool VertexDataType;
typedef bool EdgeDataType;

FILE * f;

#define MODE_ADJLIST 0
#define MODE_CASSOVARY_ADJ 1

int mode;

struct AdjConverter : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    
    /**
     *  Vertex update function.
     */
    void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
        if (vertex.id() % 10000 == 0) std::cout << vertex.id() << std::endl;
        switch(mode) {
            case MODE_ADJLIST: {
                fprintf(f, "%d %d", vertex.id(), vertex.num_outedges());
                for(int i=0; i<vertex.num_outedges(); i++) 
                    fprintf(f, " %d", vertex.outedge(i)->vertex_id());
                fprintf(f, "\n");
                break;
            }
            case MODE_CASSOVARY_ADJ: {
                fprintf(f, "%d %d\n", vertex.id(), vertex.num_outedges());
                for(int i=0; i<vertex.num_outedges(); i++) 
                    fprintf(f, "%d\n", vertex.outedge(i)->vertex_id());
                break;
            }
        }
    }
    
    /**
     * Called before an iteration starts.
     */
    void before_iteration(int iteration, graphchi_context &gcontext) {
    }
    
    /**
     * Called after an iteration has finished.
     */
    void after_iteration(int iteration, graphchi_context &gcontext) {
    }
    
    /**
     * Called before an execution interval is started.
     */
    void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
    }
    
    /**
     * Called after an execution interval has finished.
     */
    void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
    }
    
};

int main(int argc, const char ** argv) {
    /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
    graphchi_init(argc, argv);
    
    /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
    metrics m("adjconverter");
    
    /* Basic arguments for application */
    std::string filename = get_option_string("file");  // Base filename
    
    /* Detect the number of shards or preprocess an input to create them */
    int nshards          = convert_if_notexists<EdgeDataType>(filename, 
                                                              get_option_string("nshards", "auto"));
    mode = get_option_int("mode", 0);
    std::string outfile = filename + ".adj";
    f = fopen(outfile.c_str(), "w");
    
    /* Run */
    AdjConverter program;
    graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, false, m); 
    engine.set_exec_threads(1);
    engine.run(program, 1);
    
    fclose(f);
    
    /* Report execution metrics */
    metrics_report(m);
    return 0;
}


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
 * Hacky research implemenation - do not use
 * @author Aapo Kyrola
 */


#include <cmath>
#include <string>

#include "graphchi_basic_includes.hpp"
#include "util/labelanalysis.hpp"

using namespace graphchi;



/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */


typedef vid_t VertexDataType;       // vid_t is the vertex id type
typedef bool EdgeDataType;  // Not used

int iterationcount = 0;

/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct ConnectedComponentsProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    std::vector<PairContainer<vid_t> > inmemory_vertices;
    bool converged;
    
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
        }
        
        /* On subsequent iterations, find the minimum label of my neighbors */
        vid_t curmin = vertex.get_data();
        for(int i=0; i < vertex.num_edges(); i++) {
            vid_t nblabel = inmemory_vertices[vertex.edge(i)->vertex_id()].oldval(gcontext.iteration);
            if (gcontext.iteration == 0) nblabel = vertex.edge(i)->vertex_id();  // Note!
            curmin = std::min(nblabel, curmin); 
        }
        
        /* Set my label */
        if (curmin < vertex.get_data()) {
            converged = false;
        }
        vertex.set_data(curmin);
        inmemory_vertices[vertex.id()].set_newval(gcontext.iteration, curmin);
    }
    /**
     * Called before an iteration starts.
     */
    void before_iteration(int iteration, graphchi_context &info) {
        if (iteration == 0) {
            inmemory_vertices.reserve(info.nvertices);
            for(vid_t i=0; i<info.nvertices; i++) {
                inmemory_vertices[i] = PairContainer<vid_t>(i, i);
            }
        }
        iterationcount++;
        converged = iteration > 0;

    }

    
    /**
     * Called after an iteration has finished.
     */
    void after_iteration(int iteration, graphchi_context &ginfo) {
        if (converged) {
            std::cout << "Converged!" << std::endl;
            ginfo.set_last_iteration(iteration);
        }
    }
    
};

int main(int argc, const char ** argv) {
    /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
    graphchi_init(argc, argv);
    
    /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
    metrics m("sync-connected-components");
    
    /* Basic arguments for application */
    std::string filename = get_option_string("file");  // Base filename
    int niters           = get_option_int("niters", 1000); // Number of iterations (max)
    bool scheduler       = false;    // Always run with scheduler
    
    /* Process input file - if not already preprocessed */
    int nshards             = (int) convert_if_notexists<EdgeDataType>(filename, get_option_string("nshards", "auto"));
    
    if (get_option_int("onlyresult", 0) == 0) {
        /* Run */
        ConnectedComponentsProgram program;
        graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, scheduler, m); 
        engine.run(program, niters);
    }
    
    /* Run analysis of the connected components  (output is written to a file) */
    m.start_time("label-analysis");
    
    analyze_labels<vid_t>(filename);
    
    m.stop_time("label-analysis");
    
    /* Report execution metrics */
    metrics_report(m);
    
    std::cout << "Iterations: " << iterationcount << std::endl;
    
    FILE * logf = fopen("cc_log.txt", "a");
    fprintf(logf, "synchronized,%s,%d\n", filename.c_str(), iterationcount);
    fclose(logf);
    return 0;
}


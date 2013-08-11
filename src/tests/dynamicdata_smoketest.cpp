

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
 * Smoketest to test dynamically extended edge values.
 */

#define DYNAMICEDATA 1

#include <string>

#include "graphchi_basic_includes.hpp"
#include "api/dynamicdata/chivector.hpp"

using namespace graphchi;

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program.
 */
typedef vid_t VertexDataType;
typedef chivector<vid_t>  EdgeDataType;

/**
 * Smoke test.
 */
struct DynamicDataSmokeTestProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    
    /**
     *  Vertex update function.
     */
    void update(graphchi_vertex<VertexDataType, EdgeDataType > &vertex, graphchi_context &gcontext) {
        if (gcontext.iteration == 0) {
            for(int i=0; i < vertex.num_outedges(); i++) {
                chivector<vid_t> * evector = vertex.outedge(i)->get_vector();
                evector->clear();
                assert(evector->size() == 0);
                
                evector->add(vertex.id());
                assert(evector->size() == 1);
                assert(evector->get(0) == vertex.id());
            }
            
        } else {
            for(int i=0; i < vertex.num_inedges(); i++) {
                graphchi_edge<EdgeDataType> * edge = vertex.inedge(i);
                chivector<vid_t> * evector = edge->get_vector();
                assert(evector->size() >= gcontext.iteration);
                for(int j=0; j < evector->size(); j++) {
                    vid_t expected = edge->vertex_id() + j;
                    vid_t has = evector->get(j);
                    if (has != expected) {
                        std::cout << "Mismatch: " << has << " != " << expected << std::endl;
                    }
                    assert(has == expected);
                }
            }
            for(int i=0; i < vertex.num_outedges(); i++) {
                vertex.outedge(i)->get_vector()->add(vertex.id() + gcontext.iteration);
            }
        }
        vertex.set_data(gcontext.iteration + 1);
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

/**
 * Vertex callback that checks the vertex data is ok.
 */
class VertexDataChecker : public VCallback<VertexDataType> {
    int iters;
public:
    size_t total;
    
    VertexDataChecker(int iters) : iters(iters), total(0) {}
    void callback(vid_t vertex_id, VertexDataType &vecvalue) {
        assert(vecvalue == (VertexDataType)iters);
        total += (size_t) iters;
    }
};

int main(int argc, const char ** argv) {
    /* GraphChi initialization will read the command line
     arguments and the configuration file. */
    graphchi_init(argc, argv);
    
    /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
    metrics m("dynamicdata-smoketest");
    
    /* Basic arguments for application */
    std::string filename = get_option_string("file");  // Base filename
    int niters           = get_option_int("niters", 4); // Number of iterations
    bool scheduler       = false;                       // Whether to use selective scheduling

    int nshards          = convert_if_notexists<vid_t>(filename, get_option_string("nshards", "auto"));
    
    /* Run */
    DynamicDataSmokeTestProgram program;
    graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, scheduler, m);
    engine.run(program, niters);
    
    /* Check also the vertex data is ok */
    VertexDataChecker vchecker(niters);
    foreach_vertices(filename, 0, engine.num_vertices(), vchecker);
    assert(vchecker.total == engine.num_vertices() * niters);
    
    /* Report execution metrics */
    metrics_report(m);
    
    logstream(LOG_INFO) << "Smoketest passed successfully! Your system is working!" << std::endl;
    return 0;
}

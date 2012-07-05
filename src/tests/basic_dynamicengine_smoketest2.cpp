

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
 * Simple smoketest for the dynamic graph graphchi engine.
 */



#include <string>

#define SUPPORT_DELETIONS 1

#include "graphchi_basic_includes.hpp"
#include "engine/dynamic_graphs/graphchi_dynamicgraph_engine.hpp"

using namespace graphchi;

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vid_t VertexDataType;
typedef vid_t EdgeDataType;

/**
 * Smoke test. On every iteration, each vertex sets its id to be
 * id + iteration number. Vertices check whether their neighbors were
 * set correctly. This assumes that the vertices are executed in round-robin order.
 *   - Uses edges in inverse order to the first smoketest.
 */
struct SmokeTestProgram2 : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    volatile size_t ndeleted;
    
    /**
     *  Vertex update function.
     */
    void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
        int ninedges = 0;
        if (gcontext.iteration == 0) {
            for(int i=0; i < vertex.num_inedges(); i++) {
                vertex.inedge(i)->set_data(vertex.id());        
                ninedges++;
            }
        } else {
            // Keep track of the number of edegs to ensure that
            // deletion works fine.
            if (vertex.get_data() != vertex.num_inedges())  {
                logstream(LOG_ERROR) << "Discrepancy in edge counts: " << vertex.get_data() << " != " << vertex.num_inedges() << std::endl;
            }
            assert(vertex.get_data() == vertex.num_inedges());
            
            for(int i=0; i < vertex.num_outedges(); i++) {
                graphchi_edge<vid_t> * edge = vertex.outedge(i);
                vid_t outedgedata = edge->get_data();
                vid_t expected = edge->vertex_id() + gcontext.iteration - (edge->vertex_id() > vertex.id());
                if (!is_deleted_edge_value(edge->get_data())) {
                    if (outedgedata != expected) {
                        logstream(LOG_ERROR) << outedgedata << " != " << expected << std::endl;
                        assert(false);
                    }
                }
            }
            for(int i=0; i < vertex.num_inedges(); i++) {
                vertex.inedge(i)->set_data(vertex.id() + gcontext.iteration);
                
                if (std::rand()  % 4 == 1) {
                    vertex.remove_inedge(i);
                    __sync_add_and_fetch(&ndeleted, 1);
                } else {
                    ninedges++;
                }
            }
        }
        
        if (gcontext.iteration == gcontext.num_iterations - 1) {
            vertex.set_data(gcontext.iteration + 1);
        } else {
            vertex.set_data(ninedges);
        }
    }
    
    /**
     * Called before an iteration starts.
     */
    void before_iteration(int iteration, graphchi_context &gcontext) {
        ndeleted = 0;
    }
    
    /**
     * Called after an iteration has finished.
     */
    void after_iteration(int iteration, graphchi_context &gcontext) {
        if (gcontext.iteration > 0)
            assert(ndeleted > 0);
        logstream(LOG_INFO) << "Deleted: " << ndeleted << std::endl;
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
        assert(vecvalue == iters);
        total += iters;
    }
};

int main(int argc, const char ** argv) {
    /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
    graphchi_init(argc, argv);
    
    /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
    metrics m("smoketest-dynamic-engine2");
    
    /* Basic arguments for application */
    std::string filename = get_option_string("file");  // Base filename
    int niters           = get_option_int("niters", 4); // Number of iterations
    bool scheduler       = false;                       // Whether to use selective scheduling
    
    /* Detect the number of shards or preprocess an input to creae them */
    int nshards          = convert_if_notexists<EdgeDataType>(filename, 
                                                              get_option_string("nshards", "auto"));
    
    /* Run */
    SmokeTestProgram2 program;
    graphchi_dynamicgraph_engine<VertexDataType, EdgeDataType> engine(filename, nshards, scheduler, m); 
    engine.run(program, niters);
    
    /* Check also the vertex data is ok */
    VertexDataChecker vchecker(niters);
    foreach_vertices(filename, 0, engine.num_vertices(), vchecker);
    assert(vchecker.total == engine.num_vertices() * niters);
    
    /* Report execution metrics */
    metrics_report(m);
    
    logstream(LOG_INFO) << "Dynamic Engine Smoketest passed successfully! Your system is working!" << std::endl;
    return 0;
}

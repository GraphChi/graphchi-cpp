

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
 * Creates a graph with edge data for each edge and loads it and checks
 * the initial values were read correctly.
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

size_t checksum = 0;
size_t shouldbe = 0;

/**
 * Smoke test.
 */
struct DynamicDataLoaderTestProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    mutex lock;
    
    /**
     *  Vertex update function.
     */
    void update(graphchi_vertex<VertexDataType, EdgeDataType > &vertex, graphchi_context &gcontext) {
        for(int i=0; i < vertex.num_edges(); i++) {
            chivector<vid_t> * evector = vertex.edge(i)->get_vector();
            assert(evector != NULL);
            assert(evector->size() == 1);
            
            assert(evector->get(0) == vertex.id() + vertex.edge(i)->vertex_id());
            lock.lock();
            checksum += evector->get(0);
            lock.unlock();
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


void generatedata(std::string filename);
void generatedata(std::string filename) {
    const char * fname = filename.c_str();
    FILE * f = fopen(fname, "w");
    set_conf("filetype", "edgelist");
    shouldbe = 0;
    int totalVertices = 10000;
    for(int i=0; i < totalVertices; i++) {
        int nedges = random() % 200;
        for(int j=0; j < nedges; j++) {
            int dst = random() % totalVertices;
            if (dst != i) {
                fprintf(f, "%d\t%d\t%d\n", i, dst, i + dst);
                shouldbe += 2 * (i + dst); 
            }
        }
    }
    fclose(f);
}


int main(int argc, const char ** argv) {
    /* GraphChi initialization will read the command line
     arguments and the configuration file. */
    graphchi_init(argc, argv);
    
    /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
    metrics m("smoketest");
    
    /* Basic arguments for application */
    system("rm -rf /tmp/__chi_dyntest"); // Remove old

    std::string filename = "/tmp/__chi_dyntest/testgraph";  // Base filename
    mkdir("/tmp/__chi_dyntest", 0777);
    int niters           = 1; // Number of iterations
    bool scheduler       = false;                       // Whether to use selective scheduling
    
    /* Generate data */
    generatedata(filename);
    int nshards          = convert_if_notexists<int>(filename, "3");
    
    checksum = 0;
    
    /* Run */
    DynamicDataLoaderTestProgram program;
    graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, scheduler, m);
    engine.run(program, niters);
    
    // Clean up
    remove("/tmp/__chi_dyntest"); // Remove old

    /* Check */
    std::cout << "Checksum: " << checksum << ", expecting: " << shouldbe << std::endl;
    assert(shouldbe == checksum);
    
    /* Report execution metrics */
    metrics_report(m);
    
    logstream(LOG_INFO) << "Smoketest passed successfully! Your system is working!" << std::endl;
    return 0;
}

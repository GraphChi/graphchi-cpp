

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
#define DYNAMICVERTEXDATA 1   

#include <string>

#include "graphchi_basic_includes.hpp"
#include "api/dynamicdata/chivector.hpp"

using namespace graphchi;

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program.
 */
struct header_t {
    vid_t a;
    bool b;
    header_t() {}
    header_t(vid_t a, bool b): a(a), b(b) {}
};

typedef chivector<size_t, header_t> VertexDataType;
typedef chivector<vid_t, header_t>  EdgeDataType;

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
            chivector<vid_t, header_t> * evector = vertex.edge(i)->get_vector();
            assert(evector != NULL);
            
            
            // Each edge has three or one values in the chi vector
            int numelems = ((vertex.id() + vertex.edge(i)->vertex_id()) % 3 == 1 ? 3 : 1);

            for(int k=0; k < numelems ; k++) {

                vid_t expected = vertex.id() + vertex.edge(i)->vertex_id() + k;
                if (expected != evector->get(k)) {
                    logstream(LOG_ERROR) << "Vertex " << vertex.id() << ", edge dst: " << vertex.edge(i)->vertex_id() << std::endl;
                    logstream(LOG_ERROR) << "Mismatch (" << k << "): expected " << expected << " but had " << evector->get(k) << std::endl;
                }
                assert(evector->get(k) == expected);
            }
            
            assert(evector->header().a == vertex.id() + vertex.edge(i)->vertex_id());
            assert(evector->header().b == gcontext.iteration %2);
            evector->header().b  = !evector->header().b;
            
            lock.lock();
            checksum += evector->get(0);
            lock.unlock();
        }
        
         // Modify vertex data by adding values there */
         chivector<size_t, header_t> * vvector = vertex.get_vector();
         int numitems = vertex.id() % 10;
         for(int i=0; i<numitems; i++) {
             vvector->add(vertex.id() * 982192l + i); // Arbitrary
         }
        
         /* Check vertex data immediatelly */
         for(int i=0; i<numitems; i++) {
             size_t x = vvector->get(i);
             size_t expected = vertex.id() * 982192l + i;
             assert(x == expected);
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


int generatedata(std::string filename);
int generatedata(std::string filename) {
    sharder<vid_t, header_t> sharderobj(filename);
    sharderobj.start_preprocessing();
    
    std::cout << "Generating data..." << std::endl;
  
    shouldbe = 0;
    int totalVertices =  1000;
    for(int i=0; i < totalVertices; i++) {
        int nedges = random() % 10;
        for(int j=0; j < nedges; j++) {
            int dst = (totalVertices / nedges) * j + i % nedges; 
            if (dst != i) {
                std::vector<vid_t> edgevec;
                if ((i + dst) % 3 == 1) {
                    edgevec.push_back(i + dst + 1);
                    edgevec.push_back(i + dst + 2);
                } else {
                    edgevec.push_back(i + dst);
                    edgevec.push_back(i + dst + 1);
                }
                shouldbe += 2 * (i + dst);
                
                sharderobj.preprocessing_add_edge_multival(i, dst, header_t(i + dst, true), edgevec);
            }
        }
    }     
    sharderobj.end_preprocessing();

    return sharderobj.execute_sharding(get_option_string("nshards", "auto"));

}

class VertexValidator : public VCallback<VertexDataType> {
public:
    virtual void callback(vid_t vertex_id, VertexDataType &vec) {
        int numitems = vertex_id % 10;
        assert(vec.size() == numitems);
        
        for(int j=0; j < numitems; j++) {
            size_t x = vec.get(j);
            assert(x == vertex_id * 982192l + (size_t)j);
        }
    }
};





int main(int argc, const char ** argv) {
    /* GraphChi initialization will read the command line
     arguments and the configuration file. */
    graphchi_init(argc, argv);
    
    /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
    metrics m("test-dynamicedata");
    
    /* Basic arguments for application */

    std::string filename = "/tmp/__chi_dyntest/testgraph";  // Base filename
    mkdir("/tmp/__chi_dyntest", 0777);
    int niters           = 1; // Number of iterations
    bool scheduler       = false;                       // Whether to use selective scheduling
        
    /* Generate data */
    int nshards          = generatedata(filename);
   
    checksum = 0;
  
    /* Run */
    DynamicDataLoaderTestProgram program;
    graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, scheduler, m);
    engine.set_reset_vertexdata(true);
    engine.run(program, niters);
    
    /* Check */
    std::cout << "Checksum: " << checksum << ", expecting: " << shouldbe << std::endl;
    assert(shouldbe == checksum);
    
    /* Check vertex values */
    VertexValidator validator;
    foreach_vertices(filename, 0, engine.num_vertices(), validator);
    
    
    /* Clean up */
    delete_shards<EdgeDataType>(filename, 3);

    
    /* Report execution metrics */
    metrics_report(m);
    
    logstream(LOG_INFO) << "Test passed successfully! Your system is working!" << std::endl;
    return 0;
}

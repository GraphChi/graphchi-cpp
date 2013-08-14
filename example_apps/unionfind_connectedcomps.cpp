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
 * Connected components that uses the Union-Find algorithm. Requires
 * O(|V|) of RAM, but only one pass of the data. Thus much faster than
 * the completely disk based "connectedcomponents.cpp" example app.
 *
 * Highly optimized non-idiomatic GraphChi code that uses an overloaded vertex
 * class to prevent actually creating the graph in memory.
 *
 * NOTE/REMARK: THERE IS NO REAL REASON TO USE GRAPHCHI FOR THIS ALGORITHM.
 * A SIMPLE CODE THAT READ THE GRAPH ONE EDGE A TIME WOULD BE SUFFICIENT.
 *
 */

#define GRAPHCHI_DISABLE_COMPRESSION

#include <string>
#include "graphchi_basic_includes.hpp"
#include "util/labelanalysis.hpp"

using namespace graphchi;


vid_t *  sets;
unsigned int * setCounts; // Union-find


/* Find operator of Union-Find with path compression */
vid_t Find(vid_t x);
inline vid_t Find(vid_t x) {
    while (sets[x] != x) {
        x = sets[x] = sets[sets[x]];
    }
    return sets[x];
}

typedef vid_t VertexDataType;
typedef bool EdgeDataType; // not relevant

size_t ne = 0;

 class UnionFindVertex : public graphchi_vertex<VertexDataType, EdgeDataType> {
public:
    
        
    UnionFindVertex() : graphchi_vertex<VertexDataType, EdgeDataType> () {}
    
    UnionFindVertex(vid_t _id, 
                      graphchi_edge<EdgeDataType> * iptr, 
                      graphchi_edge<EdgeDataType> * optr, 
                      int indeg, 
                      int outdeg) : 
    graphchi_vertex<VertexDataType, EdgeDataType> (_id, NULL, NULL, indeg, outdeg) { 
    }
    
    inline void add_inedge(vid_t src, EdgeDataType * ptr, bool special_edge) {
        vid_t setDst = Find(this->vertexid);
        vid_t setSrc = Find(src);
        // If in same component, nothing to do, otherwise, Unite
        if (setSrc != setDst) {
            if (setCounts[setSrc] > setCounts[setDst]) {
                // A is bigger set, merge with A
                sets[setDst] = setSrc;
                setCounts[setSrc] += setCounts[setDst];
            } else {
                // or vice versa
                sets[setSrc] = setDst;
                setCounts[setDst] += setCounts[setSrc];
            }
        }
        ne++;
    }
    
       
    void add_outedge(vid_t dst, EdgeDataType * ptr, bool special_edge) {
        assert(false);
    }
    
    bool computational_edges() {
        return true;
    }
};






/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct UnionFindProgram : public GraphChiProgram<VertexDataType, EdgeDataType, UnionFindVertex> {
    
    
    /**
     *  Vertex update function.
     */
    void update(UnionFindVertex &vertex, graphchi_context &gcontext) {
        // do nothing -- all done in the special vertex class
    }
    
    /**
     * Called before an iteration starts.
     */
    void before_iteration(int iteration, graphchi_context &gcontext) {
        /* Initialize */
        sets = new vid_t[gcontext.nvertices];
        for(vid_t i=0; i<gcontext.nvertices; i++) sets[i] = i;
        setCounts = new unsigned int[gcontext.nvertices];
        // All sets start with 1
        for(vid_t i=0; i<gcontext.nvertices; i++) setCounts[i] = 1;

    }
    
    /**
     * Called after an iteration has finished.
     */
    void after_iteration(int iteration, graphchi_context &gcontext) {
        
        // Now find everyone
        logstream(LOG_INFO) << "Final finds..." << std::endl;
        for(size_t i=0; i<gcontext.nvertices; i++) {
            sets[i] = Find(i);
        }
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
    metrics m("union-find-connectedcomponents");
    
    /* Basic arguments for application */
    std::string filename = get_option_string("file");  // Base filename
    int niters           = 1;
    
    /* Detect the number of shards or preprocess an input to create them */
    int nshards          = convert_if_notexists_novalues<EdgeDataType>(filename, 
                                                              get_option_string("nshards", "auto"));
    
    // Always run with only thread only (code is not thread-safe)
    set_conf("execthreads", "1");
    
    /* Run */
    UnionFindProgram unionFind;
    graphchi_engine<VertexDataType, EdgeDataType, UnionFindVertex > engine(filename, nshards, false, m); 
    engine.set_disable_outedges(true);
    engine.set_only_adjacency(true);
    engine.set_modifies_inedges(false);
    engine.set_disable_vertexdata_storage();
    engine.run(unionFind, niters);
    
    /* Write vertex data */
    std::string outputfile = filename_vertex_data<VertexDataType>(filename);
    
    
    FILE * f = fopen(outputfile.c_str(), "w");
    fwrite(sets, sizeof(vid_t), engine.num_vertices(), f);
    fclose(f);
    
    /* Analyze */
    analyze_labels<vid_t>(filename);

    /* Report execution metrics */
    metrics_report(m);
    return 0;
}


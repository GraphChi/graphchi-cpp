
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
 * Alternative "functional" API for GraphChi. The API is implemented as a 
 * layer on top of the standard API, but uses a specialized engine "functional_engine",
 * which processes the graph data in different order. Namely, it first loads in-edges,
 * then executes updates, and finally writes new values (broadcasts) to out-edges.
 */


#ifndef GRAPHCHI_FUNCTIONALAPI_DEF
#define GRAPHCHI_FUNCTIONALAPI_DEF

#include <assert.h>

#include "api/graph_objects.hpp"
#include "api/graphchi_context.hpp"
#include "engine/functional/functional_engine.hpp"
#include "metrics/metrics.hpp"
#include "graphchi_types.hpp"

#include "api/functional/functional_defs.hpp"
#include "api/functional/functional_semisync.hpp"
#include "api/functional/functional_bulksync.hpp"
#include "preprocessing/conversions.hpp"

namespace graphchi {
       
    /**
      * Superclass for kernels
      */
     template <typename FVertexDataType, typename FEdgeDataType>
     struct functional_kernel {
        
        typedef FVertexDataType VertexDataType;
        typedef FEdgeDataType EdgeDataType;
        
        functional_kernel() {}
         
        /* Initial value - on first iteration */
         virtual VertexDataType initial_value(graphchi_context &info, vertex_info& myvertex) = 0;        
        /* Called before first "gather" */
         virtual VertexDataType reset()  = 0;        
        // Note: Unweighted version, edge value should also be passed
        // "Gather"
        virtual EdgeDataType op_neighborval(graphchi_context &info, vertex_info& myvertex, vid_t nbid, EdgeDataType nbval)= 0;         
        // "Sum"
        virtual EdgeDataType plus(VertexDataType curval, EdgeDataType toadd) = 0;
         // "Apply"
        virtual VertexDataType compute_vertexvalue(graphchi_context &ginfo, vertex_info& myvertex, EdgeDataType nbvalsum) = 0;        
        // "Scatter
        virtual EdgeDataType value_to_neighbor(graphchi_context &info, vertex_info& myvertex, vid_t nbid, VertexDataType myval) = 0;        
    }; 

    
    
    /** 
     * Run a functional kernel with unweighted edges.
     * The semantics of this API are
     * less well-defined than the standard one, because this API is "semi-synchronous". That is, 
     * inside a sub-interval, new values of neighbors are not observed, but 
     * next sub-interval will observe the new values. 
     * 
     * See application "pagerank_functional" for an example. 
     * @param KERNEL needs to be a class/struct that subclasses the functional_kernel
     * @param filename base filename
     * @param nshards number of shards
     * @param niters number of iterations to run
     * @param _m metrics object
     */
    template <class KERNEL>
    void run_functional_unweighted_semisynchronous(std::string filename, int niters, metrics &_m) {
        FunctionalProgramProxySemisync<KERNEL> program;
         
        /* Process input file - if not already preprocessed */
        int nshards           
            = convert_if_notexists<typename FunctionalProgramProxySemisync<KERNEL>::EdgeDataType>(filename, get_option_string("nshards", "auto"));
    
        functional_engine<typename FunctionalProgramProxySemisync<KERNEL>::VertexDataType, 
            typename FunctionalProgramProxySemisync<KERNEL>::EdgeDataType,
            typename FunctionalProgramProxySemisync<KERNEL>::fvertex_t > 
                engine(filename, nshards, false, _m);
        
        
        engine.set_modifies_inedges(false); // Important
        engine.set_modifies_outedges(true); // Important
        engine.run(program, niters);
    }
    
    
    /** 
     * Run a functional kernel with unweighted edges in the bulk-synchronous model.
     * Note: shards need to have space to store two values for each edge.
     * 
     * See application "pagerank_functional" for an example. 
     * @param filename base filename
     * @param nshards number of shards
     * @param niters number of iterations to run
     * @param _m metrics object
     */
    template <class KERNEL>
    void run_functional_unweighted_synchronous(std::string filename, int niters, metrics &_m) {
        FunctionalProgramProxyBulkSync<KERNEL> program;
        int nshards           
            = convert_if_notexists<typename FunctionalProgramProxyBulkSync<KERNEL>::EdgeDataType>(filename, get_option_string("nshards", "auto"));
     
        functional_engine<typename FunctionalProgramProxyBulkSync<KERNEL>::VertexDataType,
            typename FunctionalProgramProxyBulkSync<KERNEL>::EdgeDataType,
            typename FunctionalProgramProxyBulkSync<KERNEL>::fvertex_t > 
                engine(filename, nshards, false, _m);
        engine.set_modifies_inedges(false); // Important
        engine.set_modifies_outedges(true); // Important
        engine.set_enable_deterministic_parallelism(false); // Bulk synchronous does not need consistency.
        engine.run(program, niters);
    }
    
}

#endif


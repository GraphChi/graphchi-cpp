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
 * Semi-synchronous implementation of the functional API.
 */


#ifndef GRAPHCHI_FUNCTIONAL_SEMISYNC_DEF
#define GRAPHCHI_FUNCTIONAL_SEMISYNC_DEF

#include <assert.h>


#include "api/graph_objects.hpp"
#include "api/graphchi_context.hpp"
#include "api/functional/functional_defs.hpp"

#include "metrics/metrics.hpp"
#include "graphchi_types.hpp"

namespace graphchi {

template <typename KERNEL>
class functional_vertex_unweighted_semisync  {
public:
    
    typedef typename KERNEL::VertexDataType VT;
    typedef typename KERNEL::EdgeDataType ET;
    
    VT cumval;
    
    KERNEL kernel;
    vertex_info vinfo;
    graphchi_context * gcontext;
    
    // Dummy
    bool inc;
    bool outc;
    bool scheduled;
    bool modified;
    bool parallel_safe;
    VT * dataptr;

    functional_vertex_unweighted_semisync() {}
    
    functional_vertex_unweighted_semisync(graphchi_context &ginfo, vid_t _id, int indeg, int outdeg) { 
        vinfo.indegree = indeg;
        vinfo.outdegree = outdeg;
        vinfo.vertexid = _id;
        cumval = kernel.reset();
        gcontext = &ginfo;
    }
    
    functional_vertex_unweighted_semisync(vid_t _id, 
                                 graphchi_edge<ET> * iptr, 
                                 graphchi_edge<ET> * optr, 
                                 int indeg, 
                                 int outdeg) {
        assert(false); // This should never be called.
    }
    
    void first_iteration(graphchi_context &gcontext_) {
        this->set_data(kernel.initial_value(gcontext_, vinfo));
    }
    
    // Optimization: as only memshard (not streaming shard) creates inedgers,
    // we do not need atomic instructions here!
    inline void add_inedge(vid_t src, ET * ptr, bool special_edge) {
        if (gcontext->iteration > 0) {
            get_lock(vinfo.vertexid).lock();
            cumval = kernel.plus(cumval, kernel.op_neighborval(*gcontext, vinfo, src, *ptr));
            get_lock(vinfo.vertexid).unlock();
        } 
    }
    
    void ready(graphchi_context &gcontext_) {
        this->set_data(kernel.compute_vertexvalue(gcontext_, vinfo, cumval));
    }
    
    inline void add_outedge(vid_t dst, ET * ptr, bool special_edge) {
        *ptr = kernel.value_to_neighbor(*gcontext, vinfo, dst, this->get_data());
    }
    
    bool computational_edges() {
        return true;
    }
    
    /* Outedges do not need to be read, they just need to be written */
    static bool read_outedges() {
        return false;
    }
    
    
    /**
     * Modify the vertex value. The new value will be
     * stored on disk.
     */
    virtual void set_data(VT d) {
        *(this->dataptr) = d;
        this->modified = true;
    }
    
    VT get_data() {
        return *(this->dataptr);
    }
    
};



template <typename KERNEL>
    class FunctionalProgramProxySemisync : public GraphChiProgram<typename KERNEL::VertexDataType, typename  KERNEL::EdgeDataType, functional_vertex_unweighted_semisync<KERNEL> > {
    public:  
    typedef typename KERNEL::VertexDataType VertexDataType;
    typedef typename KERNEL::EdgeDataType EdgeDataType;
    typedef functional_vertex_unweighted_semisync<KERNEL> fvertex_t;
    
    /**
     * Called before an iteration starts.
     */
    void before_iteration(int iteration, graphchi_context &info) {
    }
    
    /**
     * Called after an iteration has finished.
     */
    void after_iteration(int iteration, graphchi_context &ginfo) {
    }
    
    /**
     * Called before an execution interval is started.
     */
    void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &ginfo) {        
    }
    
    /**
     * Pagerank update function.
     */
    void update(fvertex_t &v, graphchi_context &ginfo) {
        if (ginfo.iteration == 0) {
            v.first_iteration(ginfo);
        } else { 
            v.ready(ginfo);
        }   
    }
    
};

}


#endif



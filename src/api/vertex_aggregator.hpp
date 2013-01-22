


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
 * Simple vertex-aggregators/scanners which allows reductions over all vertices
 * in an I/O efficient manner.
 */


#ifndef DEF_GRAPHCHI_VERTEX_AGGREGATOR
#define DEF_GRAPHCHI_VERTEX_AGGREGATOR

#include <errno.h>
#include <memory.h>
#include <string>

#include "graphchi_types.hpp"
#include "api/chifilenames.hpp"
#include "io/stripedio.hpp"
#include "util/ioutil.hpp"
#include "engine/auxdata/vertex_data.hpp"

namespace graphchi {
    
    /** 
      * Abstract class for callbacks that are invoked for each 
      * vertex when foreach_vertices() is called (see below).
      */
    template <typename VertexDataType>
    class VCallback {
    public:
        virtual void callback(vid_t vertex_id, VertexDataType &value) = 0;
    };
    
    
    /**
      * Foreach: a callback object is invoked for every vertex in the given range.
      * See VCallback above.
      * @param basefilename base filename
      * @param fromv first vertex
      * @param tov last vertex (exclusive)
      * @param callback user-defined callback-object.
      */
    template <typename VertexDataType>
    void foreach_vertices(std::string basefilename, vid_t fromv, vid_t tov, VCallback<VertexDataType> &callback) {
        std::string filename = filename_vertex_data<VertexDataType>(basefilename);
        metrics m("foreach");
        stripedio * iomgr = new stripedio(m);
        
        vid_t readwindow = 1024 * 1024;
        size_t numvertices = get_num_vertices(basefilename);
        vertex_data_store<VertexDataType> * vertexdata =
            new vertex_data_store<VertexDataType>(basefilename, numvertices, iomgr);
        
        vid_t st = fromv;
        vid_t en = 0;
        while(st <= tov) {
            en = st + readwindow - 1;
            if (en >= tov) en = tov - 1;
            
            if (st < en) {
                vertexdata->load(st, en);
                for(vid_t v=st; v<=en; v++) {
                    VertexDataType * vptr = vertexdata->vertex_data_ptr(v);
                    callback.callback(v, (VertexDataType&) *vptr);
                }
            }
            
            st += readwindow;
        }
        delete vertexdata;
        delete iomgr;
    }
    
    /**
      * Callback for computing a sum.
      * TODO: a functional version instead of imperative.
      */
    template <typename VertexDataType, typename SumType>
    class SumCallback : public VCallback<VertexDataType> {
    public:
        SumType accum;
        SumCallback(SumType initval) : VCallback<VertexDataType>() {
            accum = initval;
        }

        void callback(vid_t vertex_id, VertexDataType &value) {
            accum += value;
        }
    };
    
    /** 
      * Computes a sum over a range of vertices' values.
      * Type SumType defines the accumulator type, which may be different
      * than vertex type. For example, often vertex value is 32-bit 
      * integer, but the sum will need to be 64-bit integer.
      * @param basefilename base filename
      * @param fromv first vertex
      * @param tov last vertex (exclusive)
      */
    template <typename VertexDataType, typename SumType>
    SumType sum_vertices(std::string base_filename, vid_t fromv, vid_t tov) {
        SumCallback<VertexDataType, SumType> sumc(0);
        foreach_vertices<VertexDataType>(base_filename, fromv, tov, sumc);
        return sumc.accum;
    }
    
}


#endif



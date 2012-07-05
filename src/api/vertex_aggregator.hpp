


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
#include "util/ioutil.hpp"

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
        int f = open(filename.c_str(), O_RDONLY);
        if (f < 0) {
            logstream(LOG_ERROR) << "Could not open file: " << filename << 
            " error: " << strerror(errno) << std::endl;
            assert(false);
        }
        size_t bufsize = 1024 * 1024; // Read one megabyte a time    
        vid_t nbuf = (vid_t) (bufsize / sizeof(VertexDataType));
        bufsize = sizeof(VertexDataType) * nbuf; 
        
        VertexDataType * buffer = (VertexDataType*) calloc(nbuf, sizeof(VertexDataType));
        
        for(vid_t v=fromv; v < tov; v += nbuf) {
            size_t nelements = std::min(tov, v + nbuf) - v;
            preada(f, buffer, nelements * sizeof(VertexDataType), v * sizeof(VertexDataType));
            
            for(int i=0; i < (int)nelements; i++) {
                callback.callback(i + v, buffer[i]);
            }
        }
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

        virtual void callback(vid_t vertex_id, VertexDataType &value) {
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



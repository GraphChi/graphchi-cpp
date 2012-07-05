


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
    
    template <typename VertexDataType>
    class VCallback {
    public:
        virtual void callback(vid_t vertex_id, VertexDataType &value) = 0;
    };
    
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
    
    template <typename VertexDataType, typename SumType>
    SumType sum_vertices(std::string base_filename, vid_t fromv, vid_t tov) {
        SumCallback<VertexDataType, SumType> sumc(0);
        foreach_vertices<VertexDataType>(base_filename, fromv, tov, sumc);
        return sumc.accum;
    }
    
}


#endif



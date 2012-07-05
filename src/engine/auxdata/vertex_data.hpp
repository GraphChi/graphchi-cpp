

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
 * The class manages vertex values (vertex data).
 */

/* Note: This class shares a lot of code with the degree_data.hpp. It might be
   useful to have a common base class "sequential-file". */

#ifndef DEF_GRAPHCHI_VERTEXDATA
#define DEF_GRAPHCHI_VERTEXDATA

#include <stdlib.h>
#include <string>
#include <assert.h>

#include "graphchi_types.hpp"
#include "api/chifilenames.hpp"
#include "io/stripedio.hpp"
#include "util/ioutil.hpp"

namespace graphchi {

    template <typename VertexDataType>
    class vertex_data_store {

    protected:
        
        stripedio * iomgr;
        
        /* Current range of vertices in memory */
        vid_t vertex_st;
        vid_t vertex_en;
        
        std::string filename;
        int filedesc;
        
        VertexDataType * loaded_chunk;


        virtual void open_file(std::string base_filename) {
            filedesc = iomgr->open_session(filename.c_str(), false);
        }
        
    public:
        
        vertex_data_store(std::string base_filename, size_t nvertices, stripedio * iomgr) : iomgr(iomgr), loaded_chunk(NULL){
            vertex_st = vertex_en = 0;
            filename = filename_vertex_data<VertexDataType>(base_filename);
            check_size(nvertices);
            iomgr->allow_preloading(filename);
            open_file(filename);
        }
        
        virtual ~vertex_data_store() {
            iomgr->close_session(filedesc);
            iomgr->wait_for_writes();
            if (loaded_chunk != NULL) {
                iomgr->managed_release(filedesc, &loaded_chunk);
            }    
        }
        
        void check_size(size_t nvertices) {
            checkarray_filesize<VertexDataType>(filename, nvertices);
        }
        
        void clear(size_t nvertices) {
            check_size(0);
            check_size(nvertices);
        }
        
        /**
         * Loads a chunk of vertex values
         * @param vertex_st first vertex id
         * @param vertex_en last vertex id, inclusive
         */
        virtual void load(vid_t _vertex_st, vid_t _vertex_en) {
            assert(_vertex_en >= _vertex_st);
            vertex_st = _vertex_st;
            vertex_en = _vertex_en;
            
            size_t datasize = (vertex_en - vertex_st + 1)* sizeof(VertexDataType);
            size_t datastart = vertex_st * sizeof(VertexDataType);
            
            if (loaded_chunk != NULL) {
                iomgr->managed_release(filedesc, &loaded_chunk);
            }
            
            iomgr->managed_malloc(filedesc, &loaded_chunk, datasize, datastart);
            iomgr->managed_preada_now(filedesc, &loaded_chunk, datasize, datastart);
        }
        
        /**
          * Saves the current chunk of vertex values
          */
        virtual void save(bool async=false) {
            assert(loaded_chunk != NULL); 
            size_t datasize = (vertex_en - vertex_st + 1) * sizeof(VertexDataType);
            size_t datastart = vertex_st * sizeof(VertexDataType);
            if (async) {
                iomgr->managed_pwritea_async(filedesc, &loaded_chunk, datasize, datastart, false);
            } else {
                iomgr->managed_pwritea_now(filedesc, &loaded_chunk, datasize, datastart);
            }
        }
        
        
        /**
         * Returns id of the first vertex currently in memory. Fails if nothing loaded yet.
         */
        vid_t first_vertex_id() {
            assert(loaded_chunk != NULL);
            return vertex_st; 
        }  
        
        
        VertexDataType * vertex_data_ptr(vid_t vertexid) {
            assert(vertexid >= vertex_st && vertexid <= vertex_en);
            return &loaded_chunk[vertexid - vertex_st];
        }   
        
        
    };
}

#endif


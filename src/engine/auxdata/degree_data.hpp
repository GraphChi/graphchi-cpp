 
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
 * The class manages information about vertex degree, and allows
 * sequential block access to the degree data file.
 */

#ifndef DEF_GRAPHCHI_DEGREE_DATA
#define DEF_GRAPHCHI_DEGREE_DATA

#include <fstream>
#include <assert.h>
#include <string>
#include <stdlib.h>

#include "graphchi_types.hpp"
#include "io/stripedio.hpp"

namespace graphchi {
    
    
    struct degree {
        int indegree;
        int outdegree;
    };
    
    
    class degree_data {
    
    
    protected:
        /* Current range of vertices in memory */
        vid_t vertex_st;
        vid_t vertex_en;
        stripedio * iomgr;
        
        /* Current chunk in memory */
        degree * loaded_chunk;
        std::string filename;
        int filedesc;
     
        virtual void open_file(std::string base_filename) {
            filename = filename_degree_data(base_filename);
            filedesc = iomgr->open_session(filename.c_str(), false);
        }
        
    public:
        
        /**
          * Constructor
          * @param base_filename base file prefix
          */
        degree_data(std::string base_filename, stripedio * iomgr) : iomgr(iomgr), loaded_chunk(NULL) {
            vertex_st = vertex_en = 0;
            open_file(base_filename);
        }
        
        virtual ~degree_data() {
            if (loaded_chunk != NULL) {
                iomgr->managed_release(filedesc, &loaded_chunk);
            }        
            iomgr->close_session(filedesc);
        }
                
        /**
          * Loads a chunk of vertex degrees
          * @param vertex_st first vertex id
          * @param vertex_en last vertex id, inclusive
          */
        virtual void load(vid_t _vertex_st, vid_t _vertex_en) {
            assert(_vertex_en >= _vertex_st);
            vertex_st = _vertex_st;
            vertex_en = _vertex_en;
            
            size_t datasize = (vertex_en - vertex_st + 1) * sizeof(degree);
            size_t datastart = vertex_st * sizeof(degree);
            
            if (loaded_chunk != NULL) {
                iomgr->managed_release(filedesc, &loaded_chunk);
            }
            
            iomgr->managed_malloc(filedesc, &loaded_chunk, datasize, datastart);
            iomgr->managed_preada_now(filedesc, &loaded_chunk, datasize, datastart);
        }
        
        /**
          * Returns id of the first vertex currently in memory. Fails if nothing loaded yet.
          */
        vid_t first_vertex_id() {
            assert(loaded_chunk != NULL);
            return vertex_st; 
        }  
        
        virtual void set_degree(vid_t vertexid, int indegree, int outdegree) {
            assert(vertexid >= vertex_st && vertexid <= vertex_en);
            loaded_chunk[vertexid - vertex_st].indegree = indegree;
            loaded_chunk[vertexid - vertex_st].outdegree = outdegree;
        }
        
        virtual void set_degree(vid_t vertexid, degree d) {
            assert(vertexid >= vertex_st && vertexid <= vertex_en);
            loaded_chunk[vertexid - vertex_st] = d;
        }
        
        inline degree get_degree(vid_t vertexid) {
            assert(vertexid >= vertex_st && vertexid <= vertex_en);
            return loaded_chunk[vertexid - vertex_st];
        }   
        
        void save() {
            size_t datasize = (vertex_en - vertex_st + 1) * sizeof(degree);
            size_t datastart = vertex_st * sizeof(degree);

            iomgr->managed_pwritea_now(filedesc, &loaded_chunk, datasize, datastart);
        }
        
        void ensure_size(vid_t maxid) {
            iomgr->truncate(filedesc, (1 + maxid) * sizeof(degree));
        }
        
    };
    
}


#endif


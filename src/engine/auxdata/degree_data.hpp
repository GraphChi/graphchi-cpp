 
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
#include <sys/mman.h>

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
        std::string base_filename;

        bool modified;
        
        int filedesc;
        
        bool use_mmap;
        degree * mmap_file;
        size_t mmap_length;
     
        virtual void open_file(std::string base_filename) {
            filename = filename_degree_data(base_filename);
            modified = false;
            if (!use_mmap) {
                filedesc = iomgr->open_session(filename.c_str(), false);
            } else {
                mmap_length = get_filesize(filename);
                filedesc = open(filename.c_str(), O_RDWR);
                mmap_file = (degree *) mmap(NULL, mmap_length, PROT_READ | PROT_WRITE, MAP_SHARED, filedesc, 0);
                assert(mmap_file);
            }
        }
        
    public:
        
        /**
          * Constructor
          * @param base_filename base file prefix
          */
        degree_data(std::string base_filename, stripedio * iomgr) : iomgr(iomgr), loaded_chunk(NULL) {
            vertex_st = vertex_en = 0;
            this->base_filename = base_filename;
            use_mmap = get_option_int("mmap", 0);  // Whether to mmap the degree file to memory
            if (use_mmap) {
                logstream(LOG_INFO) << "Use memory mapping for degree data." << std::endl;
            }
            open_file(base_filename);
        }
        
        virtual ~degree_data() {
            if (!use_mmap) {
                if (loaded_chunk != NULL) {
                    iomgr->managed_release(filedesc, &loaded_chunk);
                }        
                iomgr->close_session(filedesc);
            } else {
                if (modified) {
                    msync(mmap_file, mmap_length, MS_SYNC);
                }
                munmap(mmap_file, mmap_length);
                close(filedesc);
            }
        }
                
        /**
          * Loads a chunk of vertex degrees
          * @param vertex_st first vertex id
          * @param vertex_en last vertex id, inclusive
          */
        virtual void load(vid_t _vertex_st, vid_t _vertex_en) {
            if (!use_mmap) {
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
        }
        
        /**
          * Returns id of the first vertex currently in memory. Fails if nothing loaded yet.
          */
        vid_t first_vertex_id() {
            if (!use_mmap) {
                assert(loaded_chunk != NULL);
                return vertex_st;
            } else {
                return 0;
            }
        }  
        
        virtual void set_degree(vid_t vertexid, int indegree, int outdegree) {
            modified = true;
            if (!use_mmap) {
                assert(vertexid >= vertex_st && vertexid <= vertex_en);
                loaded_chunk[vertexid - vertex_st].indegree = indegree;
                loaded_chunk[vertexid - vertex_st].outdegree = outdegree;
            } else {
                mmap_file[vertexid].indegree = indegree;
                mmap_file[vertexid].outdegree = outdegree;
            }
        }
        
        virtual void set_degree(vid_t vertexid, degree d) {
            modified = true;
            if (!use_mmap) {
                assert(vertexid >= vertex_st && vertexid <= vertex_en);
                loaded_chunk[vertexid - vertex_st] = d;
            } else {
                mmap_file[vertexid] = d;
            }
        }
        
        inline degree get_degree(vid_t vertexid) {
            if (!use_mmap) {
                assert(vertexid >= vertex_st && vertexid <= vertex_en);
                return loaded_chunk[vertexid - vertex_st];
            } else {
                return mmap_file[vertexid];
            }
        }
        
        void save() {
            if (!use_mmap) {
                size_t datasize = (vertex_en - vertex_st + 1) * sizeof(degree);
                size_t datastart = vertex_st * sizeof(degree);
                iomgr->managed_pwritea_now(filedesc, &loaded_chunk, datasize, datastart);
            }
        }
        
        void ensure_size(vid_t maxid) {
            if (!use_mmap) {
                iomgr->truncate(filedesc, (1 + maxid) * sizeof(degree));
            } else {
                munmap(mmap_file, mmap_length);
                ftruncate(filedesc, (1 + maxid) * sizeof(degree));
                close(filedesc);
                open_file(base_filename);
            }
        }
        
    };
    
}


#endif


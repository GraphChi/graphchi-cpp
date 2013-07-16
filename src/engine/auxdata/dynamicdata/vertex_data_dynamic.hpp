

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
 * The class manages vertex values (vertex data) when the
 * vertex data is dynamic. That is, the vertex data type must
 * be a chivector.
 *
 * To enable dynamically sized data, vertex data must be stored in
 * small (1 million-vertex) blocks. 
 */ 


#ifndef DYNAMICVERTEXDATA
 ERROR(DYNAMICVERTEXDATA NEEDS TO BE DEFINED)
#endif

#ifndef DEF_GRAPHCHI_VERTEXDATA
#define DEF_GRAPHCHI_VERTEXDATA

#include <stdlib.h>
#include <string>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#include <assert.h>

#include "graphchi_types.hpp"
#include "api/chifilenames.hpp"
#include "io/stripedio.hpp"
#include "util/ioutil.hpp"
#include "api/dynamicdata/chivector.hpp"
#include "shards/dynamicdata/dynamicblock.hpp"

namespace graphchi {

    template <typename VertexDataType>
    struct vdblock_t {
        int blockid;
        int fd;
        uint8_t* data;
        dynamicdata_block<VertexDataType> * dblock;
        vdblock_t(int bid) : blockid(bid), data(NULL), dblock(NULL) {}
    };
    
    template <typename VertexDataType>
    class vertex_data_store {
        
        typedef vdblock_t<VertexDataType> vdblock;
    protected:
        
        stripedio * iomgr;
        
        /* Current range of vertices in memory */
        vid_t vertex_st;
        vid_t vertex_en;
        
        std::string dirname;
        size_t verticesperblock;
        
        VertexDataType * loaded_chunk;
        std::vector<vdblock> loadedblocks; // Blocks currently in memory
 
        
    public:
        
        vertex_data_store(std::string base_filename, size_t nvertices, stripedio * iomgr) : iomgr(iomgr), loaded_chunk(NULL){
            vertex_st = vertex_en = 0;
            verticesperblock = 1024 * 1024;

            dirname = filename_vertex_data<VertexDataType>(base_filename) + ".dynamic_blockdir";
            check_size(nvertices);            
        }    
        
        virtual ~vertex_data_store() {
            iomgr->wait_for_writes();
            releaseblocks();
        }
        
        
        void check_size(size_t nvertices) {
            int nblocks = (nvertices - 1) / verticesperblock + 1;
            for(int i=0; i < nblocks; i++) {
                init_block(i);
            }
        }
        
        void clear(size_t nvertices) {
            int nblocks = (nvertices - 1) / verticesperblock + 1;
            for(int i=0; i < nblocks; i++) {
                std::string bfilename = blockfilename(i);
                if (file_exists(bfilename)) {
                    remove(bfilename.c_str());
                }
                delete_block_uncompressed_sizefile(bfilename);
            }
        }
        
    private:
        std::string blockfilename(int blockid) {
            std::stringstream ss;
            ss << dirname;
            ss << "/";
            ss << blockid;
            return ss.str();
        }
        
        void releaseblocks() {
            for(int i=0; i < (int)loadedblocks.size(); i++) {
                delete(loadedblocks[i].dblock);
                iomgr->managed_release(loadedblocks[i].fd, &loadedblocks[i].data);
                iomgr->close_session(loadedblocks[i].fd);
                loadedblocks[i].data = NULL;
                loadedblocks[i].dblock = NULL;
            }
            loadedblocks.clear();
        }
        
        void init_block(int blockid) {
            std::string bfilename = blockfilename(blockid);
            if (!file_exists(bfilename)) {
                mkdir(dirname.c_str(), 0777);
                size_t initsize = verticesperblock * sizeof(typename VertexDataType::sizeword_t);
                int f = open(bfilename.c_str(),  O_RDWR | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
                uint8_t * zeros = (uint8_t *) calloc(verticesperblock, sizeof(typename VertexDataType::sizeword_t));
                write_compressed(f, zeros, initsize);
                free(zeros);

                write_block_uncompressed_size(bfilename, initsize);
                close(f);
            }
        }
    
    
        
        vdblock load_block(int blockid) {
            vdblock db(blockid);
            
            std::string blockfname = blockfilename(blockid);
            db.fd = iomgr->open_session(blockfname, false, true);
            int realsize = get_block_uncompressed_size(blockfname, -1);
            assert(realsize > 0);
            
            iomgr->managed_malloc(db.fd, &db.data, realsize, 0);
            iomgr->managed_preada_now(db.fd, &db.data, realsize, 0);
            db.dblock = new dynamicdata_block<VertexDataType>(verticesperblock, (uint8_t *)db.data, realsize);
            return db;
        }
        
        void write_block(vdblock &block) {
            int realsize;
            uint8_t * outdata;
            block.dblock->write(&outdata, realsize);
            std::string blockfname = blockfilename(block.blockid);
            iomgr->managed_pwritea_now(block.fd, &outdata, realsize, 0); /* Need to write whole block in the compressed regime */
            write_block_uncompressed_size(blockfname, realsize);
            free(outdata);
        }
        
    public:
        
        /**
         * Loads a chunk of vertex values
         * @param vertex_st first vertex id
         * @param vertex_en last vertex id, inclusive
         */
        virtual void load(vid_t _vertex_st, vid_t _vertex_en) {
            assert(_vertex_en >= _vertex_st);
            vertex_st = _vertex_st;
            vertex_en = _vertex_en;
            
            releaseblocks();
            
            int min_blockid = vertex_st / verticesperblock;
            int max_blockid = vertex_en / verticesperblock;
            for(int i=min_blockid; i <= max_blockid; i++) {
                loadedblocks.push_back(load_block(i));
            }
        }
        
        /**
          * Saves the current chunk of vertex values
          */
        virtual void save(bool async=false) {
            for(int i=0; i <  (int)loadedblocks.size(); i++) {
                write_block(loadedblocks[i]);
            }
        }
        
        
        /**
         * Returns id of the first vertex currently in memory. Fails if nothing loaded yet.
         */
        vid_t first_vertex_id() {
            return vertex_st; 
        }  
        
        
        VertexDataType * vertex_data_ptr(vid_t vertexid) {
            int blockid = vertexid / verticesperblock;
            int firstloaded = loadedblocks[0].blockid;
            dynamicdata_block<VertexDataType> * dynblock = loadedblocks[blockid - firstloaded].dblock;
            return dynblock->edgevec(vertexid % verticesperblock);
        }   
        
        
    };
}

#endif 

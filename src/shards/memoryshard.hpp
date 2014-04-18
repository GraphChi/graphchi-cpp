
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
 * The memory shard. This class should only be accessed internally by the GraphChi engine.
 */

#ifdef DYNAMICEDATA
#include "shards/dynamicdata/memoryshard.hpp"
#else

#ifndef DEF_GRAPHCHI_MEMSHARD
#define DEF_GRAPHCHI_MEMSHARD


#include <iostream>
#include <cstdio>
#include <sstream>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <string>

#include "api/graph_objects.hpp"
#include "metrics/metrics.hpp"
#include "io/stripedio.hpp"
#include "graphchi_types.hpp"


namespace graphchi {
    
    
    
    template <typename VT, typename ET, typename svertex_t = graphchi_vertex<VT, ET> >
    class memory_shard {
        
        stripedio * iomgr;
        
        std::string filename_edata;
        std::string filename_adj;
        
        vid_t range_st;
        vid_t range_end;
        size_t adjfilesize;
        size_t edatafilesize;
        
        vid_t streaming_offset_vid;
        size_t streaming_offset; // The offset where streaming should continue
        size_t range_start_offset; // First byte for this range's vertices (used for writing only outedges)
        size_t range_start_edge_ptr;
        size_t streaming_offset_edge_ptr;
        uint8_t * adjdata;
        char ** edgedata;
        int * doneptr;
        std::vector<size_t> blocksizes;
        uint64_t chunkid;
        
        std::vector<int> block_edatasessions;
        int adj_session;
        
        bool async_edata_loading;
        bool is_loaded;
        bool disable_async_writes;
        bool enable_parallel_loading;
        size_t blocksize;
        metrics &m;
        std::vector<shard_index> index;
        
    public:
        bool only_adjacency;
        
        memory_shard(stripedio * iomgr,
                     std::string _filename_edata,
                     std::string _filename_adj,
                     vid_t _range_start,
                     vid_t _range_end,
                     size_t _blocksize,
                     metrics &_m) : iomgr(iomgr), filename_edata(_filename_edata),
        filename_adj(_filename_adj),
        range_st(_range_start), range_end(_range_end), blocksize(_blocksize),  m(_m) {
            adjdata = NULL;
            only_adjacency = false;
            is_loaded = false;
            adj_session = -1;
            edgedata = NULL;
            doneptr = NULL;
            enable_parallel_loading = true;
            disable_async_writes = false;
            async_edata_loading = !svertex_t().computational_edges();
#ifdef SUPPORT_DELETIONS
            async_edata_loading = false; // See comment above for memshard, async_edata_loading = false;
#endif
        }
        
        ~memory_shard() {
            int nblocks = (int) block_edatasessions.size();
            
            for(int i=0; i < nblocks; i++) {
                if (edgedata[i] != NULL && block_edatasessions[i] != CACHED_SESSION_ID) {
                    iomgr->managed_release(block_edatasessions[i], &edgedata[i]);
                    iomgr->close_session(block_edatasessions[i]);
                }
            }
            if (adj_session >= 0) {
                if (adjdata != NULL) iomgr->managed_release(adj_session, &adjdata);
                iomgr->close_session(adj_session);
            }
            if (edgedata != NULL)
                free(edgedata);
            edgedata = NULL;
            if (doneptr != NULL) {
                free(doneptr);
            }
        }
        
        void set_disable_async_writes(bool b) {
            disable_async_writes = b;
        }
        
        void disable_parallel_loading() {
            enable_parallel_loading = false;
        }
        
        void commit(bool commit_inedges, bool commit_outedges) {
            if (block_edatasessions.size() == 0 || only_adjacency) return;
            assert(is_loaded);
            metrics_entry cm = m.start_time();
            
            /**
             * This is an optimization that is relevant only if memory shard
             * has been used in a case where only out-edges are considered.
             * Out-edges are in a continuous "window", while in-edges are
             * scattered all over the shard
             */
            int nblocks = (int) block_edatasessions.size();
            
            if (commit_inedges) {
                int start_stream_block = (int) (range_start_edge_ptr / blocksize);
                
#pragma omp parallel for
                for(int i=0; i < nblocks; i++) {
                    /* Write asynchronously blocks that will not be needed by the sliding windows on
                     this iteration. */
                    if (i >= start_stream_block || disable_async_writes) {
                        if (block_edatasessions[i] != CACHED_SESSION_ID) {
                            // Try to include in cache. If succeeds, do not release.
                            if (false == iomgr->get_block_cache().consider_caching(
                                                                                   iomgr->get_session_filename(block_edatasessions[i]), edgedata[i], blocksizes[i], true)) {
                                iomgr->managed_pwritea_now(block_edatasessions[i], &edgedata[i], blocksizes[i], 0);
                                iomgr->managed_release(block_edatasessions[i], &edgedata[i]);
                                iomgr->close_session(block_edatasessions[i]);
                            } else {
                                iomgr->close_session(block_edatasessions[i]);
                                block_edatasessions[i] = CACHED_SESSION_ID;
                            }
                        }
                        edgedata[i] = NULL;
                        
                    } else {
                        if (block_edatasessions[i] != CACHED_SESSION_ID) {
                            iomgr->managed_pwritea_async(block_edatasessions[i], &edgedata[i], blocksizes[i], 0, true, true);
                        }
                        edgedata[i] = NULL;
                    }
                }
            } else if (commit_outedges) {
                size_t last = streaming_offset_edge_ptr;
                if (last == 0){
                    // rollback
                    last = edatafilesize;
                }
                //char * bufp = ((char*)edgedata + range_start_edge_ptr);
                int startblock = (int) (range_start_edge_ptr / blocksize);
                int endblock = (int) (last / blocksize);
#pragma omp parallel for
                for(int i=0; i < nblocks; i++) {
                    if (block_edatasessions[i] != CACHED_SESSION_ID) {
                        if (false == iomgr->get_block_cache().consider_caching(
                                                                               iomgr->get_session_filename(block_edatasessions[i]), edgedata[i], blocksizes[i], true)) {
                            if (i >= startblock && i <= endblock) {
                                iomgr->managed_pwritea_now(block_edatasessions[i], &edgedata[i], blocksizes[i], 0);
                            }
                            iomgr->managed_release(block_edatasessions[i], &edgedata[i]);
                            iomgr->close_session(block_edatasessions[i]);
                        } else {
                            iomgr->close_session(block_edatasessions[i]);
                            block_edatasessions[i] = CACHED_SESSION_ID;
                        }
                    }
                    edgedata[i] = NULL;
                }
            } else {
                for(int i=0; i < nblocks; i++) {
                    if (block_edatasessions[i] != CACHED_SESSION_ID) {
                        iomgr->close_session(block_edatasessions[i]);
                    }
                }
            }
            
            m.stop_time(cm, "memshard_commit");
            
            iomgr->managed_release(adj_session, &adjdata);
            // FIXME: this is duplicated code from destructor
            for(int i=0; i < nblocks; i++) {
                if (edgedata[i] != NULL) {
                    if (block_edatasessions[i] != CACHED_SESSION_ID) {
                        iomgr->managed_release(block_edatasessions[i], &edgedata[i]);
                    }
                }
            }
            block_edatasessions.clear();
            is_loaded = false;
        }
        
        bool loaded() {
            return is_loaded;
        }
        
    private:
        
        /**
          * Load sparse index for the shard
          */
        std::vector<shard_index> load_index() {
            std::string indexfile = filename_shard_adjidx(filename_adj);
            if (!file_exists(indexfile)) {
                logstream(LOG_DEBUG) << "Cannot find index: " << indexfile << std::endl;
                /* Create faux index */
                std::vector<shard_index> singletonidx;
                singletonidx.push_back(shard_index(0, 0, 0));
                return singletonidx;
            }
            
            shard_index * idxraw;
            int f = open(indexfile.c_str(), O_RDONLY);
            size_t sz = readfull(f, &idxraw);
            
            int nidx = (int) (sz / sizeof(shard_index));
            std::vector<shard_index> idx;
            idx.push_back(shard_index(0, 0, 0));  // Implicit
            for(int i=0; i<nidx; i++) {
                idx.push_back(idxraw[i]);
            }
            
            free(idxraw);
            close(f);
            return idx;
        }
        
        void load_edata() {
            assert(blocksize % sizeof(ET) == 0);
            int nblocks = (int) (edatafilesize / blocksize + (edatafilesize % blocksize != 0));
            edgedata = (char **) calloc(nblocks, sizeof(char*));
            size_t compressedsize = 0;
            int blockid = 0;
            
            if (!async_edata_loading) {
                doneptr = (int *) malloc(nblocks * sizeof(int));
                for(int i=0; i < nblocks; i++) doneptr[i] = 1;
            }
            
            while(blockid < nblocks) {
                std::string block_filename = filename_shard_edata_block(filename_edata, blockid, blocksize);
                if (file_exists(block_filename)) {
                    size_t fsize = std::min(edatafilesize - blocksize * blockid, blocksize);
                    
                    compressedsize += get_filesize(block_filename);
                    blocksizes.push_back(fsize);
                    
                    /* Check if cached */
                    void * cachedblock = iomgr->get_block_cache().get_cached(block_filename);
                    if (cachedblock != NULL) {
                        // Cached
                        block_edatasessions.push_back(CACHED_SESSION_ID);
                        edgedata[blockid] = (char*)cachedblock;
                        if (!async_edata_loading) {
                            doneptr[blockid] = 0;
                        }
                    } else {
                        int blocksession = iomgr->open_session(block_filename, false, true); // compressed
                        block_edatasessions.push_back(blocksession);
                        
                        edgedata[blockid] = NULL;
                        iomgr->managed_malloc(blocksession, &edgedata[blockid], fsize, 0);
                        if (async_edata_loading) {
                            iomgr->managed_preada_async(blocksession, &edgedata[blockid], fsize, 0);
                        } else {
                            iomgr->managed_preada_async(blocksession, &edgedata[blockid], fsize, 0, (volatile int *)&doneptr[blockid]);
                        }
                    }
                    blockid++;
                    
                } else {
                    if (blockid == 0) {
                        logstream(LOG_ERROR) << "Shard block file did not exists:" << block_filename << std::endl;
                    }
                    if (blockid < nblocks) {
                        logstream(LOG_ERROR) << "Did not find block " << block_filename << std::endl;
                        logstream(LOG_ERROR) << "Going to exit..." << std::endl;
                    }
                    break;
                }
            }
            
            logstream(LOG_DEBUG) << "Compressed/full size: " << compressedsize * 1.0 / edatafilesize <<
            " number of blocks: " << nblocks << std::endl;
            assert(blockid == nblocks);
            
        }
        
        
    public:
        
        // TODO: recycle ptr!
        void load() {
            is_loaded = true;
            adjfilesize = get_filesize(filename_adj);
            
#ifdef SUPPORT_DELETIONS
            async_edata_loading = false;  // Currently we encode the deleted status of an edge into the edge value (should be changed!),
            // so we need the edge data while loading
#endif
            
            //preada(adjf, adjdata, adjfilesize, 0);
            
            adj_session = iomgr->open_session(filename_adj, true);
            iomgr->managed_malloc(adj_session, &adjdata, adjfilesize, 0);
            
            /* Load in parallel: replaces older stream solution */
            size_t bufsize = 16 * 1024 * 1024;
            int n = (int) (adjfilesize / bufsize + 1);
            
#pragma omp parallel for
            for(int i=0; i < n; i++) {
                size_t toread = std::min(adjfilesize - i * bufsize, (size_t)bufsize);
                iomgr->preada_now(adj_session, adjdata + i * bufsize, toread, i * bufsize, true);
            }
            
            
            /* Initialize edge data asynchonous reading */
            if (!only_adjacency) {
                edatafilesize = get_shard_edata_filesize<ET>(filename_edata);
                load_edata();
            }
            
            
            // Now start creating vertices
            
            streaming_offset = 0;
            streaming_offset_vid = 0;
            streaming_offset_edge_ptr = 0;
            range_start_offset = adjfilesize;
            range_start_edge_ptr = edatafilesize;
            
            
            // Get index
            index = load_index();
        }
        
        
        
        void load_vertices(vid_t window_st, vid_t window_en, std::vector<svertex_t> & prealloc, bool inedges=true, bool outedges=true) {
            /* Find file size */
            m.start_time("memoryshard_create_edges");
            
            assert(adjdata != NULL);
            
            int nblocks = (int) (edatafilesize / blocksize + (edatafilesize % blocksize != 0));
           
            bool setoffset = false;
            bool setrangeoffset = false;
            
            if (!enable_parallel_loading) {
                index.clear();
                index.push_back(shard_index(0, 0, 0));
            }

#pragma omp parallel for schedule(dynamic, 1)
            for(int chunk=0; chunk < (int)index.size(); chunk++) {
                /* Parallelized loading of adjacency data ... */
                uint8_t * ptr = adjdata + index[chunk].filepos;
                uint8_t * end = adjdata + (chunk < (int) index.size() - 1 ? index[chunk + 1].filepos :  adjfilesize);
                vid_t vid = index[chunk].vertexid;
                vid_t viden = (chunk < (int) index.size() - 1 ? index[chunk + 1].vertexid :  0xffffffffu);
                size_t edgeptr = index[chunk].edgecounter * sizeof(ET);
                size_t edgeptr_end =  (chunk < (int) index.size() - 1 ? index[chunk + 1].edgecounter * sizeof(ET) : edatafilesize);

                bool contains_range_end = vid < range_end && viden > range_end;
                bool contains_range_st = vid <= range_st && viden > range_st;
                
                
                // Optimization:
                if (!inedges && (vid > window_en || viden < window_st)) {
                    continue;
                }
                
                if (!async_edata_loading && !only_adjacency) {
                    /* Wait until blocks loaded (non-asynchronous version) */
                    for(int blid=(int)edgeptr/blocksize; blid<=(int)(edgeptr_end /blocksize); blid++) {
                        if (blid < nblocks) {
                            while(doneptr[blid] != 0) { usleep(10); }
                        }
                    }
                }
                
                
                while(ptr < end) {
                   if (contains_range_end) {
                        if (!setoffset && vid > range_end) {
                            // This is where streaming should continue. Notice that because of the
                            // non-zero counters, this might be a bit off.
                            streaming_offset = ptr-adjdata;
                            streaming_offset_vid = vid;
                            streaming_offset_edge_ptr = edgeptr;
                            setoffset = true;
                        }
                   }
                    if (contains_range_st) {
                        if (!setrangeoffset && vid >= range_st) {
                            range_start_offset = ptr-adjdata;
                            range_start_edge_ptr = edgeptr;
                            setrangeoffset = true;
                        }
                    }
                    
                    uint8_t ns = *ptr;
                    int n;
                    
                    ptr += sizeof(uint8_t);
                    
                    if (ns == 0x00) {
                        // next value tells the number of vertices with zeros
                        uint8_t nz = *ptr;
                        ptr += sizeof(uint8_t);
                        vid++;
                        vid += nz;
                        continue;
                    }
                    
                    if (ns == 0xff) {  // If 255 is not enough, then stores a 32-bit integer after.
                        n = *((uint32_t*)ptr);
                        ptr += sizeof(uint32_t);
                    } else {
                        n = ns;
                    }
                    svertex_t* vertex = NULL;
                    
                    if (vid>=window_st && vid <=window_en) { // TODO: Make more efficient
                        vertex = &prealloc[vid-window_st];
                        if (!vertex->scheduled) vertex = NULL;
                    }
                    bool any_edges = false;
                    while(--n>=0) {
                        int blockid = (int) (edgeptr / blocksize);
                       
                        vid_t target = *((vid_t*) ptr);
                        ptr += sizeof(vid_t);
                        if (vertex != NULL && outedges)
                        {
                            char * eptr = (only_adjacency ? NULL  : &(edgedata[blockid][edgeptr % blocksize]));
                            vertex->add_outedge(target, (only_adjacency ? NULL : (ET*) eptr), false);
                        }
                        
                        if (target >= window_st)  {
                            if (target <= window_en) {                        /* In edge */
                                if (inedges) {
                                    svertex_t & dstvertex = prealloc[target - window_st];
                                    if (dstvertex.scheduled) {
                                        any_edges = true;
                                        //  assert(only_adjacency ||  edgeptr < edatafilesize);
                                        char * eptr = (only_adjacency ? NULL  : &(edgedata[blockid][edgeptr % blocksize]));
                                        
                                        dstvertex.add_inedge(vid,  (only_adjacency ? NULL : (ET*) eptr), false);
                                        dstvertex.parallel_safe = dstvertex.parallel_safe && (vertex == NULL); // Avoid if
                                    }
                                }
                            } else { // Note, we cannot skip if there can be "special edges". FIXME so dirty.
                                // This vertex has no edges any more for this window, bail out
                                if (vertex == NULL) {
                                    ptr += sizeof(vid_t) * n;
                                    edgeptr += (n + 1) * sizeof(ET);
                                    break;
                                }
                            }
                        }
                        edgeptr += sizeof(ET);
                        
                    }
                    
                    if (any_edges && vertex != NULL) {
                        vertex->parallel_safe = false;
                    }
                    vid++;
                }
            }
            m.stop_time("memoryshard_create_edges", false);
        }
        
        size_t offset_for_stream_cont() {
            return streaming_offset;
        }
        vid_t offset_vid_for_stream_cont() {
            return streaming_offset_vid;
        }
        size_t edata_ptr_for_stream_cont() {
            return streaming_offset_edge_ptr;
        }
        
        
        
        
    };
};

#endif
#endif
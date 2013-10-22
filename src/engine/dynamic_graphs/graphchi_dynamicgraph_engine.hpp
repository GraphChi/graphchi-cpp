

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
 * Engine for graphs that change. This is in alpha-stage now.
 */


#ifndef GRAPHCHI_DYNAMICGRAPHENGINE_DEF
#define GRAPHCHI_DYNAMICGRAPHENGINE_DEF

#include <stdlib.h>
#include <vector>

#include "engine/graphchi_engine.hpp"
#include "engine/dynamic_graphs/edgebuffers.hpp"
#include "logger/logger.hpp"


namespace graphchi {
    
    /**
     * The actual engine
     */
    
    template <typename VertexDataType, typename EdgeDataType, typename svertex_t = graphchi_vertex<VertexDataType, EdgeDataType> >
    class graphchi_dynamicgraph_engine : public graphchi_engine<VertexDataType, EdgeDataType, svertex_t> {
    public:
        typedef graphchi_engine<VertexDataType, EdgeDataType>  base_engine;
        typedef edge_buffer_flat<EdgeDataType> edge_buffer; 
        
        graphchi_dynamicgraph_engine(std::string base_filename, int nshards, bool selective_scheduling, metrics &_m) :
        graphchi_engine<VertexDataType, EdgeDataType, svertex_t>(base_filename, nshards, selective_scheduling, _m){
            _m.set("engine", "dynamicgraphs");
            added_edges = 0;
            maxshardsize = 200 * 1024 * 1024;
        }
        
    protected:
        
        /**
         * Bookkeeping of buffered and deleted edges.
         */
        std::vector< std::vector< edge_buffer * > > new_edge_buffers;
        std::vector<int> deletecounts;
        std::vector<std::string> shard_suffices;
        
        vid_t max_vertex_id;
        size_t max_edge_buffer;
        size_t last_commit;
        size_t added_edges;
        std::string state;
        size_t maxshardsize;
        size_t edges_in_shards;
        size_t orig_edges;
        
        /**
         * Concurrency control
         */
        mutex schedulerlock;
        mutex shardlock;
        
        /** 
         * Preloading will interfere with the operation.
         */
        virtual bool disable_preloading() {
            return true;
        }
        
        /** 
          * Create a dynamic version of the degree file.
          */
        virtual degree_data * create_degree_handler() {
            /* FIXME: This is bad software design - we should not have a filename dependency here. */
            std::string orig_degree_file = filename_degree_data(this->base_filename);
            std::string dynsuffix = ".dynamic";
            std::string dynamic_degree_file = filename_degree_data(this->base_filename + dynsuffix);
            cp(orig_degree_file, dynamic_degree_file);
            return new degree_data(this->base_filename + dynsuffix, this->iomgr);
        }
        virtual size_t num_edges() {
            shardlock.lock();
            size_t ne = 0;
            for(int i=0; i < this->nshards; i++) {
                ne += this->sliding_shards[i]->num_edges();
                for(int j=0; j < (int) new_edge_buffers[i].size(); j++)
                    ne += new_edge_buffers[i][j]->size();
            }
            shardlock.unlock();
            return ne;
        }
        
    public:
        
        size_t num_edges_safe() {
            return added_edges + orig_edges;
        }
        
        size_t num_buffered_edges() {
            return added_edges - last_commit;
        }
        
    protected:
        void init_buffers() {
            max_edge_buffer = get_option_long("max_edgebuffer_mb", 1000) * 1024 * 1024 / sizeof(created_edge<EdgeDataType>);
            
            // Save old so if there are existing edges, they can be moved
            std::vector< std::vector< edge_buffer * > > tmp_new_edge_buffers;
            for(int i=0; i < this->nshards; i++) {
                std::vector<edge_buffer *> shardbuffers = std::vector<edge_buffer *>();
                for(int j=0; j < this->nshards; j++) {
                    shardbuffers.push_back(new edge_buffer());
                }
                tmp_new_edge_buffers.push_back(shardbuffers);
            }
            
            // Move old edges. This is not the fastest way... but takes only about 0.05 secs
            // on the twitter experiment
            int i = 0;
            for(typename std::vector< std::vector< edge_buffer * > >::iterator oldit = new_edge_buffers.begin(); 
                oldit != new_edge_buffers.end(); ++oldit) {
                for(typename std::vector< edge_buffer *>::iterator bufit = oldit->begin(); bufit != oldit->end(); ++bufit) {
                    edge_buffer &buffer_for_window = **bufit;
                    for(unsigned int ebi = 0; ebi < buffer_for_window.size(); ebi++ ) {
                        created_edge<EdgeDataType> * edge = buffer_for_window[ebi];
                        int shard = get_shard_for(edge->dst);
                        int srcshard = get_shard_for(edge->src);
                        i++;
                        tmp_new_edge_buffers[shard][srcshard]->add(*edge);            
                    }
                    delete *bufit;
                }
            }
            
            std::cout << "TRANSFERRED " << i << " EDGES OVER." << std::endl;
            
            new_edge_buffers = tmp_new_edge_buffers;
        }
        
        
        /**
         * In the beginning of run, we copy the shards into dynamic versions.
         */
        // Should be changed to read the file in smaller chunks
        size_t cp(std::string origfile, std::string dstfile, bool zeroout=false) {
            char * buf;
            int f = open(origfile.c_str(), O_RDONLY);    
            size_t len = readfull(f, &buf);
            std::cout << "Length: " << len << std::endl;
            std::cout << origfile << " ----> " << dstfile << std::endl;
            
            close(f);
            remove(dstfile.c_str());
            int of = open(dstfile.c_str(), O_WRONLY | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
            assert(of >= 0);
            if (zeroout) {
                memset(buf, 0, len);
            }
            writea(of, buf, len);
            
            assert(get_filesize(origfile) == get_filesize(dstfile));
            close(of);
            free(buf);
            return len;
        }
        
        // Copy the edata directory
        void cpedata(std::string origfile, std::string dstfile, bool zeroout=false) {
            cp(origfile + ".size", dstfile + ".size");
            std::string dirname = dirname_shard_edata_block(dstfile, base_engine::blocksize);
            mkdir(dirname.c_str(), 0777);
            size_t edatasize = get_shard_edata_filesize<EdgeDataType>(origfile);
            int nblocks = (int) ((edatasize / base_engine::blocksize) + (edatasize % base_engine::blocksize == 0 ? 0 : 1));
            for(int i=0; i < nblocks; i++) {
                std::string origblockname = filename_shard_edata_block(origfile, i, base_engine::blocksize);
                std::string dstblockname = filename_shard_edata_block(dstfile, i, base_engine::blocksize);
                cp(origblockname, dstblockname);
            }
        }
        
        
        virtual typename base_engine::memshard_t * create_memshard(vid_t interval_st, vid_t interval_en) {
            int p = this->exec_interval;
            std::string adj_filename = filename_shard_adj(this->base_filename, 0, 0) + ".dyngraph" + shard_suffices[p];          
            std::string edata_filename = filename_shard_edata<EdgeDataType>(this->base_filename, 0, 0) + ".dyngraph" + shard_suffices[p];
            return new typename base_engine::memshard_t(this->iomgr,
                                                        edata_filename,
                                                        adj_filename,  
                                                        interval_st, 
                                                        interval_en,
                                                        base_engine::blocksize,
                                                        this->m);
        }
        
        
        /**
         * Initialize streaming shards in the start of each iteration.
         */
        virtual void initialize_sliding_shards() {
            state = "initialize-shards";
            shardlock.lock();
            if (this->sliding_shards.empty()) {
                for(int p=0; p < this->nshards; p++) {
                    std::string adj_filename = filename_shard_adj(this->base_filename, 0, 0) + ".dyngraph" + shard_suffices[p];          
                    std::string edata_filename = filename_shard_edata<EdgeDataType>(this->base_filename, 0, 0) + ".dyngraph" + shard_suffices[p];
                    
                    this->sliding_shards.push_back(
                                                   new typename base_engine::slidingshard_t(this->iomgr, edata_filename, 
                                                                                            adj_filename,
                                                                                            this->intervals[p].first, 
                                                                                            this->intervals[p].second, 
                                                                                            this->blocksize, 
                                                                                            this->m, 
                                                                                            !this->modifies_outedges, 
                                                                                            false));
                }
            } else {
                for(int p=0; p < this->nshards; p++) {
                    if (this->sliding_shards[p] == NULL) {
                        std::string adj_filename = filename_shard_adj(this->base_filename, 0, 0) + ".dyngraph" + shard_suffices[p];          
                        std::string edata_filename = filename_shard_edata<EdgeDataType>(this->base_filename, 0, 0) + ".dyngraph" + shard_suffices[p];
                        
                        this->sliding_shards[p] =  new typename base_engine::slidingshard_t(this->iomgr, edata_filename, 
                                                                                            adj_filename,
                                                                                            this->intervals[p].first, 
                                                                                            this->intervals[p].second, 
                                                                                            this->blocksize, 
                                                                                            this->m, 
                                                                                            !this->modifies_outedges, 
                                                                                            false);
                    }
                }
            }
            shardlock.unlock();
            edges_in_shards = num_edges();
            if (orig_edges == 0) orig_edges = edges_in_shards;
        }
        
        void prepare_clean_slate() {
            logstream(LOG_INFO) << "Preparing clean slate..." << std::endl;
            for(int shard=0; shard < this->nshards; shard++) {
                shard_suffices.push_back(get_part_str(shard, this->nshards));
                
                std::string edata_filename = filename_shard_edata<EdgeDataType>(this->base_filename, shard, this->nshards);
                std::string adj_filename = filename_shard_adj(this->base_filename, shard, this->nshards);
                std::string dest_adj = filename_shard_adj(this->base_filename, 0, 0) + ".dyngraph" + shard_suffices[shard];          
                std::string dest_edata = filename_shard_edata<EdgeDataType>(this->base_filename, 0, 0) + ".dyngraph" + shard_suffices[shard];
                
                cpedata(edata_filename, dest_edata, true);
                cp(adj_filename, dest_adj);
                cp(filename_shard_adjidx(adj_filename), filename_shard_adjidx(dest_adj));

            }
        }
        
        int get_shard_for(vid_t dst) {
            for(int i=0; i < this->nshards; i++) {
                if (dst >= this->intervals[i].first && dst <= this->intervals[i].second) {
                    return i;
                }
            }
            return this->nshards - 1; // Last shard
        }
        
    public:       
        bool add_edge(vid_t src, vid_t dst, EdgeDataType edata) {
            if (src == dst) {
                logstream(LOG_WARNING) << "WARNING : tried to add self-edge!" << std::endl;
                return true;
            }
            if (this->iter < 1) {
                logstream(LOG_WARNING) << "Tried to add edge before first iteration has passed" << std::endl;
                usleep(1000000);
                return false;
            }
            if (added_edges - last_commit > 1.2 * max_edge_buffer) {
                logstream(LOG_INFO) << "Over 20% of max buffer... hold on...." << std::endl;
                usleep(1000000); // Sleep 1 sec
                return false;
            }
            this->modification_lock.lock();
            added_edges++;
            int shard = get_shard_for(dst);
            int srcshard = get_shard_for(src);
            /* Maintain max vertex id */
            vid_t prev_max_id = max_vertex_id;
            max_vertex_id = std::max(max_vertex_id, dst);
            max_vertex_id = std::max(max_vertex_id, src);
            
            // Extend degree and vertex data files
            if (max_vertex_id>prev_max_id) {
                this->degree_handler->ensure_size(this->max_vertex_id); // Expand the file
                
                // Expand scheduler
                if (this->scheduler != NULL) {
                    schedulerlock.lock();
                    this->scheduler->resize(1 + max_vertex_id);
                    schedulerlock.unlock();
                }
            }
            
            // Add edge to buffers
            new_edge_buffers[shard][srcshard]->add(src, dst, edata);            
            this->modification_lock.unlock();
            return true;
        }
        
        void add_task(vid_t vid) {
            if (this->scheduler != NULL) {
                this->modification_lock.lock();
                this->scheduler->add_task(vid);                
                this->modification_lock.unlock();
            }
        }
       
    protected:
        void incorporate_buffered_edges(int window, vid_t window_st, vid_t window_en, std::vector<svertex_t> & vertices) {
            // Lock acquired
            int ncreated = 0;
            // First outedges
            for(int shard=0; shard<this->nshards; shard++) {
                edge_buffer &buffer_for_window = *new_edge_buffers[shard][window];
                for(unsigned int ebi=0; ebi<buffer_for_window.size(); ebi++) {
                    created_edge<EdgeDataType> * edge = buffer_for_window[ebi];
                    if (edge->src >= window_st && edge->src <= window_en) {
                        if (vertices[edge->src-window_st].scheduled) {
                            if (vertices[edge->src-window_st].scheduled)
                                vertices[edge->src-window_st].add_outedge(edge->dst, &edge->data, false);
                            ncreated++;
                        }
                    }
                }
            }
            
            // Then inedges
            for(int w=0; w<this->nshards; w++) {
                edge_buffer &buffer_for_window = *new_edge_buffers[window][w];
                for(unsigned int ebi=0; ebi<buffer_for_window.size(); ebi++) {
                    created_edge<EdgeDataType> * edge = buffer_for_window[ebi];
                    if (edge->dst >= window_st && edge->dst <= window_en) {
                        if (vertices[edge->dst - window_st].scheduled) {
                            assert(edge->data < 1e20);
                            if (vertices[edge->dst-window_st].scheduled)
                                vertices[edge->dst - window_st].add_inedge(edge->src, &edge->data, false);
                            ncreated++;
                        }
                    }
                }
            }
            logstream(LOG_INFO) << "::: Used " << ncreated << " buffered edges." << std::endl;
        }
        
        bool incorporate_new_edge_degrees(int window, vid_t window_st, vid_t window_en) {
            bool modified = false;
            // First outedges
            for(int shard=0; shard < this->nshards; shard++) {
                edge_buffer &buffer_for_window = *new_edge_buffers[shard][window];
                for(unsigned int ebi=0; ebi<buffer_for_window.size(); ebi++) {
                    created_edge<EdgeDataType> * edge = buffer_for_window[ebi];
                    if (edge->src >= window_st && edge->src <= window_en) {
                        if (!edge->accounted_for_outc) {
                            degree d = this->degree_handler->get_degree(edge->src);
                            d.outdegree++;
                            this->degree_handler->set_degree(edge->src, d);
                            
                            modified = true;
                            edge->accounted_for_outc = true;
                        }
                    }
                }
            }
            
            // Then inedges
            for(int w=0; w < this->nshards; w++) {
                edge_buffer &buffer_for_window = *new_edge_buffers[window][w];
                for(unsigned int ebi=0; ebi<buffer_for_window.size(); ebi++) {
                    created_edge<EdgeDataType> * edge = buffer_for_window[ebi];
                    if (edge->dst >= window_st && edge->dst <= window_en) {
                        if (!edge->accounted_for_inc) {
                            degree d = this->degree_handler->get_degree(edge->dst);
                            d.indegree++;
                            this->degree_handler->set_degree(edge->dst, d);                            
                            edge->accounted_for_inc = true;
                            modified = true;
                        }
                    }
                }
            }
            return modified;
        }
        
        void adjust_degrees_for_deleted(std::vector< svertex_t > &vertices, vid_t window_st) {
#ifdef SUPPORT_DELETIONS
            
            bool somechanged = false;
            for(int i=0; i < (int)vertices.size(); i++) {
                svertex_t &v = vertices[i];
                if (v.scheduled) {
                    this->degree_handler->set_degree(v.id(), v.inc, v.outc);
                    somechanged = somechanged || (v.deleted_inc + v.deleted_outc > 0);
                    degree deg = this->degree_handler->get_degree(v.id());
                    
                    if (!(deg.indegree >=0 && deg.outdegree >= 0)) {
                        std::cout << "Degree discrepancy: " << deg.indegree << " " << deg.outdegree << std::endl;
                    }
                    assert(deg.indegree >=0 && deg.outdegree >= 0);
                }
            }
            if (somechanged) {
                this->degree_handler->save();
            }
#endif
        }
        
        virtual vid_t determine_next_window(vid_t iinterval, vid_t fromvid, vid_t maxvid, size_t membudget) {
            /* Load degrees */
            this->degree_handler->load(fromvid, maxvid);
            if (incorporate_new_edge_degrees(iinterval, fromvid, maxvid)) {
                this->degree_handler->save();
            }
            
            size_t memreq = 0;
            int max_interval = maxvid - fromvid;
            for(int i=0; i < max_interval; i++) {
                degree deg = this->degree_handler->get_degree(fromvid + i);
                int inc = deg.indegree;
                int outc = deg.outdegree;
                
                // Raw data and object cost included
                memreq += sizeof(svertex_t) + (sizeof(EdgeDataType) + sizeof(vid_t) + 
                                               sizeof(graphchi_edge<EdgeDataType>))*(outc + inc);
                if (memreq > membudget) {
                    return fromvid + i - 1;  // Previous was enough
                }
            }
            return maxvid;
        }
        
        
        virtual void load_before_updates(std::vector<svertex_t> &vertices) {  
            state = "load-edges";

            this->base_engine::load_before_updates(vertices);
            
#ifdef SUPPORT_DELETIONS
            for(unsigned int i=0; i < (unsigned int)vertices.size(); i++) {
                deletecounts[this->exec_interval] += vertices[i].deleted_inc;
            }
        #endif
            
            state = "execute-updates";
        }
        
        
        virtual void init_vertices(std::vector<svertex_t> &vertices, graphchi_edge<EdgeDataType> * &edata) {
            base_engine::init_vertices(vertices, edata);
            incorporate_buffered_edges(this->exec_interval, this->sub_interval_st, this->sub_interval_en, vertices);
        }
        
        
        virtual void initialize_iter() {
            this->intervals[this->nshards - 1].second = max_vertex_id;
            this->vertex_data_handler->check_size(max_vertex_id + 1);
            initialize_sliding_shards();
            
            /* Deleted edge tracking */
            deletecounts.clear();
            for(int p=0; p < this->nshards; p++) 
                deletecounts.push_back(0);
        }
        
        virtual void iteration_finished() {
            if (this->iter < this->niters - 1) {
                // Flush and restart stream shards before commiting edges
                for(int p=0; p < this->nshards; p++) {
                    this->sliding_shards[p]->flush();
                    this->sliding_shards[p]->set_offset(0, 0, 0);
                }
                
                this->iomgr->wait_for_writes();
                
                commit_graph_changes();
            }
        }
        
        virtual void initialize_before_run() {
            prepare_clean_slate();
            init_buffers();

            max_vertex_id = (vid_t) (this->num_vertices() - 1);
            
            this->vertex_data_handler->clear(this->num_vertices());
            orig_edges = 0;
        }
        
        
        /* */
        virtual void load_after_updates(std::vector<svertex_t> &vertices) {
            this->base_engine::load_after_updates(vertices);
            adjust_degrees_for_deleted(vertices, this->sub_interval_st);
        }   
        
    public:
        void finish_after_iters(int extra_iters) {
            this->chicontext.last_iteration = this->chicontext.iteration + extra_iters;
        }
        
    protected:
        
        
#define BBUF 32000000
        
        size_t curadjfilepos;

        /**
         * Code for committing changes to disk.
         */
        void commit_graph_changes() {            
            // Count deleted
            size_t ndeleted = 0;
            for(size_t i=0; i < deletecounts.size(); i++) {
                ndeleted += deletecounts[i];
            }
            
            // TODO: remove ad hoc limits, move to configuration.
            // Perhaps do some cost estimation?
            logstream(LOG_DEBUG) << "Total deleted: " << ndeleted << " total edges: " << this->num_edges() << std::endl;

            if (added_edges - last_commit < max_edge_buffer * 0.8 && ndeleted < this->num_edges() * 0.1) {
                std::cout << "==============================" << std::endl;
                std::cout << "No time to commit yet.... Only " << (added_edges - last_commit) << " / " << max_edge_buffer
                << " in buffers" << std::endl;
                return;
            }
            
            
            bool rangeschanged = false;
            state = "commit-ingests";
            vid_t maxwindow = 4000000; // FIXME: HARDCODE
            size_t mem_budget = this->membudget_mb * 1024 * 1024;
            this->modification_lock.lock();
            
            // Clean up sliding shards
            // NOTE: there is a problem since this will waste
            // io-sessions
            std::vector<int> edgespershard;
            for(int p=0; p < this->nshards; p++) {
                edgespershard.push_back(this->sliding_shards[p]->num_edges());
            }
            
            std::vector<std::pair<vid_t, vid_t> > newranges;
            std::vector<std::string> newsuffices;
            
            char iterstr[128];
            sprintf(iterstr, "%d", this->iter);
            
            size_t min_buffer_in_shard_to_commit = max_edge_buffer / this->nshards / 2;
            
            std::vector<bool> was_commited(this->nshards, true);
            
            for(int shard=0; shard < this->nshards; shard++) {
                std::vector<edge_buffer*> &shard_buffer = new_edge_buffers[shard];
                
                // Check there are any new edges
                size_t bufedges = 0;
                for(int w=0; w < this->nshards; w++) {
                    bufedges += shard_buffer[w]->size();
                    
                }
                
                if (bufedges < min_buffer_in_shard_to_commit && deletecounts[shard] * 1.0 / edgespershard[shard] < 0.2) {
                    logstream(LOG_DEBUG) << shard << ": not enough edges for shard: " << bufedges << " deleted:" << deletecounts[shard] << "/" << edgespershard[shard] << std::endl;
                    newranges.push_back(this->intervals[shard]);
                    newsuffices.push_back(shard_suffices[shard]);
                    was_commited[shard] = false;
                    continue;
                } else {
                    logstream(LOG_DEBUG) << shard << ": going to rewrite, deleted:" << deletecounts[shard] << "/" << edgespershard[shard] << " bufedges: " << bufedges << std::endl;
                    shardlock.lock();
                    delete this->sliding_shards[shard];
                    this->sliding_shards[shard] = NULL;
                    shardlock.unlock();
                }
                std::string origshardfile = filename_shard_edata<EdgeDataType>(this->base_filename, 0, 0) + ".dyngraph" + shard_suffices[shard];
                std::string origadjfile = filename_shard_adj(this->base_filename, 0, 0) + ".dyngraph" + shard_suffices[shard];
                
                // Get file size
                off_t sz = get_shard_edata_filesize<EdgeDataType>(origshardfile);
                
                int outparts = ( sz >= (off_t) maxshardsize ? 2 : 1);
                
                vid_t splitpos = 0;
                std::cout << "Size: " << sz << " vs. maxshardsize: " << maxshardsize << std::endl;
                if (sz > (off_t)maxshardsize) {
                    rangeschanged = true;
                    // Compute number edges (not including ingested ones!)
                    size_t halfedges = (sz / sizeof(EdgeDataType)) / 2;
                    // Correct to include estimate of ingested ones
                    for(int w=0; w < this->nshards; w++) {
                        halfedges += new_edge_buffers[shard][w]->size() / 2;
                    }
                    size_t nedges = 0;
                    
                    vid_t st = this->intervals[shard].first;
                    splitpos = st + (this->intervals[shard].second - st) / 2;
                    bool found = false;
                    while(st < this->intervals[shard].second) {
                        vid_t en = std::min(st + maxwindow, this->intervals[shard].second);
                        this->degree_handler->load(st, en);
                        int nv = en - st + 1;
                        
                        for(int i=0; i<nv; i++) {
                            nedges += this->degree_handler->get_degree(st + i).indegree;
                            if (nedges >= halfedges) {
                                splitpos = i+st-1;
                                found = true;
                                break;
                            }
                        }
                        if (found) break;
                        st = en+1;
                    }
                    assert(splitpos > this->intervals[shard].first && splitpos < this->intervals[shard].second);
                }
                
                for(int splits=0; splits<outparts; splits++) { // Note: this is not super-efficient because we do the operation twice in case of split
                    typename base_engine::slidingshard_t * curshard = 
                    new typename base_engine::slidingshard_t(this->iomgr, origshardfile, origadjfile, 
                                                             this->intervals[shard].first, this->intervals[shard].second, 
                                                             base_engine::blocksize, this->m, true);
                    
                    
                    std::string suffix = "";
                    char partstr[128];
                    sprintf(partstr, "%d", shard);
                    if (splits == 0) {
                        suffix = std::string(partstr);
                    } else {
                        suffix = std::string(partstr) + ".split";  
                    }
                    suffix = suffix + ".i" + std::string(iterstr);
                    newsuffices.push_back(suffix);
                    curadjfilepos = 0;
                    std::string outfile_edata = filename_shard_edata<EdgeDataType>(this->base_filename, 0, 0) + ".dyngraph" + suffix;
                    std::string outfile_edata_dirname = dirname_shard_edata_block(outfile_edata, base_engine::blocksize);
                    mkdir(outfile_edata_dirname.c_str(), 0777);
                    std::string outfile_adj = filename_shard_adj(this->base_filename, 0, 0) + ".dyngraph" + suffix;
                    
                    vid_t splitstart = this->intervals[shard].first;
                    vid_t splitend = this->intervals[shard].second;
                    if (shard == this->nshards - 1) splitend = max_vertex_id;
                    
                    // This is looking more and more hacky
                    if (outparts == 2) {
                        if (splits==0) splitend = splitpos;
                        else splitstart = splitpos+1;
                    }
                    newranges.push_back(std::pair<vid_t,vid_t>(splitstart, splitend)); 
                    
                    // Create the adj file
                    int f = open(outfile_adj.c_str(), O_WRONLY | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
                    int err = ftruncate(f, 0);
                    if (err != 0) {
                        logstream(LOG_ERROR) << "Error truncating " << outfile_adj << ", error: " << strerror(errno) << std::endl;
                    }
                    assert(err == 0);
                    /* Create edge data file */
                    int ef = open(outfile_edata.c_str(), O_WRONLY | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
                    err = ftruncate(ef, 0);
                    if (err != 0) {
                        logstream(LOG_ERROR) << "Error truncating " << outfile_edata << ", error: " << strerror(errno) << std::endl;
                    }
                    assert(err == 0);
                    char * buf = (char*) malloc(BBUF); 
                    char * bufptr = buf;
                    char * ebuf = (char*) malloc(BBUF);
                    char * ebufptr = ebuf;
                    size_t tot_edatabytes = 0;
                    
                    // Index file
                    std::string indexfile = filename_shard_adjidx(outfile_adj);
                    int idxf = open(indexfile.c_str(),  O_WRONLY | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
                    size_t last_index_output = 0;
                    size_t index_interval_edges = 1024 * 1024;
                    size_t edgecounter = 0;
                    assert(idxf>0);

                    
                    // Now create a new shard file window by window
                    for(int window=0; window < this->nshards; window++) {
                        vid_t range_st = this->intervals[window].first;
                        vid_t range_en = this->intervals[window].second;
                        if (window == this->nshards - 1) range_en = max_vertex_id;
                        edge_buffer &buffer_for_window = *new_edge_buffers[shard][window];
                        
                        for(vid_t window_st=range_st; window_st<range_en; ) {
                            // Check how much we can read
                            vid_t window_en = determine_next_window(window, window_st, 
                                                                    std::min(range_en, window_st + (vid_t)maxwindow), mem_budget);
                            // Create vertices
                            int nvertices = window_en-window_st+1;
                            std::vector< svertex_t > vertices(nvertices, svertex_t());
                            /* Allocate edge data: to do this, need to compute sum of in & out edges */
                            graphchi_edge<EdgeDataType> * edata = NULL;
                            size_t num_edges=0;
                            for(int i=0; i<nvertices; i++) {
                                degree d = this->degree_handler->get_degree(i + window_st);
                                num_edges += d.indegree+d.outdegree;
                            }
                            size_t ecounter = 0;
                            edata = (graphchi_edge<EdgeDataType>*)malloc(num_edges * sizeof(graphchi_edge<EdgeDataType>));
                            for(int i=0; i<(int)nvertices; i++) {
                                //  int inc = degrees[i].indegree;
                                degree d = this->degree_handler->get_degree(i + window_st);
                                int outc = d.outdegree;
                                vertices[i] = svertex_t(window_st+i, &edata[ecounter], 
                                                        &edata[ecounter+0], 0, outc);
                                vertices[i].scheduled = true; // guarantee that shard will read it
                                ecounter += 0 + outc;
                            }
                            
                            // Read vertices in
                            curshard->read_next_vertices(nvertices, window_st, vertices, false, true);
                            
                            // Incorporate buffered edges
                            for(unsigned int ebi=0; ebi<buffer_for_window.size(); ebi++) {
                                created_edge<EdgeDataType> * edge = buffer_for_window[ebi];
                                if (edge->src >= window_st && edge->src <= window_en) {
                                    vertices[edge->src-window_st].add_outedge(edge->dst, &edge->data, false);
                                    
                                }
                            }
                            this->iomgr->wait_for_reads();
                            
                            // If we are splitting, need to adjust counts
                            std::vector<int> adjusted_counts(vertices.size(), 0);
                            for(int iv=0; iv< (int)vertices.size(); iv++) adjusted_counts[iv] = vertices[iv].outc;
                            
                            if (outparts == 2) {
                                // do actual counts by removing the edges not in this split
                                for(int iv=0; iv< (int)vertices.size(); iv++) {
                                    svertex_t &vertex = vertices[iv];
                                    for(int i=0; i<vertex.outc; i++) {
                                        if (!(vertex.outedge(i)->vertexid >= splitstart && vertex.outedge(i)->vertexid <= splitend)) {
                                            adjusted_counts[iv]--;  
                                        }
                                    }
                                }
                            }   
                            
#ifdef SUPPORT_DELETIONS
                            // Adjust counts to remove deleted edges
                            for(int iv=0; iv< (int)vertices.size(); iv++) {
                                svertex_t &vertex = vertices[iv];
                                for(int i=0; i<vertex.outc; i++) {
                                    if (is_deleted_edge_value(vertex.outedge(i)->get_data())) {
                                        adjusted_counts[iv]--;  
                                        assert(false);
                                    }
                                }
                            }
                            
                            // Adjust degrees
                            //   adjust_degrees_for_deleted(vertices, window_st); // Double counting problem, that is why commented out.
#endif
                            
                            size_t ne = 0;
                            for(vid_t curvid=window_st; curvid<=window_en;) {
                                int iv = curvid - window_st;
                                svertex_t &vertex = vertices[iv];
                                int count = adjusted_counts[iv];                            
                                if (count == 0) {
                                    // Check how many next ones are zeros
                                    int nz=0;
                                    curvid++;
                                    for(; curvid <= window_en && nz<254; curvid++) {
                                        if (adjusted_counts[curvid - window_st] == 0) {
                                            nz++;
                                        } else {
                                            break;
                                        }
                                    }
                                    uint8_t nnz = (uint8_t)nz;
                                    // Write zero
                                    bwrite<uint8_t>(f, buf, bufptr, 0);
                                    bwrite<uint8_t>(f, buf, bufptr, nnz);
                                } else {
                                    
                                    // Write index
                                    if (edgecounter - last_index_output >= index_interval_edges) {
                                        size_t curfpos = curadjfilepos;
                                        shard_index sidx(curvid, curfpos, edgecounter);
                                        size_t a = write(idxf, &sidx, sizeof(shard_index));
                                        assert(a>0);
                                        last_index_output = edgecounter;
                                    }

                                    
                                    if (count < 255) {
                                        uint8_t x = (uint8_t)count;
                                        bwrite<uint8_t>(f, buf, bufptr, x);
                                    } else {
                                        bwrite<uint8_t>(f, buf, bufptr, 0xff);
                                        bwrite<uint32_t>(f, buf, bufptr, (uint32_t)count);
                                    }
                                    
                                    for(int i=0; i<vertex.outc; i++) {
                                        if (vertex.outedge(i)->vertexid >= splitstart && vertex.outedge(i)->vertexid <= splitend) {
#ifdef SUPPORT_DELETIONS
                                            if (is_deleted_edge_value(vertex.outedge(i)->get_data())) {
                                                assert(false);
                                                
                                            } 
#endif
                                            bwrite(f, buf, bufptr,  vertex.outedge(i)->vertexid);
                                            bwrite_edata<EdgeDataType>(ebuf, ebufptr, vertex.outedge(i)->get_data(), tot_edatabytes, outfile_edata);
                                            ne++;
                                            edgecounter++;
                                        } else assert(outparts == 2);
                                    }
                                    curvid++;
                                }
                            } 
                            free(edata);
                            window_st = window_en+1;
                        }
                        
                    } // end window
                    
                    // Flush buffers
                    writea(f, buf, bufptr-buf);
                    
                    edata_flush<EdgeDataType>(ebuf, ebufptr, outfile_edata, tot_edatabytes);
                    
                    // Write .size file for the edata firectory
                    std::string sizefilename = outfile_edata + ".size";
                    std::ofstream ofs(sizefilename.c_str());
                    ofs << tot_edatabytes;
                    ofs.close();

                    // Release
                    free(buf); 
                    free(ebuf);
                    
                    delete curshard;
                    close(f);
                    close(ef);
                    close(idxf);
                    
                    this->iomgr->wait_for_writes();
                } // splits
                
                // Delete old shard
                std::string old_file_adj = filename_shard_adj(this->base_filename, 0, 0) + ".dyngraph" + shard_suffices[shard];          
                std::string old_file_edata = filename_shard_edata<EdgeDataType>(this->base_filename, 0, 0) + ".dyngraph" + shard_suffices[shard];
                std::string old_blockdir =  dirname_shard_edata_block(old_file_edata, base_engine::blocksize);
                std::string old_file_adj_idx = filename_shard_adjidx(old_file_adj);
                remove(old_file_adj.c_str());
                remove(old_blockdir.c_str());
                remove(old_file_adj_idx.c_str());

                std::string old_sizefilename = old_file_edata + ".size";
                remove(old_sizefilename.c_str());
            }
            
            // Clear buffers
            for(int shard=0; shard < this->nshards; shard++) {
                if (was_commited[shard]) {
                    for (int win=0; win < this->nshards; win++) {
                        edge_buffer &buffer_for_window = *new_edge_buffers[shard][win];
                        for(unsigned int ebi=0; ebi<buffer_for_window.size(); ebi++) {
                            created_edge<EdgeDataType> * edge = buffer_for_window[ebi];                            
                            if (!edge->accounted_for_outc) {
                                std::cout << "Edge not accounted (out)! " << edge->src << " -- " << edge->dst << std::endl;
                            }
                            if (!edge->accounted_for_inc) {
                                std::cout << "Edge not accounted (in)! " << edge->src << " -- " << edge->dst << std::endl;
                            }
                            
                            assert(edge->accounted_for_inc);
                            assert(edge->accounted_for_outc);
                        }
                        buffer_for_window.clear();
                    }
                }
            }
            
            // Update number of shards:
            last_commit = added_edges;
            this->intervals = newranges;
            shard_suffices = newsuffices;
            this->nshards = (int) this->intervals.size();
            
            /* If the vertex intervals change, need to recreate the shard objects. */
            if (rangeschanged) {
                shardlock.lock();
                for (int i=0; i<(int)this->sliding_shards.size(); i++) {
                    if (this->sliding_shards[i] != NULL) delete this->sliding_shards[i];
                }
                this->sliding_shards.clear();
                shardlock.unlock();
            }
            /* Write meta-file with the number of vertices */
            std::string numv_filename = base_engine::base_filename + ".numvertices";
            FILE * f = fopen(numv_filename.c_str(), "w");
            fprintf(f, "%lu\n", base_engine::num_vertices());
            fclose(f);
            
            init_buffers();
            this->modification_lock.unlock();
        }
        
        
        template <typename T>
        void bwrite(int f, char * buf, char * &bufptr, T val) {
            curadjfilepos += sizeof(T);
            if (bufptr+sizeof(T)-buf>=BBUF) {
                writea(f, buf, bufptr-buf);
                bufptr = buf;
            }
            *((T*)bufptr) = val;
            bufptr += sizeof(T);
        }
        
        
        template <typename T>
        void edata_flush(char * buf, char * bufptr, std::string & shard_filename, size_t totbytes) {
            int blockid = (int) ((totbytes - sizeof(T)) / base_engine::blocksize);
            int len = (int) (bufptr - buf);
            assert(len <= (int)base_engine::blocksize);
            
            std::string block_filename = filename_shard_edata_block(shard_filename, blockid, base_engine::blocksize);
            int f = open(block_filename.c_str(), O_RDWR | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
            write_compressed(f, buf, len);
            close(f);
        }
        
        template <typename T>
        void bwrite_edata(char * buf, char * &bufptr, T val, size_t & totbytes, std::string & shard_filename) {            
            if ((int) (bufptr + sizeof(T) - buf) > (int)base_engine::blocksize) {
                edata_flush<T>(buf, bufptr, shard_filename, totbytes);
                bufptr = buf;
            }
            totbytes += sizeof(T);
            *((T*)bufptr) = val;
            bufptr += sizeof(T);
        }
        

        
        /** 
          * HTTP admin
          */
    public:
        std::string get_info_json() {
            std::stringstream json;
            
            this->httplock.lock();
            
            /**
              * FIXME: too much duplicate with graphchi_engine
              */
            json << "{";
            json << "\"state\" : \"" << state << "\",\n";
            json << "\"file\" : \"" << this->base_filename << "\",\n";
            json << "\"numOfShards\": " << this->nshards << ",\n";
            json << "\"iteration\": " << this->chicontext.iteration << ",\n";
            json << "\"numIterations\": " << this->chicontext.num_iterations << ",\n";
            json << "\"runTime\": " << this->chicontext.runtime() << ",\n";
            
            json << "\"updates\": " << this->nupdates << ",\n";
            json << "\"nvertices\": " << this->chicontext.nvertices << ",\n";
            json << "\"edges\": " << num_edges_safe() << ",\n";

            json << "\"edgesInBuffers\": " << added_edges << ",\n";

            json << "\"interval\":" << this->exec_interval << ",\n";
            json << "\"windowStart\":" << this->sub_interval_st << ",";
            json << "\"windowEnd\": " << this->sub_interval_en << ",";
            json << "\"shards\": [";
            
            
            shardlock.lock();
            for(int p=0; p < (int) this->sliding_shards.size(); p++) {
                if (p>0) json << ",";
                
                typename base_engine::slidingshard_t *  shard = this->sliding_shards[p];
                if (shard != NULL) {
                    json << "{";
                    json << "\"p\": " << p << ", ";
                    json << shard->get_info_json();
                    json << "}";
                } else {
                    json << "{";
                    json << "\"p\": " << p << ", ";
                    json << "\"state\": \"recreated\"";
                    json << "}";
                }
            }
            shardlock.unlock();
            
            json << "]";
            
            std::map<std::string, std::string>::iterator it;
            for(it=this->json_params.begin(); it != this->json_params.end(); ++it) {
                json << ", \"" << it->first << "\":\"";
                json << it->second << "\"";
            }
            
            json << "}";
            
            this->httplock.unlock();
            return json.str();
        }

        
        
    }; // End class
    
}; // End namespace


#endif



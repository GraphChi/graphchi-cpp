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
 * Sharder converts a graph into shards which the GraphChi engine
 * can process.
 */

/**
 * @section TODO
 * Change all C-style IO to Unix-style IO.
 */


#ifndef GRAPHCHI_SHARDER_DEF
#define GRAPHCHI_SHARDER_DEF


#include <iostream>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <omp.h>
#include <errno.h>
#include <sstream>
#include <string>

#include "api/chifilenames.hpp"
#include "api/graphchi_context.hpp"
#include "graphchi_types.hpp"
#include "io/stripedio.hpp"
#include "logger/logger.hpp"
#include "metrics/metrics.hpp"
#include "metrics/reps/basic_reporter.hpp"
#include "preprocessing/formats/binary_adjacency_list.hpp"
#include "shards/memoryshard.hpp"
#include "shards/slidingshard.hpp"
#include "util/ioutil.hpp"
#include "util/qsort.hpp"

namespace graphchi {
    
#define SHARDER_BUFSIZE (64 * 1024 * 1024)
    
    enum ProcPhase  { COMPUTE_INTERVALS=1, SHOVEL=2 };
    
    
    template <typename EdgeDataType>
    struct edge_with_value {
        vid_t src;
        vid_t dst;
        EdgeDataType value;
        
        edge_with_value(vid_t src, vid_t dst, EdgeDataType value) : src(src), dst(dst), value(value) {}
        
        bool stopper() { return src == 0 && dst == 0; }
    };
    
    template <typename EdgeDataType>
    bool edge_t_src_less(const edge_with_value<EdgeDataType> &a, const edge_with_value<EdgeDataType> &b) {
        if (a.src == b.src) return a.dst < b.dst;
        return a.src < b.src;
    }
    
    template <typename EdgeDataType>
    class sharder {
        
        typedef edge_with_value<EdgeDataType> edge_t;
        
    protected:
        std::string basefilename;
        
        vid_t max_vertex_id;
        
        /* Sharding */
        int nshards;
        std::vector< std::pair<vid_t, vid_t> > intervals;
        int phase;
        
        int * edgecounts;
        int vertexchunk;
        size_t nedges;
        std::string prefix;
        
        int * shovelfs;
        
        edge_t ** bufs;
        int * bufptrs;
        size_t bufsize;
        size_t edgedatasize;
        
        metrics m;
        
        binary_adjacency_list_writer<EdgeDataType> * preproc_writer;
        
    public:
        
        sharder(std::string basefilename) : basefilename(basefilename), m("sharder"), preproc_writer(NULL) {            bufs = NULL;
            edgedatasize = sizeof(EdgeDataType);
        }
        
        
        virtual ~sharder() {
            if (preproc_writer != NULL) {
                delete preproc_writer;
            }
        }
        
        std::string preprocessed_name() {
            std::stringstream ss;
            ss << basefilename;
            ss << "." <<  sizeof(EdgeDataType) << "B.bin";
            return ss.str();
        }
        
        /**
         * Checks if the preprocessed binary temporary file of a graph already exists,
         * so it does not need to be recreated.
         */
        bool preprocessed_file_exists() {
            int f = open(preprocessed_name().c_str(), O_RDONLY);
            if (f >= 0) {
                close(f);
                return true;
            } else {
                return false;
            }
        }
        
        /**
         * Call to start a preprocessing session.
         */
        void start_preprocessing() {
            if (preproc_writer != NULL) {
                logstream(LOG_FATAL) << "start_preprocessing() already called! Aborting." << std::endl;
            }   
            
            m.start_time("preprocessing");
            std::string tmpfilename = preprocessed_name() + ".tmp";
            preproc_writer = new binary_adjacency_list_writer<EdgeDataType>(tmpfilename);
            
            
            logstream(LOG_INFO) << "Started preprocessing: " << basefilename << " --> " << tmpfilename << std::endl;
            
            /* Write the maximum vertex id place holder - to be filled later */
            max_vertex_id = 0;
        }
        
        /**
         * Call to finish the preprocessing session.
         */
        void end_preprocessing() {
            assert(preproc_writer != NULL);
            
            preproc_writer->finish();
            delete preproc_writer;
            preproc_writer = NULL;
            
            /* Rename temporary file */ 
            std::string tmpfilename = preprocessed_name() + ".tmp";
            rename(tmpfilename.c_str(), preprocessed_name().c_str());
            
            assert(preprocessed_file_exists());
            logstream(LOG_INFO) << "Finished preprocessing: " << basefilename << " --> " << preprocessed_name() << std::endl;
            m.stop_time("preprocessing");
        }
        
        /**
         * Add edge to be preprocessed
         */
        void preprocessing_add_edge(vid_t from, vid_t to, EdgeDataType val) {
            preproc_writer->add_edge(from, to, val);
            max_vertex_id = std::max(std::max(from, to), max_vertex_id);
        }
        
        /**
         * Add edge without value to be preprocessed
         */
        void preprocessing_add_edge(vid_t from, vid_t to) {
            preproc_writer->add_edge(from, to);
            max_vertex_id = std::max(std::max(from, to), max_vertex_id);
        }
        
        /** Buffered write function */
        template <typename T>
        void bwrite(int f, char * buf, char * &bufptr, T val) {
            if (bufptr + sizeof(T) - buf >= SHARDER_BUFSIZE) {
                writea(f, buf, bufptr - buf);
                bufptr = buf;
            }
            *((T*)bufptr) = val;
            bufptr += sizeof(T);
        }
        
        
        
        
        
        /**
         * Executes sharding.
         * @param nshards_string the number of shards as a number, or "auto" for automatic determination
         */
        int execute_sharding(std::string nshards_string) {
            m.start_time("execute_sharding");
            determine_number_of_shards(nshards_string);
            
            for(int phase=1; phase <= 2; ++phase) {
                /* Start the sharing process */
                
                binary_adjacency_list_reader<EdgeDataType> reader(preprocessed_name());
                
                /* Read max vertex id */
                max_vertex_id = (vid_t) reader.get_max_vertex_id();
                logstream(LOG_INFO) << "Max vertex id: " << max_vertex_id << std::endl; 
                
                this->start_phase(phase);
                
                reader.read_edges(this);
                
                this->end_phase();
            }
            /* Write the shards */
            write_shards();
            
            m.stop_time("execute_sharding");
            
            /* Print metrics */
            basic_reporter basicrep;
            m.report(basicrep);
            
            return nshards;
        }
        
        /**
         * Sharding. This code might be hard to read - modify very carefully! 
         */
    protected:
        
        virtual void determine_number_of_shards(std::string nshards_string) {
            assert(preprocessed_file_exists());
            if (nshards_string.find("auto") != std::string::npos || nshards_string == "0") {
                logstream(LOG_INFO) << "Determining number of shards automatically." << std::endl;
                
                int membudget_mb = get_option_int("membudget_mb", 1024);
                logstream(LOG_INFO) << "Assuming available memory is " << membudget_mb << " megabytes. " << std::endl;
                logstream(LOG_INFO) << " (This can be defined with configuration parameter 'membudget_mb')" << std::endl;
                
                binary_adjacency_list_reader<EdgeDataType> reader(preprocessed_name());
                size_t numedges = reader.get_numedges();
                
                double max_shardsize = membudget_mb * 1024. * 1024. / 8;
                logstream(LOG_INFO) << "Determining maximum shard size: " << (max_shardsize / 1024. / 1024.) << " MB." << std::endl;
                
                nshards = (int) ( 2 + (numedges * sizeof(EdgeDataType) / max_shardsize) + 0.5); 
            } else {
                nshards = atoi(nshards_string.c_str());
            }
            
            logstream(LOG_INFO) << "Number of shards to be created: " << nshards << std::endl;
            assert(nshards > 1);
        }
        
        void compute_partitionintervals() {
            size_t edges_per_part = nedges / nshards + 1;
            
            logstream(LOG_INFO) <<  "Number of shards: " << nshards << std::endl;
            logstream(LOG_INFO)  << "Edges per shard: " << edges_per_part << std::endl;
            logstream(LOG_INFO)  << "Max vertex id: " << max_vertex_id << std::endl;
            
            vid_t cur_st = 0;
            size_t edgecounter=0;
            std::string fname = filename_intervals(basefilename, nshards);
            FILE * f = fopen(fname.c_str(), "w");
            
            if (f == NULL) {
                logstream(LOG_ERROR) << "Could not open file: " << fname << " error: " <<
                strerror(errno) << std::endl;
            }
            assert(f != NULL);
            
            vid_t i = 0;
            while(nshards > (int) intervals.size()) {
                i += vertexchunk;
                edgecounter += edgecounts[i / vertexchunk];
                if (edgecounter >= edges_per_part || (i >= max_vertex_id)) {
                    intervals.push_back(std::pair<vid_t,vid_t>(cur_st, i + (i >= max_vertex_id)));
                    logstream(LOG_INFO) << "Interval: " << cur_st << " - " << i << std::endl;
                    fprintf(f, "%u\n", i + (i == max_vertex_id));
                    cur_st = i + 1;
                    edgecounter = 0;
                }
            }
            fclose(f);
            assert(nshards == (int)intervals.size());
            
            logstream(LOG_INFO) << "Computed intervals." << std::endl;
        }
        
        std::string shovel_filename(int shard) {
            std::stringstream ss;
            ss << basefilename << shard << "." << nshards << ".shovel";
            return ss.str(); 
        }
        
        void start_phase(int p) {
            phase = p;
            lastpart = 0;
            logstream(LOG_INFO) << "Starting phase: " << phase << std::endl;
            switch (phase) {
                case COMPUTE_INTERVALS:
                    /* To compute the intervals, we need to keep track of the vertex degrees.
                     If there is not enough memory to store degree for each vertex, we combine
                     degrees of successive vertice. This results into less accurate shard split,
                     but in practice it hardly matters. */
                    vertexchunk = (int) (max_vertex_id * sizeof(int) / (1024 * 1024 * get_option_long("membudget_mb", 1024)));
                    if (vertexchunk<1) vertexchunk = 1;                    
                    edgecounts = (int*)calloc( max_vertex_id / vertexchunk + 1, sizeof(int));
                    nedges = 0;
                    break;
                    
                case SHOVEL:
                    shovelfs = new int[nshards];
                    bufs = new edge_t*[nshards];
                    bufptrs =  new int[nshards];
                    bufsize = (1024 * 1024 * get_option_long("membudget_mb", 1024)) / nshards / 4;
                    while(bufsize % sizeof(edge_t) != 0) bufsize++;
                    
                    logstream(LOG_DEBUG)<< "Shoveling bufsize: " << bufsize << std::endl;
                    
                    for(int i=0; i < nshards; i++) {
                        std::string fname = shovel_filename(i);
                        shovelfs[i] = open(fname.c_str(), O_WRONLY | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
                        if (shovelfs[i] < 0) {
                            logstream(LOG_ERROR) << "Could not create a temporary file " << fname <<
                            " error: " << strerror(errno) << std::endl;
                        }
                        assert(shovelfs[i] >= 0);
                        bufs[i] = (edge_t*) malloc(bufsize);
                        bufptrs[i] = 0;
                    }
                    break;
            }
        }
        
        void end_phase() {
            logstream(LOG_INFO) << "Ending phase: " << phase << std::endl;
            switch (phase) {
                case COMPUTE_INTERVALS:
                    compute_partitionintervals();
                    free(edgecounts);
                    edgecounts = NULL;
                    break;
                case SHOVEL:
                    for(int i=0; i<nshards; i++) {
                        writea(shovelfs[i], bufs[i], sizeof(edge_t) * (bufptrs[i]));
                        close(shovelfs[i]);
                        free(bufs[i]);
                    }
                    free(bufs);
                    free(bufptrs);
                    break;
            }    
        }
        
        
        int lastpart;
        
        void swrite(int shard, edge_t et) {
            bufs[shard][bufptrs[shard]++] = et;
            if (bufptrs[shard] * sizeof(edge_t) >= bufsize) {
                writea(shovelfs[shard], bufs[shard], sizeof(edge_t) * bufptrs[shard]);
                bufptrs[shard] = 0;
            }
            
        }
    
        void receive_edge(vid_t from, vid_t to, EdgeDataType value) {
            if (to == from) {
                logstream(LOG_WARNING) << "Tried to add self-edge " << from << "->" << to << std::endl;
                return;
            }
            if (from > max_vertex_id || to > max_vertex_id) {
                logstream(LOG_ERROR) << "Tried to add an edge with too large from/to values. From:" << 
                from << " to: "<< to << " max: " << max_vertex_id << std::endl;
                assert(false);
            }
            switch (phase) {
                case COMPUTE_INTERVALS:
                    edgecounts[to / vertexchunk]++;
                    nedges++;
                    break;
                case SHOVEL:
                    bool found=false;
                    for(int i=0; i < nshards; i++) {
                        int shard = (lastpart + i) % nshards;
                        if (to >= intervals[shard].first && to <= intervals[shard].second) {
                            swrite(shard, edge_t(from, to, value));
                            lastpart = shard;  // Small optimizations, which works if edges are in order for each vertex - not much though
                            found = true;
                            break;
                        }
                    }
                    if(!found) {
                        logstream(LOG_ERROR) << "Shard not found for : " << to << std::endl; 
                    }
                    assert(found);
                    break;
            }
        }
        
        
        
        /** 
         * Write the shard by sorting the shovel file and compressing the
         * adjacency information.
         * To support different shard types, override this function!
         */
        virtual void write_shards() {
            for(int shard=0; shard < nshards; shard++) {
                logstream(LOG_INFO) << "Starting final processing for shard: " << shard << std::endl;
                
                std::string shovelfname = shovel_filename(shard);
                std::string fname = filename_shard_adj(basefilename, shard, nshards);
                std::string edfname = filename_shard_edata<EdgeDataType>(basefilename, shard, nshards);
                
                edge_t * shovelbuf;
                int shovelf = open(shovelfname.c_str(), O_RDONLY);
                size_t shovelsize = readfull(shovelf, (char**) &shovelbuf);
                size_t numedges = shovelsize / sizeof(edge_t);
                
                logstream(LOG_DEBUG) << "Shovel size:" << shovelsize << " edges: " << numedges << std::endl;
                
                quickSort(shovelbuf, (int)numedges, edge_t_src_less<EdgeDataType>);
                
                // Create the final file
                int f = open(fname.c_str(), O_WRONLY | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
                if (f < 0) {
                    logstream(LOG_ERROR) << "Could not open " << fname << " error: " << strerror(errno) << std::endl;
                }
                assert(f >= 0);
                int trerr = ftruncate(f, 0);
                assert(trerr == 0);
                
                /* Create edge data file */
                int ef = open(edfname.c_str(), O_WRONLY | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
                if (ef < 0) {
                    logstream(LOG_ERROR) << "Could not open " << edfname << " error: " << strerror(errno) << std::endl;
                    
                }
                assert(ef >= 0);
                
                char * buf = (char*) malloc(SHARDER_BUFSIZE); 
                char * bufptr = buf;
                char * ebuf = (char*) malloc(SHARDER_BUFSIZE);
                char * ebufptr = ebuf;
                
                vid_t curvid=0;
                size_t istart = 0;
                for(size_t i=0; i <= numedges; i++) {
                    edge_t edge = (i < numedges ? shovelbuf[i] : edge_t(0, 0, EdgeDataType())); // Last "element" is a stopper
                    bwrite<EdgeDataType>(ef, ebuf, ebufptr, EdgeDataType(edge.value));
                    
                    if ((edge.src != curvid)) {
                        // New vertex
                        size_t count = i - istart;
                        assert(count>0 || curvid==0);
                        if (count>0) {
                            if (count < 255) {
                                uint8_t x = (uint8_t)count;
                                bwrite<uint8_t>(f, buf, bufptr, x);
                            } else {
                                bwrite<uint8_t>(f, buf, bufptr, 0xff);
                                bwrite<uint32_t>(f, buf, bufptr,(uint32_t)count);
                            }
                        }
                        
                        for(size_t j=istart; j<i; j++) {
                            bwrite(f, buf, bufptr,  shovelbuf[j].dst);
                        }
                        
                        istart = i;
                        
                        // Handle zeros
                        if (!edge.stopper()) {
                            if (edge.src - curvid > 1 || (i == 0 && edge.src>0)) {
                                int nz = edge.src-curvid-1;
                                if (i == 0 && edge.src>0) nz = edge.src; // border case with the first one
                                do {
                                    bwrite<uint8_t>(f, buf, bufptr, 0);
                                    nz--;
                                    int tnz = std::min(254, nz);
                                    bwrite<uint8_t>(f, buf, bufptr, (uint8_t) tnz);
                                    nz -= tnz;
                                } while (nz>0);
                            }
                        }
                        curvid = edge.src;
                    }
                }
                
                /* Flush buffers and free memory */
                writea(f, buf, bufptr - buf);
                free(buf);
                free(shovelbuf);
                close(f);
                close(shovelf);
                
                writea(ef, ebuf, ebufptr - ebuf);
                close(ef);
                
                free(ebuf);
                remove(shovelfname.c_str()); 
                
            }
            
            create_degree_file();
        }
        
        
        typedef char dummy_t;
        
        typedef sliding_shard<int, dummy_t> slidingshard_t;
        typedef memory_shard<int, dummy_t> memshard_t;
        
        
        void create_degree_file() {
            // Initialize IO
            stripedio * iomgr = new stripedio(m);
            std::vector<slidingshard_t * > sliding_shards;
            
            int subwindow = 5000000;
            std::cout << "Subwindow : " << subwindow << std::endl;
            m.set("subwindow", (size_t)subwindow);
            
            int loadthreads = 4;
            
            m.start_time("degrees.runtime");
            
            /* Initialize streaming shards */
            int blocksize = get_option_int("blocksize", 1024 * 1024);
            
            for(int p=0; p < nshards; p++) {
                std::cout << "Initialize streaming shard: " << p << std::endl;
                sliding_shards.push_back(
                                         new slidingshard_t(iomgr, filename_shard_edata<EdgeDataType>(basefilename, p, nshards), 
                                                            filename_shard_adj(basefilename, p, nshards), intervals[p].first, 
                                                            intervals[p].second, 
                                                            blocksize, m, true));
            }
            for(int p=0; p < nshards; p++) sliding_shards[p]->only_adjacency = true;
            
            graphchi_context ginfo;
            ginfo.nvertices = 1 + intervals[nshards - 1].second;
            ginfo.scheduler = NULL;
            
            
            std::string outputfname = filename_degree_data(basefilename);
            
            int degreeOutF = open(outputfname.c_str(), O_RDWR | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
            if (degreeOutF < 0) {
                logstream(LOG_ERROR) << "Could not create: " << degreeOutF << std::endl;
            }
            assert(degreeOutF >= 0);
            int trerr = ftruncate(degreeOutF, ginfo.nvertices * sizeof(int) * 2);
            assert(trerr == 0);
            
            for(int window=0; window<nshards; window++) {
                metrics_entry mwi = m.start_time();
                
                vid_t interval_st = intervals[window].first;
                vid_t interval_en = intervals[window].second;
                
                /* Flush stream shard for the window */
                sliding_shards[window]->flush();
                
                /* Load shard[window] into memory */
                memshard_t memshard(iomgr, filename_shard_edata<EdgeDataType>(basefilename, window, nshards), filename_shard_adj(basefilename, window, nshards),
                                    interval_st, interval_en, m);
                memshard.only_adjacency = true;
                logstream(LOG_INFO) << "Interval: " << interval_st << " " << interval_en << std::endl;
                
                for(vid_t subinterval_st=interval_st; subinterval_st < interval_en; ) {
                    vid_t subinterval_en = std::min(interval_en, subinterval_st + subwindow);
                    logstream(LOG_INFO) << "(Degree proc.) Sub-window: [" << subinterval_st << " - " << subinterval_en << "]" << std::endl;
                    assert(subinterval_en >= subinterval_st && subinterval_en <= interval_en);
                    
                    /* Preallocate vertices */
                    metrics_entry men = m.start_time();
                    int nvertices = subinterval_en-subinterval_st+1;
                    std::vector< graphchi_vertex<int, dummy_t> > vertices(nvertices, graphchi_vertex<int, dummy_t>()); // preallocate
                    
                    
                    for(int i=0; i < nvertices; i++) {
                        vertices[i] = graphchi_vertex<int, dummy_t>(subinterval_st + i, NULL, NULL, 0, 0);
                        vertices[i].scheduled =  true;
                    } 
                    
                    metrics_entry me = m.start_time();
                    omp_set_num_threads(loadthreads);
#pragma omp parallel for
                    for(int p=-1; p < nshards; p++)  {
                        if (p == (-1)) {
                            // if first window, now need to load the memshard
                            if (memshard.loaded() == false) {
                                memshard.load();
                            }
                            
                            /* Load vertices from memshard (only inedges for now so can be done in parallel) */
                            memshard.load_vertices(subinterval_st, subinterval_en, vertices);
                        } else {
                            /* Stream forward other than the window partition */
                            if (p != window) {
                                sliding_shards[p]->read_next_vertices(nvertices, subinterval_st, vertices, false);
                            }
                        }
                    }
                    
                    m.stop_time(me, "stream_ahead", window,  true);  
                    
                    
                    metrics_entry mev = m.start_time();
                    // Read first current values
                    
                    int * vbuf = (int*) malloc(nvertices*sizeof(int)*2);
                    
                    for(int i=0; i<nvertices; i++) {
                        vbuf[2 * i] = vertices[i].num_inedges();
                        vbuf[2 * i +1] = vertices[i].num_outedges();
                    }
                    pwritea(degreeOutF, vbuf, nvertices * sizeof(int) * 2, subinterval_st * sizeof(int) * 2);
                    
                    free(vbuf);
                    
                    // Move window
                    subinterval_st = subinterval_en+1;
                }
                /* Move the offset of the window-shard forward */
                sliding_shards[window]->set_offset(memshard.offset_for_stream_cont(), memshard.offset_vid_for_stream_cont(),
                                                   memshard.edata_ptr_for_stream_cont());   
            }
            close(degreeOutF);
            m.stop_time("degrees.runtime");
            delete iomgr;
        }
        
        friend class binary_adjacency_list_reader<EdgeDataType>;
    }; // End class sharder
    
    
    
}; // namespace


#endif




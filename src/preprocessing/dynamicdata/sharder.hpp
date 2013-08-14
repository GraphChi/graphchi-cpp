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
 * can process. For DYNAMICEDATA
 */
 

#ifndef DYNAMICEDATA
  error(Can be used only with DYNAMICEDATA)
#endif 

#ifndef GRAPHCHI_SHARDER_DEF
#define GRAPHCHI_SHARDER_DEF


#include <iostream>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <vector>
#include <omp.h>
#include <errno.h>
#include <sstream>
#include <string>
#include <vector>

#include "api/chifilenames.hpp"
#include "api/graphchi_context.hpp"
#include "graphchi_types.hpp"
#include "io/stripedio.hpp"
#include "logger/logger.hpp"
#include "engine/auxdata/degree_data.hpp"
#include "metrics/metrics.hpp"
#include "metrics/reps/basic_reporter.hpp"
#include "shards/memoryshard.hpp"
#include "shards/slidingshard.hpp"
#include "output/output.hpp"
#include "util/ioutil.hpp"
#include "util/radixSort.hpp"
#include "util/kwaymerge.hpp"
#ifdef DYNAMICEDATA
#include "util/qsort.hpp"
#endif 
namespace graphchi {
     
    
#define SHARDER_BUFSIZE (64 * 1024 * 1024)
    
    enum ProcPhase  { COMPUTE_INTERVALS=1, SHOVEL=2 };
 
    
    template <typename VectorElementType, typename HeaderDataType>
    struct edge_with_value {
        vid_t src;
        vid_t dst;
        HeaderDataType hdr;
        std::vector<VectorElementType> value;
        
        // For dynamic edge data, we need to know if the value needs to be added
        // to the vector, or are we storing an empty vector.
        bool is_chivec_value;
        edge_with_value() {}
        
        edge_with_value(vid_t src, vid_t dst, std::vector<VectorElementType> value) : src(src), dst(dst), value(value) {
        }
        
        edge_with_value(vid_t src, vid_t dst, std::vector<VectorElementType> value, HeaderDataType hdr) : src(src), dst(dst), value(value), hdr(hdr) {
        }
        
        // Order primarily by dst, then by src
        bool operator< (edge_with_value<VectorElementType, HeaderDataType> &x2) {
            return (dst < x2.dst);
        }
        
        void reade(int f) {
            read(f, &src, sizeof(vid_t));
            read(f, &dst, sizeof(vid_t));
            uint16_t nvalues;
            read(f, &nvalues, sizeof(uint16_t));
            value.resize(nvalues);
            read(f, &value[0], sizeof(VectorElementType) * nvalues);
            read(f, &hdr, sizeof(HeaderDataType));
        }
        
        void writee(int f) {
            writea(f, &src, sizeof(vid_t));
            writea(f, &dst, sizeof(vid_t));
            uint16_t nvalues = value.size();
            assert(value.size() < 1<<16);
            writea(f, &nvalues, sizeof(uint16_t));
            writea(f, &value[0], sizeof(VectorElementType) * nvalues);
            writea(f, &hdr, sizeof(HeaderDataType));
        }
        
        
        bool stopper() { return src == 0 && dst == 0; }
    };
    
    template <typename VectorElementType, typename HeaderDataType>
    bool edge_t_src_less(const edge_with_value<VectorElementType, HeaderDataType> &a, const edge_with_value<VectorElementType, HeaderDataType> &b) {
        if (a.src == b.src) {
            return a.dst < b.dst;
        }
        return a.src < b.src;
    }
    
    template <typename VectorElementType, typename HeaderDataType>
    bool edge_t_dst_less(const edge_with_value<VectorElementType, HeaderDataType> &a, const edge_with_value<VectorElementType, HeaderDataType> &b) {
        return a.dst < b.dst;
    }
    
    template <class VectorElementType, typename HeaderDataType>
    struct dstF {inline vid_t operator() (edge_with_value<VectorElementType, HeaderDataType> a) {return a.dst;} };
    
    template <class VectorElementType, typename HeaderDataType>
    struct srcF {inline vid_t operator() (edge_with_value<VectorElementType, HeaderDataType> a) {return a.src;} };
    
  
    
    template <typename VectorElementType, typename HeaderDataType>
    struct shard_flushinfo {
        std::string shovelname;
        size_t numedges;
        edge_with_value<VectorElementType, HeaderDataType> * buffer;
        vid_t max_vertex;
        
        shard_flushinfo(std::string shovelname, vid_t max_vertex, size_t numedges, edge_with_value<VectorElementType, HeaderDataType> * buffer) :
        shovelname(shovelname), numedges(numedges), buffer(buffer), max_vertex(max_vertex) {}
        
        void flush() {
            /* Sort */
            // TODO: remove duplicates here!
            logstream(LOG_INFO) << "Sorting shovel: " << shovelname << ", max:" << max_vertex << std::endl;
            iSort(buffer, (intT)numedges, (intT)max_vertex, dstF<VectorElementType, HeaderDataType>());
            logstream(LOG_INFO) << "Sort done." << shovelname << std::endl;
            int f = open(shovelname.c_str(), O_WRONLY | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
            //writea(f, buffer, numedges * sizeof(edge_with_value<VectorElementType, HeaderDataType>));
            
            writea(f, &numedges, sizeof(size_t));
            
            /* Write the edges into file. This is quite a bit more complicated than with non-dynamic data... */
            for(size_t i=0; i<numedges; i++) {
                edge_with_value<VectorElementType, HeaderDataType> &edge = buffer[i];
                edge.writee(f);
            }
            
            close(f);
            free(buffer);
        }
    };
    
    // Run in a thread
    template <typename VectorElementType, typename HeaderDataType>
    static void * shard_flush_run(void * _info) {
        shard_flushinfo<VectorElementType, HeaderDataType> * task = (shard_flushinfo<VectorElementType, HeaderDataType>*)_info;
        task->flush();
        return NULL;
    }
    
    
    template <typename VectorElementType, typename HeaderDataType=dummy>
    struct shovel_merge_source : public merge_source<edge_with_value<VectorElementType, HeaderDataType> > {
        
        size_t bufsize_bytes;
        std::string shovelfile;
        size_t idx;
        size_t bufidx;
        std::vector<edge_with_value<VectorElementType, HeaderDataType> > buffer;
        int f;
        size_t numedges;
        size_t bufsize_edges;
        
        shovel_merge_source(size_t bufsize_bytes, std::string shovelfile) : bufsize_bytes(bufsize_bytes), 
        shovelfile(shovelfile), idx(0), bufidx(0) {
            f = open(shovelfile.c_str(), O_RDONLY);
            
            if (f < 0) {
                logstream(LOG_ERROR) << "Could not open shovel file: " << shovelfile << std::endl;
                printf("Error: %d, %s\n", errno, strerror(errno));
            }
            
            assert(f>=0);
            
            bufsize_edges = bufsize_bytes / sizeof(edge_with_value<VectorElementType, HeaderDataType>);
            read(f, &numedges, sizeof(size_t));
            load_next();
        }
        
        virtual ~shovel_merge_source() {}
        
        void finish() {
            close(f);
            remove(shovelfile.c_str());
            buffer.clear();
        }
        
        void load_next() {
            size_t nread = 0;
            buffer.clear();
            while(nread < bufsize_bytes) {
                edge_with_value<VectorElementType, HeaderDataType> edge;
                edge.reade(f);
                buffer.push_back(edge);
                nread += sizeof(vid_t) * 2 + sizeof(uint16_t) + sizeof(HeaderDataType) + edge.value.size() * sizeof(VectorElementType);
            }
            bufidx = 0;
        }
        
        bool has_more() {
            return idx < numedges;
        }
        
        edge_with_value<VectorElementType, HeaderDataType> & next() {
            if (bufidx == bufsize_edges) {
                load_next();
            }
            idx++;
            if (idx == numedges) {
                edge_with_value<VectorElementType, HeaderDataType> &x = buffer[bufidx++];
                finish();
                return x;
            }
            return buffer[bufidx++];
        }
    };
    
    template <typename VectorElementType, typename HeaderDataType=dummy>
    class sharder : public merge_sink<edge_with_value<VectorElementType, HeaderDataType> > {
        
        typedef edge_with_value<VectorElementType, HeaderDataType> edge_t;
        
    protected:
        std::string basefilename;
        
        vid_t max_vertex_id;
        
        /* Sharding */
        int nshards;
        std::vector< std::pair<vid_t, vid_t> > intervals;
        
        int phase;
        
        int vertexchunk;
        size_t nedges;
        std::string prefix;
        
        int compressed_block_size;
        
        int * bufptrs;
        size_t bufsize;
        size_t edgedatasize;
        size_t ebuffer_size;
        size_t edges_per_block;
        
        vid_t filter_max_vertex;
        
        
        bool no_edgevalues;
        edge_t last_added_edge;
        
        metrics m;
        
        
        size_t curshovel_idx;
        size_t shovelsize;
        int numshovels;
        size_t shoveled_edges;
        bool shovel_sorted;
        edge_with_value<VectorElementType, HeaderDataType> * curshovel_buffer;
        std::vector<pthread_t> shovelthreads;
        
    public:
        
        sharder(std::string basefilename) : basefilename(basefilename), m("sharder") {          
            
            edgedatasize = sizeof(VectorElementType);
            no_edgevalues = false;
            compressed_block_size = 4096 * 1024;
            filter_max_vertex = 0;
            curshovel_buffer = NULL;
            while (compressed_block_size % sizeof(VectorElementType) != 0) compressed_block_size++;
            edges_per_block = compressed_block_size / sizeof(VectorElementType);
        }
        
        
        virtual ~sharder() {
            if (curshovel_buffer == NULL) free(curshovel_buffer);
        }
  
        
        void set_max_vertex_id(vid_t maxid) {
            filter_max_vertex = maxid;
        }
        
        void set_no_edgevalues() {
            no_edgevalues = true;
        }
        
        /**
         * Call to start a preprocessing session.
         */
        void start_preprocessing() {
            m.start_time("preprocessing");
            numshovels = 0;
            shovelsize = (1024l * 1024l * size_t(get_option_int("membudget_mb", 1024)) / 4l / sizeof(edge_with_value<VectorElementType, HeaderDataType>));
            curshovel_idx = 0;
            
            logstream(LOG_INFO) << "Starting preprocessing, shovel size: " << shovelsize << std::endl;
            
            curshovel_buffer = (edge_with_value<VectorElementType, HeaderDataType> *) calloc(shovelsize, sizeof(edge_with_value<VectorElementType, HeaderDataType>));
            
            assert(curshovel_buffer != NULL);
            
            shovelthreads.clear();
            
            /* Write the maximum vertex id place holder - to be filled later */
            max_vertex_id = 0;
            shoveled_edges = 0;
        }
        
        /**
         * Call to finish the preprocessing session.
         */
        void end_preprocessing() {
            m.stop_time("preprocessing");
            flush_shovel(false);
        }
        
        void flush_shovel(bool async=true) {
            /* Flush in separate thread unless the last one */
            shard_flushinfo<VectorElementType, HeaderDataType> * flushinfo = new shard_flushinfo<VectorElementType, HeaderDataType>(shovel_filename(numshovels), max_vertex_id, curshovel_idx, curshovel_buffer);
            
            if (!async) {
                curshovel_buffer = NULL;
                flushinfo->flush();
                
                /* Wait for threads to finish */
                logstream(LOG_INFO) << "Waiting shoveling threads..." << std::endl;
                for(int i=0; i < (int)shovelthreads.size(); i++) {
                    pthread_join(shovelthreads[i], NULL);
                }
            } else {
                if (shovelthreads.size() > 2) {
                    logstream(LOG_INFO) << "Too many outstanding shoveling threads..." << std::endl;

                    for(int i=0; i < (int)shovelthreads.size(); i++) {
                        pthread_join(shovelthreads[i], NULL);
                    }
                    shovelthreads.clear();
                }
                curshovel_buffer = (edge_with_value<VectorElementType, HeaderDataType> *) calloc(shovelsize, sizeof(edge_with_value<VectorElementType, HeaderDataType>));
                pthread_t t;
                int ret = pthread_create(&t, NULL, shard_flush_run<VectorElementType, HeaderDataType>, (void*)flushinfo);
                shovelthreads.push_back(t);
                assert(ret>=0);
            }
            numshovels++;
            curshovel_idx=0;
        }
        
        /**
         * Add edge to be preprocessed with a value.
         */
        void preprocessing_add_edge(vid_t from, vid_t to, VectorElementType val, bool input_value=false) {
            if (from == to) {
                // Do not allow self-edges
                return;
            }
            assert(!input_value);
            edge_with_value<VectorElementType, HeaderDataType> e(from, to, std::vector<VectorElementType>(0));

            e.is_chivec_value = input_value;
            last_added_edge = e;
            curshovel_buffer[curshovel_idx++] = e;
            if (curshovel_idx == shovelsize) {
                flush_shovel();
            }
            
            max_vertex_id = std::max(std::max(from, to), max_vertex_id);
            shoveled_edges++;
        }
        
         void preprocessing_add_edge_multival(vid_t from, vid_t to, HeaderDataType hdr, std::vector<VectorElementType> & vals) {
             if (from == to) {
                 // Do not allow self-edges
                 return;
             }
             edge_with_value<VectorElementType, HeaderDataType> e(from, to, vals, hdr);
             
             e.is_chivec_value = true;
             curshovel_buffer[curshovel_idx++] = e;
             if (curshovel_idx == shovelsize) {
                 flush_shovel();
             }
             
             max_vertex_id = std::max(std::max(from, to), max_vertex_id);
             shoveled_edges++;
        }
        
         
        /**
         * Add edge without value to be preprocessed
         */
        void preprocessing_add_edge(vid_t from, vid_t to) {
            preprocessing_add_edge(from, to, VectorElementType());
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
        
        int blockid;
        
        template <typename T>
        void edata_flush(char * buf, char * bufptr, std::string & shard_filename, size_t totbytes) {
            int len = (int) (bufptr - buf);
            
            m.start_time("edata_flush");
            
            std::string block_filename = filename_shard_edata_block(shard_filename, blockid, compressed_block_size);
            int f = open(block_filename.c_str(), O_RDWR | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
            write_compressed(f, buf, len);
            close(f);
            
            m.stop_time("edata_flush");
            
            
             // Write block's uncompressed size
            write_block_uncompressed_size(block_filename, len);
            blockid++;
        }
        
        template <typename T>
        void bwrite_edata(char * &buf, char * &bufptr, T val, size_t & totbytes, std::string & shard_filename,
                          size_t & edgecounter) {
            if (no_edgevalues) return;
            
            if (edgecounter == edges_per_block) {
                edata_flush<T>(buf, bufptr, shard_filename, totbytes);
                bufptr = buf;
                edgecounter = 0;
            }
            
            // Check if buffer is big enough
            if (bufptr - buf + sizeof(T) > ebuffer_size) {
                ebuffer_size *= 2;
                logstream(LOG_DEBUG) << "Increased buffer size to: " << ebuffer_size << std::endl;
                size_t ptroff = bufptr - buf; // Remember the offset
                buf = (char *) realloc(buf, ebuffer_size);
                bufptr = buf + ptroff;
            }
            
            totbytes += sizeof(T);
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
            if (nshards_string.find("auto") != std::string::npos || nshards_string == "0") {
                logstream(LOG_INFO) << "Determining number of shards automatically." << std::endl;
                
                int membudget_mb = get_option_int("membudget_mb", 1024);
                logstream(LOG_INFO) << "Assuming available memory is " << membudget_mb << " megabytes. " << std::endl;
                logstream(LOG_INFO) << " (This can be defined with configuration parameter 'membudget_mb')" << std::endl;
                
                size_t numedges = shoveled_edges; 
                
                double max_shardsize = membudget_mb * 1024. * 1024. / 8;
                logstream(LOG_INFO) << "Determining maximum shard size: " << (max_shardsize / 1024. / 1024.) << " MB." << std::endl;
                                
                // For dynamic edge data, more working memory is needed, thus the number of shards is larger.
                nshards = (int) ( 2 + 4 * (numedges * sizeof(VectorElementType) / max_shardsize) + 0.5);
                
            } else {
                nshards = atoi(nshards_string.c_str());
            }
            assert(nshards > 0);
            logstream(LOG_INFO) << "Number of shards to be created: " << nshards << std::endl;
        }
        
        
    protected:
        
        void one_shard_intervals() {
            assert(nshards == 1);
            std::string fname = filename_intervals(basefilename, nshards);
            FILE * f = fopen(fname.c_str(), "w");
            intervals.push_back(std::pair<vid_t,vid_t>(0, max_vertex_id));
            fprintf(f, "%u\n", max_vertex_id);
            fclose(f);
            
            /* Write meta-file with the number of vertices */
            std::string numv_filename = basefilename + ".numvertices";
            f = fopen(numv_filename.c_str(), "w");
            fprintf(f, "%u\n", 1 + max_vertex_id);
            fclose(f);
            
            assert(nshards == (int)intervals.size());
        }
        
        
        std::string shovel_filename(int idx) {
            std::stringstream ss;
            ss << basefilename << sizeof(VectorElementType) << "." << idx << ".shovel";
            return ss.str();
        }
        
        
        int lastpart;
        
        
        
        
        degree * degrees;
        
        virtual void finish_shard(int shard, edge_t * shovelbuf, size_t shovelsize) {
            m.start_time("shard_final");
            blockid = 0;
            size_t edgecounter = 0;
            
            logstream(LOG_INFO) << "Starting final processing for shard: " << shard << std::endl;
            
            std::string fname = filename_shard_adj(basefilename, shard, nshards);
            std::string edfname = filename_shard_edata<VectorElementType>(basefilename, shard, nshards);
            std::string edblockdirname = dirname_shard_edata_block(edfname, compressed_block_size);
            
            /* Make the block directory */
            if (!no_edgevalues)
                mkdir(edblockdirname.c_str(), 0777);
            size_t numedges = shovelsize / sizeof(edge_t);
            
            logstream(LOG_DEBUG) << "Shovel size:" << shovelsize << " edges: " << numedges << std::endl;
            
            m.start_time("finish_shard.sort");

            quickSort(shovelbuf, (int)numedges, edge_t_src_less<VectorElementType, HeaderDataType>);
            m.stop_time("finish_shard.sort");

            // Create the final file
            int f = open(fname.c_str(), O_WRONLY | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
            if (f < 0) {
                logstream(LOG_ERROR) << "Could not open " << fname << " error: " << strerror(errno) << std::endl;
            }
            assert(f >= 0);
            int trerr = ftruncate(f, 0);
            assert(trerr == 0);
            
            char * buf = (char*) malloc(SHARDER_BUFSIZE);
            char * bufptr = buf;
            
            char * ebuf = (char*) malloc(compressed_block_size);
            ebuffer_size = compressed_block_size;
            char * ebufptr = ebuf;
            
            vid_t curvid=0;
            vid_t lastdst = 0xffffffff;
            int jumpover = 0;
            size_t num_uniq_edges = 0;
            size_t last_edge_count = 0;
            size_t istart = 0;
            size_t tot_edatabytes = 0;
            for(size_t i=0; i <= numedges; i++) {
                if (i % 10000000 == 0) logstream(LOG_DEBUG) << i << " / " << numedges << std::endl;
                i += jumpover;  // With dynamic values, there might be several values for one edge, and thus the edge repeated in the data.
                jumpover = 0;
                edge_t edge = (i < numedges ? shovelbuf[i] : edge_t(0, 0, std::vector<VectorElementType>())); // Last "element" is a stopper
                
                if (lastdst == edge.dst && edge.src == curvid) {
                    // Currently not supported
                    logstream(LOG_ERROR) << "Duplicate edge in the stream - aborting" << std::endl;
                    assert(false);
                }
                lastdst = edge.dst;
                
                if (!edge.stopper()) {

                    
                    /* If we have dynamic edge data, we need to write the header of chivector - if there are edge values */
                     bwrite_edata<HeaderDataType>(ebuf, ebufptr, shovelbuf[i].hdr, tot_edatabytes, edfname, edgecounter);
                    
                    if (edge.is_chivec_value) {
                        // Need to check how many values for this edge
                        int count = shovelbuf[i].value.size();
                        
                        assert(count < 32768);
                        typename chivector<VectorElementType, HeaderDataType>::sizeword_t szw;
                        ((uint16_t *) &szw)[0] = (uint16_t)count;  // Sizeword with length and capacity = count
                        ((uint16_t *) &szw)[1] = (uint16_t)count;
                        bwrite_edata<typename chivector<VectorElementType, HeaderDataType>::sizeword_t>(ebuf, ebufptr, szw, tot_edatabytes, edfname, edgecounter);
                        for(int j=0; j < count; j++) {
                                bwrite_edata<VectorElementType>(ebuf, ebufptr, shovelbuf[i].value[j], tot_edatabytes, edfname, edgecounter);
                        }
                  
                    
                        jumpover = count - 1; // Jump over
                    } else {
                        // Just write size word with zero
                        bwrite_edata<int>(ebuf, ebufptr, 0, tot_edatabytes, edfname, edgecounter);
                    }
                    num_uniq_edges++;
                    edgecounter++; // Increment edge counter here --- notice that dynamic edata case makes two or more calls to bwrite_edata before incrementing
                }
                if (degrees != NULL && edge.src != edge.dst) {
                    degrees[edge.src].outdegree++;
                    degrees[edge.dst].indegree++;
                }
                
                if ((edge.src != curvid) || edge.stopper()) {
                    // New vertex

                    size_t count = num_uniq_edges - 1 - last_edge_count;
                    last_edge_count = num_uniq_edges - 1;
                    if (edge.stopper()) count++;  
                    assert(count>0 || curvid==0);
                    if (count>0) {
                        if (count < 255) {
                            uint8_t x = (uint8_t)count;
                            bwrite<uint8_t>(f, buf, bufptr, x);
                        } else {
                            bwrite<uint8_t>(f, buf, bufptr, 0xff);
                            bwrite<uint32_t>(f, buf, bufptr, (uint32_t)count);
                        }
                    }
                    
                    // Special dealing with dynamic edata because some edges can be present multiple
                    // times in the shovel.
                    for(size_t j=istart; j < i; j++) {
                        if (j == istart || shovelbuf[j - 1].dst != shovelbuf[j].dst) {
                            bwrite(f, buf, bufptr,  shovelbuf[j].dst);
                        }
                    }
                    istart = i;
                    istart += jumpover;
                    
                    // Handle zeros
                    if (!edge.stopper()) {
                        if (edge.src - curvid > 1 || (i == 0 && edge.src>0)) {
                            int nz = edge.src - curvid - 1;
                            if (i == 0 && edge.src > 0) nz = edge.src; // border case with the first one
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
            
            /* Write edata size file */
            if (!no_edgevalues) {
                edata_flush<VectorElementType>(ebuf, ebufptr, edfname, tot_edatabytes);
                
                std::string sizefilename = edfname + ".size";
                std::ofstream ofs(sizefilename.c_str());

                ofs << num_uniq_edges * sizeof(int); // For dynamic edge data, write the number of edges.                
                ofs.close();
            }
            free(ebuf);
            
            m.stop_time("shard_final");
        }
        
        /* Begin: Kway -merge sink interface */
        
        size_t edges_per_shard;
        size_t cur_shard_counter;
        size_t shard_capacity;
        size_t sharded_edges;
        int shardnum;
        edge_with_value<VectorElementType, HeaderDataType> * sinkbuffer;
        vid_t prevvid;
        vid_t this_interval_start;
        
        virtual void add(edge_with_value<VectorElementType, HeaderDataType> val) {
            if (cur_shard_counter >= edges_per_shard && val.dst != prevvid) {
                createnextshard();
            }
            
            if (cur_shard_counter == shard_capacity) {
                /* Really should have a way to limit shard sizes, but probably not needed in practice */
                logstream(LOG_WARNING) << "Shard " << shardnum << " overflowing! " << cur_shard_counter << " / " << shard_capacity << std::endl;
                shard_capacity = (size_t) (1.2 * shard_capacity);
                sinkbuffer = (edge_with_value<VectorElementType, HeaderDataType>*) realloc(sinkbuffer, shard_capacity * sizeof(edge_with_value<VectorElementType, HeaderDataType>));
            }
            
            sinkbuffer[cur_shard_counter++] = val;
            prevvid = val.dst;
            sharded_edges++;
        }
        
        void createnextshard() {
            assert(shardnum < nshards);
            intervals.push_back(std::pair<vid_t, vid_t>(this_interval_start, (shardnum == nshards - 1 ? max_vertex_id : prevvid)));
            this_interval_start = prevvid + 1;
            finish_shard(shardnum++, sinkbuffer, cur_shard_counter * sizeof(edge_with_value<VectorElementType, HeaderDataType>));
            sinkbuffer = (edge_with_value<VectorElementType, HeaderDataType> *) malloc(shard_capacity * sizeof(edge_with_value<VectorElementType, HeaderDataType>));
            cur_shard_counter = 0;
            
            // Adjust edges per hard so that it takes into account how many edges have been spilled now
            logstream(LOG_INFO) << "Remaining edges: " << (shoveled_edges - sharded_edges) << " remaining shards:" << (nshards - shardnum)
                << " edges per shard=" << edges_per_shard << std::endl;
            if (shardnum < nshards) edges_per_shard = (shoveled_edges - sharded_edges) / (nshards - shardnum);
            logstream(LOG_INFO) << "Edges per shard: " << edges_per_shard << std::endl;
            
        }
        
        virtual void done() {
            createnextshard();
            if (shoveled_edges != sharded_edges) {
                logstream(LOG_INFO) << "Shoveled " << shoveled_edges << " but sharded " << sharded_edges << " edges" << std::endl;
            }
            assert(shoveled_edges == sharded_edges);
            
            logstream(LOG_INFO) << "Created " << shardnum << " shards, expected: " << nshards << std::endl;
            assert(shardnum <= nshards);
            free(sinkbuffer);
            sinkbuffer = NULL;
            
            /* Write intervals */
            std::string fname = filename_intervals(basefilename, nshards);
            FILE * f = fopen(fname.c_str(), "w");
            
            if (f == NULL) {
                logstream(LOG_ERROR) << "Could not open file: " << fname << " error: " <<
                strerror(errno) << std::endl;
            }
            assert(f != NULL);
            for(int i=0; i<(int)intervals.size(); i++) {
               fprintf(f, "%u\n", intervals[i].second);
            }
            fclose(f);
            
            /* Write meta-file with the number of vertices */
            std::string numv_filename = basefilename + ".numvertices";
            f = fopen(numv_filename.c_str(), "w");
            fprintf(f, "%u\n", 1 + max_vertex_id);
            fclose(f);
        }
        
        /* End: Kway -merge sink interface */
        
        
        
        /**
         * Write the shard by sorting the shovel file and compressing the
         * adjacency information.
         * To support different shard types, override this function!
         */
        virtual void write_shards() {
            
            size_t membudget_mb = (size_t) get_option_int("membudget_mb", 1024);
            
            // Check if we have enough memory to keep track
            // of the vertex degrees in-memory (heuristic)
            degrees = (degree *) calloc(1 + max_vertex_id, sizeof(degree));
            
            // KWAY MERGE
            sharded_edges = 0;
            edges_per_shard = shoveled_edges / nshards + 1;
            shard_capacity = edges_per_shard / 2 * 3;  // Shard can go 50% over
            shardnum = 0;
            this_interval_start = 0;
            sinkbuffer = (edge_with_value<VectorElementType, HeaderDataType> *) calloc(shard_capacity, sizeof(edge_with_value<VectorElementType, HeaderDataType>));
            logstream(LOG_INFO) << "Edges per shard: " << edges_per_shard << " nshards=" << nshards << " total: " << shoveled_edges << std::endl;
            cur_shard_counter = 0;
            
            /* Initialize kway merge sources */
            size_t B = membudget_mb * 1024 * 1024 / 2 / numshovels;
            while (B % sizeof(edge_with_value<VectorElementType, HeaderDataType>) != 0) B++;
            logstream(LOG_INFO) << "Buffer size in merge phase: " << B << std::endl;
            prevvid = (-1);
            std::vector< merge_source<edge_with_value<VectorElementType, HeaderDataType> > *> sources;
            for(int i=0; i < numshovels; i++) {
                sources.push_back(new shovel_merge_source<VectorElementType, HeaderDataType>(B, shovel_filename(i)));
            }
            
            kway_merge<edge_with_value<VectorElementType, HeaderDataType> > merger(sources, this);
            merger.merge();
            
            // Delete sources
            for(int i=0; i < (int)sources.size(); i++) {
                delete (shovel_merge_source<VectorElementType, HeaderDataType> *)sources[i];
            }
            
            
            
            std::string degreefname = filename_degree_data(basefilename);
            int degreeOutF = open(degreefname.c_str(), O_RDWR | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
            if (degreeOutF < 0) {
                logstream(LOG_ERROR) << "Could not create: " << degreeOutF << std::endl;
                assert(degreeOutF >= 0);
            }
            
            writea(degreeOutF, degrees, sizeof(degree) * (1 + max_vertex_id));
            free(degrees);
            close(degreeOutF);
        }
        
        
        typedef char dummy_t;
        
        typedef sliding_shard<int, dummy_t> slidingshard_t;
        typedef memory_shard<int, dummy_t> memshard_t;
        
        
    }; // End class sharder
    
    
  
}; // namespace


#endif




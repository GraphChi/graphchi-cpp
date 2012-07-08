
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
 * Class representing a binary adjacency list format used by the
 * sharder. Note, this format does not comply with standard (if there are
 * any) formats.
 *
 * File format supports edges with and without values. 
 */

#ifndef DEF_GRAPHCHI_BINADJLIST_FORMAT
#define DEF_GRAPHCHI_BINADJLIST_FORMAT

#include <assert.h>
#include <stdio.h>
#include <stdint.h> 
#include <unistd.h>
#include <errno.h>
#include <string>

#include "graphchi_types.hpp"
#include "logger/logger.hpp"
#include "util/ioutil.hpp"

namespace graphchi {
    
#define FORMAT_VERSION 20120705   // Format version is the date it was conceived
    
    /**
     * Header struct
     */
    struct bin_adj_header {
        int  format_version;
        uint64_t max_vertex_id;    // Note, use 64-bit to be future-proof.
        bool contains_edge_values;
        uint32_t edge_value_size;
        uint64_t numedges;
    };
    
    /**
      * Internal container class.
      */
    template <typename EdgeDataType>
    struct edge_with_value_badj {
        vid_t vertex;
        EdgeDataType value;
        edge_with_value_badj() {}
        edge_with_value_badj(vid_t v, EdgeDataType x) : vertex(v), value(x) {}
    };
    
    template <typename EdgeDataType>
    class binary_adjacency_list_reader {
        std::string filename;
        int fd;
        size_t fpos;
        size_t blocklen;
        size_t blocksize;
        size_t total_to_process;
        char * block;
        char * blockptr;

        bin_adj_header header;
        
        template <typename U>
        inline U read_val() {
            if (blockptr + sizeof(U) > block + blocklen) {
                // Read
                blocklen = std::min(blocksize, total_to_process - fpos);
                preada(fd, block, blocklen, fpos);
                blockptr = block;
            }
            U res = *((U*)blockptr);
            blockptr += sizeof(U);
            fpos += sizeof(U);
            return res;
        }
        
    public:
        binary_adjacency_list_reader(std::string filename) : filename(filename) {
            fd = open(filename.c_str(), O_RDONLY);
            if (fd < 0) {
                logstream(LOG_FATAL) << "Could not open file: " << filename << " error: " <<
                strerror(errno) << std::endl;
            }
            assert(fd >= 0);
            
            blocksize = (size_t) get_option_long("preprocessing.bufsize", 64 * 1024 * 1024);
            block = (char*) malloc(blocksize);
            blockptr = block;
            total_to_process = get_filesize(filename);
            blocklen = 0;
            fpos = 0;

            header = read_val<bin_adj_header>();
            std::cout << header.format_version << std::endl;
            assert(header.format_version == FORMAT_VERSION);
            std::cout << "Read header: " << header.max_vertex_id << " " << header.numedges << std::endl;
        }
        
        ~binary_adjacency_list_reader() {
            if (block != NULL) free(block);
            close(fd);
        }
        
        template <class Callback>
        void read_edges(Callback * callback) {
            size_t nedges = 0;
          
            /* Note, header has been read in the beginning */
            do {
                
                if (nedges % 10000000 == 0) {
                    logstream(LOG_DEBUG) << (fpos * 1.0 / total_to_process * 100) << "%" << std::endl;
                }                
                vid_t from;
                vid_t to;
                int adjlen;
                EdgeDataType val = EdgeDataType();
                
                from = read_val<vid_t>();
                adjlen = (int) read_val<uint8_t>();
                assert(adjlen > 0);
                for(int i=0; i < adjlen; i++) {
                    to = read_val<vid_t>();
                    if (header.contains_edge_values) {
                        val = read_val<EdgeDataType>();
                    }
                    callback->receive_edge(from, to, val);
                    nedges++;
                }
                
            } while (nedges < header.numedges);
        }
        
        size_t get_max_vertex_id() {
            return header.max_vertex_id;
        }
        
        size_t get_numedges() {
            return header.numedges;
        }
    };
    
    
    template <typename EdgeDataType>
    class binary_adjacency_list_writer {
        
    private:
        
        std::string filename;
        int fd;
        bin_adj_header header;
        int bufsize;
        
        char * buf;
        char * bufptr;
        
        bool initialized;
        
        edge_with_value_badj<EdgeDataType> samev_buf[256];
        vid_t lastid;
        uint8_t counter;
        
    public:
        binary_adjacency_list_writer(std::string filename) : filename(filename) {
            bufsize = (int) get_option_int("preprocessing.bufsize", 64 * 1024 * 1024);
            assert(bufsize > 1024 * 1024);
            fd = open(filename.c_str(), O_WRONLY | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
            if (fd < 0) {
                logstream(LOG_FATAL) << "Could not open file " << filename << " for writing. " <<
                " Error: " << strerror(errno) << std::endl;
            }
            
            header.format_version = FORMAT_VERSION;
            header.max_vertex_id = 0;
            header.contains_edge_values = false;
            header.numedges = 0;
            header.edge_value_size = (uint32_t) sizeof(EdgeDataType);
            
            buf = (char*) malloc(bufsize);
            bufptr = buf;
            bwrite<bin_adj_header>(fd, buf, bufptr,  header);
            counter = 0;
            lastid = 0;
            initialized = false;
            assert(fd >= 0);
        }
        
        ~binary_adjacency_list_writer() {
            if (buf != NULL) delete buf;
        }
        
    protected:
        
        void write_header() {
            logstream(LOG_DEBUG) << "Write header: max vertex: " << header.max_vertex_id << std::endl;
            pwrite(fd, &header, sizeof(bin_adj_header), 0);
        }
        
        /** 
          * Write edges for the current vertex (lastid)
          */
        void flush() {
            if (counter != 0) {
                bwrite<vid_t>(fd, buf, bufptr, lastid);
                bwrite<uint8_t>(fd, buf, bufptr, counter);
                for(int i=0; i < counter; i++) {
                    bwrite<vid_t>(fd, buf, bufptr, samev_buf[i].vertex);
                    if (header.contains_edge_values) {
                        bwrite<EdgeDataType>(fd, buf, bufptr, samev_buf[i].value);
                    }
                }
                header.numedges += (uint64_t)counter;
                counter = 0;
            }
        }
        
        void _addedge(vid_t from, vid_t to, EdgeDataType val) {
            if (from == to) return; // Filter self-edges
            
            if (from == lastid && counter > 0) {
                samev_buf[counter++] = edge_with_value_badj<EdgeDataType>(to, val);
            } else {
                flush();
                lastid = from;
                samev_buf[counter++] = edge_with_value_badj<EdgeDataType>(to, val);
            }
            if (counter == 255) {
                /* Flush */
                flush();
                counter = 0;
            }
            if (from > header.max_vertex_id || to > header.max_vertex_id) {
                header.max_vertex_id = std::max(from, to);
            }
        }
        
        
    public:
        void add_edge(vid_t from, vid_t to, EdgeDataType val) {
            if (!initialized) {
                header.contains_edge_values = true;
                initialized = true;
            }
            if (!header.contains_edge_values) {
                logstream(LOG_ERROR) << "Tried to add edge with a value, although previously added one with a value!" << std::endl;
            }
            assert(header.contains_edge_values);
            
            _addedge(from, to, val);
        }
        
        void add_edge(vid_t from, vid_t to) {
            if (!initialized) {
                header.contains_edge_values = false;
                initialized = true;
            }
            if (header.contains_edge_values) {
                logstream(LOG_ERROR) << "Tried to add edge without a value, although previously added edge with a value!" << std::endl;
            }
            
            assert(!header.contains_edge_values);
            _addedge(from, to, EdgeDataType());
        }
        
        void finish() {
            flush();

            /* Write rest of the buffer out */
            writea(fd, buf, bufptr - buf);
            free(buf);
            buf = NULL;
            
            write_header();
            close(fd);
        }
        /** Buffered write function */
        template <typename T>
        void bwrite(int f, char * buf, char * &bufptr, T val) {
            if (bufptr + sizeof(T) - buf >=  bufsize) {
                writea(f, buf, bufptr - buf);
                bufptr = buf;
            }
            *((T*)bufptr) = val;
            bufptr += sizeof(T);
        }

        
    };
}

#endif

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
 * Edge buffers used by the dynamic graph engine. 
 */

#ifndef DEF_GRAPHCHI_EDGEBUFFERS
#define DEF_GRAPHCHI_EDGEBUFFERS

#include <stdlib.h>
#include <vector> 


namespace graphchi {
    
    /** 
     * Class for buffered edges. These are edges that
     * are currently in memory, and waiting to be commited to disk.
     */
    template <typename EdgeDataType>
    struct created_edge {
        vid_t src;
        vid_t dst;
        EdgeDataType data;
        bool accounted_for_outc;
        bool accounted_for_inc;
        created_edge(vid_t src, vid_t dst, EdgeDataType _data) : src(src), dst(dst), data(_data), accounted_for_outc(false),
        accounted_for_inc(false) {}
    };
    
#define EDGE_BUFFER_CHUNKSIZE 65536
    
    /**
     * Efficient chunked edge-buffer with very low memory-overhead (compared
     * to just using a std-vector.
     */
    template <typename ET>
    class edge_buffer_flat {
        
        unsigned int count;
        std::vector<created_edge<ET> *> bufs;
        
    public:    
        
        edge_buffer_flat() : count(0) {
        }
        
        ~edge_buffer_flat() {
            clear();
        }
        
        void clear() {
            for(int i=0; i< (int)bufs.size(); i++) {
                free(bufs[i]);
            }   
            bufs.clear();       
            count = 0;
        }
        
        unsigned int size() {
            return count;
        }
        
        created_edge<ET> * operator[](unsigned int i) {
            return &bufs[i / EDGE_BUFFER_CHUNKSIZE][i % EDGE_BUFFER_CHUNKSIZE];
        }
        
        void add(vid_t src, vid_t dst, ET data) {
            add(created_edge<ET>(src, dst, data));
        }
        
        void add(created_edge<ET> cedge) {
            int idx = count++;
            int bufidx = idx / EDGE_BUFFER_CHUNKSIZE;
            if (bufidx == (int) bufs.size()) {
                bufs.push_back((created_edge<ET>*)calloc(sizeof(created_edge<ET>), EDGE_BUFFER_CHUNKSIZE));
            }
            bufs[bufidx][idx % EDGE_BUFFER_CHUNKSIZE] = cedge;
        }
        
    private:
        // Disable value copying
        edge_buffer_flat(const edge_buffer_flat&);
        edge_buffer_flat& operator=(const edge_buffer_flat&);
    };


};

#endif

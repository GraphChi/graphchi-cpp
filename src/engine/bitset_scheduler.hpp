
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
 * Bitset scheduler.
 */

#ifndef DEF_GRAPHCHI_BITSETSCHEDULER
#define DEF_GRAPHCHI_BITSETSCHEDULER

#include "graphchi_types.hpp"
#include "api/ischeduler.hpp"
#include "util/dense_bitset.hpp"

namespace graphchi {
    
    class bitset_scheduler : public ischeduler {
    private:
        dense_bitset bitset;
    public:
        bool has_new_tasks;
        
        bitset_scheduler(int nvertices) : bitset(nvertices) {
        }
        
        virtual ~bitset_scheduler() {}
        
        inline void add_task(vid_t vertex) {
            bitset.set_bit(vertex);
            has_new_tasks = true;
        }
        
        void resize(vid_t maxsize) {
            bitset.resize(maxsize);
        }
        
        inline bool is_scheduled(vid_t vertex) {
            return bitset.get(vertex);
        }
        
        inline void remove_task(vid_t vertex) {
            bitset.clear_bit(vertex);
        }
        
        void remove_tasks(vid_t fromvertex, vid_t tovertex) {
            bitset.clear_bits(fromvertex, tovertex);
        }
        
        void add_task_to_all() {
            has_new_tasks = true;
            bitset.setall();
        }
    };
    
}


#endif


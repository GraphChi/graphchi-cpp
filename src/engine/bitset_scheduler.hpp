
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
        dense_bitset * curiteration_bitset;
        dense_bitset * nextiteration_bitset;
    public:
        bool has_new_tasks;
        
        bitset_scheduler(int nvertices) {
            curiteration_bitset = new dense_bitset(nvertices);
            nextiteration_bitset = new dense_bitset(nvertices);
        }
        
        void new_iteration(int iteration) {
            if (iteration > 0) {
                // Swap
                dense_bitset * tmp = curiteration_bitset;
                curiteration_bitset = nextiteration_bitset;
                nextiteration_bitset = tmp;
                nextiteration_bitset->clear();
            }   
            
        }
        
        virtual ~bitset_scheduler() {
            delete nextiteration_bitset;
            delete curiteration_bitset;
        }
        
        inline void add_task(vid_t vertex, bool also_this_iteration=false) {
            nextiteration_bitset->set_bit(vertex);
            if (also_this_iteration) {
                // If possible, add to schedule already this iteration
                curiteration_bitset->set_bit(vertex);
            }
            has_new_tasks = true;
        }
        
        void resize(vid_t maxsize) {
            curiteration_bitset->resize(maxsize);
            nextiteration_bitset->resize(maxsize);
            
        }
        
        inline bool is_scheduled(vid_t vertex) {
            return curiteration_bitset->get(vertex);
        }
        
        void remove_tasks(vid_t fromvertex, vid_t tovertex) {
            nextiteration_bitset->clear_bits(fromvertex, tovertex);
        }
        
        
        
        void add_task_to_all() {
            has_new_tasks = true;
            curiteration_bitset->setall();
        }
        
        size_t num_tasks() { 
            size_t n = 0;
            for(vid_t i=0; i < curiteration_bitset->size(); i++) {
                n += curiteration_bitset->get(i);
            }
            return n;
        }
        
    };
    
}


#endif


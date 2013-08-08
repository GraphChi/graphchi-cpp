

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
 * GraphChiProgram must be subclassed by GraphChi user programs.
 * They can define an update function (run for each vertex), and
 * call backs for iteration and interval beginning and ending.
 */

#ifndef GRAPHCHI_PROGRAM_DEF
#define GRAPHCHI_PROGRAM_DEF

#include "api/graph_objects.hpp"
#include "api/graphchi_context.hpp"

namespace graphchi {
    
    template <typename VertexDataType_, typename EdgeDataType_,
                typename vertex_t = graphchi_vertex<VertexDataType_, EdgeDataType_> >
    class GraphChiProgram {
        
    public:
        typedef VertexDataType_ VertexDataType;
        typedef EdgeDataType_ EdgeDataType;
        
        virtual ~GraphChiProgram() {}
        
        /**
         * Called before an iteration starts.
         */
        virtual void before_iteration(int iteration, graphchi_context &gcontext) {
        }
        
        /**
         * Called after an iteration has finished.
         */
        virtual void after_iteration(int iteration, graphchi_context &gcontext) {
        }
        
        /**
         * Support for the new "rinse" method. An app can ask the vertices currently in
         * memory be updated again before moving to new interval or iteration.
         */
        virtual bool repeat_updates(graphchi_context &gcontext) {
            return false;
        }
        
        
        
        /**
         * Called before an execution interval is started.
         */
        virtual void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
        }
        
        /**
         * Called after an execution interval has finished.
         */
        virtual void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
        }
        
        /**
         * Update function.
         */
        virtual void update(vertex_t &v, graphchi_context &gcontext)  = 0;    
    };

}

#endif


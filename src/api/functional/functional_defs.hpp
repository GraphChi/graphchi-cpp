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
 * Functional API defs.
 */
#ifndef GRAPHCHI_FUNCTIONALDEFS_DEF
#define GRAPHCHI_FUNCTIONALDEFS_DEF

#include "api/graphchi_program.hpp"
#include <vector>
#include "util/pthread_tools.hpp"

namespace graphchi {
    
    struct vertex_info {
        vid_t vertexid;
        int indegree;
        int outdegree;
    };
    
    /* Special sparse locking */
    mutex & get_lock(vid_t vertexid) {
        static std::vector<mutex> locks(1024);
        return locks[vertexid % 1024];
    }

};


#endif

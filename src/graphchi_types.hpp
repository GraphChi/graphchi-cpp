
/*
 Copyright [2012] [Aapo Kyrola, Guy Blelloch, Carlos Guestrin / Carnegie Mellon University]
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#ifndef DEF_GRAPHCHI_TYPES
#define DEF_GRAPHCHI_TYPES


#include <stdint.h>

namespace graphchi {
    
    typedef uint32_t vid_t;
    
    
    /** 
      * PairContainer encapsulates a pair of values of some type.
      * Useful for bulk-synchronuos computation.
      */
    template <typename ET>
    struct PairContainer {
        ET left;
        ET right;
        
        PairContainer() {
            left = ET();
            right = ET();
        }
        
        PairContainer(ET a, ET b) {
            left = a;
            right = b;
        }
        
        ET & oldval(int iter) {
            return (iter % 2 == 0 ? left : right);
        }
        
        void set_newval(int iter, ET x) {
            if (iter % 2 == 0) {
                right = x;
            } else {
                left = x;
            }
        }
    };
    
    struct shard_index {
        vid_t vertexid;
        size_t filepos;
        size_t edgecounter;
        shard_index(vid_t vertexid, size_t filepos, size_t edgecounter) : vertexid(vertexid), filepos(filepos), edgecounter(edgecounter) {}
    };
    

}


#endif




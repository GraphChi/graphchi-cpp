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
 * Variable size typed vector (type must be a plain old datatype) that
 * allows adding and removing of elements. 
 */


#ifndef DEF_GRAPHCHI_CHIVECTOR
#define DEF_GRAPHCHI_CHIVECTOR

#include <vector>
#include <stdint.h>

namespace graphchi {

    
template <typename T>
class chivector {
    
    uint16_t * sizeptr;
    uint16_t origsize;
    T * data;
    std::vector<T> * extensions;  // TODO: use a more memory efficient system?
    
public:
    chivector() {}
    
    chivector(uint16_t * sizeptr, T * dataptr) : sizeptr(sizeptr), data(dataptr) {
        origsize = *sizeptr;
        extensions = NULL;
    }
    
    void write(T * dest) {
        int sz = (int) this->size();
        for(int i=0; i < sz; i++) {
            dest[i] = get(i);  // TODO: use memcpy
        }
    }
    
public:
    uint16_t size() {
        return *sizeptr;
    }
    
    void add(T val) {
        *sizeptr += 1;
        if (*sizeptr > origsize) {
            if (extensions == NULL) extensions = new std::vector<T>();
            extensions->push_back(val);
        } else {
            data[*sizeptr - 1] = val;
        }
    }
    
    T get(int idx) {
        if (idx >= origsize) {
            return (* extensions)[idx - (int)origsize];
        } else {
            return data[idx];
        }
    }
    
    void remove(int idx) {
        assert(false);
    }
    
    int find(T val) {
        assert(false);
        return -1;
    }
    
    void clear() {
        *sizeptr = 0;
    }
    
    // TODO: iterators

    
    
      
};
    
}

#endif

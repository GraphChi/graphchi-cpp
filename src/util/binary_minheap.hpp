
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
 * Capacity bounded minheap.
 */

#ifndef DEF_MINHEAP_GRAPHCHI
#define DEF_MINHEAP_GRAPHCHI

#include <assert.h>
#include <stdlib.h>


template <typename T>
class binary_minheap {
    T * values;
    int capacity;
    int sz;
public:
    binary_minheap(int capacity) :  capacity(capacity), sz(0) {
        values = (T*) calloc(capacity, sizeof(T));
    } 
    
    ~binary_minheap() {
        delete values;
    }
    
    inline int parent(int i) { return (i+1)/2-1; }
    inline int left(int i)   { return i*2 + 1; }
    inline int right(int i)  { return i*2 + 2; }
    inline void incrHeapSize() { sz++; assert(sz <= capacity); }   
    inline void decrHeapSize() { sz--; assert(sz >= 0); }   
    
    void insert(T element) {
        incrHeapSize();
        // percolate up
        int pos = sz - 1;   // http://www.cs.cmu.edu/~adamchik/15-121/lectures/Binary%20Heaps/heaps.html
        for(; pos > 0 && element < values[parent(pos)];  pos=parent(pos)) 
            values[pos] = values[parent(pos)];
        values[pos] = element;
    }   
    
    bool empty() { return sz == 0; }
    
    void minHeapify(int i) {
        int l = left(i);
        int r = right(i);
        int smallest;
        if (l < sz &&  values[l] < values[i]) smallest = l;
        else smallest = i;
        if (r < sz &&  values[r] < values[smallest]) smallest = r;
        if (smallest != i) {
            // exchange
            T tmp = values[i];
            values[i] = values[smallest];
            values[smallest] = tmp;
            minHeapify(smallest);
        }
    }
    
    T min() { return values[0]; }
    void extractMin() {
        values[0] = values[sz - 1];
        decrHeapSize();
        minHeapify(0);
    }
    
};

#endif
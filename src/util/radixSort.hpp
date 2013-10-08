// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _S_RADIX_INCLUDED
#define _S_RADIX_INCLUDED

#include <iostream>
#include <algorithm>
#include <math.h>
#include "graphchi_types.hpp"





namespace graphchi {
    
     
    
    // Cannot be greater than 8 without changing definition of bIndexT
    //    from unsigned char to unsigned int (or unsigned short)
#define MAX_RADIX 8
#define BUCKETS (1 << MAX_RADIX)
    
    // a type that must hold MAX_RADIX bits
    typedef unsigned char bIndexT;
    typedef long long int intT;
    
    // input in A, output in B
    template <class E, class F>
    void radixStep(E* A, E* B, bIndexT *Tmp, intT* counts,
                   intT n, intT m, F extract) {
        for (intT i = 0; i < m; i++)  counts[i] = 0;
        for (intT j = 0; j < n; j++) {
            intT k = Tmp[j] = extract(A[j]);
            counts[k]++;
        }
        intT s = 0;
        for (intT i = 0; i < m; i++) {
            s += counts[i];
            counts[i] = s;
        }
        for (intT j = n-1; j >= 0; j--) {
            intT x =  --counts[Tmp[j]];
            B[x] = A[j];
        }
    }
    
    // a function to extract "bits" bits starting at bit location "offset"
    template <class E, class F>
    struct eBits {
         intT _mask;  intT _offset; F _f; 
        eBits(int bits, intT offset, F f): _mask((1<<bits)-1),
        _offset(offset), _f(f) {}
        intT operator() (E p) {return _mask&(_f(p)>>_offset);}
    };
    
    template <class T>
    intT log2Up(T i) {
        intT a=0;
        while ((1L << a) <= i) a++;
        return a;
    }
    
    // Radix sort with low order bits first
    template <class E, class F>
    void iSort(E *A, intT n, intT m, F f) {
        intT bits = log2Up(m);
        
        // temporary space
        E* B = (E*) malloc(sizeof(E)*n);
        bIndexT* Tmp = (bIndexT*) malloc(sizeof(bIndexT)*n);
        intT* counts = (intT*) malloc(sizeof(intT)*BUCKETS);
        
        intT rounds = 1+(bits-1)/MAX_RADIX;
        intT rbits = 1+(bits-1)/rounds;
        intT bitOffset = 0;
        bool flipped = 0;
        
        while (bitOffset < bits) {
            if (bitOffset+rbits > bits) rbits = bits-bitOffset;
            if (flipped)
                radixStep(B, A, Tmp, counts, n, 1L << rbits,
                          eBits<E,F>(rbits,bitOffset,f));
            else
                radixStep(A, B, Tmp, counts, n, 1L << rbits,
                          eBits<E,F>(rbits,bitOffset,f));
            bitOffset += rbits;
            flipped = !flipped;
        }
        
        if (flipped)
            for (intT i=0; i < n; i++)
                A[i] = B[i];
        
        free(B); free(Tmp); free(counts);
    }
}


#endif // _S_RADIX_INCLUDED

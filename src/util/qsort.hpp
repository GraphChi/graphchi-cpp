// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2010 Guy Blelloch and Harsha Vardhan Simhadri and the PBBS team
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

 
#ifndef GRAPHCHI_QSORT_INCLUDED
#define GRAPHCHI_QSORT_INCLUDED

#include <algorithm>
#include <vector>

template <class E, class BinPred>
void insertionSort(E* A, int n, BinPred f) {
    for (int i=0; i < n; i++) {
        E v = A[i];
        E* B = A + i;
        while (--B >= A && f(v,*B)) *(B+1) = *B;
        *(B+1) = v;
    }
}


#define ISORT 25

template <class E, class BinPred>
E median(E a, E b, E c, BinPred f) {
    return  f(a,b) ? (f(b,c) ? b : (f(a,c) ? c : a)) 
    : (f(a,c) ? a : (f(b,c) ? c : b));
}

 // Partly copied from PBBS
template <class E, class BinPred>
void quickSort(E* A, int n, BinPred f) {
    if (n < ISORT) insertionSort(A, n, f);
    else {
        E p = A[rand() % n]; // Random pivot
        E* L = A;   // below L are less than pivot
        E* M = A;   // between L and M are equal to pivot
        E* R = A+n-1; // above R are greater than pivot
        while (1) {
            while (!f(p,*M)) {
                if (f(*M,p)) std::swap(*M,*(L++));
                if (M >= R) break; 
                M++;
            }
            while (f(p,*R)) R--;
            if (M >= R) break; 
            std::swap(*M,*R--); 
            if (f(*M,p)) std::swap(*M,*(L++));
            M++;
        }
        quickSort(A, (int) (L-A), f);
        quickSort(M, (int) (A+n-M), f); // Exclude all elts that equal pivot
    }
} 


#endif  



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


#ifndef DEF_MERGE
#define DEF_MERGE

template <class ET, class F> 
void merge(ET* S1, int l1, ET* S2, int l2, ET* R, F f) {
    ET* pR = R; 
    ET* pS1 = S1; 
    ET* pS2 = S2;
    ET* eS1 = S1+l1; 
    ET* eS2 = S2+l2;
    while (true) {
        *pR++ = f(*pS2,*pS1) ? *pS2++ : *pS1++;
        if (pS1==eS1) {std::copy(pS2,eS2,pR); break;}
        if (pS2==eS2) {std::copy(pS1,eS1,pR); break;}
    }
}


#endif


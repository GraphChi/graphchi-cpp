


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
 * Context object which contains information about the graph
 * and on-going computation.
 */

#ifndef DEF_GRAPHCHI_CONTEXT
#define DEF_GRAPHCHI_CONTEXT

#include <vector>
#include <assert.h>
#include <omp.h>
#include <sys/time.h>

#include "graphchi_types.hpp"
#include "api/ischeduler.hpp"

namespace graphchi {
    
    struct graphchi_context {

        size_t nvertices;
        size_t nedges;
        
        ischeduler * scheduler;
        int iteration;
        int num_iterations;
        int last_iteration;
        int execthreads;
        std::vector<double> deltas;
        timeval start;
        std::string filename;
        double last_deltasum;
        
        graphchi_context() : scheduler(NULL), iteration(0), last_iteration(-1) {
            gettimeofday(&start, NULL);
            last_deltasum = 0.0;
        }
        
        double runtime() {
            timeval end;
            gettimeofday(&end, NULL);
            return end.tv_sec-start.tv_sec+ ((double)(end.tv_usec-start.tv_usec))/1.0E6;
        }
        
        /** 
          * Set a termination iteration.
          */
        void set_last_iteration(int _last_iteration) {
            last_iteration = _last_iteration;
        }
        
        void reset_deltas(int nthreads) {
            deltas = std::vector<double>(nthreads, 0.0);
        }
        
        double get_delta() {
            double d = 0.0;
            for(int i=0; i < (int)deltas.size(); i++) {
                d += deltas[i];
            }
            last_deltasum = d;
            return d;
        }
        
        inline bool isnan(double x) {
            return !(x<0 || x>=0);
        }
        
        /**
          * Method for keeping track of the amount of change in computation.
          * An update function may broadcast a numerical "delta" value that is 
          * automatically accumulated (in thread-safe way).
          * @param delta
          */
        void log_change(double delta) {
            deltas[omp_get_thread_num()] += delta;  
            assert(delta >= 0);
            assert(!isnan(delta)); /* Sanity check */
        }
    };
    
}


#endif


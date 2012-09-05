

/**
 * @file
 * @author  Danny Bickson, Based on Code of Aapo Kyrola
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
 * Common code for SGD implementations.
 */



#ifndef DEF_SGDHPP
#define DEF_SGDHPP

#include <assert.h>
#include <cmath>
#include <errno.h>
#include <string>
#include <stdint.h>

#include "../../example_apps/matrix_factorization/matrixmarket/mmio.h"
#include "../../example_apps/matrix_factorization/matrixmarket/mmio.c"

#include "api/chifilenames.hpp"
#include "api/vertex_aggregator.hpp"
#include "preprocessing/sharder.hpp"

#include "eigen_wrapper.hpp"
using namespace graphchi;


#ifndef NLATENT
#define NLATENT 20   // Dimension of the latent factors. You can specify this in compile time as well (in make).
#endif

double sgd_lambda = 1e-3;
double sgd_gamma = 1e-3;
double sgd_step_dec = 0.9;
double minval = -1e100;
double maxval = 1e100;
std::string training;
std::string validation;
std::string test;
uint M, N, Me, Ne, Le, K;
size_t L;
double globalMean = 0;

/// RMSE computation
double rmse=0.0;


// Hackish: we need to count the number of left
// and right vertices in the bipartite graph ourselves.
vid_t max_left_vertex =0 ;
vid_t max_right_vertex = 0;

struct vertex_data {
    double pvec[NLATENT];
    double rmse;
    double bias;
 
    vertex_data() {
        for(int k=0; k < NLATENT; k++) 
           pvec[k] =  drand48(); 
        rmse = 0;
        bias = 0;
    }
    
    double dot(const vertex_data &oth) const {
        double x=0;
        for(int i=0; i<NLATENT; i++) x+= oth.pvec[i]*pvec[i];
        return x;
    }
    
};

#endif



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

double sgd_lambda = 1e-3; //sgd step size
double sgd_gamma = 1e-3;  //sgd regularization
double sgd_step_dec = 0.9; //sgd step decrement
double minval = -1e100;    //min allowed rating
double maxval = 1e100;     //max allowed rating
std::string training;      //training input file
std::string validation;    //validation input file
std::string test;          //test input file
uint M;                    //number of users
uint N;                    //number of items
uint Me;                   //number of users (validation file)      
uint Ne;                   //number of items (validation file)
uint Le;                   //number of ratings (validation file)
uint K;                    //unused
size_t L;                  //number of ratings (training file)
double globalMean = 0;     //global mean rating - unused
double rmse=0.0;           //current error

struct vertex_data {
    double pvec[NLATENT]; //storing the feature vector
    double rmse;          //tracking rmse
    double bias;
 
    vertex_data() {
        for(int k=0; k < NLATENT; k++) 
           pvec[k] =  drand48(); 
        rmse = 0;
        bias = 0;
    }
   
    //dot product 
    double dot(const vertex_data &oth) const {
        double x=0;
        for(int i=0; i<NLATENT; i++) x+= oth.pvec[i]*pvec[i];
        return x;
    }
    
};

#endif

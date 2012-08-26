

/**
 * @file
 * @author  Danny Bickson
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
 * Common code for SVDPP implementations.
 */



#ifndef DEF_SVDPPHPP
#define DEF_SVDPPHPP

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



struct svdpp_params{
  float itmBiasStep;
  float itmBiasReg;
  float usrBiasStep;
  float usrBiasReg;
  float usrFctrStep;
  float usrFctrReg;
  float itmFctrStep;
  float itmFctrReg; //gamma7
  float itmFctr2Step;
  float itmFctr2Reg;
  float step_dec;

  svdpp_params(){
    itmBiasStep = 1e-4f;
    itmBiasReg = 1e-4f;
    usrBiasStep = 1e-4f;
    usrBiasReg = 2e-4f;
    usrFctrStep = 1e-4f;
    usrFctrReg = 2e-4f;
    itmFctrStep = 1e-4f;
    itmFctrReg = 1e-4f; //gamma7
    itmFctr2Step = 1e-4f;
    itmFctr2Reg = 1e-4f;
    step_dec = 0.9;
  }
};

svdpp_params svdpp;

double minval = -1e100;
double maxval = 1e100;
std::string training;
std::string validation;
std::string test;
int M, N, Me, Ne, Le;
size_t L;


/// RMSE computation
double rmse=0.0;
double globalMean = 0;

// Hackish: we need to count the number of left
// and right vertices in the bipartite graph ourselves.
uint max_left_vertex =0 ;
uint max_right_vertex = 0;

struct vertex_data {
  double pvec[NLATENT];
  double weight[NLATENT];
  double rmse;
  double bias;

  vertex_data() {
    for(int k=0; k < NLATENT; k++) {
      pvec[k] =  drand48(); 
      weight[k] = drand48();
    }
    rmse = 0;
    bias = 0;
  }

};





#include "io.hpp"

#endif

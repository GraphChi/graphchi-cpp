

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

// See note above about Eigen
#include "Eigen/Dense"
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include "Eigen/Sparse"
#include "Eigen/Cholesky"
#include "Eigen/Eigenvalues"
#include "Eigen/SVD"
using namespace Eigen;
  

typedef MatrixXd mat;
typedef VectorXd vec;
typedef VectorXi ivec;
typedef MatrixXi imat;
typedef SparseVector<double> sparse_vec;

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
int M, N, L, Me, Ne, Le;
double globalMean = 0;

/// RMSE computation
double rmse=0.0;


// Hackish: we need to count the number of left
// and right vertices in the bipartite graph ourselves.
vid_t max_left_vertex =0 ;
vid_t max_right_vertex = 0;

struct vertex_data {
    double d[NLATENT];
    double rmse;
    
    vertex_data() {
        for(int k=0; k < NLATENT; k++) 
           d[k] =  drand48(); 
        rmse = 0;
    }
    
    double dot(vertex_data &oth) const {
        double x=0;
        for(int i=0; i<NLATENT; i++) x+= oth.d[i]*d[i];
        return x;
    }
    
};


#include "io.hpp"




#endif

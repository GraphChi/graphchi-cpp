

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
 * Common code for ALS implementations.
 */



#ifndef DEF_ALSHPP
#define DEF_ALSHPP

#include <assert.h>
#include <cmath>
#include <errno.h>
#include <string>
#include <stdint.h>

#include "matrixmarket/mmio.h"
#include "matrixmarket/mmio.c"

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
#define NLATENT 5   // Dimension of the latent factors. You can specify this in compile time as well (in make).
#endif

double LAMBDA = 0.065;

/// RMSE computation
double rmse=0.0;
mutex rmselock;


// Hackish: we need to count the number of left
// and right vertices in the bipartite graph ourselves.
vid_t max_left_vertex =0 ;
vid_t max_right_vertex = 0;

struct latentvec_t {
    double d[NLATENT];
    
    latentvec_t() {
    }
    
    void init() {
        for(int k=0; k < NLATENT; k++) d[k] =  0.001 * (std::rand() % 1000); 
    }
    
    double & operator[] (int idx) {
        return d[idx];
    }
    bool operator!=(const latentvec_t &oth) const {
        for(int i=0; i<NLATENT; i++) { if (d[i] != oth.d[i]) return true; }
        return false;
    }
    
    double dot(latentvec_t &oth) const {
        double x=0;
        for(int i=0; i<NLATENT; i++) x+= oth.d[i]*d[i];
        return x;
    }
    
};



struct als_factor_and_weight {
    latentvec_t factor;
    float weight;
    
    als_factor_and_weight() {}
    
    als_factor_and_weight(float obs) {
        weight = obs;
        factor.init();
    }
};


 
 /**
 * Create a bipartite graph from a matrix. Each row corresponds to vertex
 * with the same id as the row number (0-based), but vertices correponsing to columns
 * have id + num-rows.
 */
template <typename als_edge_type>
int convert_matrixmarket_for_ALS(std::string base_filename) {
    // Note, code based on: http://math.nist.gov/MatrixMarket/mmio/c/example_read.c
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    uint M, N;
    size_t nz;   
    
    /**
     * Create sharder object
     */
    int nshards;
    if ((nshards = find_shards<als_edge_type>(base_filename, get_option_string("nshards", "auto")))) {
        if (check_origfile_modification_earlier<als_edge_type>(base_filename, nshards)) {
            logstream(LOG_INFO) << "File " << base_filename << " was already preprocessed, won't do it again. " << std::endl;
            logstream(LOG_INFO) << "If this is not intended, please delete the shard files and try again. " << std::endl;
            return nshards;
        }
    }
    
    sharder<als_edge_type> sharderobj(base_filename);
    sharderobj.start_preprocessing();
    
    
    if ((f = fopen(base_filename.c_str(), "r")) == NULL) {
        logstream(LOG_ERROR) << "Could not open file: " << base_filename << ", error: " << strerror(errno) << std::endl;
        exit(1);
    }
    
    
    if (mm_read_banner(f, &matcode) != 0)
    {
        logstream(LOG_ERROR) << "Could not process Matrix Market banner. File: " << base_filename << std::endl;
        logstream(LOG_ERROR) << "Matrix must be in the Matrix Market format. " << std::endl;
        exit(1);
    }
    
    
    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */
    
    if (mm_is_complex(matcode) || !mm_is_sparse(matcode))
    {
        logstream(LOG_ERROR) << "Sorry, this application does not support complex values and requires a sparse matrix." << std::endl;
        logstream(LOG_ERROR) << "Market Market type: " << mm_typecode_to_str(matcode) << std::endl;
        exit(1);
    }
    
    /* find out size of sparse matrix .... */
    
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0) {
        logstream(LOG_ERROR) << "Failed reading matrix size: error=" << ret_code << std::endl;
        exit(1);
    }
    
    
    logstream(LOG_INFO) << "Starting to read matrix-market input. Matrix dimensions: " 
    << M << " x " << N << ", non-zeros: " << nz << std::endl;
    
    if (M < 5 || N < 5 || nz < 10) {
        logstream(LOG_ERROR) << "File is suspiciously small. Something wrong? File: " << base_filename << std::endl;
        assert(M < 5 || N < 5 || nz < 10);
    }   


    for (size_t i=0; i<nz; i++)
    {
        uint I, J;
        double val;
        int rc = fscanf(f, "%u %u %lg\n", &I, &J, &val);
        if (rc != 3)
          logstream(LOG_FATAL)<<"Error reading line: " << i << std::endl;
        I--;  /* adjust from 1-based to 0-based */
        J--;
        
        sharderobj.preprocessing_add_edge(I, M + J, als_edge_type((float)val));
    }
    sharderobj.end_preprocessing();
    
    if (f !=stdin) fclose(f);
    
    
    logstream(LOG_INFO) << "Now creating shards." << std::endl;
    
    // Shard with a specified number of shards, or determine automatically if not defined
    nshards = sharderobj.execute_sharding(get_option_string("nshards", "auto"));
    
    return nshards;
}

struct  MMOutputter : public VCallback<latentvec_t> {
    FILE * outf;
    MMOutputter(std::string fname, vid_t nvertices)  {
        MM_typecode matcode;
        mm_initialize_typecode(&matcode);
        mm_set_matrix(&matcode);
        mm_set_array(&matcode);
        mm_set_real(&matcode);
        
        outf = fopen(fname.c_str(), "w");
        assert(outf != NULL);
        mm_write_banner(outf, matcode);
        mm_write_mtx_array_size(outf, nvertices, NLATENT); 
    }
    
    void callback(vid_t vertex_id, latentvec_t &vec) {
        for(int i=0; i < NLATENT; i++) {
            fprintf(outf, "%lf\n", vec.d[i]);
        }
    }
    
    ~MMOutputter() {
        if (outf != NULL) fclose(outf);
    }
    
};

void output_als_result(std::string filename, vid_t numvertices, vid_t max_left_vertex) {
    MMOutputter mmoutput_left(filename + "_U.mm", max_left_vertex + 1);
    foreach_vertices<latentvec_t>(filename, 0, max_left_vertex + 1, mmoutput_left);
    
    
    MMOutputter mmoutput_right(filename + "_V.mm", numvertices - max_left_vertex - 1);
    foreach_vertices<latentvec_t>(filename, max_left_vertex + 1, numvertices-1, mmoutput_right);
    logstream(LOG_INFO) << "ALS output files (in matrix market format): " << filename + "_U.mm" <<
    ", " << filename + "_V.mm" << std::endl;
}



#endif

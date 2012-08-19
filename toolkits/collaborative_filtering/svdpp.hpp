

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
 * Common code for SVDPPPP implementations.
 */



#ifndef DEF_SVDPPPPHPP
#define DEF_SVDPPPPHPP

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
  itmBiasStep = 1e-1f;
  itmBiasReg = 1e-3f;
  usrBiasStep = 1e-1f;
  usrBiasReg = 5e-3f;
  usrFctrStep = 1e-1f;
  usrFctrReg = 2e-2f;
  itmFctrStep = 1e-1f;
  itmFctrReg = 1e-2f; //gamma7
  itmFctr2Step = 1e-1f;
  itmFctr2Reg = 1e-3f;
  step_dec = 0.9;
 }
};

svdpp_params svdpp;

double minval = -1e100;
double maxval = 1e100;
std::string training;
std::string validation;
std::string test;
int M, N;


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
    }
    
    void init() {
        for(int k=0; k < NLATENT; k++) {
           pvec[k] =  drand48(); 
           weight[k] = drand48();
        }
        rmse = 0;
        bias = 0;
    }
    
    double & operator[] (int idx) {
        return pvec[idx];
    }
    
    double dot(vertex_data &oth) const {
        double x=0;
        for(int i=0; i<NLATENT; i++) x+= oth.pvec[i]*pvec[i];
        return x;
    }
    
};



struct svdpp_factor_and_weight {
    vertex_data factor;
    float weight;
    
    svdpp_factor_and_weight() {}
    
    svdpp_factor_and_weight(float obs) {
        weight = obs;
        factor.init();
    }
};


 
 /**
 * Create a bipartite graph from a matrix. Each row corresponds to vertex
 * with the same id as the row number (0-based), but vertices correponsing to columns
 * have id + num-rows.
 */
template <typename svdpp_edge_type>
int convert_matrixmarket_for_SVDPP(std::string base_filename) {
    // Note, code based on: http://math.nist.gov/MatrixMarket/mmio/c/example_read.c
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int nz;   
    
    /**
     * Create sharder object
     */
    int nshards;
    if ((nshards = find_shards<svdpp_edge_type>(base_filename, get_option_string("nshards", "auto")))) {
        logstream(LOG_INFO) << "File " << base_filename << " was already preprocessed, won't do it again. " << std::endl;
        logstream(LOG_INFO) << "If this is not intended, please delete the shard files and try again. " << std::endl;
        return nshards;
    }   
    
    sharder<svdpp_edge_type> sharderobj(base_filename);
    sharderobj.start_preprocessing();
    
    if ((f = fopen(base_filename.c_str(), "r")) == NULL) {
        logstream(LOG_FATAL) << "Could not open file: " << base_filename << ", error: " << strerror(errno) << std::endl;
    }
    
    
    if (mm_read_banner(f, &matcode) != 0)
        logstream(LOG_FATAL) << "Could not process Matrix Market banner. File: " << base_filename << std::endl;
    
    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */
    
    if (mm_is_complex(matcode) || !mm_is_sparse(matcode))
        logstream(LOG_FATAL) << "Sorry, this application does not support complex values and requires a sparse matrix." << std::endl;
    
    /* find out size of sparse matrix .... */
    
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0) {
        logstream(LOG_FATAL) << "Failed reading matrix size: error=" << ret_code << std::endl;
    }
    
    
    logstream(LOG_INFO) << "Starting to read matrix-market input. Matrix dimensions: " 
    << M << " x " << N << ", non-zeros: " << nz << std::endl;
    
    
    if (!sharderobj.preprocessed_file_exists()) {
        for (int i=0; i<nz; i++)
        {
            int I, J;
            double val;
            int rc = fscanf(f, "%d %d %lg\n", &I, &J, &val);
            if (rc != 3)
              logstream(LOG_FATAL)<<"Error processing input file " << base_filename << " at data row " << i <<std::endl;
            I--;  /* adjust from 1-based to 0-based */
            J--;
            globalMean += val; 
            sharderobj.preprocessing_add_edge(I, M + J, svdpp_edge_type((float)val));
        }
        sharderobj.end_preprocessing();
        
    } else {
        logstream(LOG_INFO) << "Matrix already preprocessed, just run sharder." << std::endl;
    }
    
    fclose(f);
    globalMean /= nz;
    
    logstream(LOG_INFO) << "Global mean is: " << globalMean << " Now creating shards." << std::endl;
    
    // Shard with a specified number of shards, or determine automatically if not defined
    nshards = sharderobj.execute_sharding(get_option_string("nshards", "auto"));
    
    return nshards;
}

struct  MMOutputter : public VCallback<vertex_data> {
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
    
    void callback(vid_t vertex_id, vertex_data &vec) {
        for(int i=0; i < NLATENT; i++) {
            fprintf(outf, "%lf\n", vec.pvec[i]);
        }
    }
    
    ~MMOutputter() {
        if (outf != NULL) fclose(outf);
    }
    
};

void output_svdpp_result(std::string filename, vid_t numvertices, vid_t max_left_vertex) {
    MMOutputter mmoutput_left(filename + "_U.mm", max_left_vertex + 1);
    foreach_vertices<vertex_data>(filename, 0, max_left_vertex + 1, mmoutput_left);
    
    
    MMOutputter mmoutput_right(filename + "_V.mm", numvertices - max_left_vertex - 2);
    foreach_vertices<vertex_data>(filename, max_left_vertex + 1, numvertices-1, mmoutput_right);
    logstream(LOG_INFO) << "SVDPPPP output files (in matrix market format): " << filename + "_U.mm" <<
    ", " << filename + "_V.mm" << std::endl;
}



#endif

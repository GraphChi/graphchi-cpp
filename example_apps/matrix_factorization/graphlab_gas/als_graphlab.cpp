

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
 * Program for running ALS-matrix factorizatino toolkit from
 * GraphLab. This is an example of GraphLab v2.1 programs development
 * for GraphChi.
 */

#include <string>
#include <algorithm>
#include <stdint.h>
#include <memory.h>
#include <cstdlib>

#include "graphchi_basic_includes.hpp"

#include "../matrixmarket/mmio.h"
#include "../matrixmarket/mmio.c"
#include "api/graphlab2_1_GAS_api/graphlab.hpp"

#include "als_vertex_program.hpp"

using namespace graphchi;
using namespace graphlab;

// Forward declaration
int convert_matrixmarket_for_ALS_graphlab(std::string filename);

size_t vertex_data::NLATENT = 5;
double als_vertex_program::TOLERANCE = 1e-3;
double als_vertex_program::LAMBDA = 0.01;
size_t als_vertex_program::MAX_UPDATES = -1;

int main(int argc, const char ** argv) {
    /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
    graphchi_init(argc, argv);
    
    /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
    metrics m("als-graphlab");
    

    /* Basic arguments for application. NOTE: File will be automatically 'sharded'. */
    std::string filename = get_option_string("file");    // Base filename
    int niters           = get_option_int("niters", 4);  // Number of iterations
    
    /* Preprocess data if needed, or discover preprocess files */
    int nshards = convert_matrixmarket_for_ALS_graphlab(filename);
    
    /* Run */
    std::vector<vertex_data> * vertices =
        run_graphlab_vertexprogram<als_vertex_program>(filename, nshards, niters, false, m, false, false);
    
    /* Error aggregation */
    error_aggregator final_error = 
            run_graphlab_edge_aggregator<als_vertex_program, error_aggregator>(filename, nshards,    
                                                        error_aggregator::map, error_aggregator::finalize, vertices, m);
    
    std::cout << "Final train error: " << final_error.train_error << std::endl; 
  
    /* TODO: write output latent matrices */
    delete vertices;
    /* Report execution metrics */
    metrics_report(m);
    return 0;
}


/**
 * Create a bipartite graph from a matrix. Each row corresponds to vertex
 * with the same id as the row number (0-based), but vertices correponsing to columns
 * have id + num-rows.
 */
int convert_matrixmarket_for_ALS_graphlab(std::string base_filename) {
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
    if ((nshards = find_shards<edge_data>(base_filename, get_option_string("nshards", "auto")))) {
        logstream(LOG_INFO) << "File " << base_filename << " was already preprocessed, won't do it again. " << std::endl;
        logstream(LOG_INFO) << "If this is not intended, please delete the shard files and try again. " << std::endl;
        return nshards;
    }   
    
    sharder<edge_data> sharderobj(base_filename);
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
    

    for (int i=0; i<nz; i++)
    {
        int I, J;
        double val;
        fscanf(f, "%d %d %lg\n", &I, &J, &val);
        I--;  /* adjust from 1-based to 0-based */
        J--;
         
        
        sharderobj.preprocessing_add_edge(I, M + J, edge_data(val, edge_data::TRAIN));
    }
    sharderobj.end_preprocessing();

    if (f !=stdin) fclose(f);
    
    
    logstream(LOG_INFO) << "Now creating shards." << std::endl;
    
    // Shard with a specified number of shards, or determine automatically if not defined
    nshards = sharderobj.execute_sharding(get_option_string("nshards", "auto"));
    
    return nshards;
}

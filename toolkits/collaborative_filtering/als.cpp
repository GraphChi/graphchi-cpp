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
 * Matrix factorizatino with the Alternative Least Squares (ALS) algorithm.
 * This code is based on GraphLab's implementation of ALS by Joey Gonzalez
 * and Danny Bickson (CMU). A good explanation of the algorithm is 
 * given in the following paper:
 *    Large-Scale Parallel Collaborative Filtering for the Netflix Prize
 *    Yunhong Zhou, Dennis Wilkinson, Robert Schreiber and Rong Pan
 *    http://www.springerlink.com/content/j1076u0h14586183/
 *
 * Faster version of ALS, which stores latent factors of vertices in-memory.
 * Thus, this version requires more memory. See the version "als_edgefactors"
 * for a low-memory implementation.
 *
 *
 * In the code, we use movie-rating terminology for clarity. This code has been
 * tested with the Netflix movie rating challenge, where the task is to predict
 * how user rates movies in range from 1 to 5.
 *
 * This code is has integrated preprocessing, 'sharding', so it is not necessary
 * to run sharder prior to running the matrix factorization algorithm. Input
 * data must be provided in the Matrix Market format (http://math.nist.gov/MatrixMarket/formats.html).
 *
 * ALS uses free linear algebra library 'Eigen'. See Readme_Eigen.txt for instructions
 * how to obtain it.
 *
 * At the end of the processing, the two latent factor matrices are written into files in 
 * the matrix market format. 
 *
 * @section USAGE
 *
 * bin/example_apps/matrix_factorization/als_edgefactors file <matrix-market-input> niters 5
 *
 * 
 */



#include <string>
#include <algorithm>

#include "graphchi_basic_includes.hpp"

/* ALS-related classes are contained in als.hpp */
#include "als.hpp"

using namespace graphchi;


/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef latentvec_t VertexDataType;
typedef float EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<latentvec_t> latent_factors_inmem;
 /**
  compute validation rmse
 */
void test_predictions() {
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int vM, vN, nz;   
    
    if ((f = fopen(test.c_str(), "r")) == NULL) {
       return; //missing validaiton data, nothing to compute
    }
    FILE * fout = fopen((test + ".predict").c_str(),"w");
    if (fout == NULL)
       logstream(LOG_FATAL)<<"Failed to open test prediction file for writing"<<std::endl;
    
    if (mm_read_banner(f, &matcode) != 0)
        logstream(LOG_FATAL) << "Could not process Matrix Market banner. File: " << test << std::endl;
    
    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */
    if (mm_is_complex(matcode) || !mm_is_sparse(matcode))
        logstream(LOG_FATAL) << "Sorry, this application does not support complex values and requires a sparse matrix." << std::endl;
    
    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &vM, &vN, &nz)) !=0) {
        logstream(LOG_FATAL) << "Failed reading matrix size: error=" << ret_code << std::endl;
    }

    if (vM != M || vN != N)
      logstream(LOG_FATAL)<<"Input size of test matrix must be identical to training matrix, namely " << M << "x" << N << std::endl;

    mm_write_banner(fout, matcode);
    mm_write_mtx_crd_size(fout ,M,N,nz); 
 
    for (int i=0; i<nz; i++)
    {
            int I, J;
            double val;
            int rc = fscanf(f, "%d %d %lg\n", &I, &J, &val);
            if (rc != 3)
              logstream(LOG_FATAL)<<"Error when reading input file: " << i << std::endl;
            I--;  /* adjust from 1-based to 0-based */
            J--;
    	    double prediction = latent_factors_inmem[I].dot(latent_factors_inmem[J]);        
            prediction = std::max(prediction, minval);
            prediction = std::min(prediction, maxval);
            fprintf(fout, "%d %d %12.8lg\n", I+1, J+1, prediction);
    }
    fclose(f);
    fclose(fout);

    logstream(LOG_INFO)<<"Finished writing " << nz << " predictions to file: " << test << ".predict" << std::endl;
}

 
 /**
  compute validation rmse
 */
void validation_rmse() {
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int vM, vN, nz;   
    
    if ((f = fopen(validation.c_str(), "r")) == NULL) {
       return; //missing validaiton data, nothing to compute
    }
    
    if (mm_read_banner(f, &matcode) != 0)
        logstream(LOG_FATAL) << "Could not process Matrix Market banner. File: " << validation << std::endl;
    
    
    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */
    
    if (mm_is_complex(matcode) || !mm_is_sparse(matcode))
        logstream(LOG_FATAL) << "Sorry, this application does not support complex values and requires a sparse matrix." << std::endl;
    
    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &vM, &vN, &nz)) !=0) {
        logstream(LOG_FATAL) << "Failed reading matrix size: error=" << ret_code << std::endl;
    }
    if (vM != M || vN != N)
      logstream(LOG_FATAL)<<"Input size of validation matrix must be identical to training matrix, namely " << M << "x" << N << std::endl;

    
    double validation_rmse = 0;   
 
    for (int i=0; i<nz; i++)
    {
            int I, J;
            double val;
            int rc = fscanf(f, "%d %d %lg\n", &I, &J, &val);
	   
            if (rc != 3)
              logstream(LOG_FATAL)<<"Error when reading input file: " << i << std::endl;
            if (val < minval || val > maxval)
              logstream(LOG_FATAL)<<"Value is out of range: " << val << " should be: " << minval << " to " << maxval << std::endl;
            I--;  /* adjust from 1-based to 0-based */
            J--;
            
    	    double prediction = latent_factors_inmem[I].dot(latent_factors_inmem[J]);        
            prediction = std::max(prediction, minval);
            prediction = std::min(prediction, maxval);
            validation_rmse += (prediction - val)*(prediction-val);
    }
    fclose(f);

    logstream(LOG_INFO)<<"Validation RMSE: " << sqrt(validation_rmse/pengine->num_edges())<< std::endl;
}


                                       
/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct ALSVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    
    // Helper
    virtual void set_latent_factor(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, latentvec_t &fact) {
        vertex.set_data(fact); // Note, also stored on disk. This is non-optimal...
        latent_factors_inmem[vertex.id()] = fact;
    }
    
    /**
     * Called before an iteration starts.
     */
    void before_iteration(int iteration, graphchi_context &gcontext) {
        if (iteration == 0) {
            latent_factors_inmem.resize(gcontext.nvertices); // Initialize in-memory vertices.
        }
    }
    
    /**
     *  Vertex update function.
     */
    void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
                if (gcontext.iteration == 0) {
            /* On first iteration, initialize vertex (and its edges). This is usually required, because
             on each run, GraphChi will modify the data files. To start from scratch, it is easiest
             do initialize the program in code. Alternatively, you can keep a copy of initial data files. */

            latentvec_t latentfac;
            latentfac.init();
            set_latent_factor(vertex, latentfac);
        /* Hack: we need to count ourselves the number of vertices on left
           and right side of the bipartite graph.
           TODO: maybe there should be specialized support for bipartite graphs in GraphChi?
        */
        if (vertex.num_outedges() > 0) {
            // Left side on the bipartite graph
            max_left_vertex = std::max(vertex.id(), max_left_vertex);
        } else {
            max_right_vertex = std::max(vertex.id(), max_right_vertex);
        }

   } else {
            mat XtX(NLATENT, NLATENT); 
            XtX.setZero();
            vec Xty(NLATENT);
            Xty.setZero();
            
            // Compute XtX and Xty (NOTE: unweighted)
            for(int e=0; e < vertex.num_edges(); e++) {
                float observation = vertex.edge(e)->get_data();                
                latentvec_t & nbr_latent = latent_factors_inmem[vertex.edge(e)->vertex_id()];
                for(int i=0; i<NLATENT; i++) {
                    Xty(i) += nbr_latent[i] * observation;
                    for(int j=i; j < NLATENT; j++) {
                        XtX(j,i) += nbr_latent[i] * nbr_latent[j];
                    }
                }
            }
            
            // Symmetrize
            for(int i=0; i <NLATENT; i++)
                for(int j=i + 1; j< NLATENT; j++) XtX(i,j) = XtX(j,i);
            
            // Diagonal
            for(int i=0; i < NLATENT; i++) XtX(i,i) += (lambda) * vertex.num_edges();
            
            // Solve the least squares problem with eigen using Cholesky decomposition
            vec veclatent = XtX.ldlt().solve(Xty);
            
            // Convert to plain doubles (this is useful because now the output data by GraphCHI
            // is plain binary double matrix that can be read, for example, by Matlab).
            latentvec_t newlatent;
            for(int i=0; i < NLATENT; i++) newlatent[i] = veclatent[i];
            newlatent.rmse = 0; 
            
            bool compute_rmse = (vertex.num_outedges() > 0);
            if (compute_rmse) { // Compute RMSE only on "right side" of bipartite graph
                for(int e=0; e < vertex.num_edges(); e++) {        
                    // Compute RMSE
                    double observation = vertex.edge(e)->get_data();
                    latentvec_t & nbr_latent =  latent_factors_inmem[vertex.edge(e)->vertex_id()];
                    double prediction = nbr_latent.dot(newlatent);
                    prediction = std::min(maxval, prediction);
		    prediction = std::max(minval, prediction);
                    newlatent.rmse += (prediction - observation) * (prediction - observation);                
                    
                }
                
            }
            
            set_latent_factor(vertex, newlatent); 
            
            if (vertex.id() % 100000 == 1) {
                std::cout <<  gcontext.iteration << ": " << vertex.id() << std::endl;
            }
        }
        
         }
    
    
    
    /**
     * Called after an iteration has finished.
     */
    void after_iteration(int iteration, graphchi_context &gcontext) {
       rmse = 0;
#pragma omp parallel for reduction(+:rmse)
       for (uint i=0; i< max_left_vertex; i++){
         rmse += latent_factors_inmem[i].rmse;
       }
       logstream(LOG_INFO)<<"Training RMSE: " << sqrt(rmse/pengine->num_edges()) << std::endl;
       validation_rmse();
    }
    
    /**
     * Called before an execution interval is started.
     */
    void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
    }
    
    /**
     * Called after an execution interval has finished.
     */
    void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
    }
    
};

int main(int argc, const char ** argv) {
    /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
    graphchi_init(argc, argv);
    
    /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
    metrics m("als-inmemory-factors");
    
    /* Basic arguments for application. NOTE: File will be automatically 'sharded'. */
    training = get_option_string("training");    // Base filename
    validation = get_option_string("validation", "");
    test = get_option_string("test", "");

    if (validation == "")
       validation += training + "e";  
    if (test == "")
       test += training + "t";

    int niters    = get_option_int("niters", 6);  // Number of iterations
    maxval        = get_option_float("maxval", 1e100);
    minval        = get_option_float("minval", -1e100);
    lambda        = get_option_float("lambda", 0.065);
    
    bool scheduler       = false;                        // Selective scheduling not supported for now.
    
    /* Preprocess data if needed, or discover preprocess files */
    int nshards = convert_matrixmarket_for_ALS<float>(training);
    
    /* Run */
    ALSVerticesInMemProgram program;
    graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, scheduler, m); 
    engine.set_modifies_inedges(false);
    engine.set_modifies_outedges(false);
    pengine = &engine;
    engine.run(program, niters);
        
    m.set("train_rmse", rmse);
    m.set("latent_dimension", NLATENT);
    
    /* Output latent factor matrices in matrix-market format */
    vid_t numvertices = engine.num_vertices();
    assert(numvertices == max_right_vertex + 1); // Sanity check
    output_als_result(training, numvertices, max_left_vertex);
    test_predictions();    
    
    /* Report execution metrics */
    metrics_report(m);
    return 0;
}

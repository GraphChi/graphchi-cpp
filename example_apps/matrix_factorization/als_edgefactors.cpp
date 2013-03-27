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
 * Matrix factorization with the Alternative Least Squares (ALS) algorithm.
 * This code is based on GraphLab's implementation of ALS by Joey Gonzalez
 * and Danny Bickson (CMU). A good explanation of the algorithm is 
 * given in the following paper:
 *    Large-Scale Parallel Collaborative Filtering for the Netflix Prize
 *    Yunhong Zhou, Dennis Wilkinson, Robert Schreiber and Rong Pan
 *    http://www.springerlink.com/content/j1076u0h14586183/
 *
 * There are two versions of the ALS in the example applications. This version
 * is slower, but works with very low memory. In this implementation, a vertex
 * writes its D-dimensional latent factor to its incident edges. See application
 * "als_vertices_inmem" for a faster version, which requires more memory.
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
typedef als_factor_and_weight EdgeDataType;  // Edges store the "rating" of user->movie pair
                                             // and the latent factor of their incident vertex.

/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct ALSEdgeFactorsProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {
    mutex lock;
    
    // Helper
    virtual void set_latent_factor(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, latentvec_t &fact) {
        vertex.set_data(fact);
        for(int i=0; i < vertex.num_edges(); i++) {
            als_factor_and_weight factwght = vertex.edge(i)->get_data();
            factwght.factor = fact;
            vertex.edge(i)->set_data(factwght);   // Note that neighbors override the values they have written to edges.
                                                  // This is ok, because vertices are always executed in same order.
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
        } else {
            mat XtX(NLATENT, NLATENT); 
            XtX.setZero();
            vec Xty(NLATENT);
            Xty.setZero();
            
            // Compute XtX and Xty (NOTE: unweighted)
            for(int e=0; e < vertex.num_edges(); e++) {
                float observation = vertex.edge(e)->get_data().weight;                
                latentvec_t nbr_latent = vertex.edge(e)->get_data().factor;
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
            for(int i=0; i < NLATENT; i++) XtX(i,i) += (LAMBDA) * vertex.num_edges();
            
            // Solve the least squares problem with eigen using Cholesky decomposition
            vec veclatent = XtX.ldlt().solve(Xty);
            
            // Convert to plain doubles (this is useful because now the output data by GraphCHI
            // is plain binary double matrix that can be read, for example, by Matlab).
            latentvec_t newlatent;
            for(int i=0; i < NLATENT; i++) newlatent[i] = veclatent[i];
            
            
            double sqerror = 0;
            bool compute_rmse = (gcontext.iteration == gcontext.num_iterations-1 && vertex.num_outedges() == 0);
            if (compute_rmse) { // Compute RMSE only on "right side" of bipartite graph
                for(int e=0; e < vertex.num_edges(); e++) {        
                    // Compute RMSE
                    float observation = vertex.edge(e)->get_data().weight;
                    latentvec_t nbr_latent = vertex.edge(e)->get_data().factor;
                    double prediction = nbr_latent.dot(newlatent);
                    sqerror += (prediction - observation) * (prediction - observation);                
                    
                }
                rmselock.lock();
                rmse += sqerror;
                rmselock.unlock();
                
                if (vertex.id() % 5000 == 1) {
                    logstream(LOG_DEBUG) << "Computed RMSE for : " << vertex.id() << std::endl;
                }
            }
            
            set_latent_factor(vertex, newlatent); 
            
            if (vertex.id() % 100000 == 1) {
                std::cout <<  gcontext.iteration << ": " << vertex.id() << std::endl;
            }
        }
        
        /* Hack: we need to count ourselves the number of vertices on left
           and right side of the bipartite graph.
           TODO: maybe there should be specialized support for bipartite graphs in GraphChi?
        */
        if (vertex.num_outedges() > 0) {
            // Left side on the bipartite graph
            if (vertex.id() > max_left_vertex) {
                lock.lock();
                max_left_vertex = std::max(vertex.id(), max_left_vertex);
                lock.unlock();
            }
        } else {
            if (vertex.id() > max_right_vertex) {
                lock.lock();
                max_right_vertex = std::max(vertex.id(), max_right_vertex);
                lock.unlock();
            }
        }

    }
    
    /**
     * Called before an iteration starts.
     */
    void before_iteration(int iteration, graphchi_context &gcontext) {
    }
    
    /**
     * Called after an iteration has finished.
     */
    void after_iteration(int iteration, graphchi_context &gcontext) {
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
    metrics m("als-edgefactors");
    
    /* Basic arguments for application. NOTE: File will be automatically 'sharded'. */
    std::string filename = get_option_string("file");    // Base filename
    int niters           = get_option_int("niters", 6);  // Number of iterations
    bool scheduler       = false;                        // Selective scheduling not supported for now.
    
    /* Preprocess data if needed, or discover preprocess files */
    int nshards = convert_matrixmarket_for_ALS<als_factor_and_weight>(filename);
    
    /* Run */
    ALSEdgeFactorsProgram program;
    graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, scheduler, m); 
    engine.set_enable_deterministic_parallelism(false);
    engine.run(program, niters);
        
    /* Report result (train RMSE) */
    double trainRMSE = sqrt(rmse / (1.0 * engine.num_edges()));
    m.set("train_rmse", trainRMSE);
    m.set("latent_dimension", NLATENT);
    std::cout << "Latent factor dimension: " << NLATENT << " - train RMSE: " << trainRMSE << std::endl;
    
    /* Output latent factor matrices in matrix-market format */
    vid_t numvertices = engine.num_vertices();
    assert(numvertices == max_right_vertex + 1); // Sanity check
    output_als_result(filename, numvertices, max_left_vertex);
    
    
    /* Report execution metrics */
    metrics_report(m);
    return 0;
}

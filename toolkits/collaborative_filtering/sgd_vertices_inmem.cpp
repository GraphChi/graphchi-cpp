/**
 * @file
 * @author  Danny Bickson (based on code template by Aapo Kyorla)
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
 * Matrix factorizatino with the Alternative Least Squares (SGD) algorithm.
 *
 * 
 */



#include <string>
#include <algorithm>

#include "graphchi_basic_includes.hpp"

/* SGD-related classes are contained in als.hpp */
#include "sgd.hpp"

using namespace graphchi;

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef latentvec_t VertexDataType;
typedef float EdgeDataType;  // Edges store the "rating" of user->movie pair
                                        
/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct SGDVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    std::vector<latentvec_t> latent_factors_inmem;
    
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
     * Called after an iteration has finished.
     */
    void after_iteration(int iteration, graphchi_context &gcontext) {
       sgd_lambda *= sgd_step_dec;
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
            latentvec_t & user = latent_factors_inmem[vertex.id()]; 
            for(int e=0; e < vertex.num_edges(); e++) {
                float observation = vertex.edge(e)->get_data();                
                latentvec_t & movie = latent_factors_inmem[vertex.edge(e)->vertex_id()];
                float estScore = user.dot(movie);
                float err = observation - estScore;
                for (int i=0; i< NLATENT; i++){
                   movie.d[i] += sgd_gamma*(err*user.d[i] - sgd_lambda*movie.d[i]);
                   user.d[i] += sgd_gamma*(err*movie.d[i] - sgd_lambda*user.d[i]);
                } 
            }

            
            
            
            /*double sqerror = 0;
            bool compute_rmse = (gcontext.iteration == gcontext.num_iterations-1 && vertex.num_outedges() == 0);
            if (compute_rmse) { // Compute RMSE only on "right side" of bipartite graph
                for(int e=0; e < vertex.num_edges(); e++) {        
                    // Compute RMSE
                    float observation = vertex.edge(e)->get_data();
                    latentvec_t & nbr_latent =  latent_factors_inmem[vertex.edge(e)->vertex_id()];
                    double prediction = nbr_latent.dot(newlatent);
                    sqerror += (prediction - observation) * (prediction - observation);                
                    
                }
                rmselock.lock();
                rmse += sqerror;
                rmselock.unlock();
                
                if (vertex.id() % 5000 == 1) {
                    logstream(LOG_DEBUG) << "Computed RMSE for : " << vertex.id() << std::endl;
                }
            }*/
            
            //set_latent_factor(vertex, newlatent); 
            
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
            max_left_vertex = std::max(vertex.id(), max_left_vertex);
        } else {
            max_right_vertex = std::max(vertex.id(), max_right_vertex);
        }

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
    metrics m("sgd-inmemory-factors");
    
    /* Basic arguments for application. NOTE: File will be automatically 'sharded'. */
    std::string filename = get_option_string("file");    // Base filename
    int niters           = get_option_int("niters", 6);  // Number of iterations
    sgd_lambda    = get_option_float("sgd_lambda", 1e-3);
    sgd_gamma     = get_option_float("sgd_gamma", 1e-3);
    sgd_step_dec  = get_option_float("sgd_step_dec", 0.9);

    /* Preprocess data if needed, or discover preprocess files */
    int nshards = convert_matrixmarket_for_SGD<float>(filename);
    
    /* Run */
    SGDVerticesInMemProgram program;
    graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, false, m); 
    engine.set_modifies_inedges(false);
    engine.set_modifies_outedges(false);
    engine.run(program, niters);
        
    /* Report result (train RMSE) */
    double trainRMSE = sqrt(rmse / (1.0 * engine.num_edges()));
    m.set("train_rmse", trainRMSE);
    m.set("latent_dimension", NLATENT);
    std::cout << "Latent factor dimension: " << NLATENT << " - train RMSE: " << trainRMSE << std::endl;
    
    /* Output latent factor matrices in matrix-market format */
    vid_t numvertices = engine.num_vertices();
    assert(numvertices == max_right_vertex + 1); // Sanity check
    output_sgd_result(filename, numvertices, max_left_vertex);
    
    
    /* Report execution metrics */
    metrics_report(m);
    return 0;
}

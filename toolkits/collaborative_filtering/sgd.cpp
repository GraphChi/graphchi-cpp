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
 * Matrix factorization with the Stochastic Gradient Descent (SGD) algorithm.
 *
 * 
 */



#include <string>
#include <algorithm>

#include "graphchi_basic_includes.hpp"

/* SGD-related classes are contained in sgd.hpp */
#include "sgd.hpp"

using namespace graphchi;


/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef float EdgeDataType;  // Edges store the "rating" of user->movie pair
    
graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

#include "rmse.hpp"

/** compute a missing value based on SGD algorithm */
float sgd_predict(const vertex_data& user, 
                const vertex_data& movie, 
                const float rating, 
                double & prediction){
 

  prediction = 0;
  for (int j=0; j< NLATENT; j++)
    prediction += user.d[j]* movie.d[j];  

  //truncate prediction to allowed values
  prediction = std::min((double)prediction, maxval);
  prediction = std::max((double)prediction, minval);
  //return the squared error
  float err = rating - prediction;
  assert(!std::isnan(err));
  return err*err; 
 
}




                                        
/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct SGDVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    // Helper
    virtual void set_latent_factor(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, vertex_data &fact) {
        vertex.set_data(fact); // Note, also stored on disk. This is non-optimal...
        latent_factors_inmem[vertex.id()] = fact;
    }
    
    /**
     * Called before an iteration starts.
     */
    void before_iteration(int iteration, graphchi_context &gcontext) {
        if (iteration == 0) {
            latent_factors_inmem.resize(gcontext.nvertices); // Initialize in-memory vertices.
            assert(M > 0 && N > 0);
            max_left_vertex = M-1;
            max_right_vertex = M+N-1;
        }
        rmse = 0;
    }

  /**
     * Called after an iteration has finished.
     */
    void after_iteration(int iteration, graphchi_context &gcontext) {
       sgd_lambda *= sgd_step_dec;
       validation_rmse(&sgd_predict);
       rmse = 0;
#pragma omp parallel for reduction(+:rmse)
       for (uint i=0; i< max_left_vertex; i++){
         rmse += latent_factors_inmem[i].rmse;
       }
       logstream(LOG_INFO)<<"Training RMSE: " << sqrt(rmse/pengine->num_edges()) << std::endl;
    }
         
    /**
     *  Vertex update function.
     */
    void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
        if (gcontext.iteration == 0) {
            /* On first iteration, initialize vertex (and its edges). This is usually required, because
             on each run, GraphChi will modify the data files. To start from scratch, it is easiest
             do initialize the program in code. Alternatively, you can keep a copy of initial data files. */

            vertex_data latentfac;
            latentfac.init();
            set_latent_factor(vertex, latentfac);
        } else {
	    if ( vertex.num_outedges() > 0){
            vertex_data & user = latent_factors_inmem[vertex.id()]; 
            user.rmse = 0; 
            for(int e=0; e < vertex.num_edges(); e++) {
                float observation = vertex.edge(e)->get_data();                
                vertex_data & movie = latent_factors_inmem[vertex.edge(e)->vertex_id()];
                double estScore;
                user.rmse += sgd_predict(user, movie, observation, estScore);
                double err = observation - estScore;
                if (std::isnan(err) || std::isinf(err))
                  logstream(LOG_FATAL)<<"SGD got into numerical error. Please tune step size using --sgd_gamma and sgd_lambda" << std::endl;
                for (int i=0; i< NLATENT; i++){
                   movie.d[i] += sgd_gamma*(err*user.d[i] - sgd_lambda*movie.d[i]);
                   user.d[i] += sgd_gamma*(err*movie.d[i] - sgd_lambda*user.d[i]);
                }
                user.rmse +=  err*err;
            }
            set_latent_factor(vertex, user);
            }
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

struct  MMOutputter{
  FILE * outf;
  MMOutputter(std::string fname, uint start, uint end, std::string comment)  {
    MM_typecode matcode;
    set_matcode(matcode);     
    outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, end-start, NLATENT); 
    for (uint i=start; i < end; i++)
      for(int j=0; j < NLATENT; j++) {
        fprintf(outf, "%lf\n", latent_factors_inmem[i].d[j]);
    }
  }

  ~MMOutputter() {
    if (outf != NULL) fclose(outf);
  }

};
void output_sgd_result(std::string filename, vid_t numvertices, vid_t max_left_vertex) {
  MMOutputter mmoutput_left(filename + "_U.mm", 0, max_left_vertex + 1, "This file contains SGD output matrix U. In each row NLATENT factors of a single user node.");
  MMOutputter mmoutput_right(filename + "_V.mm", max_left_vertex +1 ,numvertices - max_left_vertex - 1, "This file contains SGD  output matrix V. In each row NLATENT factors of a single item node.");

  logstream(LOG_INFO) << "SGD output files (in matrix market format): " << filename << "_U.mm" <<
                                                                             ", " << filename + "_V.mm " << std::endl;
}


int main(int argc, const char ** argv) {
    logstream(LOG_WARNING)<<"GraphChi Collaborative filtering library is written by Danny Bickson (c). Send any "
     " comments or bug reports to danny.bickson@gmail.com " << std::endl;
    
    //* GraphChi initialization will read the command line arguments and the configuration file. */
    graphchi_init(argc, argv);
    
    /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
    metrics m("sgd-inmemory-factors");
    
    /* Basic arguments for application. NOTE: File will be automatically 'sharded'. */
    training = get_option_string("training");    // Base training
    validation = get_option_string("validation", "");
    test = get_option_string("test", "");

    if (validation == "")
       validation += training + "e";  
    if (test == "")
       test += training + "t";

    int niters    = get_option_int("niters", 6);  // Number of iterations
    sgd_lambda    = get_option_float("sgd_lambda", 1e-3);
    sgd_gamma     = get_option_float("sgd_gamma", 1e-3);
    sgd_step_dec  = get_option_float("sgd_step_dec", 0.9);
    maxval        = get_option_float("maxval", 1e100);
    minval        = get_option_float("minval", -1e100);

    /* Preprocess data if needed, or discover preprocess files */
    int nshards = convert_matrixmarket<float>(training);
    
    /* Run */
    SGDVerticesInMemProgram program;
    graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
    engine.set_modifies_inedges(false);
    engine.set_modifies_outedges(false);
    pengine = &engine;
    engine.run(program, niters);
        
    /* Output latent factor matrices in matrix-market format */
    vid_t numvertices = engine.num_vertices();
    assert(numvertices == max_right_vertex + 1); // Sanity check
    output_sgd_result(training, numvertices, max_left_vertex);
    test_predictions(&sgd_predict);    
    
    
    /* Report execution metrics */
    metrics_report(m);
    return 0;
}

/**
 * @file
 * @author  Danny Bickson
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright [2012] [Carnegie Mellon University]
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
 * Algorithm is described in the papers:
 * 1) Matrix Factorization Techniques for Recommender Systems Yehuda Koren, Robert Bell, Chris Volinsky. In IEEE Computer, Vol. 42, No. 8. (07 August 2009), pp. 30-37. 
 * 2) Takács, G, Pilászy, I., Németh, B. and Tikk, D. (2009). Scalable Collaborative Filtering Approaches for Large Recommender Systems. Journal of Machine Learning Research, 10, 623-656.
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
#include "io.hpp"

/** compute a missing value based on SGD algorithm */
float sgd_predict(const vertex_data& user, 
    const vertex_data& movie, 
    const float rating, 
    double & prediction){


  prediction = user.dot(movie);

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


  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    sgd_lambda *= sgd_step_dec;
    training_rmse(iteration, gcontext);
    validation_rmse(&sgd_predict, gcontext);
  }

  /**
   *  Vertex update function.
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
    //go over all user nodes
    if ( vertex.num_outedges() > 0){
      vertex_data & user = latent_factors_inmem[vertex.id()]; 
      user.rmse = 0; 
      //go over all ratings
      for(int e=0; e < vertex.num_edges(); e++) {
        float observation = vertex.edge(e)->get_data();                
        vertex_data & movie = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        double estScore;
        user.rmse += sgd_predict(user, movie, observation, estScore);
        double err = observation - estScore;
        if (std::isnan(err) || std::isinf(err))
          logstream(LOG_FATAL)<<"SGD got into numerical error. Please tune step size using --sgd_gamma and sgd_lambda" << std::endl;
        Map<vec> movie_vec(movie.pvec, NLATENT);
        Map<vec> user_vec(user.pvec, NLATENT);
        //NOTE: the following code is not thread safe, since potentially several
        //user nodes may updates this item gradient vector concurrently. However in practice it
        //did not matter in terms of accuracy on a multicore machine.
        //if you like to defend the code, you can define a global variable
        //mutex mymutex;
        //
        //and then do: mymutex.lock()
        movie_vec += sgd_gamma*(err*user_vec - sgd_lambda*movie_vec);
        //and here add: mymutex.unlock();
        user_vec += sgd_gamma*(err*movie_vec - sgd_lambda*user_vec);
      }
    }

  }

};

//struct for writing the output feature vectors into file
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
        fprintf(outf, "%1.12e\n", latent_factors_inmem[i].pvec[j]);
      }
  }

  ~MMOutputter() {
    if (outf != NULL) fclose(outf);
  }

};

//dump output to file
void output_sgd_result(std::string filename, vid_t numvertices, vid_t max_left_vertex) {
  MMOutputter mmoutput_left(filename + "_U.mm", 0, M, "This file contains SGD output matrix U. In each row NLATENT factors of a single user node.");
  MMOutputter mmoutput_right(filename + "_V.mm", M ,numvertices,  "This file contains SGD  output matrix V. In each row NLATENT factors of a single item node.");

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

  int niters    = get_option_int("max_iter", 6);  // Number of iterations
  sgd_lambda    = get_option_float("sgd_lambda", 1e-3);
  sgd_gamma     = get_option_float("sgd_gamma", 1e-3);
  sgd_step_dec  = get_option_float("sgd_step_dec", 0.9);
  maxval        = get_option_float("maxval", 1e100);
  minval        = get_option_float("minval", -1e100);
  bool quiet    = get_option_int("quiet", 0);
  if (quiet)
    global_logger().set_log_level(LOG_ERROR);
  halt_on_rmse_increase = get_option_int("halt_on_rmse_increase", 0);
  load_factors_from_file = get_option_int("load_factors_from_file", 0);

  parse_implicit_command_line();

  mytimer.start();

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket<float>(training);
  latent_factors_inmem.resize(M+N); // Initialize in-memory vertices.
 if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, NLATENT);
    load_matrix_market_matrix(training + "_V.mm", M, NLATENT);
  }

 
  /* Run */
  SGDVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  engine.set_disable_vertexdata_storage();  
  engine.set_modifies_inedges(false);
  engine.set_modifies_outedges(false);
  pengine = &engine;
  engine.run(program, niters);

  /* Output latent factor matrices in matrix-market format */
  vid_t numvertices = engine.num_vertices();
  output_sgd_result(training, numvertices, M);
  test_predictions(&sgd_predict);    


  /* Report execution metrics */
  metrics_report(m);
  return 0;
}

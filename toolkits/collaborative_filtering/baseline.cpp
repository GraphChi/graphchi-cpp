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
 * Matrix factorization with the Stochastic Gradient Descent (Baseline) algorithm.
 * Algorithm is described in the papers:
 * 1) Matrix Factorization Techniques for Recommender Systems Yehuda Koren, Robert Bell, Chris Volinsky. In IEEE Computer, Vol. 42, No. 8. (07 August 2009), pp. 30-37. 
 * 2) Takács, G, Pilászy, I., Németh, B. and Tikk, D. (2009). Scalable Collaborative Filtering Approaches for Large Recommender Systems. Journal of Machine Learning Research, 10, 623-656.
 *
 * 
 */



#include "common.hpp"
#include "eigen_wrapper.hpp"

//types of algorithms supported when computing prediction
enum{
  GLOBAL_MEAN = 0, USER_MEAN = 1, ITEM_MEAN = 2
};

int algo = GLOBAL_MEAN;
std::string algorithm;

struct vertex_data {
  double mean_rating; 
  vec pvec;

  vertex_data() {
    mean_rating = 0; 
  }
  void set_val(int index, float val){
    pvec[index] = val;
  }
  float get_val(int index){
    return pvec[index];
  }

};

#include "util.hpp"


typedef vertex_data VertexDataType;
typedef float EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

#include "rmse.hpp"
#include "io.hpp"


/** compute a missing value based on SGD algorithm */
float baseline_predict(const vertex_data& user, 
    const vertex_data& movie, 
    const float rating, 
    double & prediction, 
    void * extra = NULL){


  prediction = globalMean;
  if (algo == USER_MEAN)
    prediction = user.mean_rating;
  else if (algo == ITEM_MEAN)
    prediction = movie.mean_rating;

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
struct BaselineVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {


  void after_iteration(int iteration, graphchi_context &gcontext) {
    training_rmse(iteration, gcontext, algo == ITEM_MEAN);
    validation_rmse(&baseline_predict, gcontext);
  }

  /**
   *  Vertex update function.
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
    //go over all user nodes
    if ( vertex.num_outedges() > 0 && (algo == GLOBAL_MEAN || algo == USER_MEAN)){
      vertex_data & user = latent_factors_inmem[vertex.id()]; 

      //go over all ratings
      if (algo == USER_MEAN){
          user.mean_rating = 0;
        for(int e=0; e < vertex.num_edges(); e++) {
          float observation = vertex.edge(e)->get_data();                
          user.mean_rating += observation;
        }
        if (vertex.num_edges() > 0)
          user.mean_rating /= vertex.num_edges();
      }

      //go over all ratings
      for(int e=0; e < vertex.num_edges(); e++) {
        double prediction;
        float observation = vertex.edge(e)->get_data();                
        vertex_data & movie = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        rmse_vec[omp_get_thread_num()] += baseline_predict(user, movie, observation, prediction);
      }
    }
    else if (vertex.num_inedges() > 0 && algo == ITEM_MEAN){
      vertex_data & user = latent_factors_inmem[vertex.id()]; 
      user.mean_rating = 0;
      //go over all ratings
      for(int e=0; e < vertex.num_edges(); e++) {
        float observation = vertex.edge(e)->get_data();                
        user.mean_rating += observation;
      } 
      if (vertex.num_edges() > 0)
        user.mean_rating /= vertex.num_edges();

      for(int e=0; e < vertex.num_edges(); e++) {
        float observation = vertex.edge(e)->get_data();                
        double prediction;
        vertex_data & movie = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        rmse_vec[omp_get_thread_num()] += baseline_predict(movie, user, observation, prediction);
      }
    }
  }
};

//struct for writing the output feature vectors into file
struct  MMOutputter2{
  FILE * outf;
  MMOutputter2(std::string fname, uint start, uint end, std::string comment)  {
    MM_typecode matcode;
    set_matcode(matcode);     
    outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, end-start, 1); 
    for (uint i=start; i < end; i++)
      fprintf(outf, "%1.12e\n", latent_factors_inmem[i].mean_rating);
  }

  ~MMOutputter2() {
    if (outf != NULL) fclose(outf);
  }

};

//dump output to file
void output_baseline_result(std::string filename) {
  if (algo == USER_MEAN){
    MMOutputter2 mmoutput_left(filename + ".baseline_user", 0, M, "This file contains Baseline output matrix U. In each row rating mean a single user node.");
  }
  else if (algo == ITEM_MEAN){
    MMOutputter2 mmoutput_right(filename + ".baseline_item", M ,M+N,  "This file contains Baseline  output vector V. In each row rating mean of a single item node.");

  }
  logstream(LOG_INFO) << "Baseline output files (in matrix market format): " << filename << (algo == USER_MEAN ? ".baseline_user" : ".baseline_item") << std::endl;
}


int main(int argc, const char ** argv) {

  print_copyright();

  //* GraphChi initialization will read the command line arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("sgd-inmemory-factors");

  algorithm     = get_option_string("algorithm", "global_mean");
  if (algorithm == "global_mean")
    algo = GLOBAL_MEAN;
  else if (algorithm == "user_mean")
    algo = USER_MEAN;
  else if (algorithm == "item_mean")
    algo = ITEM_MEAN;
  else logstream(LOG_FATAL)<<"Unsupported algorithm name. Should be --algorithm=XX where XX is one of [global_mean,user_mean,item_mean] for example --algorithm=global_mean" << std::endl;


  parse_command_line_args();
  mytimer.start();

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket<float>(training, 0, 0, 3, TRAINING, false);
  init_feature_vectors<std::vector<vertex_data> >(M+N, latent_factors_inmem, false);
  rmse_vec = zeros(number_of_omp_threads());
  print_config();

  /* Run */
  BaselineVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine); 
  pengine = &engine;
  engine.run(program, 1);

  if (algo == USER_MEAN || algo == ITEM_MEAN)
    output_baseline_result(training);
  test_predictions(&baseline_predict);    

  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  return 0;
}

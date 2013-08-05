/**
 * @file
 * @author  Danny Bickson, CMU
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
 * This code implements the paper:
 * Lee, D..D., and Seung, H.S., (2001), 'Algorithms for Non-negative Matrix
 * Factorization', Adv. Neural Info. Proc. Syst. 13, 556-562.
 *
 * 
 */



#include "common.hpp"
#include "eigen_wrapper.hpp"

const double epsilon = 1e-16;
struct vertex_data {
  vec pvec;

  vertex_data() {
   pvec = zeros(D); 
  }
  void set_val(int index, float val){
    pvec[index] = val;
  }
  float get_val(int index){
    return pvec[index];
  }
};



typedef vertex_data VertexDataType;
typedef float EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
graphchi_engine<VertexDataType, EdgeDataType> * pvalidation_engine = NULL; 
std::vector<vertex_data> latent_factors_inmem;
vec sum_of_item_latent_features, sum_of_user_latent_feautres;
int iter;

#include "rmse.hpp"
#include "rmse_engine.hpp"
#include "io.hpp"

/** compute a missing value based on NMF algorithm */
float nmf_predict(const vertex_data& user, 
    const vertex_data& movie, 
    const float rating, 
    double & prediction, 
    void * extra = NULL){

  prediction = dot_prod(user.pvec, movie.pvec);
  //truncate prediction to allowed values
  prediction = std::min((double)prediction, maxval);
  prediction = std::max((double)prediction, minval);
  //return the squared error
  float err = rating - prediction;
  assert(!std::isnan(err));
  return err*err; 

}

/* sum up all item data vectors */
void pre_user_iter(){
  sum_of_item_latent_features = zeros(D);
  for (uint i=M; i<M+N; i++){
    vertex_data & data = latent_factors_inmem[i];
    sum_of_item_latent_features += data.pvec;
  }
}

/* sum up all user data vectors */
void pre_movie_iter(){
  sum_of_user_latent_feautres = zeros(D);
  for (uint i=0; i<M; i++){
    vertex_data & data = latent_factors_inmem[i];
    sum_of_user_latent_feautres += data.pvec;
  }
}




/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct NMFVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {


  /**
   * Called before an iteration starts.
   */
  void before_iteration(int iteration, graphchi_context &gcontext) {
    reset_rmse(gcontext.execthreads);
    iter = iteration;
    if (iteration > 0) {
      if (iteration % 2 == 1)
        pre_user_iter();
      else pre_movie_iter();
    }
  }

  /**
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {

    if (gcontext.iteration == 0){
      if (vertex.num_outedges() == 0 && vertex.id() < M)
        logstream(LOG_FATAL)<<"NMF algorithm can not work when the row " << vertex.id() << " of the matrix contains all zeros" << std::endl;
      for(int e=0; e < vertex.num_edges(); e++) {
        float observation = vertex.edge(e)->get_data();                
        if (observation < 0 ){
          logstream(LOG_FATAL)<<"Found a negative entry in matirx row " << vertex.id() << " with value: " << observation << std::endl;
        }
      }
      return;   
    }

    bool isuser = (vertex.id() < M);
    if ((iter % 2 == 1 && !isuser) ||
        (iter % 2 == 0 && isuser))
      return;
    
    vec ret = zeros(D);

    vertex_data & vdata = latent_factors_inmem[vertex.id()];
    
    for(int e=0; e < vertex.num_edges(); e++) {
      float observation = vertex.edge(e)->get_data();                
      vertex_data & nbr_latent = latent_factors_inmem[vertex.edge(e)->vertex_id()];
      double prediction;
      rmse_vec[omp_get_thread_num()] += nmf_predict(vdata, nbr_latent, observation, prediction);
      if (prediction == 0)
        logstream(LOG_FATAL)<<"Got into numerical error! Please submit a bug report." << std::endl;
      ret += nbr_latent.pvec * (observation / prediction);
    }
    
    vec px;
    if (isuser)
      px = sum_of_item_latent_features;
    else 
      px = sum_of_user_latent_feautres;
    for (int i=0; i<D; i++){
      assert(px[i] != 0);
      vdata.pvec[i] *= ret[i] / px[i];
      if (vdata.pvec[i] < epsilon)
        vdata.pvec[i] = epsilon;
    }
  }



  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    //print rmse every other iteration, since 2 iterations are considered one NMF round
    int now = iteration % 2;
    if (now == 0){
      training_rmse(iteration/2, gcontext);
      run_validation(pvalidation_engine, gcontext);
    }
  }
};



void output_nmf_result(std::string filename){
  MMOutputter_mat<vertex_data> user_mat(filename + "_U.mm", 0, M, "This file contains NMF output matrix U. In each row D factors of a single user node.", latent_factors_inmem);
  MMOutputter_mat<vertex_data> item_mat(filename + "_V.mm", M, M+N, "This file contains NMF  output matrix V. In each row D factors of a single item node.", latent_factors_inmem);
  logstream(LOG_INFO) << "NMF output files (in matrix market format): " << filename << "_U.mm" <<
                                                                           ", " << filename + "_V.mm " << std::endl;
}

int main(int argc, const char ** argv) {


  print_copyright(); 
 
  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);
  metrics m("nmf-inmemory-factors");

  parse_command_line_args();
  parse_implicit_command_line();

  niters *= 2; //each NMF iteration is composed of two sub iters

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket<float>(training, 0, 0, 3, TRAINING, false);
  init_feature_vectors<std::vector<vertex_data> >(M+N, latent_factors_inmem, !load_factors_from_file);
  if (validation != ""){
    int vshards = convert_matrixmarket<EdgeDataType>(validation, 0, 0, 3, VALIDATION, false);
    if (vshards != -1)
       init_validation_rmse_engine<VertexDataType, EdgeDataType>(pvalidation_engine, vshards, &nmf_predict);
  }
 
  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, D);
    load_matrix_market_matrix(training + "_V.mm", M, D);
  }

  sum_of_item_latent_features = zeros(D);
  sum_of_user_latent_feautres = zeros(D);

  /* Run */
  NMFVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output latent factor matrices in matrix-market format */
  output_nmf_result(training);
  test_predictions(&nmf_predict);    

  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);

  return 0;
}

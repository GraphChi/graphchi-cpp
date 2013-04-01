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
 * Tensor factorization with the Alternative Least Squares (ALS) algorithm.
 * Algorithm is described in: Tensor Decompositions, Alternating Least Squares and other Tales. P. Comon, X. Luciani and A. L. F. de Almeida. Special issue, Journal of Chemometrics. In memory of R. Harshman.
 * August 16, 2009
 * 
 */



#include "common.hpp"
#include "eigen_wrapper.hpp"

double lambda = 0.065;

bool is_user(vid_t id){ return id < M; }
bool is_item(vid_t id){ return id >= M && id < N; }
bool is_time(vid_t id){ return id >= M+N; }

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

struct edge_data {
  double weight;
  double time;

  edge_data() { weight = time = 0; }

  edge_data(double weight, double time) : weight(weight), time(time) { }
};


/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef edge_data EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
graphchi_engine<VertexDataType, EdgeDataType> * pvalidation_engine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

#include "io.hpp"
#include "rmse.hpp"
#include "rmse_engine4.hpp"

float als_tensor_predict(const vertex_data& user, 
    const vertex_data& movie, 
    const float rating, 
    double & prediction, 
    void * extra){

  vertex_data * time_node = (vertex_data*)extra;
  assert(time_node != NULL && time_node->pvec.size() == D);
  prediction = dot3(user.pvec, movie.pvec, time_node->pvec);
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
struct ALSVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

   /*
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
    vertex_data & vdata = latent_factors_inmem[vertex.id()];
    mat XtX = mat::Zero(D, D); 
    vec Xty = vec::Zero(D);

    bool compute_rmse = is_user(vertex.id()); 
    // Compute XtX and Xty (NOTE: unweighted)
    for(int e=0; e < vertex.num_edges(); e++){
      float observation = vertex.edge(e)->get_data().weight;                
      uint time = (uint)vertex.edge(e)->get_data().time;
      assert(time >= 0 && time < M+N+K);
      assert(time != vertex.id());
      vertex_data & nbr_latent = latent_factors_inmem[vertex.edge(e)->vertex_id()];
      vertex_data & time_node = latent_factors_inmem[time];
      assert(time != vertex.id() && time != vertex.edge(e)->vertex_id());
      vec XY = nbr_latent.pvec.cwiseProduct(time_node.pvec);
      Xty += XY * observation;
      XtX.triangularView<Eigen::Upper>() += XY * XY.transpose();
      if (compute_rmse) {
        double prediction;
        rmse_vec[omp_get_thread_num()] += als_tensor_predict(vdata, nbr_latent, observation, prediction, (void*)&time_node);
      }
    }
    double regularization = lambda;
    if (regnormal)
      lambda *= vertex.num_edges();
    for(int i=0; i < D; i++) XtX(i,i) += regularization;

    // Solve the least squares problem with eigen using Cholesky decomposition
    vdata.pvec = XtX.selfadjointView<Eigen::Upper>().ldlt().solve(Xty);
  }


 /**
   * Called before an iteration is started.
   */
  void before_iteration(int iteration, graphchi_context &gcontext) {
    reset_rmse(gcontext.execthreads);
  }


  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    training_rmse(iteration, gcontext);
    run_validation4(pvalidation_engine, gcontext);
  }


};



void output_als_result(std::string filename) {
  MMOutputter_mat<vertex_data> user_mat(filename + "_U.mm", 0, M, "This file contains tensor-ALS output matrix U. In each row D factors of a single user node.", latent_factors_inmem);
  MMOutputter_mat<vertex_data> item_mat(filename + "_V.mm", M ,M+N, "This file contains tensor-ALS  output matrix V. In each row D factors of a single item node.", latent_factors_inmem);
  MMOutputter_mat<vertex_data> time_mat(filename + "_T.mm", M+N ,M+N+K, "This file contains tensor-ALS  output matrix T. In each row D factors of a single time node.", latent_factors_inmem);
  logstream(LOG_INFO) << "tensor - ALS output files (in matrix market format): " << filename << "_U.mm" <<
                                                                           ", " << filename + "_V.mm " << filename + "_T.mm" << std::endl;
}

int main(int argc, const char ** argv) {


  print_copyright();  

/* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("als-tensor-inmemory-factors");

  lambda        = get_option_float("lambda", 0.065);
  parse_command_line_args();
  parse_implicit_command_line();

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket4<edge_data>(training, true);
  init_feature_vectors<std::vector<vertex_data> >(M+N+K, latent_factors_inmem, !load_factors_from_file);
  if (validation != ""){
    int vshards = convert_matrixmarket4<EdgeDataType>(validation, true, M==N, VALIDATION);
    init_validation_rmse_engine<VertexDataType, EdgeDataType>(pvalidation_engine, vshards, &als_tensor_predict, false, true, 1);
   }


if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, D);
    load_matrix_market_matrix(training + "_V.mm", M, D);
    load_matrix_market_matrix(training + "_T.mm", M+N, D);
  }


  /* Run */
  ALSVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output test predictions in matrix-market format */
  output_als_result(training);
  test_predictions3(&als_tensor_predict);    

  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  return 0;
}

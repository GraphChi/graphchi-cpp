/**
 * @file
 * @author  Danny Bickson
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright [2012] Carnegie Mellon University]
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
 *
 * @section USAGE
 *
 *
 * 
 */



#include "common.hpp"
double lambda = 1e-3;

struct vertex_data {
  double pvec[NLATENT];
  double rmse;

  vertex_data() {
    for(int k=0; k < NLATENT; k++) pvec[k] =  drand48(); 
    rmse = 0;
  }

  double dot(const vertex_data &oth) const {
    double x=0;
    for(int i=0; i<NLATENT; i++) x+= oth.pvec[i]*pvec[i];
    return x;
  }

};

struct edge_data {
  double weight;
  double time;

  edge_data() { weight = time = 0; }

  edge_data(double weight, double time) : weight(weight), time(time) { }
};


using namespace graphchi;


/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef edge_data  EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;


#include "rmse.hpp"
#include "io.hpp"

/** compute a missing value based on WALS algorithm */
float wals_predict(const vertex_data& user, 
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
struct WALSVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

  /**
   *  Vertex update function.
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
    vertex_data & vdata = latent_factors_inmem[vertex.id()];
    vdata.rmse = 0;
    mat XtX = mat::Zero(NLATENT, NLATENT); 
    vec Xty = vec::Zero(NLATENT);

    bool compute_rmse = (vertex.num_outedges() > 0);
    // Compute XtX and Xty (NOTE: unweighted)
    for(int e=0; e < vertex.num_edges(); e++) {
      const edge_data & edge = vertex.edge(e)->get_data();                
      vertex_data & nbr_latent = latent_factors_inmem[vertex.edge(e)->vertex_id()];
      Map<vec> X(nbr_latent.pvec, NLATENT);
      Xty += X * edge.weight * edge.time;
      XtX.triangularView<Eigen::Upper>() += X * X.transpose() * edge.time;
      if (compute_rmse) {
        double prediction;
        vdata.rmse += wals_predict(vdata, nbr_latent, edge.weight, prediction) * edge.time;
      }
    }
    // Diagonal
    for(int i=0; i < NLATENT; i++) XtX(i,i) += (lambda); // * vertex.num_edges();
    // Solve the least squares problem with eigen using Cholesky decomposition
    Map<vec> vdata_vec(vdata.pvec, NLATENT);
    vdata_vec = XtX.selfadjointView<Eigen::Upper>().ldlt().solve(Xty);
  }



  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    training_rmse(iteration, gcontext);
    validation_rmse(&wals_predict, gcontext, 4);
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
        fprintf(outf, "%1.12e\n", latent_factors_inmem[i].pvec[j]);
      }
  }

  ~MMOutputter() {
    if (outf != NULL) fclose(outf);
  }

};


void output_als_result(std::string filename) {
  MMOutputter mmoutput_left(filename + "_U.mm", 0, M, "This file contains WALS output matrix U. In each row NLATENT factors of a single user node.");
  MMOutputter mmoutput_right(filename + "_V.mm", M, M+N, "This file contains WALS  output matrix V. In each row NLATENT factors of a single item node.");
  logstream(LOG_INFO) << "WALS output files (in matrix market format): " << filename << "_U.mm" <<
                                                                            ", " << filename + "_V.mm " << std::endl;
}

int main(int argc, const char ** argv) {

  print_copyright();
  
  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("als-inmemory-factors");

  lambda        = get_option_float("lambda", 0.065);
  parse_command_line_args();
  parse_implicit_command_line();
  if (unittest == 1){
    if (training == "") training = "test_wals"; 
    niters = 100;
  }

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket4<edge_data>(training);
  latent_factors_inmem.resize(M+N); // Initialize in-memory vertices.
  assert(M > 0 && N > 0);

  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, NLATENT);
    load_matrix_market_matrix(training + "_V.mm", M, NLATENT);
  }


  /* Run */
  WALSVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output latent factor matrices in matrix-market format */
  output_als_result(training);
  test_predictions(&wals_predict);    

  if (unittest == 1){
    if (dtraining_rmse > 0.03)
      logstream(LOG_FATAL)<<"Unit test 1 failed. Training RMSE is: " << training_rmse << std::endl;
    if (dvalidation_rmse > 0.61)
      logstream(LOG_FATAL)<<"Unit test 1 failed. Validation RMSE is: " << validation_rmse << std::endl;

  }
 
  /* Report execution metrics */
  metrics_report(m);
  return 0;
}

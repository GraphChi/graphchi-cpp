/**
 * @file
 * @author  Danny Bickson, based on code by Aapo Kyrola
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
 */


#include "eigen_wrapper.hpp"
#include "common.hpp"

using namespace graphchi;

double lambda = 0.065;




struct vertex_data {
  vec pvec;
  double rmse;

  vertex_data() {
    pvec = zeros(D);
    rmse = 0;
  }
};


/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef float EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

#include "io.hpp"
#include "rmse.hpp"

/** compute a missing value based on ALS algorithm */
float als_predict(const vertex_data& user, 
    const vertex_data& movie, 
    const float rating, 
    double & prediction){


  prediction = dot_prod(user.pvec, movie.pvec);
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



  /**
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
    vertex_data & vdata = latent_factors_inmem[vertex.id()];
    vdata.rmse = 0;
    mat XtX = mat::Zero(NLATENT, NLATENT); 
    vec Xty = vec::Zero(NLATENT);

    bool compute_rmse = (vertex.num_outedges() > 0);
    // Compute XtX and Xty (NOTE: unweighted)
    for(int e=0; e < vertex.num_edges(); e++) {
      float observation = vertex.edge(e)->get_data();                
      vertex_data & nbr_latent = latent_factors_inmem[vertex.edge(e)->vertex_id()];
      Xty += nbr_latent.pvec * observation;
      XtX.triangularView<Eigen::Upper>() += nbr_latent.pvec * nbr_latent.pvec.transpose();
      if (compute_rmse) {
        double prediction;
        vdata.rmse += als_predict(vdata, nbr_latent, observation, prediction);
      }
    }

    for(int i=0; i < NLATENT; i++) XtX(i,i) += (lambda); // * vertex.num_edges();

    // Solve the least squares problem with eigen using Cholesky decomposition
    vdata.pvec = XtX.selfadjointView<Eigen::Upper>().ldlt().solve(Xty);
  }



  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    training_rmse(iteration, gcontext);
    validation_rmse(&als_predict, gcontext);
  }


};

struct  MMOutputter{
  FILE * outf;
  MMOutputter(std::string fname, uint start, uint end, std::string comment)  {
    assert(start < end);
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
  MMOutputter mmoutput_left(filename + "_U.mm", 0, M , "This file contains ALS output matrix U. In each row NLATENT factors of a single user node.");
  MMOutputter mmoutput_right(filename + "_V.mm", M  ,M+N, "This file contains ALS  output matrix V. In each row NLATENT factors of a single item node.");
  logstream(LOG_INFO) << "ALS output files (in matrix market format): " << filename << "_U.mm" <<
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
  D             = get_option_int("D", D);
  parse_command_line_args();
  parse_implicit_command_line();


  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket<float>(training);
  init_feature_vectors<std::vector<vertex_data> >(M+N, latent_factors_inmem, !load_factors_from_file);

  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, NLATENT);
    load_matrix_market_matrix(training + "_V.mm", M, NLATENT);
  }

  /* Run */
  ALSVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);


  /* Output latent factor matrices in matrix-market format */
  output_als_result(training);
  test_predictions(&als_predict);    

  if (unittest == 1){
    if (dtraining_rmse > 0.03)
      logstream(LOG_FATAL)<<"Unit test 1 failed. Training RMSE is: " << training_rmse << std::endl;
    if (dvalidation_rmse > 1.03)
      logstream(LOG_FATAL)<<"Unit test 1 failed. Validation RMSE is: " << dvalidation_rmse << std::endl;

  }
  
  /* Report execution metrics */
  metrics_report(m);
  return 0;
}

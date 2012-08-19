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
 * Matrix factorization with the Stochastic Gradient Descent (SVDPP) algorithm.
 *
 * 
 */



#include <string>
#include <algorithm>

#include "graphchi_basic_includes.hpp"

/* SVDPP-related classes are contained in svdpp.hpp */
#include "svdpp.hpp"

using namespace graphchi;


/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef float EdgeDataType;  // Edges store the "rating" of user->movie pair
    
graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

/** compute a missing value based on SVD++ algorithm */
float svdpp_predict(const vertex_data& user, const vertex_data& movie, const float rating, double & prediction){
  //\hat(r_ui) = \mu + 
  prediction = globalMean;
  // + b_u  +    b_i +
  prediction += user.bias + movie.bias;
  // + q_i^T   *(p_u      +sqrt(|N(u)|)\sum y_j)
  //prediction += dot_prod(movie.pvec,(user.pvec+user.weight));
  for (int j=0; j< NLATENT; j++)
    prediction += movie.pvec[j] * (user.pvec[j] + user.weight[j]);

  prediction = std::min((double)prediction, maxval);
  prediction = std::max((double)prediction, minval);
  float err = rating - prediction;
  if (std::isnan(err))
    logstream(LOG_FATAL)<<"Got into numerical errors. Try to decrease step size using svdpp)" << std::endl;
  return err*err; 
}



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

  if ((M > 0 && N > 0 ) && (vM != M || vN != N))
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
    double prediction = 0;
    svdpp_predict(latent_factors_inmem[I], latent_factors_inmem[J], 0, prediction);        
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
    logstream(LOG_ERROR) << "Failed reading matrix size: error=" << ret_code << std::endl;
  }
  //if (vM != M || vN != N) //TODO
  //  logstream(LOG_FATAL)<<"Input size of validation matrix must be identical to training matrix, namely " << M << "x" << N << std::endl;


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

    double prediction = 0;
    validation_rmse += svdpp_predict(latent_factors_inmem[I], latent_factors_inmem[J], val, prediction);
  }
  fclose(f);

  logstream(LOG_INFO)<<"Validation RMSE: " << sqrt(validation_rmse/pengine->num_edges())<< std::endl;
}

/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct SVDPPVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

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
    }
  }

  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    svdpp.itmFctrStep *= svdpp.step_dec;
    svdpp.itmFctr2Step *= svdpp.step_dec;
    svdpp.usrFctrStep *= svdpp.step_dec;
    svdpp.itmBiasStep *= svdpp.step_dec;
    svdpp.usrBiasStep *= svdpp.step_dec;

    validation_rmse();
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
      /* Hack: we need to count ourselves the number of vertices on left
         and right side of the bipartite graph.
TODO: maybe there should be specialized support for bipartite graphs in GraphChi?
*/
      if (vertex.num_outedges() > 0) {
        // Left side on the bipartite graph
        if (vertex.id() > max_left_vertex) {
          //lock.lock();
          max_left_vertex = std::max(vertex.id(), max_left_vertex);
          //lock.unlock();
        }
      } else {
        if (vertex.id() > max_right_vertex) {
          //lock.lock();
          max_right_vertex = std::max(vertex.id(), max_right_vertex);
          //lock.unlock();
        }
      }

    } else {
      if ( vertex.num_outedges() > 0){
        vertex_data & user = latent_factors_inmem[vertex.id()]; 

        user.rmse = 0; 
        memset(user.weight, 0, sizeof(double)*NLATENT);
        for(int e=0; e < vertex.num_edges(); e++) {
          vertex_data & movie = latent_factors_inmem[vertex.edge(e)->vertex_id()]; 
          for (int i=0; i< NLATENT; i++)
            user.weight[i] += movie.weight[i];

        }
        // sqrt(|N(u)|) 
        float usrNorm = double(1.0/sqrt(vertex.num_outedges()));
        //sqrt(|N(u)| * sum_j y_j
        for (int j=0; j< NLATENT; j++)
          user.weight[j] *= usrNorm;

        vec step = zeros(NLATENT);

        // main algorithm, see Koren's paper, just below below equation (16)
        for(int e=0; e < vertex.num_edges(); e++) {
          vertex_data & movie = latent_factors_inmem[vertex.edge(e)->vertex_id()]; 
          float observation = vertex.edge(e)->get_data();                
          double estScore;
          user.rmse += svdpp_predict(user, movie,observation, estScore); 
          // e_ui = r_ui - \hat{r_ui}
          float err = observation - estScore;
          assert(!std::isnan(user.rmse));
          vec itmFctr = init_vec(movie.pvec, NLATENT);
          vec usrFctr = init_vec(user.pvec, NLATENT);

          //q_i = q_i + gamma2     *(e_ui*(p_u      +  sqrt(N(U))\sum_j y_j) - gamma7    *q_i)
          for (int j=0; j< NLATENT; j++)
            movie.pvec[j] += svdpp.itmFctrStep*(err*(usrFctr[j] +  user.weight[j])             - svdpp.itmFctrReg*itmFctr[j]);
          //p_u = p_u + gamma2    *(e_ui*q_i   -gamma7     *p_u)
          for (int j=0; j< NLATENT; j++)
            user.pvec[j] += svdpp.usrFctrStep*(err *itmFctr[j] - svdpp.usrFctrReg*usrFctr[j]);
          step += err*itmFctr;

          //b_i = b_i + gamma1*(e_ui - gmma6 * b_i) 
          movie.bias += svdpp.itmBiasStep*(err-svdpp.itmBiasReg* movie.bias);
          //b_u = b_u + gamma1*(e_ui - gamma6 * b_u)
          user.bias += svdpp.usrBiasStep*(err-svdpp.usrBiasReg* user.bias);
        }

        step *= float(svdpp.itmFctr2Step*usrNorm);
        //gamma7 
        double mult = svdpp.itmFctr2Step*svdpp.itmFctr2Reg;
        for(int e=0; e < vertex.num_edges(); e++) {
          vertex_data&  movie = latent_factors_inmem[vertex.edge(e)->vertex_id()];
          //y_j = y_j  +   gamma2*sqrt|N(u)| * q_i - gamma7 * y_j
          for (int j=0; j< NLATENT; j++)
            movie.weight[j] +=  step[j]                    -  mult  * movie.weight[j];
        }


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

int main(int argc, const char ** argv) {
  logstream(LOG_WARNING)<<"GraphChi Collaborative filtering library is written by Danny Bickson (c). Send any "
    " comments or bug reports to danny.bickson@gmail.com " << std::endl;

  //* GraphChi initialization will read the command line arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("svdpp-inmemory-factors");

  /* Basic arguments for application. NOTE: File will be automatically 'sharded'. */
  training = get_option_string("training");    // Base training
  validation = get_option_string("validation", "");
  test = get_option_string("test", "");

  if (validation == "")
    validation += training + "e";  
  if (test == "")
    test += training + "t";

  int niters        = get_option_int("niters", 6);  // Number of iterations
  svdpp.step_dec  =   get_option_float("svdpp_step_dec", 0.9);
  svdpp.itmBiasStep  =   get_option_float("svdpp_item_bias_step", 1e-3);
  svdpp.itmBiasReg =   get_option_float("svdpp_item_bias_reg", 1e-3);
  svdpp.usrBiasStep  =   get_option_float("svdpp_user_bias_step", 1e-3);
  svdpp.usrBiasReg  =   get_option_float("svdpp_user_bias_reg", 1e-3);
  svdpp.usrFctrStep  =   get_option_float("svdpp_user_factor_step", 1e-3);
  svdpp.usrFctrReg  =   get_option_float("svdpp_user_factor_reg", 1e-3);
  svdpp.itmFctr2Reg =   get_option_float("svdpp_user_factor2_reg", 1e-3);
  svdpp.itmFctr2Step =   get_option_float("svdpp_user_factor2_step", 1e-3);

  maxval            = get_option_float("maxval", 1e100);
  minval            = get_option_float("minval", -1e100);

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket_for_SVDPP<float>(training);

  /* Run */
  SVDPPVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  engine.set_modifies_inedges(false);
  engine.set_modifies_outedges(false);
  pengine = &engine;
  engine.run(program, niters);

  /* Output latent factor matrices in matrix-market format */
  vid_t numvertices = engine.num_vertices();
  assert(numvertices == max_right_vertex + 1); // Sanity check
  output_svdpp_result(training, numvertices, max_left_vertex);
  test_predictions();    


  /* Report execution metrics */
  metrics_report(m);
  return 0;
}

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
 * Matrix factorization with the Koren's SVD++ algorithm.
 * Algorithm described in the paper:
 * 
 */



#include "common.hpp"
#include "eigen_wrapper.hpp"

struct svdpp_params{
  float itmBiasStep;
  float itmBiasReg;
  float usrBiasStep;
  float usrBiasReg;
  float usrFctrStep;
  float usrFctrReg;
  float itmFctrStep;
  float itmFctrReg; //gamma7
  float itmFctr2Step;
  float itmFctr2Reg;
  float step_dec;

  svdpp_params(){
    itmBiasStep = 1e-4f;
    itmBiasReg = 1e-4f;
    usrBiasStep = 1e-4f;
    usrBiasReg = 2e-4f;
    usrFctrStep = 1e-4f;
    usrFctrReg = 2e-4f;
    itmFctrStep = 1e-4f;
    itmFctrReg = 1e-4f; //gamma7
    itmFctr2Step = 1e-4f;
    itmFctr2Reg = 1e-4f;
    step_dec = 0.9;
  }
};

svdpp_params svdpp;
#define BIAS_POS -1

struct vertex_data {
  vec pvec;
  vec weight;
  double bias;

  vertex_data() {
    pvec = zeros(D);
    weight = zeros(D);
    bias = 0;
  }
  void set_val(int index, float val){
    if (index == BIAS_POS)
      bias = val;
    else if (index < D)
      pvec[index] = val;
    else weight[index-D] = val;
  }
  float get_val(int index){
    if (index== BIAS_POS)
      return bias;
    else if (index < D)
      return pvec[index];
    else return weight[index-D];
  }
};



/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef float EdgeDataType;  // Edges store the "rating" of user->movie pair
    
graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
graphchi_engine<VertexDataType, EdgeDataType> * pvalidation_engine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

#include "io.hpp"
#include "rmse.hpp"
#include "rmse_engine.hpp"

/** compute a missing value based on SVD++ algorithm */
float svdpp_predict(const vertex_data& user, const vertex_data& movie, const float rating, double & prediction, void * extra = NULL){
  //\hat(r_ui) = \mu + 
  prediction = globalMean;
  // + b_u  +    b_i +
  prediction += user.bias + movie.bias;
  // + q_i^T   *(p_u      +sqrt(|N(u)|)\sum y_j)
  //prediction += dot_prod(movie.pvec,(user.pvec+user.weight));
  for (int j=0; j< D; j++)
    prediction += movie.pvec[j] * (user.pvec[j] + user.weight[j]);

  prediction = std::min((double)prediction, maxval);
  prediction = std::max((double)prediction, minval);
  float err = rating - prediction;
  if (std::isnan(err))
    logstream(LOG_FATAL)<<"Got into numerical errors. Try to decrease step size using the command line: svdpp_user_bias_step, svdpp_item_bias_step, svdpp_user_factor2_step, svdpp_user_factor_step, svdpp_item_step" << std::endl;
  return err*err; 
}


/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct SVDPPVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

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
    svdpp.itmFctrStep *= svdpp.step_dec;
    svdpp.itmFctr2Step *= svdpp.step_dec;
    svdpp.usrFctrStep *= svdpp.step_dec;
    svdpp.itmBiasStep *= svdpp.step_dec;
    svdpp.usrBiasStep *= svdpp.step_dec;

    training_rmse(iteration, gcontext);
    validation_rmse(&svdpp_predict, gcontext);
  }

  /**
   *  Vertex update function.
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
      if ( vertex.num_outedges() > 0){
        vertex_data & user = latent_factors_inmem[vertex.id()]; 

        memset(&user.weight[0], 0, sizeof(double)*D);
        for(int e=0; e < vertex.num_outedges(); e++) {
          vertex_data & movie = latent_factors_inmem[vertex.edge(e)->vertex_id()]; 
          user.weight += movie.weight;

        }
        // sqrt(|N(u)|) 
        float usrNorm = double(1.0/sqrt(vertex.num_outedges()));
        //sqrt(|N(u)| * sum_j y_j
        user.weight *= usrNorm;

        vec step = zeros(D);

        // main algorithm, see Koren's paper, just below below equation (16)
        for(int e=0; e < vertex.num_outedges(); e++) {
          vertex_data & movie = latent_factors_inmem[vertex.edge(e)->vertex_id()]; 
          float observation = vertex.edge(e)->get_data();                
          double estScore;
          rmse_vec[omp_get_thread_num()] += svdpp_predict(user, movie,observation, estScore); 
          // e_ui = r_ui - \hat{r_ui}
          float err = observation - estScore;
          assert(!std::isnan(rmse_vec[omp_get_thread_num()]));
          vec itmFctr = movie.pvec;
          vec usrFctr = user.pvec;

          //q_i = q_i + gamma2     *(e_ui*(p_u      +  sqrt(N(U))\sum_j y_j) - gamma7    *q_i)
          for (int j=0; j< D; j++)
            movie.pvec[j] += svdpp.itmFctrStep*(err*(usrFctr[j] +  user.weight[j])             - svdpp.itmFctrReg*itmFctr[j]);
          //p_u = p_u + gamma2    *(e_ui*q_i   -gamma7     *p_u)
          for (int j=0; j< D; j++)
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
          movie.weight +=  step                    -  mult  * movie.weight;
        }
      }
  }




};


void output_svdpp_result(std::string filename) {
  MMOutputter_mat<vertex_data> user_output(filename + "_U.mm", 0, M, "This file contains SVD++ output matrix U. In each row D factors of a single user node. Then additional D weight factors.", latent_factors_inmem, 2*D);
  MMOutputter_mat<vertex_data> item_output(filename + "_V.mm", M ,M+N, "This file contains SVD++ output matrix V. In each row D factors of a single item node.", latent_factors_inmem);
  MMOutputter_vec<vertex_data> bias_user_vec(filename + "_U_bias.mm", 0, M, BIAS_POS, "This file contains SVD++ output bias vector. In each row a single user bias.", latent_factors_inmem);
  MMOutputter_vec<vertex_data> bias_mov_vec(filename + "_V_bias.mm", M, M+N, BIAS_POS, "This file contains SVD++ output bias vector. In each row a single item bias.", latent_factors_inmem);
  MMOutputter_scalar gmean(filename + "_global_mean.mm", "This file contains SVD++ global mean which is required for computing predictions.", globalMean);

  logstream(LOG_INFO) << "SVDPP output files (in matrix market format): " << filename << "_U.mm" <<
                                                                             ", " << filename + "_V.mm, " << filename << "_U_bias.mm, " << filename << "_V_bias.mm, " << filename << "_global_mean.mm" << std::endl;
}

void svdpp_init(){
  srand48(time(NULL));
  latent_factors_inmem.resize(M+N);

#pragma omp parallel for
  for(int i = 0; i < (int)(M+N); ++i){
    vertex_data & data = latent_factors_inmem[i];
    data.pvec = zeros(D);
    if (i < (int)M) //user node
      data.weight = zeros(D);
    for (int j=0; j<D; j++)
      latent_factors_inmem[i].pvec[j] = drand48();
  }
  logstream(LOG_INFO) << "SVD++ initialization ok" << std::endl;

}

int main(int argc, const char ** argv) {

  print_copyright();

  //* GraphChi initialization will read the command line arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("svdpp-inmemory-factors");

  svdpp.step_dec  =   get_option_float("svdpp_step_dec", 0.9);
  svdpp.itmBiasStep  =   get_option_float("svdpp_item_bias_step", 1e-3);
  svdpp.itmBiasReg =   get_option_float("svdpp_item_bias_reg", 1e-3);
  svdpp.usrBiasStep  =   get_option_float("svdpp_user_bias_step", 1e-3);
  svdpp.usrBiasReg  =   get_option_float("svdpp_user_bias_reg", 1e-3);
  svdpp.usrFctrStep  =   get_option_float("svdpp_user_factor_step", 1e-3);
  svdpp.usrFctrReg  =   get_option_float("svdpp_user_factor_reg", 1e-3);
  svdpp.itmFctrReg =   get_option_float("svdpp_item_factor_reg", 1e-3);
  svdpp.itmFctrStep =   get_option_float("svdpp_item_factor_step", 1e-3);
  svdpp.itmFctr2Reg =   get_option_float("svdpp_item_factor2_reg", 1e-3);
  svdpp.itmFctr2Step =   get_option_float("svdpp_item_factor2_step", 1e-3);

  parse_command_line_args();
  parse_implicit_command_line();

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket<EdgeDataType>(training, 0, 0, 3, TRAINING, false);
  if (validation != ""){
    int vshards = convert_matrixmarket<EdgeDataType>(validation, 0, 0, 3, VALIDATION, false);
    init_validation_rmse_engine<VertexDataType, EdgeDataType>(pvalidation_engine, vshards, &svdpp_predict);
  }

  svdpp_init();

  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, 2*D);
    load_matrix_market_matrix(training + "_V.mm", M, D);
    load_matrix_market_vector(training + "_U_bias.mm", BIAS_POS, false, true, 0); 
    load_matrix_market_vector(training + "_V_bias.mm", BIAS_POS, false, true, M); 
    vec gm = load_matrix_market_vector(training + "_global_mean.mm", false, true);
    globalMean = gm[0];
 }

  /* Run */
  SVDPPVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output latent factor matrices in matrix-market format */
  output_svdpp_result(training);
  test_predictions(&svdpp_predict);    


  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  return 0;
}

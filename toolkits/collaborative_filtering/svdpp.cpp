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



#include "graphchi_basic_includes.hpp"
#include "common.hpp"
using namespace graphchi;


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

struct vertex_data {
  double pvec[NLATENT];
  double weight[NLATENT];
  double rmse;
  double bias;

  vertex_data() {
    for(int k=0; k < NLATENT; k++) {
      pvec[k] =  drand48(); 
      weight[k] = drand48();
    }
    rmse = 0;
    bias = 0;
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

#include "rmse.hpp"
#include "io.hpp"

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
    logstream(LOG_FATAL)<<"Got into numerical errors. Try to decrease step size using the command line: svdpp_user_bias_step, svdpp_item_bias_step, svdpp_user_factor2_step, svdpp_user_factor_step, svdpp_item_step" << std::endl;
  return err*err; 
}


/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct SVDPPVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {


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

        user.rmse = 0; 
        memset(user.weight, 0, sizeof(double)*NLATENT);
        for(int e=0; e < vertex.num_outedges(); e++) {
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
        for(int e=0; e < vertex.num_outedges(); e++) {
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




};

struct  MMOutputter_bias{
  FILE * outf;
  MMOutputter_bias(std::string fname, uint start, uint end, std::string comment)  {
    MM_typecode matcode;
    set_matcode(matcode);
    outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, end-start, 1); 
    for (uint i=start; i< end; i++)
       fprintf(outf, "%1.12e\n", latent_factors_inmem[i].bias);
  }


  ~MMOutputter_bias() {
    if (outf != NULL) fclose(outf);
  }

};

struct  MMOutputter_global_mean {
  FILE * outf;
  MMOutputter_global_mean(std::string fname, std::string comment)  {
    MM_typecode matcode;
    set_matcode(matcode);
    outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, 1, 1); 
    fprintf(outf, "%1.12e\n", globalMean);
  }

  ~MMOutputter_global_mean() {
    if (outf != NULL) fclose(outf);
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



void output_svdpp_result(std::string filename) {
  MMOutputter mmoutput_left(filename + "_U.mm", 0, M, "This file contains SVD++ output matrix U. In each row NLATENT factors of a single user node.");
  MMOutputter mmoutput_right(filename + "_V.mm", M ,M+N, "This file contains SVD++ output matrix V. In each row NLATENT factors of a single item node.");
  MMOutputter_bias mmoutput_bias_left(filename + "_U_bias.mm", 0, M, "This file contains SVD++ output bias vector. In each row a single user bias.");
  MMOutputter_bias mmoutput_bias_right(filename + "_V_bias.mm", M, M+N, "This file contains SVD++ output bias vector. In each row a single item bias.");
  MMOutputter_global_mean gmean(filename + "_global_mean.mm", "This file contains SVD++ global mean which is required for computing predictions.");

  logstream(LOG_INFO) << "SVDPP output files (in matrix market format): " << filename << "_U.mm" <<
                                                                             ", " << filename + "_V.mm, " << filename << "_U_bias.mm, " << filename << "_V_bias.mm, " << filename << "_global_mean.mm" << std::endl;
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
  int nshards = convert_matrixmarket<float>(training);
  latent_factors_inmem.resize(M+N); // Initialize in-memory vertices.
  assert(M > 0 && N > 0);

  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, NLATENT);
    load_matrix_market_matrix(training + "_V.mm", M, NLATENT);
    vec user_bias = load_matrix_market_vector(training +"_U_bias.mm", false, true);
    vec item_bias = load_matrix_market_vector(training +"_V_bias.mm", false, true);
    for (uint i=0; i<M+N; i++){
      latent_factors_inmem[i].bias = ((i<M)?user_bias[i] : item_bias[i-M]);
    }
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
  metrics_report(m);
  return 0;
}

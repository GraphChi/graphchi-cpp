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
 * This code implements the PMF (probablistic matrix factorization) algorithm
 * as explained in Liang Xiong et al SDM 2010 paper.
 * 
 */


#include "eigen_wrapper.hpp"
#include "common.hpp"
#include "prob.hpp"

double lambda = 0.065;
int pmf_burn_in = 10;//number of iterations for burn in (itermediate solutions are thrown)
int pmf_additional_output = 0;
int debug = 0;

/* variables for PMF */
double nuAlpha = 1;
double Walpha = 1;
double nu0 = D;
double alpha = 0;
double beta = 1;
vec beta0 = init_vec("1", 1);
//vec mu0T = init_vec("1", 1);
mat W0;
//mat W0T;
double iWalpha;
mat iW0;
//mat iW0T;
mat A_U, A_V;// A_T;
vec mu_U, mu_V; //, mu_T;
int iiter = 0;

vec validation_avgprod; //vector for storing temporary aggregated predictions for the MCMC method
vec test_avgprod; //vector for strogin temporary aggregated predictions for the MCMC method
size_t rmse_index = 0;
int rmse_type = 0;

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
   float weight;
   float avgprd;
   edge_data() { weight = 0; avgprd = 0; }
   edge_data(double weight): weight(weight) { avgprd = 0; }
};

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef edge_data EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

#include "io.hpp"
#include "rmse.hpp"

/** compute a missing value based on PMF algorithm */
float pmf_predict(const vertex_data& user, 
    const vertex_data& movie, 
    const float rating, 
    double & prediction,
    void * pedge){


  prediction = dot_prod(user.pvec, movie.pvec);
  //truncate prediction to allowed values
  prediction = std::min((double)prediction, maxval);
  prediction = std::max((double)prediction, minval);

  float err = 0; 
  if (iiter > pmf_burn_in){
     if (pedge){
       if (iiter == pmf_burn_in+1)
         (*(float*)pedge) = 0;
       (*(float*)pedge) += prediction;
       err = pow(((*(float*)pedge) / (iiter - pmf_burn_in)) - rating, 2);
     }
  }
  else {
     err = pow(prediction - rating,2);
  }
  assert(!std::isnan(err));
  if (!pedge) 
        rmse_index++;
  return err; 

}


void init_self_pot(){

  W0 = eye(D);
  //W0T = eye(D);
  iWalpha = 1.0/Walpha;
  iW0 = inv(W0);
  //iW0T = inv(W0T);
  nu0 = D;

  A_U = eye(D); //cov prior for users
  A_V = eye(D); //cov prior for movies
  //A_T = eye(D); //cov prior for time nodes

  mu_U = zeros(D); mu_V = zeros(D);// mu_T = zeros(D);
  //printf("nuAlpha=%g, Walpha=%g, mu0=%d, muT=%g, nu=%g, "
  //       "beta=%g, W=%g, WT=%g pmf_burn_in=%d\n", nuAlpha, Walpha, 0, 
  //       mu0T[0], nu0, beta0[0], W0(1,1), W0T(1,1), pmf_burn_in);


  //test_randn(); 
  //test_wishrnd();
  //test_wishrnd2(); 
  //test_chi2rnd();
  //test_wishrnd3();
  //test_mvnrndex();
}

/**
 * sample the noise level 
 * Euqation A.2 in Xiong paper
 */
void sample_alpha(double res2){
  
  if (debug)
  printf("res is %g\n", res2); 
  
  double res = res2;
  if (nuAlpha > 0){
    double nuAlpha_ =nuAlpha+ L;
    mat iWalpha_(1,1);
    set_val(iWalpha_, 0,0,iWalpha + res);
    mat iiWalpha_ = zeros(1,1);
    iiWalpha_ = inv(iWalpha_);
    alpha = get_val(wishrnd(iiWalpha_, nuAlpha_),0,0);
    assert(alpha != 0);

    if (debug)
      std::cout<<"Sampling from alpha" <<nuAlpha_<<" "<<iWalpha<<" "<< iiWalpha_<<" "<<alpha<<endl;
    //printf("sampled alpha is %g\n", alpha); 
  }
}

mat calc_MMT(int start_pos, int end_pos, vec &Umean){

  int batchSize = 1000;
  mat U = zeros(batchSize,D);
  mat MMT = zeros(D,D);
  int cnt = 0;

  for (int i=start_pos; i< end_pos; i++){
    if ((i-start_pos) % batchSize == 0){
      U=zeros(batchSize, D);
      cnt = 1;
    }

    const vertex_data * data= &latent_factors_inmem[i];
    vec mean = data->pvec;
    Umean += mean;
    for (int s=0; s<D; s++)
      U(i%batchSize,s)=mean(s);
    if (debug && (i==start_pos || i == end_pos-1))
      std::cout<<" clmn "<<i<< " vec: " << mean <<std::endl;

    if ((cnt  == batchSize) || (cnt < batchSize && i == end_pos-1)){
      MMT = MMT+transpose(U)*U;
    }
    cnt++;
  }
  Umean /= (end_pos-start_pos);
  if (debug)
    cout<<"mean: "<<Umean<<endl;

  assert(MMT.rows() == D && MMT.cols() == D);
  assert(Umean.size() == D);
  return MMT;
}


// sample movie nodes hyperprior
// according to equation A.3 in Xiong paper.
void sample_U(){

  vec Umean = zeros(D);
  mat UUT = calc_MMT(0,M,Umean);
  
  double beta0_ = beta0[0] + M;
  vec mu0_ = (M*Umean)/beta0_;
  double nu0_ = nu0 +M;
  vec dMu = - Umean;
  if (debug)
    std::cout<<"dMu:"<<dMu<<"beta0: "<<beta0[0]<<" beta0_ "<<beta0_<<" nu0_ " <<nu0_<<" mu0_ " << mu0_<<endl;

  mat UmeanT = M*outer_product(Umean, Umean); 
  assert(UmeanT.rows() == D && UmeanT.cols() == D);
  mat dMuT = (beta0[0]/beta0_)*UmeanT;
  mat iW0_ = iW0 + UUT - UmeanT + dMuT;
  mat W0_; 
  bool ret =inv(iW0_, W0_);
  assert(ret);
  mat tmp = (W0_+transpose(W0_))*0.5;
  if (debug)
    std::cout<<iW0<<UUT<<UmeanT<<dMuT<<W0_<<tmp<<nu0_<<endl;
  A_U = wishrnd(tmp, nu0_);
  mat tmp2;  
  ret =  inv(beta0_ * A_U, tmp2);
  assert(ret);
  mu_U = mvnrndex(mu0_, tmp2, D, 0);
  if (debug)
    std::cout<<"Sampling from U" <<A_U<<" "<<mu_U<<" "<<Umean<<" "<<W0_<<tmp<<endl;
}

// sample user nodes hyperprior
// according to equation A.4 in Xiong paper
void sample_V(){

  vec Vmean = zeros(D);
  mat VVT = calc_MMT(M, M+N, Vmean);   

  double beta0_ = beta0[0] + N;
  vec mu0_ = (N*Vmean)/beta0_;
  double nu0_ = nu0 +N;
  vec dMu = - Vmean;
  if (debug)
    std::cout<<"dMu:"<<dMu<<"beta0: "<<beta0[0]<<" beta0_ "<<beta0_<<" nu0_ " <<nu0_<<endl;
  mat VmeanT = N*outer_product(Vmean, Vmean); 
  assert(VmeanT.rows() == D && VmeanT.cols() == D);
  mat dMuT =  (beta0[0]/beta0_)*VmeanT;
  mat iW0_ = iW0 + VVT - VmeanT + dMuT;
  mat W0_;
  bool ret = inv(iW0_, W0_);
  assert(ret);
  mat tmp = (W0_+transpose(W0_))*0.5;
  if (debug)
    std::cout<<"iW0: "<<iW0<<" VVT: "<<VVT<<" VmeanT: "<<VmeanT<<" dMuT: " <<dMuT<<"W0_"<< W0_<<" tmp: " << tmp<<" nu0_: "<<nu0_<<endl;
  A_V = wishrnd(tmp, nu0_);
  mat tmp2; 
  ret = inv(beta0_*A_V, tmp2);
  assert(ret);
  mu_V = mvnrndex(mu0_, tmp2, D, 0);
  if (debug)
    std::cout<<"Sampling from V: A_V" <<A_V<<" mu_V: "<<mu_V<<" Vmean: "<<Vmean<<" W0_: "<<W0_<<" tmp: "<<tmp<<endl;
}



void sample_hyperpriors(double res){
    sample_alpha(res);
    sample_U();
    sample_V();
    //if (tensor) 
    //  sample_T();
}

void output_pmf_result(std::string filename) {
  MMOutputter_mat<vertex_data> user_mat(filename + "_U.mm", 0, M , "This file contains PMF output matrix U. In each row D factors of a single user node.", latent_factors_inmem);
  MMOutputter_mat<vertex_data> item_mat(filename + "_V.mm", M  ,M+N, "This file contains PMF  output matrix V. In each row D factors of a single item node.", latent_factors_inmem);
  logstream(LOG_INFO) << "PMF output files (in matrix market format): " << filename << "_U.mm" <<
                                                                           ", " << filename + "_V.mm " << std::endl;
}



/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct PMFVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

  /**
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
    vertex_data & vdata = latent_factors_inmem[vertex.id()];
    bool isuser = vertex.id() < M;
    mat XtX = mat::Zero(D, D); 
    vec Xty = vec::Zero(D);

    bool compute_rmse = (vertex.num_outedges() > 0);
    // Compute XtX and Xty (NOTE: unweighted)
    for(int e=0; e < vertex.num_edges(); e++) {
      const edge_data & edge = vertex.edge(e)->get_data();
      float observation = edge.weight;                
      vertex_data & nbr_latent = latent_factors_inmem[vertex.edge(e)->vertex_id()];
      Xty += nbr_latent.pvec * observation;
      XtX.triangularView<Eigen::Upper>() += nbr_latent.pvec * nbr_latent.pvec.transpose();
      if (compute_rmse) {
        double prediction;
        rmse_vec[omp_get_thread_num()] += pmf_predict(vdata, nbr_latent, observation, prediction, (void*)&edge.avgprd);
        vertex.edge(e)->set_data(edge);
      }
    }

    double regularization = lambda;
    if (regnormal)
      lambda *= vertex.num_edges();
    for(int i=0; i < D; i++) XtX(i,i) += regularization;

    // Solve the least squares problem with eigen using Cholesky decomposition
    mat iAi_;
    bool ret =inv((isuser? A_U : A_V) + alpha *  XtX, iAi_);
    assert(ret);
    vec mui_ =  iAi_*((isuser? (A_U*mu_U) : (A_V*mu_V)) + alpha * Xty); 
    vdata.pvec = mvnrndex(mui_, iAi_, D, 0); 
    assert(vdata.pvec.size() == D);
 }


  /**
   * Called before an iteration is started.
   */
  void before_iteration(int iteration, graphchi_context &gcontext) {
    rmse_vec = zeros(gcontext.execthreads);
  }



  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    if (iteration == pmf_burn_in){
      printf("Finished burn-in period. starting to aggregate samples\n");
    }
    if (pmf_additional_output && iiter >= pmf_burn_in){
        char buf[256];
        sprintf(buf, "%s-%d", training.c_str(), iiter-pmf_burn_in);
        output_pmf_result(buf);
    }
 
    double res = training_rmse(iteration, gcontext);
    sample_hyperpriors(res);
    rmse_index = 0;
    rmse_type = VALIDATION;
    validation_rmse(&pmf_predict, gcontext, 3, &validation_avgprod, pmf_burn_in);
    if (iteration >= pmf_burn_in){
      rmse_index = 0;
      rmse_type = TEST;
      test_predictions(&pmf_predict, &gcontext, iiter == niters-1, &test_avgprod, pmf_burn_in);
    }
    iiter++;
  }


};



void init_pmf(){
  init_self_pot();
}


int main(int argc, const char ** argv) {

  print_copyright();
 
  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("pmf-inmemory-factors");

  lambda        = get_option_float("lambda", 0.065);
  debug        = get_option_int("debug", debug);
  pmf_burn_in  = get_option_int("pmf_burn_in", pmf_burn_in);
  pmf_additional_output = get_option_int("pmf_additional_output", pmf_additional_output);
  
  parse_command_line_args();
  parse_implicit_command_line();


  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket<edge_data>(training, 0, 0, 3, TRAINING, false);
  init_feature_vectors<std::vector<vertex_data> >(M+N, latent_factors_inmem, !load_factors_from_file);
  init_pmf();

  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, D);
    load_matrix_market_matrix(training + "_V.mm", M, D);
  }

  /* Run */
  PMFVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine, true);
  pengine = &engine;
  engine.run(program, niters);

  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);

  return 0;
}

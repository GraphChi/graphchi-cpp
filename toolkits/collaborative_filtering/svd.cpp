/**  
 * Copyright (c) 2009 Carnegie Mellon University. 
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://graphchi.org
 *
 * Written by Danny Bickson
 *
 */


#include "common.hpp"
#include "types.hpp"
#include "eigen_wrapper.hpp"
#include "timer.hpp"
using namespace std;

#define GRAPHCHI_DISABLE_COMPRESSION
int nshards;
int nconv = 0;
/* Metrics object for keeping track of performance counters
     and other information. Currently required. */
metrics m("svd-inmemory-factors");
vec singular_values;

struct vertex_data {
  vec pvec;
  double value;
  double A_ii;
  vertex_data(){ value = 0; A_ii = 1; }

  void set_val(int field_type, double value) { 
    pvec[field_type] = value;
  }
  double get_val(int field_type){ return pvec[field_type]; }
  //double get_output(int field_type){ return pred_x; }
}; // end of vertex_data


/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef float EdgeDataType; 

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;


#include "io.hpp"
#include "rmse.hpp"
#include "rmse_engine.hpp"

/** compute a missing value based on SVD algorithm */
float svd_predict(const vertex_data& user, 
    const vertex_data& movie, 
    const float rating, 
    double & prediction, 
    void * extra = NULL){

  Eigen::DiagonalMatrix<double, Eigen::Dynamic> diagonal_matrix(nconv);      
  diagonal_matrix.diagonal() = singular_values;

  prediction = user.pvec.head(nconv).transpose() * diagonal_matrix * movie.pvec.head(nconv);
  //truncate prediction to allowed values
  prediction = std::min((double)prediction, maxval);
  prediction = std::max((double)prediction, minval);
  //return the squared error
  float err = rating - prediction;
  assert(!std::isnan(err));
  return err*err; 

}



/**
 *
 *  Implementation of the Lanczos algorithm, as given in:
 *  http://en.wikipedia.org/wiki/Lanczos_algorithm
 * 
 *  Code written by Danny Bickson, CMU, June 2011
 * */



//LANCZOS VARIABLES
int max_iter = 10;
int actual_vector_len;
int nv = 0;
int nsv = 0;
double tol = 1e-8;
bool finished = false;
int ortho_repeats = 3;
bool save_vectors = false;
std::string format = "matrixmarket";
int nodes = 0;

int data_size = max_iter;
#include "math.hpp"
#include "printouts.hpp"

void init_lanczos(bipartite_graph_descriptor & info){
  srand48(time(NULL));
  latent_factors_inmem.resize(info.total());
  data_size = nsv + nv+1 + max_iter;
  if (info.is_square())
    data_size *= 2;
  actual_vector_len = data_size;
#pragma omp parallel for
  for (int i=0; i< info.total(); i++){
      latent_factors_inmem[i].pvec = zeros(actual_vector_len);
  } 
  logstream(LOG_INFO)<<"Allocated a total of: " << ((double)actual_vector_len * info.total() * sizeof(double)/ 1e6) << " MB for storing vectors." << std::endl;
}

void output_svd_result(std::string filename) {
  MMOutputter_mat<vertex_data> user_mat(filename + "_U.mm", 0, M , "This file contains SVD output matrix U. In each row nconv factors of a single user node.", latent_factors_inmem, nconv);
  MMOutputter_mat<vertex_data> item_mat(filename + "_V.mm", M  ,M+N, "This file contains SVD  output matrix V. In each row nconv factors of a single item node.", latent_factors_inmem, nconv);
  logstream(LOG_INFO) << "SVD output files (in matrix market format): " << filename << "_U.mm" <<
                                                                           ", " << filename + "_V.mm " << std::endl;
}


vec lanczos( bipartite_graph_descriptor & info, timer & mytimer, vec & errest, 
            const std::string & vecfile){
   

 

   int its = 1;
   DistMat A(info);
   DistSlicedMat U(info.is_square() ? data_size/2 : 0, info.is_square() ? data_size : data_size, true, info, "U");
   DistSlicedMat V(0, data_size, false, info, "V");
   vec alpha, beta, b;
   vec sigma = zeros(data_size);
   errest = zeros(nv);
   DistVec v_0(info, 0, false, "v_0");
   if (vecfile.size() == 0)
     v_0 = randu(size(A,2));
   PRINT_VEC2("svd->V", v_0);
   
  DistDouble vnorm = norm(v_0);
  v_0=v_0/vnorm;
  PRINT_INT(nv);

  while(nconv < nsv && its < max_iter){
    std::cout<<"Starting iteration: " << its << " at time: " << mytimer.current_time() << std::endl;
    int k = nconv;
    int n = nv;
    PRINT_INT(k);
    PRINT_INT(n);

    alpha = zeros(n);
    beta = zeros(n);

    U[k] = V[k]*A._transpose();
    PRINT_VEC(U[k]);
    orthogonalize_vs_all(U, k, alpha(0));
    PRINT_VEC(U[k]);
    PRINT_VEC3("alpha", alpha, 0);

    for (int i=k+1; i<n; i++){
      std::cout <<"Starting step: " << i << " at time: " << mytimer.current_time() <<  std::endl;
      PRINT_INT(i);

      V[i]=U[i-1]*A;
      PRINT_VEC(V[i]);
      orthogonalize_vs_all(V, i, beta(i-k-1));
      PRINT_VEC(V[i]);
      
      PRINT_VEC3("beta", beta, i-k-1); 
      
      U[i] = V[i]*A._transpose();
      orthogonalize_vs_all(U, i, alpha(i-k));
      PRINT_VEC3("alpha", alpha, i-k);
     }

    V[n]= U[n-1]*A;
    orthogonalize_vs_all(V, n, beta(n-k-1));
    PRINT_VEC3("beta", beta, n-k-1);

  //compute svd of bidiagonal matrix
  
  PRINT_INT(nv);
  PRINT_NAMED_INT("svd->nconv", nconv);
  n = nv - nconv;
  PRINT_INT(n);
  alpha.conservativeResize(n);
  beta.conservativeResize(n);

  PRINT_MAT2("Q",eye(n));
  PRINT_MAT2("PT",eye(n));
  PRINT_VEC2("alpha",alpha);
  PRINT_VEC2("beta",beta);
 
  mat T=diag(alpha);
  for (int i=0; i<n-1; i++)
    set_val(T, i, i+1, beta(i));
  PRINT_MAT2("T", T);
  mat a,PT;
  svd(T, a, PT, b);
  PRINT_MAT2("Q", a);
  alpha=b.transpose();
  PRINT_MAT2("alpha", alpha);
  for (int t=0; t< n-1; t++)
     beta(t) = 0;
  PRINT_VEC2("beta",beta);
  PRINT_MAT2("PT", PT.transpose());

  
   //estiamte the error
  
  int kk = 0;
  for (int i=nconv; i < nv; i++){
    int j = i-nconv;
    PRINT_INT(j);
    sigma(i) = alpha(j);
    PRINT_NAMED_DBL("svd->sigma[i]", sigma(i));
    PRINT_NAMED_DBL("Q[j*n+n-1]",a(n-1,j));
    PRINT_NAMED_DBL("beta[n-1]",beta(n-1));
    errest(i) = abs(a(n-1,j)*beta(n-1));
    PRINT_NAMED_DBL("svd->errest[i]", errest(i));
    if (alpha(j) >  tol){
      errest(i) = errest(i) / alpha(j);
      PRINT_NAMED_DBL("svd->errest[i]", errest(i));
    }
    if (errest(i) < tol){
      kk = kk+1;
      PRINT_NAMED_INT("k",kk);
    }


    if (nconv +kk >= nsv){
      printf("set status to tol\n");
      finished = true;
    }
  }//end for
  PRINT_NAMED_INT("k",kk);


  vec v;
  if (!finished){
    
    vec swork=get_col(PT,kk); 
    PRINT_MAT2("swork", swork);
    v = zeros(size(A,1));
    for (int ttt=nconv; ttt < nconv+n; ttt++){
      v = v+swork(ttt-nconv)*(V[ttt].to_vec());
    }
    PRINT_VEC2("svd->V",V[nconv]);
    PRINT_VEC2("v[0]",v); 
  }


   
  //compute the ritz eigenvectors of the converged singular triplets
  if (kk > 0){
  PRINT_VEC2("svd->V", V[nconv]);
    
    mat tmp= V.get_cols(nconv,nconv+n)*PT;
    V.set_cols(nconv, nconv+kk, get_cols(tmp, 0, kk));
    PRINT_VEC2("svd->V", V[nconv]);
    PRINT_VEC2("svd->U", U[nconv]);
    tmp= U.get_cols(nconv, nconv+n)*a;
    U.set_cols(nconv, nconv+kk,get_cols(tmp,0,kk));
    PRINT_VEC2("svd->U", U[nconv]);
  }

  





  
  nconv=nconv+kk;
  if (finished)
    break;

  V[nconv]=v;
  PRINT_VEC2("svd->V", V[nconv]);
  PRINT_NAMED_INT("svd->nconv", nconv);

  its++;
  PRINT_NAMED_INT("svd->its", its);
  PRINT_NAMED_INT("svd->nconv", nconv);
  PRINT_NAMED_INT("nv",nv);

} // end(while)

printf(" Number of computed signular values %d",nconv);
printf("\n");
  DistVec normret(info, nconv, false, "normret");
  DistVec normret_tranpose(info, nconv, true, "normret_tranpose");
  

  for (int i=0; i < std::min(nsv,nconv); i++){
    normret = V[i]*A._transpose() -U[i]*sigma(i);
    double n1 = norm(normret).toDouble();
    PRINT_DBL(n1);
    normret_tranpose = U[i]*A -V[i]*sigma(i);
    double n2 = norm(normret_tranpose).toDouble();
    PRINT_DBL(n2);
    double err=sqrt(n1*n1+n2*n2);
    PRINT_DBL(err);
    PRINT_DBL(tol);
    if (sigma(i)>tol){
      err = err/sigma(i);
    }
    PRINT_DBL(err);
    PRINT_DBL(sigma(i));
    printf("Singular value %d \t%13.6g\tError estimate: %13.6g\n", i, sigma(i),err);
  }

  if (save_vectors){
     std::cout<<"Going to save output vectors U and V" << std::endl;
     if (nconv == 0)
       logstream(LOG_FATAL)<<"No converged vectors. Aborting the save operation" << std::endl;
  
     output_svd_result(training);   
  }
  return sigma;
}

int main(int argc,  const char *argv[]) {
 
  print_copyright();

  //* GraphChi initialization will read the command line arguments and the configuration file. */
  graphchi_init(argc, argv);

    

  std::string vecfile;

  vecfile       = get_option_string("initial_vector", "");
  debug         = get_option_int("debug", 0);
  ortho_repeats = get_option_int("ortho_repeats", 3); 
  nv = get_option_int("nv", 1);
  nsv = get_option_int("nsv", 1);
  tol = get_option_float("tol", 1e-5);
  save_vectors = get_option_int("save_vectors", 1);
  max_iter = get_option_int("max_iter", max_iter);

  parse_command_line_args();
  parse_implicit_command_line();

  if (nv < nsv){
    logstream(LOG_FATAL)<<"Please set the number of vectors --nv=XX, to be at least the number of support vectors --nsv=XX or larger" << std::endl;
  }


  //unit testing
  if (unittest == 1){
    training = "gklanczos_testA"; 
    vecfile = "gklanczos_testA_v0";
    nsv = 3; nv = 3;
    debug = true;
    //TODO core.set_ncpus(1);
  }
  else if (unittest == 2){
    training = "gklanczos_testB";
    vecfile = "gklanczos_testB_v0";
    nsv = 10; nv = 10;
    debug = true;  max_iter = 100;
    //TODO core.set_ncpus(1);
  }
  else if (unittest == 3){
    training = "gklanczos_testC";
    vecfile = "gklanczos_testC_v0";
    nsv = 25; nv = 25;
    debug = true;  max_iter = 100;
    //TODO core.set_ncpus(1);
  }
  else if (unittest == 4){
    training = "A2";
    vecfile = "A2_v0";
    nsv=3; nv = 4; 
    debug=true; max_iter=3;
  }

  std::cout << "Load matrix " << training << std::endl;
  /* Preprocess data if needed, or discover preprocess files */
  if (tokens_per_row == 3 || tokens_per_row == 2)
    nshards = convert_matrixmarket<EdgeDataType>(training,0,0,tokens_per_row);
  //else if (tokens_per_row == 4)
  //  nshards = convert_matrixmarket4<EdgeDataType>(training);
  else logstream(LOG_FATAL)<<"--tokens_per_row=XX should be either 2 or 3 input columns" << std::endl;

 
  info.rows = M; info.cols = N; info.nonzeros = L;
  assert(info.rows > 0 && info.cols > 0 && info.nonzeros > 0);

  timer mytimer; mytimer.start();
  init_lanczos(info);
  init_math(info, ortho_repeats);

  //read initial vector from file (optional)
  if (vecfile.size() > 0){
    std::cout << "Load inital vector from file" << vecfile << std::endl;
    load_matrix_market_vector(vecfile, 0, true, false);
  }  

  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;   

  vec errest;
  singular_values = lanczos(info, mytimer, errest, vecfile);
  singular_values.conservativeResize(nconv); 
  std::cout << "Lanczos finished " << mytimer.current_time() << std::endl;

  write_output_vector(training + ".singular_values", singular_values,false, "%GraphLab SVD Solver library. This file contains the singular values.");

  if (unittest == 1){
    assert(errest.size() == 3);
    for (int i=0; i< errest.size(); i++)
      assert(errest[i] < 1e-30);
  }
  else if (unittest == 2){
     assert(errest.size() == 10);
    for (int i=0; i< errest.size(); i++)
      assert(errest[i] < 1e-15);
  }
  else if (unittest == 4){
    assert(pow(singular_values[0]-  2.16097, 2) < 1e-8);
    assert(pow(singular_values[2]-  0.554159, 2) < 1e-8);
   }
 
  if (validation != ""){
    int vshards = convert_matrixmarket<EdgeDataType>(validation, 0, 0, 3, VALIDATION, false);
    graphchi_engine<VertexDataType, EdgeDataType> * pvalidation_engine = NULL;    
    init_validation_rmse_engine<VertexDataType, EdgeDataType>(pvalidation_engine, vshards, &svd_predict);
    ValidationRMSEProgram program;
    pvalidation_engine->run(program, 1);
  }
  
  test_predictions(&svd_predict);    

  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
 
  return 0;
}



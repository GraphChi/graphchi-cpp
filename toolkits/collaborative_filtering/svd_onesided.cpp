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
metrics m("svd-onesided-inmemory-factors");
vec singular_values;

struct vertex_data {
  vec pvec;
  double value;
  double A_ii;
  vertex_data(){ value = 0; A_ii = 1; }

  void set_val(int field_type, double value) { 
    pvec[field_type] = value;
  }
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
void reset_rmse(int){}


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
     if (i < info.get_start_node(false) || info.is_square())
      latent_factors_inmem[i].pvec = zeros(actual_vector_len);
     else latent_factors_inmem[i].pvec = zeros(3);
   }
   logstream(LOG_INFO)<<"Allocated a total of: " << 
     ((double)(data_size * info.num_nodes(true) +3.0*info.num_nodes(false)) * sizeof(double)/ 1e6) << " MB for storing vectors." << " rows: " << info.num_nodes(true) << std::endl;
}

vec one_sided_lanczos( bipartite_graph_descriptor & info, timer & mytimer, vec & errest, 
            const std::string & vecfile){
   

   int its = 1;
   DistMat A(info);
   int other_size_offset = info.is_square() ? data_size : 0;
   DistSlicedMat U(other_size_offset, other_size_offset + 3, true, info, "U");
   DistSlicedMat V(0, data_size, false, info, "V");
   DistVec v(info, 1, false, "v");
   DistVec u(info, other_size_offset+ 0, true, "u");
   DistVec u_1(info, other_size_offset+ 1, true, "u_1");
   DistVec tmp(info, other_size_offset + 2, true, "tmp");
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
     PRINT_VEC2("v", v);
     PRINT_VEC2("u", u);

     alpha = zeros(n);
     beta = zeros(n);

     u = V[k]*A._transpose();
     PRINT_VEC2("u",u);

     for (int i=k+1; i<n; i++){
       std::cout <<"Starting step: " << i << " at time: " << mytimer.current_time() << std::endl;
       PRINT_INT(i);

       V[i]=u*A;
       double a = norm(u).toDouble();
       u = u / a;
       multiply(V, i, a);
       PRINT_DBL(a);     
 
       double b;
       orthogonalize_vs_all(V, i, b);
       PRINT_DBL(b);
       u_1 = V[i]*A._transpose();  
       u_1 = u_1 - u*b;
       alpha(i-k-1) = a;
       beta(i-k-1) = b;
       PRINT_VEC3("alpha", alpha, i-k-1);
       PRINT_VEC3("beta", beta, i-k-1);
       tmp = u;
       u = u_1;
       u_1 = tmp;
     }

     V[n]= u*A;
     double a = norm(u).toDouble();
     PRINT_DBL(a);
     u = u/a;
     double b;
     multiply(V, n, a);
     orthogonalize_vs_all(V, n, b);
     alpha(n-k-1)= a;
     beta(n-k-1) = b;
     PRINT_VEC3("alpha", alpha, n-k-1);
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
  mat aa,PT;
  vec bb;
  svd(T, aa, PT, bb);
  PRINT_MAT2("Q", aa);
  alpha=bb.transpose();
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
    PRINT_NAMED_DBL("Q[j*n+n-1]",aa(n-1,j));
    PRINT_NAMED_DBL("beta[n-1]",beta(n-1));
    errest(i) = abs(aa(n-1,j)*beta(n-1));
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
  //nv = min(nconv+mpd, N);
  //if (nsv < 10)
  //  nv = 10;
  PRINT_NAMED_INT("nv",nv);

} // end(while)

printf(" Number of computed signular values %d",nconv);
printf("\n");
  DistVec normret(info, other_size_offset + 1, true, "normret");
  DistVec normret_tranpose(info, nconv, false, "normret_tranpose");
  for (int i=0; i < nconv; i++){
    u = V[i]*A._transpose();
    double a = norm(u).toDouble();
    u = u / a;
    if (save_vectors){
       char output_filename[256];
       sprintf(output_filename, "%s.U.%d", training.c_str(), i);
        write_output_vector(output_filename, u.to_vec(), false, "GraphLab v2 SVD output. This file contains eigenvector number i of the matrix U");
    }
    normret = V[i]*A._transpose() - u*sigma(i);
    double n1 = norm(normret).toDouble();
    PRINT_DBL(n1);
    normret_tranpose = u*A -V[i]*sigma(i);
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
     std::cout<<"Going to save output vectors V" << std::endl;
    if (nconv == 0)
       logstream(LOG_FATAL)<<"No converged vectors. Aborting the save operation" << std::endl;
    char output_filename[256];
    for (int i=0; i< nconv; i++){
        sprintf(output_filename, "%s.V.%d", training.c_str(), i);
        write_output_vector(output_filename, V[i].to_vec(), false, "GraphLab v2 SVD output. This file contains eigenvector number i of the matrix V'");
     }
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
    logstream(LOG_FATAL)<<"Please set the number of vectors --nv=XX, to be greater than the number of support vectors --nsv=XX " << std::endl;
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

  std::cout << "Load matrix " << training << std::endl;
  /* Preprocess data if needed, or discover preprocess files */
  if (tokens_per_row == 3 || tokens_per_row == 2)
    nshards = convert_matrixmarket<EdgeDataType>(training,0,0,tokens_per_row);

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
  singular_values = one_sided_lanczos(info, mytimer, errest, vecfile);
  singular_values.conservativeResize(nconv); 
  std::cout << "Lanczos finished in " << mytimer.current_time() << std::endl;

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

  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  return 0;
}



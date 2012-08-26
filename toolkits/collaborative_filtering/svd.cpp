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


#include "graphchi_basic_includes.hpp"
#include "../../example_apps/matrix_factorization/matrixmarket/mmio.h"
#include "../../example_apps/matrix_factorization/matrixmarket/mmio.c"

#include "api/chifilenames.hpp"
#include "api/vertex_aggregator.hpp"
#include "preprocessing/sharder.hpp"

#include "types.hpp"
#include "eigen_wrapper.hpp"

using namespace graphchi;
using namespace std;

#ifndef NLATENT
#define NLATENT 20   // Dimension of the latent factors. You can specify this in compile time as well (in make).
#endif

double minval = -1e100;
double maxval = 1e100;
std::string training;
std::string validation;
std::string test;
uint M, N, Me, Ne, Le;
size_t L;
double globalMean = 0;
int nshards;
vid_t max_left_vertex =0 ;
vid_t max_right_vertex = 0;
/* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("svd-inmemory-factors");


struct vertex_data {
  double pvec[NLATENT];
  double value;
  double A_ii;
  vertex_data(){ memset(pvec, 0, NLATENT*sizeof(double)); value = 0; A_ii = 1; }
  //TODO void add_self_edge(double value) { A_ii = value; }

  void set_val(double value, int field_type) { 
    pvec[field_type] = value;
  }
  //double get_output(int field_type){ return pred_x; }
}; // end of vertex_data

struct edge_data {
  float weight;
  edge_data(double weight = 0) : weight(weight) { }
  //void set_field(int pos, double val){ weight = val; }
  //double get_field(int pos){ return weight; }
};



#include "io.hpp"


/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef edge_data EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

//#include "rmse.hpp"


/**
 *
 *  Implementation of the Lanczos algorithm, as given in:
 *  http://en.wikipedia.org/wiki/Lanczos_algorithm
 * 
 *  Code written by Danny Bickson, CMU, June 2011
 * */



//LANCZOS VARIABLES
int max_iter = 10;
bool no_edge_data = false;
int actual_vector_len;
int nv = 0;
int nsv = 0;
double tol = 1e-8;
bool finished = false;
double ortho_repeats = 3;
bool update_function = false;
bool save_vectors = false;
std::string datafile; 
std::string format = "matrixmarket";
int nodes = 0;

int data_size = max_iter;
#include "math.hpp"
#include "printouts.hpp"

Axb program; //the update function instance

void init_lanczos(bipartite_graph_descriptor & info){

  data_size = nsv + nv+1 + max_iter;
  actual_vector_len = data_size;
  if (info.is_square())
     actual_vector_len = 2*data_size;

  logstream(LOG_INFO)<<"Allocated a total of: " << ((double)actual_vector_len * info.total() * sizeof(double)/ 1e6) << " MB for storing vectors." << std::endl;
}
/*
 * open a file and verify open success
 */
FILE * open_file(const char * name, const char * mode, bool optional = false){
  FILE * f = fopen(name, mode);
  if (f == NULL && !optional){
      perror("fopen failed");
      logstream(LOG_FATAL) <<" Failed to open file" << name << std::endl;
   }
  return f;
}


void load_matrix_market_vector(const std::string & filename, const bipartite_graph_descriptor & desc, 
    int type, bool optional_field, bool allow_zeros)
{
    
    int ret_code;
    MM_typecode matcode;
    uint M, N; 
    size_t i,nz;  

    logstream(LOG_INFO) <<"Going to read matrix market vector from input file: " << filename << std::endl;
  
    FILE * f = open_file(filename.c_str(), "r", optional_field);
    //if optional file not found return
    if (f== NULL && optional_field){
       return;
    }

    if (mm_read_banner(f, &matcode) != 0)
        logstream(LOG_FATAL) << "Could not process Matrix Market banner." << std::endl;

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
        logstream(LOG_FATAL) << "sorry, this application does not support " << std::endl << 
          "Market Market type: " << mm_typecode_to_str(matcode) << std::endl;

    /* find out size of sparse matrix .... */
    if (mm_is_sparse(matcode)){
       if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
          logstream(LOG_FATAL) << "failed to read matrix market cardinality size " << std::endl; 
    }
    else {
      if ((ret_code = mm_read_mtx_array_size(f, &M, &N))!= 0)
          logstream(LOG_FATAL) << "failed to read matrix market vector size " << std::endl; 
         if (N > M){ //if this is a row vector, transpose
           int tmp = N;
           N = M;
           M = tmp;
         }
         nz = M*N;
    }


    uint row,col; 
    double val;

    for (i=0; i<nz; i++)
    {
        if (mm_is_sparse(matcode)){
          int rc = fscanf(f, "%u %u %lg\n", &row, &col, &val);
          if (rc != 3){
	    logstream(LOG_FATAL) << "Failed reading input file: " << filename << "Problm at data row " << i << " (not including header and comment lines)" << std::endl;
          }
          row--;  /* adjust from 1-based to 0-based */
          col--;
        }
        else {
	  int rc = fscanf(f, "%lg\n", &val);
          if (rc != 1){
	    logstream(LOG_FATAL) << "Failed reading input file: " << filename << "Problm at data row " << i << " (not including header and comment lines)" << std::endl;
          }
          row = i;
          col = 0;
        }
       //some users have gibrish in text file - better check both I and J are >=0 as well
        assert(row >=0 && row< M);
        assert(col == 0);
        if (val == 0 && !allow_zeros)
           logstream(LOG_FATAL)<<"Zero entries are not allowed in a sparse matrix market vector. Use --zero=true to avoid this error"<<std::endl;
        //set observation value
        vertex_data & vdata = latent_factors_inmem[row];
        vdata.pvec[type] = val;
    }
    fclose(f);

}



inline void write_row(int row, int col, double val, FILE * f, bool issparse){
    if (issparse)
      fprintf(f, "%d %d %10.13g\n", row, col, val);
    else fprintf(f, "%10.13g ", val);
}

inline void write_row(int row, int col, int val, FILE * f, bool issparse){
    if (issparse)
      fprintf(f, "%d %d %d\n", row, col, val);
    else fprintf(f, "%d ", val);
}

template<typename T>
inline void set_typecode(MM_typecode & matcore);

template<>
inline void set_typecode<vec>(MM_typecode & matcode){
   mm_set_real(&matcode);
}

template<>
inline void set_typecode<ivec>(MM_typecode & matcode){
  mm_set_integer(&matcode);
}


template<typename vec>
void save_matrix_market_format_vector(const std::string datafile, const vec & output, bool issparse, std::string comment)
{
    MM_typecode matcode;                        
    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_coordinate(&matcode);

    if (issparse)
       mm_set_sparse(&matcode);
    else mm_set_dense(&matcode);

    set_typecode<vec>(matcode);

    FILE * f = fopen(datafile.c_str(),"w");
    if (f == NULL)
      logstream(LOG_FATAL)<<"Failed to open file: " << datafile << " for writing. " << std::endl;

    mm_write_banner(f, matcode); 
    if (comment.size() > 0) // add a comment to the matrix market header
      fprintf(f, "%c%s\n", '%', comment.c_str());
    if (issparse)
      mm_write_mtx_crd_size(f, output.size(), 1, output.size());
    else
      mm_write_mtx_array_size(f, output.size(), 1);

    for (int j=0; j<(int)output.size(); j++){
      write_row(j+1, 1, output[j], f, issparse);
      if (!issparse) 
        fprintf(f, "\n");
    }

    fclose(f);
}


template<typename vec>
inline void write_output_vector(const std::string & datafile, const vec& output, bool issparse, std::string comment = ""){

  logstream(LOG_INFO)<<"Going to write output to file: " << datafile << " (vector of size: " << output.size() << ") " << std::endl;
  save_matrix_market_format_vector(datafile, output,issparse, comment); 
}



vec lanczos( bipartite_graph_descriptor & info, vec & errest, 
            const std::string & vecfile){
   

   int nconv = 0;
   int its = 1;
   int mpd = 24;
   DistMat A(info);
   DistSlicedMat U(info.is_square() ? data_size : 0, info.is_square() ? 2*data_size : data_size, true, info, "U");
   DistSlicedMat V(0, data_size, false, info, "V");
   vec alpha, beta, b;
   vec sigma = zeros(data_size);
   errest = zeros(nv);
   DistVec v_0(info, 0, false, "v_0");
   if (vecfile.size() == 0)
     v_0 = randu(size(A,2));
   PRINT_VEC2("svd->V", v_0);
/* Example Usage:
  DECLARE_TRACER(classname_someevent)
  INITIALIZE_TRACER(classname_someevent, "hello world");
  Then later on...
  BEGIN_TRACEPOINT(classname_someevent)
  ...
  END_TRACEPOINT(classname_someevent)
 */
   DistDouble vnorm = norm(v_0);
   v_0=v_0/vnorm;
   PRINT_INT(nv);

   while(nconv < nsv && its < max_iter){
     logstream(LOG_INFO)<<"Starting iteration: " << its << /*" at time: " << mytimer.current_time() << */ std::endl;
     int k = nconv;
     int n = nv;
     PRINT_INT(k);
     PRINT_INT(n);

     alpha = zeros(n);
     beta = zeros(n);

     U[k] = V[k]*A._transpose();
     orthogonalize_vs_all(U, k, alpha(0));
     //alpha(0)=norm(U[k]).toDouble(); 
     PRINT_VEC3("alpha", alpha, 0);
     //U[k] = U[k]/alpha(0);

     for (int i=k+1; i<n; i++){
       logstream(LOG_INFO) <<"Starting step: " << i << /*" at time: " << mytimer.current_time() << */ std::endl;
       PRINT_INT(i);

       V[i]=U[i-1]*A;
       orthogonalize_vs_all(V, i, beta(i-k-1));
      
       //beta(i-k-1)=norm(V[i]).toDouble();
       //V[i] = V[i]/beta(i-k-1);
       PRINT_VEC3("beta", beta, i-k-1); 
      
       U[i] = V[i]*A._transpose();
       orthogonalize_vs_all(U, i, alpha(i-k));
       //alpha(i-k)=norm(U[i]).toDouble();

       //U[i] = U[i]/alpha(i-k);
       PRINT_VEC3("alpha", alpha, i-k);
     }

     V[n]= U[n-1]*A;
     orthogonalize_vs_all(V, n, beta(n-k-1));
     //beta(n-k-1)=norm(V[n]).toDouble();
     PRINT_VEC3("beta", beta, n-k-1);

  //compute svd of bidiagonal matrix
  PRINT_INT(nv);
  PRINT_NAMED_INT("svd->nconv", nconv);
  PRINT_NAMED_INT("svd->mpd", mpd);
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
  //nv = min(nconv+mpd, N);
  //if (nsv < 10)
  //  nv = 10;
  PRINT_NAMED_INT("nv",nv);

} // end(while)

printf(" Number of computed signular values %d",nconv);
printf("\n");
  DistVec normret(info, nconv, false, "normret");
  DistVec normret_tranpose(info, nconv, true, "normret_tranpose");
  for (int i=0; i < nconv; i++){
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
     if (nconv == 0)
       logstream(LOG_FATAL)<<"No converged vectors. Aborting the save operation" << std::endl;
     char output_filename[256];
     for (int i=0; i< nconv; i++){
        sprintf(output_filename, "%s.U.%d", datafile.c_str(), i);
        write_output_vector(output_filename, U[i].to_vec(), false, "GraphLab v2 SVD output. This file contains eigenvector number i of the matrix U");
        sprintf(output_filename, "%s.U.%d", datafile.c_str(), i);
        write_output_vector(output_filename, V[i].to_vec(), false, "GraphLab v2 SVD output. This file contains eigenvector number i of the matrix V'");
     }
  }
  return sigma;
}

int main(int argc,  const char *argv[]) {
  logstream(LOG_WARNING)<<"GraphChi Collaborative filtering library is written by Danny Bickson (c). Send any "
    " comments or bug reports to danny.bickson@gmail.com " << std::endl;

  //* GraphChi initialization will read the command line arguments and the configuration file. */
  graphchi_init(argc, argv);

    

  std::string vecfile;
  int unittest = 0;

  /* Basic arguments for application. NOTE: File will be automatically 'sharded'. */
  training = get_option_string("training");    // Base training
  validation = get_option_string("validation", "");
  test = get_option_string("test", "");

  if (validation == "")
    validation += training + "e";  
  if (test == "")
    test += training + "t";

  max_iter      = get_option_int("max_iter", 6);  // Number of iterations
  maxval        = get_option_float("maxval", 1e100);
  minval        = get_option_float("minval", -1e100);
  bool quiet    = get_option_int("quiet", 0);
  if (quiet)
    global_logger().set_log_level(LOG_ERROR);

  vecfile       = get_option_string("initial_vector", "");
  debug         = get_option_int("debug", 0);
  unittest      = get_option_int("unittest", 0);
  ortho_repeats = get_option_int("ortho_repeats", 3); 
  nv = get_option_int("nv", 1);
  nsv = get_option_int("nsv", 1);
  regularization = get_option_float("regularization", 1e-5);
  tol = get_option_float("tol", 1e-5);
  save_vectors = get_option_int("save_vectors", 1);
  nodes = get_option_int("nodes", 0);
  //clopts.attach_option("no_edge_data", &no_edge_data, no_edge_data, "matrix is binary (optional)");

  if (nv < nsv){
    logstream(LOG_FATAL)<<"Please set the number of vectors --nv=XX, to be at least the number of support vectors --nsv=XX or larger" << std::endl;
  }


  //unit testing
  if (unittest == 1){
    datafile = "gklanczos_testA"; 
    vecfile = "gklanczos_testA_v0";
    nsv = 3; nv = 3;
    debug = true;
    //TODO core.set_ncpus(1);
  }
  else if (unittest == 2){
    datafile = "gklanczos_testB";
    vecfile = "gklanczos_testB_v0";
    nsv = 10; nv = 10;
    debug = true;  max_iter = 100;
    //TODO core.set_ncpus(1);
  }
  else if (unittest == 3){
    datafile = "gklanczos_testC";
    vecfile = "gklanczos_testC_v0";
    nsv = 4; nv = 10;
    debug = true;  max_iter = 100;
    //TODO core.set_ncpus(1);
  }

  std::cout << "Load matrix " << datafile << std::endl;
  /* Preprocess data if needed, or discover preprocess files */
  nshards = convert_matrixmarket<edge_data>(training);
  info.rows = M; info.cols = N; info.nonzeros = L;
  assert(info.rows > 0 && info.cols > 0 && info.nonzeros > 0);
  latent_factors_inmem.resize(info.total());

  init_lanczos(info);
  init_math(info, ortho_repeats, update_function);
  if (vecfile.size() > 0){
    std::cout << "Load inital vector from file" << vecfile << std::endl;
    load_matrix_market_vector(vecfile, info, 0, true, false);
  }  

  
  vec errest;
  vec singular_values = lanczos(info, errest, vecfile);
 
  std::cout << "Lanczos finished " <</* mytimer.current_time() << */std::endl;

  //vec ret = fill_output(&core.graph(), bipartite_graph_descriptor, JACOBI_X);

  write_output_vector(datafile + ".singular_values", singular_values,false, "%GraphLab SVD Solver library. This file contains the singular values.");

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
  metrics_report(m);
 
  return 0;
}



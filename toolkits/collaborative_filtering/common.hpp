#ifndef _COMMON_H__
#define _COMMON_H__

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
 */

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <errno.h>
#include <string>

#include "util.hpp"
#include "graphchi_basic_includes.hpp"
#include "api/vertex_aggregator.hpp"
#include "preprocessing/sharder.hpp"
#include "../../example_apps/matrix_factorization/matrixmarket/mmio.h"
#include "../../example_apps/matrix_factorization/matrixmarket/mmio.c"

#include <stdio.h>
#include <limits.h>

#ifdef __APPLE__
//#include "getline.hpp" //fix for missing getline() function on MAC OS
#endif 

using namespace graphchi;

double minval = -1e100; //max allowed value in matrix
double maxval = 1e100; //min allowed value in matrix
double valrange = 1;   //range of allowed values in matrix
std::string training;
std::string validation;
std::string test;
uint M, N, K;
size_t L, Le;
uint Me, Ne;
double globalMean = 0;
double globalMean2 = 0;
double rmse=0.0;
bool load_factors_from_file = false;
int unittest = 0;
int niters = 10;
int halt_on_rmse_increase = 0;
int D = 20; //feature vector width
bool quiet = false;
int input_file_offset = 1;
int kfold_cross_validation = 0;
int kfold_cross_validation_index = 0;
int regnormal = 0; // if set to 1, compute LS regularization according to the paper "Yunhong Zhou, Dennis Wilkinson, Robert Schreiber and Rong Pan. Large-Scale Parallel Collaborative Filtering for the Netflix Prize."
int clean_cache = 0;
int R_output_format = 0; // if set to 1, all matrices and vectors are written in sparse matrix market format since
                         // R does not currently support array format (dense format).
int tokens_per_row = 3; //number of columns per input row
int allow_zeros;
int start_user=0; //start offset of user 
int end_user=INT_MAX; //end offset of user
int binary_relevance_threshold = -1; // if set, all edge values above this number will treated as binary 1
int exact_training_rmse = 0; //if 1, will compute training RMSE explicitly exact computation, this will slow down the run

/* support for different loss types (for SGD variants) */
std::string loss = "square";
enum {
  LOGISTIC = 0, SQUARE = 1, ABS = 2, AP = 3
};
const char * error_names[] = {"LOGISTIC LOSS", "RMSE", "MAE", "AP"};
int loss_type = SQUARE;
int calc_ap = 0;
int ap_number = 3; //AP@3


enum {
  TRAINING= 0, VALIDATION = 1, TEST = 2
};

void remove_cached_files(){
  //remove cached files
  int rc;
  assert(training != "");
  rc = system((std::string("rm -fR ") + training + std::string(".*")).c_str()); 
  assert(!rc);
  rc = system((std::string("rm -fR ") + training + std::string("_degs.bin")).c_str()); 
  assert(!rc);
  if (validation != ""){
    rc = system((std::string("rm -fR ") + validation + std::string(".*")).c_str()); 
    assert(!rc);
  }

}


void parse_command_line_args(){
  /* Basic arguments for application. NOTE: File will be automatically 'sharded'. */
  unittest = get_option_int("unittest", 0);
  niters    = get_option_int("max_iter", 6);  // Number of iterations
  if (unittest > 0)
    training = get_option_string("training", "");    // Base filename
  else training = get_option_string("training");
  validation = get_option_string("validation", "");
  test = get_option_string("test", "");
  D    = get_option_int("D", D);

  maxval        = get_option_float("maxval", 1e100);
  minval        = get_option_float("minval", -1e100);
  if (minval >= maxval)
    logstream(LOG_FATAL)<<"Min allowed rating (--minval) should be smaller than max allowed rating (--maxval)" << std::endl;
  valrange      = maxval - minval;
  assert(valrange > 0);
  quiet    = get_option_int("quiet", 0);
  if (quiet)
    global_logger().set_log_level(LOG_WARNING);
  halt_on_rmse_increase = get_option_int("halt_on_rmse_increase", 0);

  load_factors_from_file = get_option_int("load_factors_from_file", 0);
  input_file_offset = get_option_int("input_file_offset", input_file_offset);
  tokens_per_row = get_option_int("tokens_per_row", tokens_per_row);
  allow_zeros = get_option_int("allow_zeros", 0);
  binary_relevance_threshold = get_option_int("binary_relevance_threshold", -1);
  /* find out loss type (optional, for SGD variants only) */
  loss              = get_option_string("loss", loss);
  if (loss == "square")
    loss_type = SQUARE;
  else if (loss == "logistic")
    loss_type = LOGISTIC;
  else if (loss == "abs")
    loss_type = ABS;
  else if (loss == "ap")
    loss_type = AP;
  else logstream(LOG_FATAL)<<"Loss type should be one of [square,logistic,abs] (for example, --loss==square);" << std::endl;

  calc_ap      = get_option_int("calc_ap", calc_ap);
  if (calc_ap)
    loss_type = AP;
  ap_number    = get_option_int("ap_number", ap_number);
  kfold_cross_validation = get_option_int("kfold_cross_validation", kfold_cross_validation);
  kfold_cross_validation_index = get_option_int("kfold_cross_validation_index", kfold_cross_validation_index);
  if (kfold_cross_validation_index > 0){
    if (kfold_cross_validation_index >= kfold_cross_validation)
      logstream(LOG_FATAL)<<"kfold_cross_validation index should be between 0 to kfold_cross_validation-1 parameter" << std::endl;
  }
  if (kfold_cross_validation != 0){
    logstream(LOG_WARNING)<<"Activating kfold cross vlidation with K="<< kfold_cross_validation << std::endl;
    if (validation != "" || test != "")
      logstream(LOG_FATAL)<<"Using cross validation, validation file (--validation) and test file (--test) should be empty/" << std::endl;
    //removing tmp file (if present)
    int rc = system(("rm -fR " + training + "_kfold_tmp_file").c_str());
    if (rc != 0)
      logstream(LOG_FATAL)<<"Failed to delete temp file. Please check permissions." << std::endl;
    //linking training to validation
    rc = system(("ln -s " + training + " " + training + "_kfold_tmp_file").c_str());
    if (rc != 0)  
      logstream(LOG_FATAL)<<"Failed to link temp file. Please check permissions." << std::endl;
    validation = training + "_kfold_tmp_file";
    clean_cache = 1;
  }
  regnormal = get_option_int("regnormal", regnormal);
  clean_cache = get_option_int("clean_cache", clean_cache);

  if (clean_cache)
    remove_cached_files();

  R_output_format = get_option_int("R_output_format", R_output_format);
  start_user = get_option_int("start_user", start_user);
  end_user   = get_option_int("end_user",   end_user);
  exact_training_rmse = get_option_int("exact_training_rmse", 0);
}

template<typename T>
void set_engine_flags(T & pengine){
  pengine.set_disable_vertexdata_storage();  
  pengine.set_enable_deterministic_parallelism(false);
  pengine.set_modifies_inedges(false);
  pengine.set_modifies_outedges(false);
}
template<typename T>
void set_engine_flags(T & pengine, bool modify_outedges){
  pengine.set_disable_vertexdata_storage();  
  pengine.set_enable_deterministic_parallelism(false);
  pengine.set_modifies_inedges(false);
  pengine.set_modifies_outedges(modify_outedges);
}


void print_copyright(){
  logstream(LOG_WARNING)<<"GraphChi Collaborative filtering library is written by Danny Bickson (c). Send any "
    " comments or bug reports to danny.bickson@gmail.com " << std::endl;
}

void print_config(){
  std::cout<<"[feature_width] => [" << D << "]" << std::endl;
  std::cout<<"[users] => [" << M << "]" << std::endl;
  std::cout<<"[movies] => [" << N << "]" <<std::endl;
  std::cout<<"[training_ratings] => [" << L << "]" << std::endl;
  std::cout<<"[number_of_threads] => [" << number_of_omp_threads() << "]" <<std::endl;
  std::cout<<"[membudget_Mb] => [" << get_option_int("membudget_mb") << "]" <<std::endl; 
}

template<typename T>
void init_feature_vectors(uint size, T& latent_factors_inmem, bool randomize = true, double scale = 1.0){
  assert(size > 0);

  srand48(time(NULL));
  latent_factors_inmem.resize(size); // Initialize in-memory vertices.
  if (!randomize)
    return;

#pragma omp parallel for
  for (int i=0; i < (int)size; i++){
    for (int j=0; j<D; j++)
      latent_factors_inmem[i].pvec[j] = scale * drand48();
  } 
}
#endif //_COMMON_H__

#ifndef DEF_RMSEHPP
#define DEF_RMSEHPP
#include <iostream>
#include <iomanip>
#include <omp.h>
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
 */

#include "timer.hpp"
#include "eigen_wrapper.hpp"

timer mytimer;
double dtraining_rmse = 0;
double last_training_rmse = 0;
double dvalidation_rmse = 0;
double last_validation_rmse = 0;

/**
  compute predictions on test data
  */
void test_predictions(float (*prediction_func)(const vertex_data & user, const vertex_data & movie, float rating, double & prediction)) {
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  uint Me, Ne;
  size_t nz;   

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
  if ((ret_code = mm_read_mtx_crd_size(f, &Me, &Ne, &nz)) !=0) {
    logstream(LOG_FATAL) << "Failed reading matrix size: error=" << ret_code << std::endl;
  }

  if ((M > 0 && N > 0 ) && (Me != M || Ne != N))
    logstream(LOG_FATAL)<<"Input size of test matrix must be identical to training matrix, namely " << M << "x" << N << std::endl;

  mm_write_banner(fout, matcode);
  fprintf(fout, "%%This file contains predictions of user/item pair, one prediction in each line. The first column is user id. The second column is the item id. The third column is the computed prediction.\n");
  mm_write_mtx_crd_size(fout ,M,N,nz); 

  for (uint i=0; i<nz; i++)
  {
    int I, J;
    double val;
    int rc = fscanf(f, "%d %d %lg\n", &I, &J, &val);
    if (rc != 3)
      logstream(LOG_FATAL)<<"Error when reading input file: " << i << std::endl;
    I--;  /* adjust from 1-based to 0-based */
    J--;
    double prediction;
    (*prediction_func)(latent_factors_inmem[I], latent_factors_inmem[J+M], val, prediction);
    fprintf(fout, "%d %d %12.8lg\n", I+1, J+1, prediction);
  }
  fclose(f);
  fclose(fout);

  std::cout<<"Finished writing " << nz << " predictions to file: " << test << ".predict" << std::endl;
}

void test_predictions3(float (*prediction_func)(const vertex_data & user, const vertex_data & movie, const vertex_data & time, float rating, double & prediction), int time_offset = 0) {
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  uint Me, Ne;
  size_t nz;   

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
  if ((ret_code = mm_read_mtx_crd_size(f, &Me, &Ne, &nz)) !=0) {
    logstream(LOG_FATAL) << "Failed reading matrix size: error=" << ret_code << std::endl;
  }

  if ((M > 0 && N > 0 ) && (Me != M || Ne != N))
    logstream(LOG_FATAL)<<"Input size of test matrix must be identical to training matrix, namely " << M << "x" << N << std::endl;

  mm_write_banner(fout, matcode);
  mm_write_mtx_crd_size(fout ,M,N,nz); 

  for (uint i=0; i<nz; i++)
  {
    int I, J;
    double val;
    int time;
    int rc = fscanf(f, "%d %d %d %lg\n", &I, &J, &time, &val);
    if (rc != 4)
      logstream(LOG_FATAL)<<"Error when reading input file: " << i << std::endl;
    if (time - time_offset < 0)
      logstream(LOG_FATAL)<<"Error: we assume time values >= " << time_offset << std::endl;
    I--;  /* adjust from 1-based to 0-based */
    J--;
    double prediction;
    (*prediction_func)(latent_factors_inmem[I], latent_factors_inmem[J+M], latent_factors_inmem[time+M+N-time_offset], 1, prediction);
    fprintf(fout, "%d %d %12.8lg\n", I+1, J+1, prediction);
  }
  fclose(f);
  fclose(fout);

  logstream(LOG_INFO)<<"Finished writing " << nz << " predictions to file: " << test << ".predict" << std::endl;
}
/**
  compute validation rmse
  */
void validation_rmse(float (*prediction_func)(const vertex_data & user, const vertex_data & movie, float rating, double & prediction)
    ,graphchi_context & gcontext, int tokens_per_row = 3) {
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  size_t nz;   

  if ((f = fopen(validation.c_str(), "r")) == NULL) {
    std::cout<<std::endl;
    return; //missing validaiton data, nothing to compute
  }

  if (mm_read_banner(f, &matcode) != 0)
    logstream(LOG_FATAL) << "Could not process Matrix Market banner. File: " << validation << std::endl;

  if (mm_is_complex(matcode) || !mm_is_sparse(matcode))
    logstream(LOG_FATAL) << "Sorry, this application does not support complex values and requires a sparse matrix." << std::endl;

  /* find out size of sparse matrix .... */
  if ((ret_code = mm_read_mtx_crd_size(f, &Me, &Ne, &nz)) !=0) {
    logstream(LOG_FATAL) << "Failed reading matrix size: error=" << ret_code << std::endl;
  }
  if ((M > 0 && N > 0) && (Me != M || Ne != N))
    logstream(LOG_FATAL)<<"Input size of validation matrix must be identical to training matrix, namely " << M << "x" << N << std::endl;

  Le = nz;

  last_validation_rmse = dvalidation_rmse;
  dvalidation_rmse = 0;   
  int I, J;
  double val, time = 1.0;

  for (size_t i=0; i<nz; i++)
  {
    int rc;
    if (tokens_per_row == 3)
      rc = fscanf(f, "%d %d %lg\n", &I, &J, &val);
    else rc = fscanf(f, "%d %d %lg %lg\n", &I, &J, &time, &val);

    if (rc != tokens_per_row)
      logstream(LOG_FATAL)<<"Error when reading input file on line: " << i << " . should have" << tokens_per_row << std::endl;
    if (val < minval || val > maxval)
      logstream(LOG_FATAL)<<"Value is out of range: " << val << " should be: " << minval << " to " << maxval << std::endl;
    I--;  /* adjust from 1-based to 0-based */
    J--;
    double prediction;
    (*prediction_func)(latent_factors_inmem[I], latent_factors_inmem[J+M], val, prediction);
    dvalidation_rmse += time * pow(prediction - val, 2);
  }
  fclose(f);

  assert(Le > 0);
  dvalidation_rmse = sqrt(dvalidation_rmse / (double)Le);
  std::cout<<"  Validation RMSE: " << std::setw(10) << dvalidation_rmse << 
    " ratings_per_sec: " << std::setw(10) << (gcontext.iteration*L/mytimer.current_time()) << std::endl;
  if (halt_on_rmse_increase && dvalidation_rmse > last_validation_rmse && gcontext.iteration > 0){
    logstream(LOG_WARNING)<<"Stopping engine because of validation RMSE increase" << std::endl;
    gcontext.set_last_iteration(gcontext.iteration);
  }
}


/**
  compute validation rmse
  */
void validation_rmse3(float (*prediction_func)(const vertex_data & user, const vertex_data & movie, const vertex_data & time, float rating, double & prediction)
    ,graphchi_context & gcontext,int tokens_per_row = 4, int time_offset = 0) {
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  size_t nz;   

  if ((f = fopen(validation.c_str(), "r")) == NULL) {
    std::cout<<std::endl;
    return; //missing validaiton data, nothing to compute
  }

  if (mm_read_banner(f, &matcode) != 0)
    logstream(LOG_FATAL) << "Could not process Matrix Market banner. File: " << validation << std::endl;

  if (mm_is_complex(matcode) || !mm_is_sparse(matcode))
    logstream(LOG_FATAL) << "Sorry, this application does not support complex values and requires a sparse matrix." << std::endl;

  /* find out size of sparse matrix .... */
  if ((ret_code = mm_read_mtx_crd_size(f, &Me, &Ne, &nz)) !=0) {
    logstream(LOG_FATAL) << "Failed reading matrix size: error=" << ret_code << std::endl;
  }
  if ((M > 0 && N > 0) && (Me != M || Ne != N))
    logstream(LOG_FATAL)<<"Input size of validation matrix must be identical to training matrix, namely " << M << "x" << N << std::endl;

  Le = nz;

  last_validation_rmse = dvalidation_rmse;
  dvalidation_rmse = 0;   
  int I, J;
  double val, time = 1.0;

  for (size_t i=0; i<nz; i++)
  {
    int rc;
    rc = fscanf(f, "%d %d %lg %lg\n", &I, &J, &time, &val);

    if (rc != tokens_per_row)
      logstream(LOG_FATAL)<<"Error when reading input file on line: " << i << " . should have" << tokens_per_row << std::endl;
    if (val < minval || val > maxval)
      logstream(LOG_FATAL)<<"Value is out of range: " << val << " should be: " << minval << " to " << maxval << std::endl;
    if ((uint)time > K)
      logstream(LOG_FATAL)<<"Third column value time should be smaller than " << K << " while observed " << time << " in line : " << i << std::endl;

    I--;  /* adjust from 1-based to 0-based */
    J--;
    double prediction;
    (*prediction_func)(latent_factors_inmem[I], latent_factors_inmem[J+M], latent_factors_inmem[M+N+(uint)time-time_offset], val, prediction);
    dvalidation_rmse += pow(prediction - val, 2);
  }
  fclose(f);

  assert(Le > 0);
  dvalidation_rmse = sqrt(dvalidation_rmse / (double)Le);
  std::cout<<"  Validation RMSE: " << std::setw(10) << dvalidation_rmse << std::endl;
  if (halt_on_rmse_increase && dvalidation_rmse > last_validation_rmse && gcontext.iteration > 0){
    logstream(LOG_WARNING)<<"Stopping engine because of validation RMSE increase" << std::endl;
    gcontext.set_last_iteration(gcontext.iteration);
  }
}

void training_rmse(int iteration, graphchi_context &gcontext, bool items = false){
  last_training_rmse = dtraining_rmse;
  dtraining_rmse = 0;
  int start = 0;
  int end = M;
  if (items){
    start = M;
    end = M+N;
  }
#pragma omp parallel for reduction(+:dtraining_rmse)
  for (int i=start; i< (int)end; i++){
    dtraining_rmse += latent_factors_inmem[i].rmse;
  }
  dtraining_rmse = sqrt(dtraining_rmse / pengine->num_edges());
  std::cout<< std::setw(10) << mytimer.current_time() << ") Iteration: " << std::setw(3) <<iteration<<" Training RMSE: " << std::setw(10)<< dtraining_rmse;
}
#endif //DEF_RMSEHPP

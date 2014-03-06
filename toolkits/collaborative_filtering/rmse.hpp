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
 * File for aggregating and siplaying error mesasures and algorithm progress
 */

#include "timer.hpp"
#include "eigen_wrapper.hpp"
#include "common.hpp"
void read_matrix_market_banner_and_size(FILE * pfile, MM_typecode & matcode, uint & M, uint & N, size_t & nz, const std::string & filename);
FILE * open_file(const char * filename, const char * mode, bool optional);

timer mytimer;
double dtraining_rmse = 0;
double last_training_rmse = 0;
double dvalidation_rmse = 0;
double last_validation_rmse = 0;

int sign(double x){ if (x < 0) return -1; else if (x > 0) return 1; else return 0; }

/* compute the average of the loss after aggregating it */
double finalize_rmse(double rmse, double num_edges){
  double ret = 0;
 switch(loss_type){
    case SQUARE:
      ret = sqrt(rmse / num_edges);
  break;
    case LOGISTIC:
      ret = rmse/num_edges;
   break;
    case ABS:
      ret = rmse / num_edges;
    case AP:
      ret = rmse / num_edges;
   break;
  }
 return ret;
}

/** calc the loss measure based on the cost function */
double calc_loss(double exp_prediction, double err){
   double ret = 0;
  switch (loss_type){
    case LOGISTIC: ret= (exp_prediction * log(exp_prediction) + (1-exp_prediction)*log(1-exp_prediction));
                   break;
    case SQUARE:   ret = err*err;
                   break;
    case ABS:      ret = fabs(err);
                   break;
  }
  return ret;
}

/** calc prediction error based on the cost function */
double calc_error_f(double exp_prediction, double err){
  switch (loss_type){
    case LOGISTIC: 
      return err;
    case SQUARE:   
      return err *= (exp_prediction*(1.0-exp_prediction)*(maxval-minval));
    case ABS:      
      return err  = sign(err)*(exp_prediction*(1-exp_prediction)*(maxval-minval));
  }
  return NAN;
}


bool decide_if_edge_is_active(size_t i, int type);

/**
  compute predictions on test data
  */
void test_predictions(float (*prediction_func)(const vertex_data & user, const vertex_data & movie, float rating, double & prediction, void * extra), graphchi_context * gcontext = NULL, bool dosave = true, vec * avgprd = NULL, int pmf_burn_in = 0) {
  MM_typecode matcode;
  FILE *f;
  uint Me, Ne;
  size_t nz;   

  if (kfold_cross_validation > 0)
    test = training;

  if ((f = fopen(test.c_str(), "r")) == NULL) {
    return; //missing test data, nothing to compute
  }
  FILE * fout = NULL;
  if (dosave)
    fout = open_file((test + ".predict").c_str(),"w", false);
  
  read_matrix_market_banner_and_size(f, matcode, Me, Ne, nz, test+".predict");

  if ((M > 0 && N > 0 ) && (Me != M || Ne != N))
    logstream(LOG_FATAL)<<"Input size of test matrix must be identical to training matrix, namely " << M << "x" << N << std::endl;

  if (avgprd && gcontext->iteration == pmf_burn_in)
    *avgprd = zeros(nz);

  size_t test_ratings = nz;

  if (dosave){
    mm_write_banner(fout, matcode);
    fprintf(fout, "%%This file contains predictions of user/item pair, one prediction in each line. The first column is user id. The second column is the item id. The third column is the computed prediction.\n");
    if (kfold_cross_validation > 0)
      test_ratings = (1.0/(double)kfold_cross_validation)*nz;  
    mm_write_mtx_crd_size(fout ,M,N,test_ratings); 
  }

  for (uint i=0; i<nz; i++)
  {
    int I, J;
    double val;
    int rc = fscanf(f, "%d %d %lg\n", &I, &J, &val);
    if (rc != 3)
      logstream(LOG_FATAL)<<"Error when reading input test file, on data line " << i+2 << std::endl;
    I--;  /* adjust from 1-based to 0-based */
    J--;

    if (I < 0 || (uint)I >= M)
       logstream(LOG_FATAL)<<"Bad input " << I+1<< " in test file in line " << i+2<< " . First column should be in the range 1 to " << M << std::endl;
    if (J < 0 || (uint)J >= N)
       logstream(LOG_FATAL)<<"Bad input " << J+1<< " in test file in line " << i+2<< ". Second column should be in the range 1 to " << N << std::endl;

    if (!decide_if_edge_is_active(i, VALIDATION))
       continue;

    double prediction;
    (*prediction_func)(latent_factors_inmem[I], latent_factors_inmem[J+M], val, prediction, NULL); //TODO
    //for mcmc methods, store the sum of predictions
    if (avgprd && avgprd->size() > 0 && gcontext->iteration >= pmf_burn_in)
      avgprd->operator[](i) += prediction;

    if (dosave){
      if (avgprd && avgprd->size() > 0)
        prediction = avgprd->operator[](i) /(gcontext->iteration - pmf_burn_in); 
      fprintf(fout, "%d %d %12.8lg\n", I+1, J+1, prediction);
    }

 }
  fclose(f);
  if (dosave) 
    fclose(fout);

  if (dosave)
    std::cout<<"Finished writing " << test_ratings << " predictions to file: " << test << ".predict" << std::endl;
}

void test_predictions3(float (*prediction_func)(const vertex_data & user, const vertex_data & movie, float rating, double & prediction, void * extra), int time_offset = 0) {
  MM_typecode matcode;
  FILE *f;
  uint Me, Ne;
  size_t nz;   

  if ((f = fopen(test.c_str(), "r")) == NULL) {
    return; //missing validaiton data, nothing to compute
  }
  FILE * fout = open_file((test + ".predict").c_str(),"w", false);

  read_matrix_market_banner_and_size(f, matcode, Me, Ne, nz, test+".predict");

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
    if (time - input_file_offset < 0)
      logstream(LOG_FATAL)<<"Error: we assume time values >= " << input_file_offset << std::endl;
    I--;  /* adjust from 1-based to 0-based */
    J--;
    double prediction;
    (*prediction_func)(latent_factors_inmem[I], latent_factors_inmem[J+M], 1, prediction, (void*)&latent_factors_inmem[time+M+N-input_file_offset]);
    fprintf(fout, "%d %d %12.8lg\n", I+1, J+1, prediction);
  }
  fclose(f);
  fclose(fout);

  logstream(LOG_INFO)<<"Finished writing " << nz << " predictions to file: " << test << ".predict" << std::endl;
}

float (*prediction_func)(const vertex_data & user, const vertex_data & movie, float rating, double & prediction, void * extra);


void detect_matrix_size(std::string filename, FILE *&f, uint &_MM, uint &_NN, size_t & nz, uint nodes, size_t edges, int type);

/**
  compute validation rmse
  */
void validation_rmse(float (*prediction_func)(const vertex_data & user, const vertex_data & movie, float rating, double & prediction, void * extra)
    ,graphchi_context & gcontext, int tokens_per_row = 3, vec * avgprd = NULL, int pmf_burn_in = 0) {
  FILE *f;
  size_t nz;   

  detect_matrix_size(validation, f, Me, Ne, nz, 0, 0, VALIDATION);
  if (f == NULL)
    return;
  if ((M > 0 && N > 0) && (Me != M || Ne != N))
    logstream(LOG_FATAL)<<"Input size of validation matrix must be identical to training matrix, namely " << M << "x" << N << std::endl;

  Le = nz;
  if (avgprd != NULL && gcontext.iteration == 0)
    *avgprd = zeros(nz);


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
    dvalidation_rmse += time *(*prediction_func)(latent_factors_inmem[I], latent_factors_inmem[J+M], val, prediction, avgprd == NULL ? NULL : &avgprd->operator[](i)); 

  }
  fclose(f);

  assert(Le > 0);
  dvalidation_rmse = finalize_rmse(dvalidation_rmse , (double)Le);
  std::cout<<"  Validation  " << error_names[loss_type] << ":" << std::setw(10) << dvalidation_rmse << 
    " ratings_per_sec: " << std::setw(10) << (gcontext.iteration*L/mytimer.current_time()) << std::endl;
  if (halt_on_rmse_increase > 0 && halt_on_rmse_increase < gcontext.iteration && dvalidation_rmse > last_validation_rmse){
    logstream(LOG_WARNING)<<"Stopping engine because of validation RMSE increase" << std::endl;
    gcontext.set_last_iteration(gcontext.iteration);
  }
}


/**
  compute validation rmse
  */
void validation_rmse3(float (*prediction_func)(const vertex_data & user, const vertex_data & movie, const vertex_data & time, float rating, double & prediction)
    ,graphchi_context & gcontext,int tokens_per_row = 4, int time_offset = 0) {
  MM_typecode matcode;
  FILE *f;
  size_t nz;   

  if ((f = fopen(validation.c_str(), "r")) == NULL) {
    std::cout<<std::endl;
    return; //missing validaiton data, nothing to compute
  }

  read_matrix_market_banner_and_size(f, matcode, Me, Ne, nz, validation);
  
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
    time -= time_offset;

    if (rc != 4)
      logstream(LOG_FATAL)<<"Error when reading input file on line: " << i << " . should have 4 columns " << std::endl;
    if (val < minval || val > maxval)
      logstream(LOG_FATAL)<<"Value is out of range: " << val << " should be: " << minval << " to " << maxval << std::endl;
    if ((uint)time > K)
      logstream(LOG_FATAL)<<"Third column value time should be smaller than " << K << " while observed " << time << " in line : " << i << std::endl;

    I--;  /* adjust from 1-based to 0-based */
    J--;
    double prediction;
    dvalidation_rmse += (*prediction_func)(latent_factors_inmem[I], latent_factors_inmem[J+M], latent_factors_inmem[M+N+(uint)time], val, prediction);
  }
  fclose(f);

  assert(Le > 0);
  dvalidation_rmse = finalize_rmse(dvalidation_rmse , (double)Le);
  std::cout<<"  Validation " << error_names[loss_type] << ":" << std::setw(10) << dvalidation_rmse << std::endl;
  if (halt_on_rmse_increase >= gcontext.iteration && dvalidation_rmse > last_validation_rmse){
    logstream(LOG_WARNING)<<"Stopping engine because of validation RMSE increase" << std::endl;
    gcontext.set_last_iteration(gcontext.iteration);
  }
}

vec rmse_vec;


double training_rmse(int iteration, graphchi_context &gcontext, bool items = false){
  last_training_rmse = dtraining_rmse;
  dtraining_rmse = 0;
  double ret = 0;
  dtraining_rmse = sum(rmse_vec);
  int old_loss = loss_type;
  if (loss_type == AP)
    loss_type = SQUARE;
  ret = dtraining_rmse = finalize_rmse(dtraining_rmse, pengine->num_edges());
  std::cout<< std::setw(10) << mytimer.current_time() << ") Iteration: " << std::setw(3) <<iteration<<" Training " << error_names[loss_type] << ":"<< std::setw(10)<< dtraining_rmse;
  loss_type = old_loss;

  return ret;
}
#endif //DEF_RMSEHPP

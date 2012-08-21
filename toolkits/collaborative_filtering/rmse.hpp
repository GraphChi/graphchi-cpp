#ifndef DEF_RMSEHPP
#define DEF_RMSEHPP
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



/**
  compute predictions on test data
  */
void test_predictions(float (*prediction_func)(const vertex_data & user, const vertex_data & movie, float rating, double & prediction)) {
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  int Me, Ne, nz;   

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

  for (int i=0; i<nz; i++)
  {
    int I, J;
    double val;
    int rc = fscanf(f, "%d %d %lg\n", &I, &J, &val);
    if (rc != 3)
      logstream(LOG_FATAL)<<"Error when reading input file: " << i << std::endl;
    I--;  /* adjust from 1-based to 0-based */
    J--;
    double prediction;
    (*prediction_func)(latent_factors_inmem[I], latent_factors_inmem[J], val, prediction);
    fprintf(fout, "%d %d %12.8lg\n", I+1, J+1, prediction);
  }
  fclose(f);
  fclose(fout);

  logstream(LOG_INFO)<<"Finished writing " << nz << " predictions to file: " << test << ".predict" << std::endl;
}


/**
  compute validation rmse
  */
void validation_rmse(float (*prediction_func)(const vertex_data & user, const vertex_data & movie, float rating, double & prediction)) {
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  int nz;   

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
  if ((ret_code = mm_read_mtx_crd_size(f, &Me, &Ne, &nz)) !=0) {
    logstream(LOG_FATAL) << "Failed reading matrix size: error=" << ret_code << std::endl;
  }
  if ((M > 0 && N > 0) && (Me != M || Ne != N))
    logstream(LOG_FATAL)<<"Input size of validation matrix must be identical to training matrix, namely " << M << "x" << N << std::endl;

  Le = nz;

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
    double prediction;
    (*prediction_func)(latent_factors_inmem[I], latent_factors_inmem[J], val, prediction);
    validation_rmse += (prediction - val)*(prediction-val);
  }
  fclose(f);

  assert(Le > 0);
  logstream(LOG_INFO)<<"Validation RMSE: " << sqrt(validation_rmse/Le)<< std::endl;
}

#endif //DEF_RMSEHPP

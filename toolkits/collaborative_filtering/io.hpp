#ifndef DEF_IOHPP
#define DEF_IOHPP
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
 * Create a bipartite graph from a matrix. Each row corresponds to vertex
 * with the same id as the row number (0-based), but vertices correponsing to columns
 * have id + num-rows.
 * Line format of the type
 * [user] [item] [rating] [time/weight]
 */
template <typename als_edge_type>
int convert_matrixmarket4(std::string base_filename) {
  // Note, code based on: http://math.nist.gov/MatrixMarket/mmio/c/example_read.c
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  int nz;   

  /**
   * Create sharder object
   */
  int nshards;
  if ((nshards = find_shards<als_edge_type>(base_filename, get_option_string("nshards", "auto")))) {
    logstream(LOG_INFO) << "File " << base_filename << " was already preprocessed, won't do it again. " << std::endl;
    FILE * inf = fopen((base_filename + ".gm").c_str(), "r");
    int rc = fscanf(inf,"%d\n%d\n%lg",&M, &N, &globalMean);
    if (rc != 3)
       logstream(LOG_FATAL)<<"Failed to read global mean from file" << std::endl;
    fclose(inf);
    logstream(LOG_INFO) << "Global mean is: " << globalMean << " Now creating shards." << std::endl;
    return nshards;
  }   

  sharder<als_edge_type> sharderobj(base_filename);
  sharderobj.start_preprocessing();


  if ((f = fopen(base_filename.c_str(), "r")) == NULL) {
    logstream(LOG_FATAL) << "Could not open file: " << base_filename << ", error: " << strerror(errno) << std::endl;
  }


  if (mm_read_banner(f, &matcode) != 0)
    logstream(LOG_FATAL) << "Could not process Matrix Market banner. File: " << base_filename << std::endl;


  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */

  if (mm_is_complex(matcode) || !mm_is_sparse(matcode))
    logstream(LOG_FATAL) << "Sorry, this application does not support complex values and requires a sparse matrix." << std::endl;

  /* find out size of sparse matrix .... */

  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0) {
    logstream(LOG_FATAL) << "Failed reading matrix size: error=" << ret_code << std::endl;
  }

  logstream(LOG_INFO) << "Starting to read matrix-market input. Matrix dimensions: " 
    << M << " x " << N << ", non-zeros: " << nz << std::endl;

  int I, J;
  double val, time;
 
  if (!sharderobj.preprocessed_file_exists()) {
    for (int i=0; i<nz; i++)
    {
     int rc = fscanf(f, "%d %d %lg %lg\n", &I, &J, &val, &time);
    if (rc != 4)
        logstream(LOG_FATAL)<<"Error when reading input file: " << i << std::endl;
      I--;  /* adjust from 1-based to 0-based */
      J--;
      globalMean += val; 
      sharderobj.preprocessing_add_edge(I, M + J, als_edge_type(val, time));
    }
    sharderobj.end_preprocessing();
    globalMean /= nz;
    logstream(LOG_INFO) << "Global mean is: " << globalMean << " Now creating shards." << std::endl;
    FILE * outf = fopen((base_filename + ".gm").c_str(), "w");
    fprintf(outf, "%d\n%d\n%lg\n", M, N, globalMean);
    fclose(outf);


  } else {
    logstream(LOG_INFO) << "Matrix already preprocessed, just run sharder." << std::endl;
  }
  if (f !=stdin) fclose(f);


  logstream(LOG_INFO) << "Now creating shards." << std::endl;

  // Shard with a specified number of shards, or determine automatically if not defined
  nshards = sharderobj.execute_sharding(get_option_string("nshards", "auto"));

  return nshards;
}


/**
 * Create a bipartite graph from a matrix. Each row corresponds to vertex
 * with the same id as the row number (0-based), but vertices correponsing to columns
 * have id + num-rows.
 */
template <typename als_edge_type>
int convert_matrixmarket(std::string base_filename) {
  // Note, code based on: http://math.nist.gov/MatrixMarket/mmio/c/example_read.c
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  int nz;   

  /**
   * Create sharder object
   */
  int nshards;
  if ((nshards = find_shards<als_edge_type>(base_filename, get_option_string("nshards", "auto")))) {
    logstream(LOG_INFO) << "File " << base_filename << " was already preprocessed, won't do it again. " << std::endl;
    FILE * inf = fopen((base_filename + ".gm").c_str(), "r");
    int rc = fscanf(inf,"%d\n%d\n%lg",&M, &N, &globalMean);
    if (rc != 3)
       logstream(LOG_FATAL)<<"Failed to read global mean from file" << std::endl;
    fclose(inf);
    logstream(LOG_INFO) << "Global mean is: " << globalMean << " Now creating shards." << std::endl;
    return nshards;
  }   

  sharder<als_edge_type> sharderobj(base_filename);
  sharderobj.start_preprocessing();


  if ((f = fopen(base_filename.c_str(), "r")) == NULL) {
    logstream(LOG_ERROR) << "Could not open file: " << base_filename << ", error: " << strerror(errno) << std::endl;
    exit(1);
  }


  if (mm_read_banner(f, &matcode) != 0)
    logstream(LOG_FATAL) << "Could not process Matrix Market banner. File: " << base_filename << std::endl;


  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */

  if (mm_is_complex(matcode) || !mm_is_sparse(matcode))
    logstream(LOG_FATAL) << "Sorry, this application does not support complex values and requires a sparse matrix." << std::endl;

  /* find out size of sparse matrix .... */

  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0) {
    logstream(LOG_FATAL) << "Failed reading matrix size: error=" << ret_code << std::endl;
  }

  L=nz;

  logstream(LOG_INFO) << "Starting to read matrix-market input. Matrix dimensions: " 
    << M << " x " << N << ", non-zeros: " << nz << std::endl;


  if (!sharderobj.preprocessed_file_exists()) {
    for (int i=0; i<nz; i++)
    {
      int I, J;
      double val;
      int rc = fscanf(f, "%d %d %lg\n", &I, &J, &val);
      if (rc != 3)
        logstream(LOG_FATAL)<<"Error when reading input file: " << i << std::endl;
      I--;  /* adjust from 1-based to 0-based */
      J--;
      globalMean += val; 
      sharderobj.preprocessing_add_edge(I, M + J, als_edge_type((float)val));
    }
    sharderobj.end_preprocessing();
    globalMean /= nz;
    logstream(LOG_INFO) << "Global mean is: " << globalMean << " Now creating shards." << std::endl;
    FILE * outf = fopen((base_filename + ".gm").c_str(), "w");
    fprintf(outf, "%d\n%d\n%lg\n", M, N, globalMean);
    fclose(outf);


  } else {
    logstream(LOG_INFO) << "Matrix already preprocessed, just run sharder." << std::endl;
  }
  if (f !=stdin) fclose(f);


  logstream(LOG_INFO) << "Now creating shards." << std::endl;

  // Shard with a specified number of shards, or determine automatically if not defined
  nshards = sharderobj.execute_sharding(get_option_string("nshards", "auto"));

  return nshards;
}

void set_matcode(MM_typecode & matcode){
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_array(&matcode);
  mm_set_real(&matcode);
}


#endif

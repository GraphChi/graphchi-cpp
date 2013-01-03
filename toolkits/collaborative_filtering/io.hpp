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

#include "types.hpp"
#include "implicit.hpp"

void read_matrix_market_banner_and_size(FILE * f, MM_typecode & matcode, uint & Me, uint & Ne, size_t & nz, const std::string & filename){

  if (mm_read_banner(f, &matcode) != 0)
    logstream(LOG_FATAL) << "Could not process Matrix Market banner. File: " << filename << std::endl;

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */
  if (mm_is_complex(matcode) || !mm_is_sparse(matcode))
    logstream(LOG_FATAL) << "Sorry, this application does not support complex values and requires a sparse matrix." << std::endl;

  /* find out size of sparse matrix .... */
  if (mm_read_mtx_crd_size(f, &Me, &Ne, &nz) != 0) {
    logstream(LOG_FATAL) << "Failed reading matrix size: error" << std::endl;
  }
}

void read_global_mean(std::string base_filename, int type){
  FILE * inf = fopen((base_filename + ".gm").c_str(), "r");
  int rc;
  if (type == TRAINING)
    rc = fscanf(inf,"%d\n%d\n%ld\n%lg\n%d\n",&M, &N, &L, &globalMean, &K);
  else rc = fscanf(inf,"%d\n%d\n%ld\n%lg\n%d\n",&Me, &Ne, &Le, &globalMean2, &K);
  if (rc != 5)
    logstream(LOG_FATAL)<<"Failed to read global mean from file" << base_filename << ".gm" << std::endl;
  fclose(inf);
  if (type == TRAINING)
    logstream(LOG_INFO) << "Opened matrix size: " <<M << " x " << N << " Global mean is: " << globalMean << " time bins: " << K << " Now creating shards." << std::endl;
  else 
    logstream(LOG_INFO) << "Opened VLIDATION matrix size: " <<Me << " x " << Ne << " Global mean is: " << globalMean2 << " time bins: " << K << " Now creating shards." << std::endl;
}

void write_global_mean(std::string base_filename, int type){
  FILE * outf = fopen((base_filename + ".gm").c_str(), "w");
  if (type == TRAINING)
    fprintf(outf, "%d\n%d\n%ld\n%lg\n%d\n", M, N, L, globalMean, K);
  else 
    fprintf(outf, "%d\n%d\n%ld\n%lg\n%d\n", Me, Ne, Le, globalMean2, K);
  fclose(outf);
}

bool try_to_detect_info_file(std::string base_filename, int type, size_t & nz){
  MM_typecode matcode;
  bool info_file = false;
  FILE * ff = NULL;
  /* auto detect presence of file named base_filename.info to find out matrix market size */
  if ((ff = fopen((base_filename + ":info").c_str(), "r")) != NULL) {
    info_file = true;
    if (type == TRAINING)
      read_matrix_market_banner_and_size(ff, matcode, M, N, nz, base_filename);
    else
      read_matrix_market_banner_and_size(ff, matcode, Me, Ne, nz, base_filename);
    fclose(ff);
  }
  return info_file;
}

/**
 * Create a bipartite graph from a matrix. Each row corresponds to vertex
 * with the same id as the row number (0-based), but vertices correponsing to columns
 * have id + num-rows.
 * Line format of the type
 * [user] [item] [rating] [time/weight]
 */

template <typename als_edge_type>
int convert_matrixmarket4(std::string base_filename, bool add_time_edges = false, bool square = false, int type = TRAINING, int matlab_time_offset = 1) {
  // Note, code based on: http://math.nist.gov/MatrixMarket/mmio/c/example_read.c
  MM_typecode matcode;
  FILE *f;
  size_t nz;
  /**
   * Create sharder object
   */
  int nshards;
  if ((nshards = find_shards<als_edge_type>(base_filename, get_option_string("nshards", "auto")))) {
    logstream(LOG_INFO) << "File " << base_filename << " was already preprocessed, won't do it again. " << std::endl;
    read_global_mean(base_filename, type);
    return nshards;
  }   

  sharder<als_edge_type> sharderobj(base_filename);
  sharderobj.start_preprocessing();


  bool info_file = try_to_detect_info_file(base_filename, type, nz);
  if ((f = fopen(base_filename.c_str(), "r")) == NULL) {
    if (type == VALIDATION){
      logstream(LOG_INFO)<< "Did not find validation file: " << base_filename << std::endl;
      return -1;
    }
    logstream(LOG_FATAL) << "Could not open file: " << base_filename << ", error: " << strerror(errno) << std::endl;
  }

  /* if .info file is not present, try to find matrix market header inside the base_filename file */
  if (!info_file){
    read_matrix_market_banner_and_size(f, matcode, M, N, nz, base_filename);
  }
  if (type == TRAINING)
    logstream(LOG_INFO) << "Starting to read matrix-market input. Matrix dimensions: " 
    << M << " x " << N << ", non-zeros: " << nz << std::endl;
  else
    logstream(LOG_INFO) << "Starting to read matrix-market input. Matrix dimensions: " 
    << Me << " x " << Ne << ", non-zeros: " << nz << std::endl;

  uint I, J;
  double val, time;
  if (type == TRAINING)
    L = nz;
  else Le = nz;

  if (!sharderobj.preprocessed_file_exists()) {
    for (size_t i=0; i<nz; i++)
    {
      int rc = fscanf(f, "%d %d %lg %lg\n", &I, &J, &time, &val);
      if (rc != 4)
        logstream(LOG_FATAL)<<"Error when reading input file - line " << i << std::endl;
      if (time < 0)
        logstream(LOG_FATAL)<<"Time (third columns) should be >= 0 " << std::endl;
      I--;  /* adjust from 1-based to 0-based */
      J--;
      if (I >= M)
        logstream(LOG_FATAL)<<"Row index larger than the matrix row size " << I << " > " << M << " in line: " << i << std::endl;
      if (J >= N)
        logstream(LOG_FATAL)<<"Col index larger than the matrix col size " << J << " > " << N << " in line; " << i << std::endl;
      K = std::max((int)time, (int)K);
      time -= matlab_time_offset;
      if (time < 0 && add_time_edges)
        logstream(LOG_FATAL)<<"Time bins should be >= 1 in row " << i << std::endl;
      //avoid self edges
      if (square && I == J)
        continue;
      globalMean += val; 
      sharderobj.preprocessing_add_edge(I, (square? J : (M + J)), als_edge_type(val, time+M+N));
      //in case of a tensor, add besides of the user-> movie edge also
      //time -> user and time-> movie edges
      if (add_time_edges){
        sharderobj.preprocessing_add_edge((uint)time + M + N, I, als_edge_type(val, M+J));
        sharderobj.preprocessing_add_edge((uint)time + M + N, M+J , als_edge_type(val, I));
      }
    }

    if (type == TRAINING){
      uint toadd = 0;
      if (implicitratingtype == IMPLICIT_RATING_RANDOM)
      toadd = add_implicit_edges4(implicitratingtype, sharderobj);
      globalMean += implicitratingvalue * toadd;
      L += toadd;
       globalMean /= L;
      logstream(LOG_INFO) << "Global mean is: " << globalMean << " time bins: " << K << " . Now creating shards." << std::endl;
    }
    else {
      globalMean2 /= Le;
      logstream(LOG_INFO) << "Global mean is: " << globalMean2 << " time bins: " << K << " . Now creating shards." << std::endl;
    }
    write_global_mean(base_filename, type);

    sharderobj.end_preprocessing();

  } else {
    logstream(LOG_INFO) << "Matrix already preprocessed, just run sharder." << std::endl;
  }

  fclose(f);
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
int convert_matrixmarket(std::string base_filename, SharderPreprocessor<als_edge_type> * preprocessor = NULL, size_t nodes = 0, size_t edges = 0, int tokens_per_row = 3, int type = TRAINING) {
  // Note, code based on: http://math.nist.gov/MatrixMarket/mmio/c/example_read.c
  MM_typecode matcode;
  FILE *f;
  size_t nz;   

  std::string suffix = "";
  if (preprocessor != NULL) {
    suffix = preprocessor->getSuffix();
  }

  /**
   * Create sharder object
   */
  int nshards;
  if ((nshards = find_shards<als_edge_type>(base_filename+ suffix, get_option_string("nshards", "auto")))) {
    logstream(LOG_INFO) << "File " << base_filename << " was already preprocessed, won't do it again. " << std::endl;
    read_global_mean(base_filename, type);
    return nshards;
  }   

  sharder<als_edge_type> sharderobj(base_filename + suffix);
  sharderobj.start_preprocessing();

  bool info_file = try_to_detect_info_file(base_filename, type, nz);
  if ((f = fopen(base_filename.c_str(), "r")) == NULL) {
    if (type == VALIDATION){
      logstream(LOG_INFO)<<"Validation file: "  << base_filename << " is not found. " << std::endl;
      return -1;
    }
    logstream(LOG_FATAL) << "Could not open file: " << base_filename << ", error: " << strerror(errno) << std::endl;
  }

  if ((nodes == 0 && edges == 0) && !info_file){
    if (type == TRAINING)
      read_matrix_market_banner_and_size(f, matcode, M, N, nz, base_filename);
    else
      read_matrix_market_banner_and_size(f, matcode, Me, Ne, nz, base_filename);
  }
  else if (!info_file){
    if (type == TRAINING){
      M = N = nodes;
      nz = edges;
    }
    else {
      Me = Ne = nodes;
      nz = edges;
    }
  }
  if (type == TRAINING)
    L=nz;
  else 
    Le = nz;

  if (type == TRAINING)
    logstream(LOG_INFO) << "Starting to read matrix-market input. Matrix dimensions: " 
      << M << " x " << N << ", non-zeros: " << nz << std::endl;
  else
    logstream(LOG_INFO) << "Starting to read VALIDATION matrix-market input. Matrix dimensions: "
      << Me << " x " << Ne << ", non-zeros: " << nz << std::endl;

  uint I, J;
  double val = 1.0;
  if (!sharderobj.preprocessed_file_exists()) {
    for (size_t i=0; i<nz; i++)
    {
      if (tokens_per_row == 3){
        int rc = fscanf(f, "%u %u %lg\n", &I, &J, &val);
        if (rc != 3)
          logstream(LOG_FATAL)<<"Error when reading input file: " << i << std::endl;
      }
      else if (tokens_per_row == 2){
        int rc = fscanf(f, "%u %u\n", &I, &J);
        if (rc != 2)
          logstream(LOG_FATAL)<<"Error when reading input file: " << i << std::endl;
      }
      else assert(false);

      if (I ==987654321 || J== 987654321) //hack - to be removed later
        continue;
      I--;  /* adjust from 1-based to 0-based */
      J--;
      if (I >= M)
        logstream(LOG_FATAL)<<"Row index larger than the matrix row size " << I << " > " << M << " in line: " << i << std::endl;
      if (J >= N)
        logstream(LOG_FATAL)<<"Col index larger than the matrix col size " << J << " > " << N << " in line; " << i << std::endl;
      if (type == TRAINING)
        globalMean += val; 
      else globalMean2 += val;
      sharderobj.preprocessing_add_edge(I, M==N?J:M + J, als_edge_type((float)val));
    }
    
    if (type == TRAINING){
      uint toadd = 0;
      if (implicitratingtype == IMPLICIT_RATING_RANDOM)
        toadd = add_implicit_edges(implicitratingtype, sharderobj);
      globalMean += implicitratingvalue * toadd;
      L += toadd;
      globalMean /= L;
      logstream(LOG_INFO) << "Global mean is: " << globalMean << " Now creating shards." << std::endl;
    }
    else {
      globalMean2 /= Le;
      logstream(LOG_INFO) << "Global mean is: " << globalMean2 << " Now creating shards." << std::endl;
    }
    write_global_mean(base_filename, type); 
    sharderobj.end_preprocessing();

    if (preprocessor != NULL) {
      preprocessor->reprocess(sharderobj.preprocessed_name(), base_filename);
    }

  } else {
    logstream(LOG_INFO) << "Matrix already preprocessed, just run sharder." << std::endl;
  }
  fclose(f);


  logstream(LOG_INFO) << "Now creating shards." << std::endl;

  // Shard with a specified number of shards, or determine automatically if not defined
  nshards = sharderobj.execute_sharding(get_option_string("nshards", "auto"));
  logstream(LOG_INFO) << "Successfully finished sharding for " << base_filename + suffix << std::endl;
  logstream(LOG_INFO) << "Created " << nshards << " shards." << std::endl;

  return nshards;
}

void set_matcode(MM_typecode & matcode){
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_array(&matcode);
  mm_set_real(&matcode);
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

vec load_matrix_market_vector(const std::string & filename,  bool optional_field, bool allow_zeros)
{

  int ret_code;
  MM_typecode matcode;
  uint M, N; 
  size_t i,nz;  

  logstream(LOG_INFO) <<"Going to read matrix market vector from input file: " << filename << std::endl;

  FILE * f = open_file(filename.c_str(), "r", optional_field);
  //if optional file not found return
  if (f== NULL && optional_field){
    return zeros(1);
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

  vec ret = zeros(M);
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
    ret[row] = val;
  }
  fclose(f);
  logstream(LOG_INFO)<<"Succesfully read a vector of size: " << M << " [ " << nz << "]" << std::endl;
  return ret;
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

/** load a matrix market file into a matrix */
void load_matrix_market_matrix(const std::string & filename, int offset, int D){
  MM_typecode matcode;                        
  uint i,I,J;
  double val;
  uint rows, cols;
  size_t nnz;
  FILE * f = open_file(filename.c_str() ,"r");
  int rc = mm_read_banner(f, &matcode); 
  if (rc != 0)
    logstream(LOG_FATAL)<<"Failed to load matrix market banner in file: " << filename << std::endl;

  if (mm_is_sparse(matcode)){
    int rc = mm_read_mtx_crd_size(f, &rows, &cols, &nnz);
    if (rc != 0)
      logstream(LOG_FATAL)<<"Failed to load matrix market banner in file: " << filename << std::endl;
  }
  else { //dense matrix
    rc = mm_read_mtx_array_size(f, &rows, &cols);
    if (rc != 0)
      logstream(LOG_FATAL)<<"Failed to load matrix market banner in file: " << filename << std::endl;
    nnz = rows * cols;
  }

  for (i=0; i<nnz; i++){
    if (mm_is_sparse(matcode)){
      rc = fscanf(f, "%u %u %lg\n", &I, &J, &val);
      if (rc != 3)
        logstream(LOG_FATAL)<<"Error reading input line " << i << std::endl;
      I--; J--;
      assert(I >= 0 && I < rows);
      assert(J >= 0 && J < cols);
      //set_val(a, I, J, val);
      latent_factors_inmem[I+offset].pvec[J] = val;
    }
    else {
      rc = fscanf(f, "%lg", &val);
      if (rc != 1)
        logstream(LOG_FATAL)<<"Error reading nnz " << i << std::endl;
      I = i / D;
      J = i % cols;
      //set_val(a, I, J, val);
      latent_factors_inmem[I+offset].pvec[J] = val;
    }
  }
  logstream(LOG_INFO) << "Factors from file: loaded matrix of size " << rows << " x " << cols << " from file: " << filename << " total of " << nnz << " entries. "<< i << std::endl;
  fclose(f);
}

#endif

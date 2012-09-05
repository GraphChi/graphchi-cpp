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
/**
 * Create a bipartite graph from a matrix. Each row corresponds to vertex
 * with the same id as the row number (0-based), but vertices correponsing to columns
 * have id + num-rows.
 * Line format of the type
 * [user] [item] [rating] [time/weight]
 */

template <typename als_edge_type>
int convert_matrixmarket4(std::string base_filename, bool add_time_edges = false) {
  // Note, code based on: http://math.nist.gov/MatrixMarket/mmio/c/example_read.c
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  size_t nz;
  /**
   * Create sharder object
   */
  int nshards;
  if ((nshards = find_shards<als_edge_type>(base_filename, get_option_string("nshards", "auto")))) {
    logstream(LOG_INFO) << "File " << base_filename << " was already preprocessed, won't do it again. " << std::endl;
    FILE * inf = fopen((base_filename + ".gm").c_str(), "r");
    int rc = fscanf(inf,"%d\n%d\n%ld\n%lg\n%d\n",&M, &N, &L, &globalMean, &K);
    if (rc != 5)
      logstream(LOG_FATAL)<<"Failed to read global mean from file" << base_filename << ".gm" << std::endl;
    fclose(inf);
    if (K <= 0)
      logstream(LOG_FATAL)<<"Incorrect number of time bins K in .gm file " << base_filename << ".gm" << std::endl;

    logstream(LOG_INFO) << "Read matrix of size " << M << " x " << N << " Global mean is: " << globalMean << " time bins: " << K << " Now creating shards." << std::endl;
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
  L = nz;

  if (!sharderobj.preprocessed_file_exists()) {
    for (size_t i=0; i<nz; i++)
    {
      int rc = fscanf(f, "%d %d %lg %lg\n", &I, &J, &time, &val);
      if (rc != 4)
        logstream(LOG_FATAL)<<"Error when reading input file: " << i << std::endl;
      if (time < 0)
        logstream(LOG_FATAL)<<"Time (third columns) should be >= 0 " << std::endl;
      I--;  /* adjust from 1-based to 0-based */
      J--;
      K = std::max((int)time, (int)K);
      globalMean += val; 
      sharderobj.preprocessing_add_edge(I, M + J, als_edge_type(val, time+M+N));
      //in case of a tensor, add besides of the user-> movie edge also
      //time -> user and time-> movie edges
      if (add_time_edges){
        sharderobj.preprocessing_add_edge((uint)time + M + N, I, als_edge_type(val, M+J));
        sharderobj.preprocessing_add_edge((uint)time + M + N, M+J , als_edge_type(val, I));
      }
    }
  
    uint toadd = 0;
    if (implicitratingtype == IMPLICIT_RATING_RANDOM)
      toadd = add_implicit_edges4(implicitratingtype, sharderobj);
    globalMean += implicitratingvalue * toadd;
    L += toadd;
  
    sharderobj.end_preprocessing();
    globalMean /= L;
    logstream(LOG_INFO) << "Global mean is: " << globalMean << " time bins: " << K << " . Now creating shards." << std::endl;
    FILE * outf = fopen((base_filename + ".gm").c_str(), "w");
    fprintf(outf, "%d\n%d\n%ld\n%lg\n%d\n", M, N, L, globalMean, K);
    fclose(outf);


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
int convert_matrixmarket(std::string base_filename, SharderPreprocessor<als_edge_type> * preprocessor = NULL) {
  // Note, code based on: http://math.nist.gov/MatrixMarket/mmio/c/example_read.c
  int ret_code;
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
    FILE * inf = fopen((base_filename + ".gm").c_str(), "r");
    int rc = fscanf(inf,"%d\n%d\n%ld\n%lg\n%d\n",&M, &N, &L, &globalMean, &K);
    if (rc != 5)
      logstream(LOG_FATAL)<<"Failed to read global mean from file" << base_filename+ suffix << ".gm" << std::endl;
    fclose(inf);
    logstream(LOG_INFO) << "Opened matrix size: " <<M << " x " << N << " Global mean is: " << globalMean << " time bins: " << K << " Now creating shards." << std::endl;
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

  L=nz;

  logstream(LOG_INFO) << "Starting to read matrix-market input. Matrix dimensions: " 
    << M << " x " << N << ", non-zeros: " << nz << std::endl;


  if (!sharderobj.preprocessed_file_exists()) {
    for (size_t i=0; i<nz; i++)
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
    uint toadd = 0;
    if (implicitratingtype == IMPLICIT_RATING_RANDOM)
      toadd = add_implicit_edges(implicitratingtype, sharderobj);
    globalMean += implicitratingvalue * toadd;
    L += toadd;
  
    sharderobj.end_preprocessing();
    globalMean /= L;
    logstream(LOG_INFO) << "Global mean is: " << globalMean << " Now creating shards." << std::endl;

    if (preprocessor != NULL) {
       preprocessor->reprocess(sharderobj.preprocessed_name(), base_filename);
    }
     
    FILE * outf = fopen((base_filename + ".gm").c_str(), "w");
    fprintf(outf, "%d\n%d\n%ld\n%lg\n%d\n", M, N, L, globalMean, K);
    fclose(outf);


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



#endif

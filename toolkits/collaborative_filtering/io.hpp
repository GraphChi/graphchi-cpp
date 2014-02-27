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


void set_matcode(MM_typecode & matcode, bool sparse = false){
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  if (sparse)
    mm_set_coordinate(&matcode);
  else
    mm_set_array(&matcode);
  mm_set_real(&matcode);
}


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

void detect_matrix_size(std::string filename, FILE *&f, uint &_MM, uint &_NN, size_t & nz, uint nodes = 0, size_t edges = 0, int type = TRAINING){

  MM_typecode matcode;
  bool info_file = false;

  if (nodes == 0 && edges == 0){
    FILE * ff = NULL;
    /* auto detect presence of file named base_filename.info to find out matrix market size */
    if ((ff = fopen((filename + ":info").c_str(), "r")) != NULL) {
      info_file = true;
      read_matrix_market_banner_and_size(ff, matcode, _MM, _NN, nz, filename + ":info");
      fclose(ff);
    }
  }
  if ((f = fopen(filename.c_str(), "r")) == NULL) {
    if (type == VALIDATION){
      std::cout<<std::endl;
      return; //missing validaiton data
    }
    else logstream(LOG_FATAL)<<"Failed to open input file: " << filename << std::endl;
  }
  if (!info_file && nodes == 0 && edges == 0){
    read_matrix_market_banner_and_size(f, matcode, _MM, _NN, nz, filename);
  }
  else if (nodes > 0 && edges > 0){
    _MM = _NN = nodes;
    nz = edges;
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
    logstream(LOG_INFO) << "Opened matrix size: " <<M << " x " << N << " edges: " << L << " Global mean is: " << globalMean << " time bins: " << K << " Now creating shards." << std::endl;
  else
    logstream(LOG_INFO) << "Opened VLIDATION matrix size: " <<Me << " x " << Ne << " edges: " << Le << " Global mean is: " << globalMean2 << " time bins: " << K << " Now creating shards." << std::endl;
}

void write_global_mean(std::string base_filename, int type){
  FILE * outf = fopen((base_filename + ".gm").c_str(), "w");
  if (type == TRAINING)
    fprintf(outf, "%d\n%d\n%ld\n%lg\n%d\n", M, N, L, globalMean, K);
  else
    fprintf(outf, "%d\n%d\n%ld\n%lg\n%d\n", Me, Ne, Le, globalMean2, K);
  fclose(outf);
}

void compute_matrix_size(size_t nz, int type){
  if (kfold_cross_validation > 0){
    if (type == TRAINING)
      L = (1 - 1.0/(double)kfold_cross_validation)*nz; 
    else Le = (1.0/(double)kfold_cross_validation)*nz;  
  }
  else {
    if (type == TRAINING)
      L = nz;
    else Le = nz;
  }

  if (type == TRAINING)
    logstream(LOG_INFO) << "Starting to read matrix-market input. Matrix dimensions: "
      << M << " x " << N << ", non-zeros: " << L << std::endl;
  else
    logstream(LOG_INFO) << "Starting to read VALIDATION matrix-market input. Matrix dimensions: "
      << Me << " x " << Ne << ", non-zeros: " << Le << std::endl;

}
/** decide on training vs. validation split in case of k fold cross validation */
bool decide_if_edge_is_active(size_t i, int type){
  bool active_edge = true;
  if (type == TRAINING){
    if (kfold_cross_validation > 0 && (((int)(i % kfold_cross_validation)) == kfold_cross_validation_index))
      active_edge = false;
  }
  else if (type == VALIDATION){
    if (kfold_cross_validation > 0){
     if ((((int)(i % kfold_cross_validation)) == kfold_cross_validation_index))
      active_edge = true;
     else active_edge = false;
    }
  }
  return active_edge;
}


template<typename vertex_data>
struct  MMOutputter_vec{
  MMOutputter_vec(std::string fname, uint start, uint end, int index, std::string comment, std::vector<vertex_data> & latent_factors_inmem)  {
    MM_typecode matcode;
    set_matcode(matcode, R_output_format);
    FILE * outf = open_file(fname.c_str(), "w");
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    if (R_output_format)
      mm_write_mtx_crd_size(outf, end-start, 1, end-start);
    else
      mm_write_mtx_array_size(outf, end-start, 1);
    for (uint i=start; i< end; i++)
      if (R_output_format)
         fprintf(outf, "%d %d %12.8g\n", i-start+input_file_offset, 1, latent_factors_inmem[i].get_val(index));
      else
         fprintf(outf, "%1.12e\n", latent_factors_inmem[i].get_val(index));
    fclose(outf);
  }

};


template<typename vertex_data>
struct  MMOutputter_mat{
  MMOutputter_mat(std::string fname, uint start, uint end, std::string comment, std::vector<vertex_data> & latent_factors_inmem, int size = 0)  {
    assert(start < end);
    MM_typecode matcode;
    set_matcode(matcode, R_output_format);
    FILE * outf = open_file(fname.c_str(), "w");
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    int actual_Size = size > 0 ? size : latent_factors_inmem[start].pvec.size();

    if (R_output_format)
      mm_write_mtx_crd_size(outf, end-start, actual_Size, (end-start)*actual_Size);
    else
      mm_write_mtx_array_size(outf, end-start, actual_Size);

    for (uint i=start; i < end; i++){
      for(int j=0; j < actual_Size; j++) {
        if (R_output_format)
          fprintf(outf, "%d %d %12.8g\n", i-start+input_file_offset, j+input_file_offset, latent_factors_inmem[i].get_val(j));
        else
          fprintf(outf, "%1.12e\n", latent_factors_inmem[i].get_val(j));
      }
      }
    fclose(outf);
  }
};

struct  MMOutputter_scalar {
  MMOutputter_scalar(std::string fname, std::string comment, double val)  {
    MM_typecode matcode;
    set_matcode(matcode, R_output_format);
    FILE * outf = open_file(fname.c_str(), "w");
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());

    if (R_output_format)
      mm_write_mtx_crd_size(outf, 1, 1, 1);
    else 
      mm_write_mtx_array_size(outf, 1, 1);

    if (R_output_format)
      fprintf(outf, "%d %d %12.8g\n", 1, 1, val);
    else
      fprintf(outf, "%1.12e\n", val);
    fclose(outf);
  }

};




/**
 * Create a bipartite graph from a matrix. Each row corresponds to vertex
 * with the same id as the row number (0-based), but vertices correponsing to columns
 * have id + num-rows.
 * Line format of the type
 * [user] [item] [time/weight] [rating]
 */

template <typename als_edge_type>
int convert_matrixmarket4(std::string base_filename, bool add_time_edges = false, bool square = false, int type = TRAINING, int matlab_time_offset = 1) {
  // Note, code based on: http://math.nist.gov/MatrixMarket/mmio/c/example_read.c
  FILE *f = NULL;
  size_t nz;
  /**
   * Create sharder object
   */
  int nshards;
  if ((nshards = find_shards<als_edge_type>(base_filename, get_option_string("nshards", "auto")))) {

    if (check_origfile_modification_earlier<als_edge_type>(base_filename, nshards)) {
      logstream(LOG_INFO) << "File " << base_filename << " was already preprocessed, won't do it again. " << std::endl;
      read_global_mean(base_filename, type);
    }
    return nshards;
  }

  sharder<als_edge_type> sharderobj(base_filename);
  sharderobj.start_preprocessing();


  detect_matrix_size(base_filename, f, type == TRAINING? M:Me, type == TRAINING? N:Ne, nz);
  if (f == NULL){
    if (type == VALIDATION){
      logstream(LOG_INFO)<< "Did not find validation file: " << base_filename << std::endl;
      return -1;
    }
    else if (type == TRAINING)
      logstream(LOG_FATAL)<<"Failed to open training input file: " << base_filename << std::endl;
  }

  compute_matrix_size(nz, type); 

  uint I, J;
  double val, time;
  bool active_edge = true;

    for (size_t i=0; i<nz; i++)
    {
      int rc = fscanf(f, "%d %d %lg %lg\n", &I, &J, &time, &val);
      if (rc != 4)
        logstream(LOG_FATAL)<<"Error when reading input file - line " << i << std::endl;
      if (time < 0)
        logstream(LOG_FATAL)<<"Time (third columns) should be >= 0 " << std::endl;
      I-=input_file_offset;  /* adjust from 1-based to 0-based */
      J-=input_file_offset;
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

      active_edge = decide_if_edge_is_active(i, type);

      if (active_edge){
        if (type == TRAINING)
        globalMean += val;
        else globalMean2 += val;
        sharderobj.preprocessing_add_edge(I, (square? J : (M + J)), als_edge_type(val, time+M+N));
      }
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
int convert_matrixmarket_and_item_similarity(std::string base_filename, std::string similarity_file, int tokens_per_row, vec & degrees) {
  FILE *f = NULL, *fsim = NULL;
  size_t nz, nz_sim;
  /**
   * Create sharder object
   */
  int nshards;
  if ((nshards = find_shards<als_edge_type>(base_filename, get_option_string("nshards", "auto")))) {
    if (check_origfile_modification_earlier<als_edge_type>(base_filename, nshards)) {
      logstream(LOG_INFO) << "File " << base_filename << " was already preprocessed, won't do it again. " << std::endl;
      read_global_mean(base_filename, TRAINING);
      return nshards;
    }
  }

  sharder<als_edge_type> sharderobj(base_filename);
  sharderobj.start_preprocessing();

  detect_matrix_size(base_filename, f, M, N, nz);
  if (f == NULL)
    logstream(LOG_FATAL)<<"Failed to open training input file: " << base_filename << std::endl;
  uint N_row = 0 ,N_col = 0;
  detect_matrix_size(similarity_file, fsim, N_row, N_col, nz_sim);
  if (fsim == NULL || nz_sim == 0)
    logstream(LOG_FATAL)<<"Failed to open item similarity input file: " << similarity_file << std::endl;
  if (N_row != N || N_col != N)
    logstream(LOG_FATAL)<<"Wrong item similarity file matrix size: " << N_row <<" x " << N_col << "  Instead of " << N << " x " << N << std::endl;
  L=nz + nz_sim;

  degrees.resize(M+N);

  uint I, J;
  double val = 1.0;
  int zero_entries = 0;
  unsigned int actual_edges = 0;
    logstream(LOG_INFO) << "Starting to read matrix-market input. Matrix dimensions: "
      << M << " x " << N << ", non-zeros: " << nz << std::endl;

    for (size_t i=0; i<nz; i++){
      if (tokens_per_row == 3){
        int rc = fscanf(f, "%u %u %lg\n", &I, &J, &val);
        if (rc != 3)
          logstream(LOG_FATAL)<<"Error when reading input file in line: " << i << std::endl;
        if (val == 0 && ! allow_zeros)
          logstream(LOG_FATAL)<<"Zero weight encountered at input file line: " << i << " . Run with --allow_zeros=1 to ignore zero weights." << std::endl;
        else if (val == 0) { zero_entries++; continue; }
      }
      else if (tokens_per_row == 2){
        int rc = fscanf(f, "%u %u\n", &I, &J);
        if (rc != 2)
          logstream(LOG_FATAL)<<"Error when reading input file: " << i << std::endl;
      }
      else assert(false);

      I-=input_file_offset;  /* adjust from 1-based to 0-based */
      J-=input_file_offset;
      if (I >= M)
        logstream(LOG_FATAL)<<"Row index larger than the matrix row size " << I << " > " << M << " in line: " << i << std::endl;
      if (J >= N)
        logstream(LOG_FATAL)<<"Col index larger than the matrix col size " << J << " > " << N << " in line; " << i << std::endl;
      degrees[J+M]++;
      degrees[I]++;
      if (I< (uint)start_user || I >= (uint)end_user){
         continue;
      }
      sharderobj.preprocessing_add_edge(I, M + J, als_edge_type((float)val, 0));
      //std::cout<<"adding an edge: " <<I << " -> " << M+J << std::endl;
      actual_edges++;
    }

    logstream(LOG_DEBUG)<<"Finished loading " << actual_edges << " ratings from file: " << base_filename << std::endl;

    for (size_t i=0; i<nz_sim; i++){
      if (tokens_per_row == 3){
        int rc = fscanf(fsim, "%u %u %lg\n", &I, &J, &val);
        if (rc != 3)
          logstream(LOG_FATAL)<<"Error when reading input file: " << similarity_file << " line: " << i << std::endl;
      }
      else if (tokens_per_row == 2){
        int rc = fscanf(fsim, "%u %u\n", &I, &J);
        if (rc != 2)
          logstream(LOG_FATAL)<<"Error when reading input file: " << i << std::endl;
      }
      else assert(false);

      I-=input_file_offset;  /* adjust from 1-based to 0-based */
      J-=input_file_offset;
      if (I >= N)
        logstream(LOG_FATAL)<<"Row index larger than the matrix row size " << I << " > " << M << " in line: " << i << std::endl;
      if (J >= N)
        logstream(LOG_FATAL)<<"Col index larger than the matrix col size " << J << " > " << N << " in line; " << i << std::endl;
      if (I == J)
        logstream(LOG_FATAL)<<"Item similarity to itself found for item " << I << " in line; " << i << std::endl;
      //std::cout<<"Adding an edge between "<<M+I<< " : " << M+J << "  " << (I<J)  << " " << val << std::endl; 
      sharderobj.preprocessing_add_edge(M+I, M+J, als_edge_type(I < J? val: 0, I>J? val: 0));
      actual_edges++;
    }

    L = actual_edges;
    logstream(LOG_DEBUG)<<"Finished loading " << nz_sim << " ratings from file: " << similarity_file << std::endl;
    write_global_mean(base_filename, TRAINING);
    sharderobj.end_preprocessing();

    if (zero_entries)
      logstream(LOG_WARNING)<<"Found " << zero_entries << " edges with zero weight!" << std::endl;
    
  fclose(f);
  fclose(fsim);


  logstream(LOG_INFO) << "Now creating shards." << std::endl;

  // Shard with a specified number of shards, or determine automatically if not defined
  nshards = sharderobj.execute_sharding(get_option_string("nshards", "auto"));
  logstream(LOG_INFO) << "Successfully finished sharding for " << base_filename << std::endl;
  logstream(LOG_INFO) << "Created " << nshards << " shards." << std::endl;

  return nshards;
}


/**
 * Create a bipartite graph from a matrix. Each row corresponds to vertex
 * with the same id as the row number (0-based), but vertices correponsing to columns
 * have id + num-rows.
 */
template <typename als_edge_type>
int convert_matrixmarket(std::string base_filename, size_t nodes = 0, size_t edges = 0, int tokens_per_row = 3, int type = TRAINING, int allow_square = true) {
  // Note, code based on: http://math.nist.gov/MatrixMarket/mmio/c/example_read.c
  FILE *f;
  size_t nz;

  /**
   * Create sharder object
   */
  int nshards;
  if ((nshards = find_shards<als_edge_type>(base_filename, get_option_string("nshards", "auto")))) {
    if (check_origfile_modification_earlier<als_edge_type>(base_filename, nshards)) {
      logstream(LOG_INFO) << "File " << base_filename << " was already preprocessed, won't do it again. " << std::endl;
      read_global_mean(base_filename, type);
      return nshards;
    }
  }

  sharder<als_edge_type> sharderobj(base_filename);
  sharderobj.start_preprocessing();

  detect_matrix_size(base_filename, f, type == TRAINING?M:Me, type == TRAINING?N:Ne, nz, nodes, edges, type);
  if (f == NULL){
    if (type == TRAINING){
      logstream(LOG_FATAL)<<"Failed to open training input file: " << base_filename << std::endl;
    }
    else if (type == VALIDATION){
      logstream(LOG_INFO)<<"Validation file: "  << base_filename << " is not found. " << std::endl;
      return -1;
    }
  }

  compute_matrix_size(nz, type);   
  uint I, J;
  double val = 1.0;
  bool active_edge = true;
  int zero_entries = 0;

  for (size_t i=0; i<nz; i++)
    {
      if (tokens_per_row == 3){
        int rc = fscanf(f, "%u %u %lg\n", &I, &J, &val);
        if (rc != 3)
          logstream(LOG_FATAL)<<"Error when reading input file: " << i << std::endl;
        if (val == 0 && ! allow_zeros)
          logstream(LOG_FATAL)<<"Encountered zero edge [ " << I << " " <<J << " 0] in line: " << i << " . Run with --allow_zeros=1 to ignore zero weights." << std::endl;
        else if (val == 0){
           zero_entries++;
           continue;
        }
      }
      else if (tokens_per_row == 2){
        int rc = fscanf(f, "%u %u\n", &I, &J);
        if (rc != 2)
          logstream(LOG_FATAL)<<"Error when reading input file: " << i << std::endl;
      }
      else assert(false);

      if (I ==987654321 || J== 987654321) //hack - to be removed later
        continue;
      I-=(uint)input_file_offset;  /* adjust from 1-based to 0-based */
      J-=(uint)input_file_offset;
      if (I >= M)
        logstream(LOG_FATAL)<<"Row index larger than the matrix row size " << I+1 << " > " << M << " in line: " << i << std::endl;
      if (J >= N)
        logstream(LOG_FATAL)<<"Col index larger than the matrix col size " << J+1 << " > " << N << " in line; " << i << std::endl;
      if (minval != -1e100 && val < minval)
        logstream(LOG_FATAL)<<"Found illegal rating value: " << val << " where min value is: " << minval << std::endl;
      if (maxval != 1e100 && val > maxval)
        logstream(LOG_FATAL)<<"Found illegal rating value: " << val << " where max value is: " << maxval << std::endl;

      active_edge = decide_if_edge_is_active(i, type);

      if (active_edge){
        if (type == TRAINING)
          globalMean += val;
        else globalMean2 += val;
        sharderobj.preprocessing_add_edge(I, (M==N && allow_square)?J:M + J, als_edge_type((float)val));
      } 
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

  if (zero_entries)
     logstream(LOG_WARNING)<<"Found " << zero_entries << " zero edges!" << std::endl; 
  fclose(f);


  logstream(LOG_INFO) << "Now creating shards." << std::endl;

  // Shard with a specified number of shards, or determine automatically if not defined
  nshards = sharderobj.execute_sharding(get_option_string("nshards", "auto"));
  logstream(LOG_INFO) << "Successfully finished sharding for " << base_filename<< std::endl;
  logstream(LOG_INFO) << "Created " << nshards << " shards." << std::endl;

  return nshards;
}



void load_matrix_market_vector(const std::string & filename,
    int type, bool optional_field, bool allow_zeros, int offset = 0)
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
    vertex_data & vdata = latent_factors_inmem[row+offset];
    vdata.set_val(type, val);
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

  if (D != (int)cols)
    logstream(LOG_FATAL)<<"Wrong matrix size detected, command line argument should be --D=" << D << " instead of : " << cols << std::endl;

  for (i=0; i<nnz; i++){
    if (mm_is_sparse(matcode)){
      rc = fscanf(f, "%u %u %lg\n", &I, &J, &val);
      if (rc != 3)
        logstream(LOG_FATAL)<<"Error reading input line " << i << std::endl;
      I--; J--;
      assert(I >= 0 && I < rows);
      assert(J >= 0 && J < cols);
      //set_val(a, I, J, val);
      latent_factors_inmem[I+offset].set_val(J,val);
    }
    else {
      rc = fscanf(f, "%lg", &val);
      if (rc != 1)
        logstream(LOG_FATAL)<<"Error reading nnz " << i << std::endl;
      I = i / cols;
      J = i % cols;
      latent_factors_inmem[I+offset].set_val(J, val);
    }
  }
  logstream(LOG_INFO) << "Factors from file: loaded matrix of size " << rows << " x " << cols << " from file: " << filename << " total of " << nnz << " entries. "<< i << std::endl;
  fclose(f);
}

#endif

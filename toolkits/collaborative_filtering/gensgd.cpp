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
 * @section DESCRIPTION
 *
 * Implementation of the gensgd algorithm.
 * Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th IEEE International Conference on Data Mining (ICDM 2010), Sydney, Australia.
 * Original implementation by Qiang Yan, Chinese Academy of Science.
 * note: this code version implements the SGD version of gensgd. In the original library there are also ALS and MCMC methods.
 * Also the treatment of features is richer in gensgd. The code here can serve for a quick evaluation but the user
 * is encouraged to try gensgd as well.
 */



#include "graphchi_basic_includes.hpp"
#include "common.hpp"
#include "eigen_wrapper.hpp"
#define MAX_FEATAURES 21
#define FEATURE_WIDTH 1 //MAX NUMBER OF ALLOWED FEATURES IN TEXT FILE
float minarray[FEATURE_WIDTH];
float maxarray[FEATURE_WIDTH];
float meanarray[FEATURE_WIDTH];
int actual_features = FEATURE_WIDTH;
int total_features = 0;
bool feature_selection[MAX_FEATAURES];
std::string default_feature_str;

double gensgd_rate = 1e-02;
double gensgd_mult_dec = 0.9;
double gensgd_regw = 1e-3;
double gensgd_regv = 1e-3;
double gensgd_reg0 = 1e-1;
int D = 20; //feature vector width, can be changed on runtime using --D=XX flag
bool debug = false;
int last_item = 0;

int num_feature_bins(){
  int sum = 0;
  for (int i=0; i< total_features; i++)
    sum += ((maxarray[i] - minarray[i]) + 1);
  if (total_features > 0)
    assert(sum > 0);
  return sum;
}
int get_offset(int i){
  int offset = 0;
   if (i >= 1)
     offset += M;
   if (i >= 2)
     offset += N;
   for (int j=2; j < i; j++)
     offset += ((maxarray[j-2]-minarray[j-2])+1);
   return offset;
}
int * offsets;

bool is_user(vid_t id){ return id < M; }
bool is_item(vid_t id){ return id >= M && id < N; }
bool is_time(vid_t id){ return id >= M+N; }

inline double sum(double * pvec){
  double tsum = 0;
  for (int j=0; j< D; j++)
    tsum += pvec[j];
  return tsum;
}


struct vertex_data {
  vec pvec;
  double rmse;
  double bias;
  int last_item;

  vertex_data() {
    rmse = 0;
    bias = 0;
    last_item = 0;
  }


};

struct edge_data {
  float features[FEATURE_WIDTH];
  float weight;
  edge_data() { weight = 0; memset(features, 0, sizeof(float)*FEATURE_WIDTH); }

  edge_data(float weight, float * valarray): weight(weight) { memcpy(features, valarray, sizeof(float)*FEATURE_WIDTH); }
};



/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef edge_data EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

float gensgd_predict(const vertex_data** node_array, int node_array_size,
    const float rating, double& prediction, vec * sum){

  //vertex_data & last_item = latent_factors_inmem[M+N+K+(*user.last_item)]; //TODO, when no ratings, last item is 0
  vec sum_sqr = zeros(D);
  *sum = zeros(D);
  prediction = globalMean;
  for (int i=0; i< node_array_size; i++)
    prediction += node_array[i]->bias;
  
  for (int j=0; j< D; j++){
    for (int i=0; i< node_array_size; i++){
      sum->operator[](j) += node_array[i]->pvec[j];
      sum_sqr[j] += pow(node_array[i]->pvec[j],2);
    }
    prediction += 0.5 * (pow(sum->operator[](j),2) - sum_sqr[j]);
  }
  //truncate prediction to allowed values
  prediction = std::min((double)prediction, maxval);
  prediction = std::max((double)prediction, minval);
  //return the squared error
  float err = rating - prediction;
  assert(!std::isnan(err));
  return err*err; 

}
float gensgd_predict(const vertex_data** node_array, int node_array_size,
    const float rating, double & prediction){
   vec sum;
   return gensgd_predict(node_array, node_array_size, rating, prediction, &sum);
}

void init_gensgd(){

  srand(time(NULL));
  int nodes = M+N+num_feature_bins()+last_item*M;
  latent_factors_inmem.resize(nodes);
  offsets = new int[total_features+2+last_item];
  for (int i=0; i< total_features+2; i++){
    offsets[i] = get_offset(i);
    assert(offsets[i] < nodes);
  }
  if (last_item)
    offsets[2+total_features] = M+N+num_feature_bins();
  assert(D > 0);
  double factor = 0.1/sqrt(D);
#pragma omp parallel for
  for (int i=0; i< nodes; i++){
      latent_factors_inmem[i].pvec = (debug ? 0.1*ones(D) : (::randu(D)*factor));
  }
}

#include "io.hpp"
#include "rmse.hpp"


/**
  compute validation rmse
  */
void validation_rmse_N(float (*prediction_func)(const vertex_data ** array, int arraysize, float rating, double & prediction)
    ,graphchi_context & gcontext, int feature_num, int last_item, int * offsets, float * minarray, bool * feature_selection, int total_features, bool square = false) {

  assert(total_features <= feature_num);
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
  uint I, J;
  char * linebuf = NULL;
  char linebuf_debug[1024];
  size_t linesize;
  float * valarray = new float[total_features];
  float val;
  vertex_data ** node_array = new vertex_data*[2+total_features+last_item];

  for (size_t i=0; i<nz; i++)
  {
    /* READ LINE */
    int rc = getline(&linebuf, &linesize, f);
    if (rc == -1)
      logstream(LOG_FATAL)<<"Failed to get line: " << i << " in file: " << validation << std::endl;
    strncpy(linebuf_debug, linebuf, 1024);

    /* READ FROM */
    char *pch = strtok(linebuf,"\t,\r ");
    if (pch == NULL)
      logstream(LOG_FATAL)<<"Error reading line " << i << " [ " << linebuf_debug << " ] " << std::endl;
    I = atoi(pch); I--;
    if (I >= M)
      logstream(LOG_FATAL)<<"Row index larger than the matrix row size " << I << " > " << M << " in line: " << i << std::endl;

    /* READ TO */
    pch = strtok(NULL, "\t,\r ");
    if (pch == NULL)
      logstream(LOG_FATAL)<<"Error reading line " << i << " [ " << linebuf_debug << " ] " << std::endl;
    J = atoi(pch); J--;
    if (J >= N)
      logstream(LOG_FATAL)<<"Col index larger than the matrix col size " << J << " > " << N << " in line; " << i << std::endl;

    /* READ FEATURES */
    int index = 0;
    for (int j=0; j< feature_num; j++){
      pch = strtok(NULL, "\t,\r ");
      if (pch == NULL)
        logstream(LOG_FATAL)<<"Error reading line " << i << " feature " << j << " [ " << linebuf_debug << " ] " << std::endl;
      if (!feature_selection[j])
        continue;

      valarray[index] = atof(pch); 
      if (std::isnan(valarray[index]))
        logstream(LOG_FATAL)<<"Error reading line " << i << " feature " << j << " [ " << linebuf_debug << " ] " << std::endl;
      index++;
    }

    /* READ RATING */
    pch = strtok(NULL, "\t,\r ");
    if (pch == NULL)
      logstream(LOG_FATAL)<<"Error reading line " << i << " [ " << linebuf_debug << " ] " << std::endl;
    val = atof(pch);
    if (std::isnan(val))
      logstream(LOG_FATAL)<<"Error reading line " << i << " rating "  << " [ " << linebuf_debug << " ] " << std::endl;


    /* COMPUTE PREDICTION */
    double prediction;
    node_array[0] = &latent_factors_inmem[I];
    node_array[1] = &latent_factors_inmem[square?J:J+M];
    for (int j=0; j< total_features; j++){
      uint pos = valarray[j]+offsets[j+2]-minarray[j];
      assert(pos >= 0 && pos < latent_factors_inmem.size());
      node_array[j+2] = & latent_factors_inmem[pos];
    }
    if (last_item){
      uint pos = latent_factors_inmem[I].last_item + offsets[total_features+2];
      assert(pos < latent_factors_inmem.size());
      node_array[total_features+2] = &latent_factors_inmem[pos];
    }

    (*prediction_func)((const vertex_data**)node_array, 2+total_features+last_item, val, prediction);
    dvalidation_rmse += pow(prediction - val, 2);
  }
  fclose(f);
  delete[] node_array;

  assert(Le > 0);
  dvalidation_rmse = sqrt(dvalidation_rmse / (double)Le);
  std::cout<<"  Validation RMSE: " << std::setw(10) << dvalidation_rmse << std::endl;
  if (halt_on_rmse_increase && dvalidation_rmse > last_validation_rmse && gcontext.iteration > 0){
    logstream(LOG_WARNING)<<"Stopping engine because of validation RMSE increase" << std::endl;
    gcontext.set_last_iteration(gcontext.iteration);
  }
}

void test_predictions_N(float (*prediction_func)(const vertex_data ** node_array, int node_array_size, float rating, double & prediction), int feature_num, int * offsets, bool * feature_selection, int total_features, float * minarray,  bool square = false) {
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
  char * linebuf;
  char linebuf_debug[1024];
  size_t linesize;
  float * valarray = new float[feature_num];
  vertex_data ** node_array = new vertex_data*[2+feature_num];
  float val;
  uint I,J;

  for (uint i=0; i<nz; i++)
  {

    /* READ LINE */
    int rc = getline(&linebuf, &linesize, f);
    if (rc == -1)
      logstream(LOG_FATAL)<<"Failed to get line number: " << i << " in file: " << test <<std::endl;
    strncpy(linebuf_debug, linebuf, 1024);

    /* READ FROM */
    char *pch = strtok(linebuf,"\t,\r ");
    if (pch == NULL)
      logstream(LOG_FATAL)<<"Error reading line " << i << " [ " << linebuf_debug << " ] " << std::endl;
    I = atoi(pch); I--;
    pch = strtok(NULL, "\t,\r ");

      /* READ TO */
      if (pch == NULL)
        logstream(LOG_FATAL)<<"Error reading line " << i << " [ " << linebuf_debug << " ] " << std::endl;
    J = atoi(pch); J--;
    if (I >= M)
      logstream(LOG_FATAL)<<"Row index larger than the matrix row size " << I << " > " << M << " in line: " << i << std::endl;
    if (J >= N)
      logstream(LOG_FATAL)<<"Col index larger than the matrix col size " << J << " > " << N << " in line; " << i << std::endl;

    /* READ FEATURES */
    int index = 0;
    for (int j=0; j< feature_num; j++){
      pch = strtok(NULL, "\t,\r ");
      if (pch == NULL)
        logstream(LOG_FATAL)<<"Error reading line " << i << " feature " << j << " [ " << linebuf_debug << " ] " << std::endl;
      if (!feature_selection[j])
        continue;

      valarray[index] = atof(pch); 
      if (std::isnan(valarray[j]))
        logstream(LOG_FATAL)<<"Error reading line " << i << " feature " << j << " [ " << linebuf_debug << " ] " << std::endl;
      index++;
    }

    /* READ VAL (OPTIONAL) */
    pch = strtok(NULL, "\t,\r ");
    if (pch == NULL){ }
    else {
      val = atof(pch);
      if (std::isnan(val))
        logstream(LOG_FATAL)<<"Error reading line " << i << " rating "  << " [ " << linebuf_debug << " ] " << std::endl;
    }

    double prediction;
    node_array[0] = &latent_factors_inmem[I] + offsets[0];
    node_array[1] = &latent_factors_inmem[square?J:J+offsets[1]];
    for (int j=0; j< total_features; j++){
      uint pos = offsets[j+2] + valarray[j] - minarray[j];
      assert(pos >=0 && pos < latent_factors_inmem.size());
      node_array[j+2] = & latent_factors_inmem[pos]; 
    }
    //vec sum;
    (*prediction_func)((const vertex_data**)node_array, 2+feature_num,0 , prediction);
    fprintf(fout, "%d %d %12.8lg\n", I+1, J+1, prediction);
  }
  fclose(f);
  fclose(fout);

  logstream(LOG_INFO)<<"Finished writing " << nz << " predictions to file: " << test << ".predict" << std::endl;
}




/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct LIBFMVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

  /*
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {


  if (last_item && gcontext.iteration == 0){
    if (is_user(vertex.id()) && vertex.num_outedges() > 0) { //user node. find the last rated item and store it. we assume items are sorted by time!
      vertex_data& user = latent_factors_inmem[vertex.id()]; 
      int max_time = 0;
      for(int e=0; e < vertex.num_outedges(); e++) {
        const edge_data & edge = vertex.edge(e)->get_data();
        if (edge.features[0]-1 >= max_time){ //first feature is time
          max_time = edge.features[0]-1;
          user.last_item = vertex.edge(e)->vertex_id() - M;
        }
      }
     }
    else if (is_user(vertex.id()) && vertex.num_outedges() == 0)
      logstream(LOG_WARNING)<<"Vertex: " << vertex.id() << " with no edges: " << std::endl;
    return;
  } 
 
    //go over all user nodes
    if (is_user(vertex.id())){
      vertex_data& user = latent_factors_inmem[vertex.id()]; 
      user.rmse = 0; 
      assert(user.last_item >= 0 && user.last_item < (int)N);

      for(int e=0; e < vertex.num_outedges(); e++) {
        const edge_data & data = vertex.edge(e)->get_data();
        float rui = data.weight;
        double pui;
        vec sum;
        //vertex_data & time = latent_factors_inmem[(int)vertex.edge(e)->get_data().time - time_offset];
        vertex_data *relevant_features[2+total_features+last_item];
        relevant_features[0] = &user;
        relevant_features[1] = &latent_factors_inmem[vertex.edge(e)->vertex_id()];
        for (int i=0; i< total_features; i++){
          uint pos = get_offset(i+2) + data.features[i] - minarray[i];
          assert(pos >= 0 && pos < latent_factors_inmem.size());
          relevant_features[i+2] = &latent_factors_inmem[pos];
        }
        if (last_item){
          uint pos = M+N+num_feature_bins()+user.last_item;
          assert(pos < latent_factors_inmem.size());
          relevant_features[2+total_features] = &latent_factors_inmem[pos];
        }
        float sqErr = gensgd_predict((const vertex_data**)relevant_features, 2+total_features+last_item, rui, pui, &sum);
        float eui = pui - rui;

        globalMean -= gensgd_rate * (eui + gensgd_reg0 * globalMean);
        //user.bias -= gensgd_rate * (eui + gensgd_regw * user.bias);
        //movie.bias -= gensgd_rate * (eui + gensgd_regw * movie.bias);
        //time.bias -= gensgd_regw * (eui + gensgd_regw * time.bias);
        //assert(!std::isnan(time.bias));
        for (int i=0; i < total_features + last_item; i++){
          relevant_features[i]->bias -= gensgd_rate * (eui + gensgd_regw* relevant_features[i]->bias);
          vec grad = sum - relevant_features[i]->pvec;
          relevant_features[i]->pvec -= gensgd_rate * (eui*grad + gensgd_regv * relevant_features[i]->pvec);
        }

        //last_item.bias -= gensgd_regw * (eui + gensgd_regw * last_item.bias);
       // float grad;
        /*for(int f = 0; f < D; f++){
          // user
          grad = sum[f] - user.v[f];
          user.v[f] -= gensgd_rate * (eui * grad + gensgd_regv * user.v[f]);
          // item
          grad = sum[f] - movie.v[f];
          movie.v[f] -= gensgd_rate * (eui * grad + gensgd_regv * movie.v[f]);
          // time
          grad = sum[f] - time.pvec[f];
          time.pvec[f] -= gensgd_rate * (eui * grad + gensgd_regv * time.pvec[f]);
          // last item
          grad = sum[f] - last_item.pvec[f];
          last_item.pvec[f] -= gensgd_rate * (eui * grad + gensgd_regv * last_item.pvec[f]);

        }*/

        user.rmse += sqErr;
      }

    }

  };

  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    gensgd_rate *= gensgd_mult_dec;
    training_rmse(iteration, gcontext);
    validation_rmse_N(&gensgd_predict, gcontext, FEATURE_WIDTH, last_item, offsets, minarray, feature_selection, total_features);
  };


};

struct  MMOutputter_bias{
  FILE * outf;
  MMOutputter_bias(std::string fname, uint start, uint end, std::string comment)  {
    MM_typecode matcode;
    set_matcode(matcode);
    outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, end-start, 1); 
    for (uint i=start; i< end; i++)
      fprintf(outf, "%1.12e\n", latent_factors_inmem[i].bias);
  }


  ~MMOutputter_bias() {
    if (outf != NULL) fclose(outf);
  }

};

struct  MMOutputter_global_mean {
  FILE * outf;
  MMOutputter_global_mean(std::string fname, std::string comment)  {
    MM_typecode matcode;
    set_matcode(matcode);
    outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, 1, 1); 
    fprintf(outf, "%1.12e\n", globalMean);
  }

  ~MMOutputter_global_mean() {
    if (outf != NULL) fclose(outf);
  }

};


struct  MMOutputter{
  FILE * outf;
  MMOutputter(std::string fname, uint start, uint end, std::string comment)  {
    assert(start < end);
    MM_typecode matcode;
    set_matcode(matcode);     
    outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, end-start, latent_factors_inmem[start].pvec.size()); 
    for (uint i=start; i < end; i++)
      for(int j=0; j < latent_factors_inmem[i].pvec.size(); j++) {
        fprintf(outf, "%1.12e\n", latent_factors_inmem[i].pvec[j]);
      }
  }

  ~MMOutputter() {
    if (outf != NULL) fclose(outf);
  }

};


void output_gensgd_result(std::string filename) {
  MMOutputter mmoutput(filename + "_U.mm", 0, num_feature_bins(), "This file contains LIBFM output matrices. In each row D factors of a single user node, then item nodes, then features");
   MMOutputter_bias mmoutput_bias(filename + "_U_bias.mm", 0, num_feature_bins(), "This file contains LIBFM output bias vector. In each row a single user bias.");
  MMOutputter_global_mean gmean(filename + "_global_mean.mm", "This file contains LIBFM global mean which is required for computing predictions.");

  logstream(LOG_INFO) << " time-svd++ output files (in matrix market format): " << filename << "_U.mm" << ",  "<< filename <<  "_global_mean.mm, " << filename << "_U_bias.mm "  <<std::endl;
}

int main(int argc, const char ** argv) {


  print_copyright();  

  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("als-tensor-inmemory-factors");

  //specific command line parameters for gensgd
  gensgd_rate = get_option_float("gensgd_rate", gensgd_rate);
  gensgd_regw = get_option_float("gensgd_regw", gensgd_regw);
  gensgd_regv = get_option_float("gensgd_regv", gensgd_regv);
  gensgd_reg0 = get_option_float("gensgd_reg0", gensgd_reg0);
  gensgd_mult_dec = get_option_float("gensgd_mult_dec", gensgd_mult_dec);
  last_item = get_option_int("last_item", last_item);
  D = get_option_int("D", D);
  std::string string_features = get_option_string("features", default_feature_str);
  if (string_features != ""){
  char * pfeatures = strdup(string_features.c_str());
  char * pch = strtok(pfeatures, ",\n\r\t ");
  int node = atoi(pch);
  if (node < 0 || node >= MAX_FEATAURES)
    logstream(LOG_FATAL)<<"Feature id using the --features=XX command should be non negative, starting from zero"<<std::endl;
  feature_selection[node] = true;
  total_features++;
  while ((pch = strtok(NULL, ",\n\r\t "))!= NULL){
    node = atoi(pch);
    if (node < 0 || node >= MAX_FEATAURES)
      logstream(LOG_FATAL)<<"Feature id using the --features=XX command should be non negative, starting from zero"<<std::endl;
    feature_selection[node] = true;
    total_features++;
  }
  }

  logstream(LOG_INFO) <<"Total selected features: " << total_features << " : " << std::endl;
  for (int i=0; i < MAX_FEATAURES; i++)
  if (feature_selection[i])
    logstream(LOG_INFO)<<"Selected feature: " << i << std::endl;

  parse_command_line_args();
  parse_implicit_command_line();

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket_N<edge_data>(training, false, FEATURE_WIDTH, actual_features, minarray, maxarray, meanarray, feature_selection, total_features);
  init_gensgd();

  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, D);
    vec user_bias =      load_matrix_market_vector(training +"_U_bias.mm", false, true);
    for (uint i=0; num_feature_bins(); i++){
        latent_factors_inmem[i].bias = user_bias[i];
    }
    vec gm = load_matrix_market_vector(training + "_global_mean.mm", false, true);
    globalMean = gm[0];
}


  /* Run */
  LIBFMVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output test predictions in matrix-market format */
  output_gensgd_result(training);
  test_predictions_N(&gensgd_predict, FEATURE_WIDTH, offsets, feature_selection, total_features, minarray);    

  /* Report execution metrics */
  metrics_report(m);
  return 0;
}

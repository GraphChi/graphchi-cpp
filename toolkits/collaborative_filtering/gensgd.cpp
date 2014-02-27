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
 * Implementation of the gensgd algorithm. A generalization of SGD algorithm when there are multiple features for each
 * rating, in the form 
 * [from] [to] [feature1] [feature2] [feature3] ... [featureN] [rating] 
 * (It is also possible to dynamically specify column numbers which are relevant)
 * Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th IEEE International Conference on Data Mining (ICDM 2010), Sydney, Australia.
 * Original implementation by Qiang Yan, Chinese Academy of Science.
 * note: this code version implements the SGD version of gensgd. In the original library there are also ALS and MCMC methods.
 * Also the treatment of features is richer in gensgd. The code here can serve for a quick evaluation but the user
 * is encouraged to try gensgd as well.
 */


#include <vector>
#include "common.hpp"
#include "eigen_wrapper.hpp"
#include <omp.h>
#define GRAPHCHI_DISABLE_COMPRESSION //remove this if you want to save memory but increase runtime

double gensgd_rate0 = 1e-03;
double gensgd_rate1 = 1e-03;
double gensgd_rate2 = 1e-03;
double gensgd_rate3 = 1e-03;
double gensgd_rate4 = 1e-03;
double gensgd_mult_dec = 0.9999999;
double gensgd_regw = 1e-3;
double gensgd_regv = 1e-3;
double gensgd_reg0 = 1e-2;
bool debug = false;
std::string user_file; //optional file with user features
std::string item_file; //optional file with item features
std::string user_links; //optional file with user to user links
size_t vertex_with_no_edges = 0;
int calc_error = 0;
int has_user_titles = 0;
int has_item_titles = 0;
float cutoff = 0;
size_t new_validation_users = 0;
size_t new_test_users = 0;
int json_input = 0;
int cold_start = 0;
int binary_prediction = 0;
int verbose = 0; //print statistics about step sizes
int node_id_maps_size = 0;
bool debug2 = false;

enum _cold_start{
   NONE = 0,
   GLOBAL = 1, 
   ITEM = 3
};

vec stat1, stat2, stat3;

#include "parser.hpp"
int num_feature_bins(){
  int sum = 0;
  if (fc.hash_strings){
    assert(2+fc.total_features+fc.node_features == (int)fc.node_id_maps.size());
    for (int i=2; i < 2+fc.total_features+fc.node_features; i++){
      sum+= fc.node_id_maps[i].string2nodeid.size();
    }
  }
  else {
  }
  if (fc.total_features > 0)
    assert(sum > 0);
  return sum;
}

int calc_feature_num(){
  return 2+fc.total_features+fc.node_features;
}
void get_offsets(std::vector<int> & offsets){
  assert(offsets.size() >= 2);
  offsets[0] = 0;
  offsets[1] = M;
  if (offsets.size() >= 3)
    offsets[2] = M+N;
  if (fc.hash_strings){
    for (uint j=2; j< offsets.size()-1; j++){
      offsets[j+1] = offsets[j] + fc.node_id_maps[j].string2nodeid.size();
      logstream(LOG_DEBUG)<<"Offset " << j+1 << " is: " << offsets[j+1] << std::endl;
    }
  } else {
    assert(false);
  }

}


bool is_user(vid_t id){ return id < M; }
bool is_item(vid_t id){ return id >= M && id < M+N; }
bool is_time(vid_t id){ return id >= M+N; }

vec errors_vec;
#define BIAS_POS -1
struct vertex_data {
  vec pvec;
  double bias;
  float avg_rating;
  sparse_vec features;
  sparse_vec links; //links to other users or items

  vertex_data() {
    bias = 0;
    avg_rating = -1;
  }
  void set_val(int index, float val){
    if (index == BIAS_POS)
      bias = val;
    else pvec[index] = val;
  }
  float get_val(int index){
    if (index== BIAS_POS)
      return bias;
    else return pvec[index];
  }


};

vertex_data vertex_dummy; //placeholders for new validaiton/test users

struct edge_data {
  float features[FEATURE_WIDTH];
  float weight;
  edge_data() { weight = 0; memset(features, 0, sizeof(float)*FEATURE_WIDTH); }

  edge_data(float weight, float * valarray, int size): weight(weight) { memcpy(features, valarray, sizeof(float)*size); }
};


vertex_data real_features[FEATURE_WIDTH]; 


/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef edge_data EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;


int calc_feature_node_array_size(uint node, uint item){
  if (node != (uint)-1){
    assert(node <= M);
    assert(node < latent_factors_inmem.size());
  }
  if (item != (uint)-1){
    assert(item <= N);
    assert(fc.offsets[1]+item < latent_factors_inmem.size());
  }
  int ret =  fc.total_features+2;
  if (node != (uint)-1)
    ret += nnz(latent_factors_inmem[node].features);
  if (item != (uint)-1)
    ret += nnz(latent_factors_inmem[fc.offsets[1]+item].features);
  assert(ret > 0);
  return ret;
}



 

/* compute an edge prediction based on input features */
float compute_prediction(
    const uint I, 
    const uint J, 
    const float val, 
    double & prediction, 
    const float * valarray,
    int val_array_len, 
    float (*prediction_func)(const vertex_data ** array, int arraysize, const float * val_array, int val_array_size, float rating, double & prediction, vec * psum), 
    vec * psum, 
    vertex_data **& node_array,
    int type){

  //if (I == (uint)-1 && J == (uint)-1)
  // logstream(LOG_FATAL)<<"BUG: can not compute prediction for new user and new item" << std::endl;
 
  if (J != (uint)-1) 
    assert(J >=0 && J <= N);
  if (I != (uint)-1)
    assert(I>=0 && I <= M);


  /* COMPUTE PREDICTION */
  /* USER NODE **/
  int index = 0;
  int loc = 0;
  if (I != (uint)-1){
    node_array[index] = &latent_factors_inmem[I+fc.offsets[loc]];
    if (debug2) std::cout<<"I: "<<I<< " offset: " << I+fc.offsets[loc] << " " << head(latent_factors_inmem[fc.offsets[loc]].pvec,5).transpose() << " " << latent_factors_inmem[fc.offsets[loc]].bias << std::endl ;
    if (node_array[index]->pvec[0] >= 1e5)
      logstream(LOG_FATAL)<<"Got into numerical problem, try to decrease SGD step size" << std::endl;
  }
  else node_array[index] = &vertex_dummy; 
  index++; 
  loc++;

  /* 1) ITEM NODE */
  if (J != (uint)-1){
    assert(J+fc.offsets[index] < latent_factors_inmem.size());
    node_array[index] = &latent_factors_inmem[J+fc.offsets[loc]];
    if (debug2) std::cout<<"J: "<<J<< " offset: " << J+fc.offsets[loc] << " " << head(latent_factors_inmem[J+fc.offsets[loc]].pvec,5).transpose() <<" " << latent_factors_inmem[J+fc.offsets[loc]].bias << std::endl ;
    if (node_array[index]->pvec[0] >= 1e5)
      logstream(LOG_FATAL)<<"Got into numerical problem, try to decrease SGD step size" << std::endl;
  }
  else node_array[index] = &vertex_dummy; 
  index++; 
  loc++;

  /* 2) FEATURES GIVEN IN RATING LINE */
  for (int j=0; j< fc.total_features; j++){
    assert(fc.feature_positions[j] < FEATURE_WIDTH);
    if (fc.real_features_indicators[fc.feature_positions[j]]){
      node_array[j+index] = & real_features[j];
    } else {
      uint pos = (uint)ceil(valarray[j]+fc.offsets[j+loc]);
      if (pos < 0 || pos >= latent_factors_inmem.size())
        logstream(LOG_FATAL)<<"Bug: j is: " << j << " fc.total_features " << fc.total_features << " index : " << index << " loc: " << loc <<  
          " fc.offsets " << fc.offsets[j+loc] << " vlarray[j] " << valarray[j] << " pos: " << pos << " latent_factors_inmem.size() " << latent_factors_inmem.size() << std::endl;
      node_array[j+index] = & latent_factors_inmem[pos];
      if (debug2) std::cout<<"j+index: "<<j+index<< " offset: " << pos << " " << head(latent_factors_inmem[pos].pvec,5).transpose() << " " << latent_factors_inmem[pos].bias << std::endl;
    }
    if (node_array[j+index]->pvec[0] >= 1e5)
      logstream(LOG_FATAL)<<"Got into numerical problem, try to decrease SGD step size" << std::endl;

  }
  index+= fc.total_features;
  loc += fc.total_features;
  
  /* 3) USER FEATURES */
  if (user_file != ""){
  if (I != (uint)-1){
  int i = 0;
  FOR_ITERATOR(j, latent_factors_inmem[I+fc.offsets[0]].features){
    int pos;
    if (user_links != ""){
      pos = j.index();
      assert(pos < (int)M);
    }
    else {
      pos = j.index()+fc.offsets[loc];
      assert((uint)loc < fc.node_id_maps.size());
      assert(j.index() < (int)fc.node_id_maps[loc].string2nodeid.size());
      assert(pos >= 0 && pos < (int)latent_factors_inmem.size());
      assert(pos >= (int)fc.offsets[loc]);
    }
    //logstream(LOG_INFO)<<"setting index " << i+index << " to: " << pos << std::endl;
    node_array[i+index] = & latent_factors_inmem[pos];
    if (node_array[i+index]->pvec[0] >= 1e5)
      logstream(LOG_FATAL)<<"Got into numerical problem, try to decrease SGD step size" << std::endl;
    i++;
  }
  assert(i == nnz(latent_factors_inmem[I+fc.offsets[0]].features));
  index+= nnz(latent_factors_inmem[I+fc.offsets[0]].features);
  loc+=1;
  }
  }

  /* 4) ITEM FEATURES */
  if (item_file != ""){
  if (J != (uint)-1){
  int i=0;
  FOR_ITERATOR(j, latent_factors_inmem[J+fc.offsets[1]].features){
    uint pos = j.index()+fc.offsets[loc];
    assert(j.index() < (int)fc.node_id_maps[loc].string2nodeid.size());
    assert(pos >= 0 && pos < latent_factors_inmem.size());
    assert(pos >= (uint)fc.offsets[loc]);
    //logstream(LOG_INFO)<<"setting index " << i+index << " to: " << pos << std::endl;
    node_array[i+index] = & latent_factors_inmem[pos];
    if (node_array[i+index]->pvec[0] >= 1e5)
      logstream(LOG_FATAL)<<"Got into numerical problem, try to decrease SGD step size" << std::endl;
    i++;
  }
  assert(i == nnz(latent_factors_inmem[J+fc.offsets[1]].features));
  index+= nnz(latent_factors_inmem[J+fc.offsets[1]].features);
  loc+=1;
  }
  }

  assert(index == calc_feature_node_array_size(I,J));
  (*prediction_func)((const vertex_data**)node_array, calc_feature_node_array_size(I,J), valarray, val_array_len, val, prediction, psum);
  return pow(val - prediction,2);
} 

#include "io.hpp"
#include "../parsers/common.hpp"


/* read node features from file */
void read_node_features(std::string base_filename, bool square, feature_control & fc, bool user, bool binary) {
  FILE *f;

  if ((f = fopen(base_filename.c_str(), "r")) == NULL) {
    logstream(LOG_FATAL) << "Could not open file: " << base_filename << ", error: " << strerror(errno) << std::endl;
  }
  binary = true; //TODO
  double_map fmap;
  fc.node_id_maps.push_back(fmap);
  fc.node_features++;

  uint I, J = -1;
  char * linebuf = NULL;
  char linebuf_debug[1024];
  size_t linesize;
  size_t lines = 0;
  size_t tokens = 0;
  float val = 1;
  int missing_nodes = 0;

  while(true){
    /* READ LINE */
    int rc = getline(&linebuf, &linesize, f);
    if (rc == -1)
      break;
    strncpy(linebuf_debug, linebuf, 1024);
    lines++;
    
    //skip over header titles (if any) 
    if (lines == 1 && user && has_user_titles)
      continue;
    else if (lines == 1 && !user && has_item_titles)
      continue;


    /** READ [FROM] */
    char *pch = strtok(linebuf,ptokens);
    if (pch == NULL)
      logstream(LOG_FATAL)<<"Error reading line " << lines << " [ " << linebuf_debug << " ] " << std::endl;
    I = (uint)get_node_id(pch, user?0:1, lines, true);
    if (I == (uint)-1){ //user id was not found in map, so we do not need this users features
      missing_nodes++;
      continue;
    }

    if (user)
      assert(I >= 0 && I < M);
    else assert(I>=0  && I< N);


    /** READ USER FEATURES */
    while (pch != NULL){
      pch = strtok(NULL, ptokens);
      if (pch == NULL)
        break;
      if (binary){
        J = (uint)get_node_id(pch, 2+fc.total_features+fc.node_features-1, tokens, lines);
      }
      else { 
        pch = strtok(NULL, ptokens);
        if (pch == NULL)
          logstream(LOG_FATAL)<<"Failed to read feture value" << std::endl;
        val = atof(pch);
      }
      assert(J >= 0);
      if (user)
        assert(I < latent_factors_inmem.size());
      else assert(I+M < latent_factors_inmem.size());
      set_new(latent_factors_inmem[user? I : I+M].features, J, val);
      tokens++;
    }
  }

  assert(tokens > 0);
  logstream(LOG_DEBUG)<<"Read a total of " << lines << " node features. Tokens: " << tokens << " avg tokens: " << (lines/tokens) 
    << " user? " << user <<  " new entries: " << fc.node_id_maps[2+fc.total_features+fc.node_features-1].string2nodeid.size() << std::endl;
  if (missing_nodes > 0)
    std::cerr<<"Warning: missing: " << missing_nodes << " from node feature file: " << base_filename << " out of: " << lines << std::endl;
}


/* read node features from file */
void read_node_links(std::string base_filename, bool square, feature_control & fc, bool user, bool binary) {
  FILE *f;

  if ((f = fopen(base_filename.c_str(), "r")) == NULL) {
    logstream(LOG_FATAL) << "Could not open file: " << base_filename << ", error: " << strerror(errno) << std::endl;
  }
  //double_map fmap;
  //fc.node_id_maps.push_back(fmap);
  fc.node_links++;

  uint I, J = -1;
  char * linebuf = NULL;
  char linebuf_debug[1024];
  size_t linesize;
  size_t lines = 0;
  size_t tokens = 0;
  float val = 1;

  while(true){
    /* READ LINE */
    int rc = getline(&linebuf, &linesize, f);
    if (rc == -1)
      break;
    strncpy(linebuf_debug, linebuf, 1024);
    lines++;

    /** READ [FROM] */
    char *pch = strtok(linebuf, ptokens);
    if (pch == NULL)
      logstream(LOG_FATAL)<<"Error reading line " << lines << " [ " << linebuf_debug << " ] " << std::endl;
    I = (uint)get_node_id(pch, user? 0 : 1, user? 0 : 1, lines, true);
    if (I == (uint)-1)//user id was not found in map, we do not need this user link features
      continue; 

    if (user)
      assert(I < (uint)fc.offsets[1]);
    else assert(I < (uint)fc.offsets[2]);

    /** READ TO */  
    pch = strtok(NULL, ptokens);
    if (pch == NULL)
      logstream(LOG_FATAL)<<"Failed to read to field [ " << linebuf_debug << " ] " << std::endl;

    J = (uint)get_node_id(pch, user? 0 : 1, user? 0 : 1, lines);
    set_new(latent_factors_inmem[user? I : I+M].links, J, val);
    tokens++;
  }

  logstream(LOG_DEBUG)<<"Read a total of " << lines << " node features. Tokens: " << tokens << " user? " << user <<  " new entries: " << fc.node_id_maps[user? 0 : 1].string2nodeid.size() << std::endl;
}


#include "rmse.hpp"




/**
  compute validation rmse
  */
  void validation_rmse_N(std::string & filename, 
      float (*prediction_func)(const vertex_data ** array, int arraysize, const float * val_array, int val_array_size, float rating, double & prediction, vec * psum)
      ,graphchi_context & gcontext, 
      feature_control & fc, 
      bool square = false, 
      int type = VALIDATION) {

    assert(fc.total_features <= fc.feature_num);
    if ((filename == "") || !file_exists(filename)) {
      if ((validation != (training + "e")) && gcontext.iteration == 0)
        logstream(LOG_WARNING) << "Validation file was specified, but not found:" << validation << std::endl;
      std::cout << std::endl;
      return;
    }
    FILE *f = NULL;
    size_t nz;   

    detect_matrix_size(validation, f, Me, Ne, nz);
    if (f == NULL){
      logstream(LOG_WARNING)<<"Failed to open validation data. Skipping."<<std::endl;
      return;
    }

    if ((M > 0 && N > 0) && (Me != M || Ne != N))
      logstream(LOG_WARNING)<<"Input size of validation matrix must be identical to training matrix, namely " << M << "x" << N << std::endl;

    compute_matrix_size(nz, VALIDATION);

    last_validation_rmse = dvalidation_rmse;
    dvalidation_rmse = 0;   
    double validation_error = 0;

    std::vector<float> valarray; valarray.resize(fc.total_features);
    uint I, J;
    float val = 0.0f;

    char linebuf_debug[1024];
    for (size_t i=0; i<nz; i++)
    {
      int size = num_feature_bins();
      if (!read_line(f, validation, i, I, J, val, valarray, VALIDATION, linebuf_debug))
        logstream(LOG_FATAL)<<"Failed to read line: " << i << " in file: " <<filename << std::endl;

      bool active_edge = decide_if_edge_is_active(i, VALIDATION);

      if (active_edge){
        assert(size == num_feature_bins());
        if (I == (uint)-1 || J == (uint)-1){
          new_validation_users++;
        //  continue;
        }

        double prediction;
        vertex_data ** node_array = new vertex_data*[calc_feature_node_array_size(I,J)];
        for (int k=0; k< calc_feature_node_array_size(I,J); k++)
          node_array[k] = NULL;
        vec sum;
        //std::cout<<"Going to compute validation for : " <<I << " " <<J << " " << val << " " << size << " " << sum << " " << calc_feature_node_array_size(I,J) << " " << Le << std::endl;
        compute_prediction(I, J, val, prediction, &valarray[0], size, prediction_func, &sum, node_array, VALIDATION);
        //std::cout<<"Computed prediction is: " << prediction << std::endl;
        delete [] node_array;
        dvalidation_rmse += pow(prediction - val, 2);
        if (calc_error) 
          if ((prediction < cutoff && val > cutoff) || (prediction > cutoff && val < cutoff))
            validation_error++;
      }
    }

    fclose(f);

    int howmany = ((type == TRAINING)? L: Le);
    assert(Le > 0);
    dvalidation_rmse = sqrt(dvalidation_rmse / (double)howmany);
    std::cout<< (type == TRAINING ? "  Training RMSE: " :"  Validation RMSE: ") << std::setw(10) << dvalidation_rmse;
    if (!calc_error)
      std::cout << std::endl;
    else std::cout << (type == TRAINING? " Training Error" : " Validation error: ") << std::setw(10) << validation_error/howmany << std::endl;
    if (halt_on_rmse_increase && dvalidation_rmse > last_validation_rmse && gcontext.iteration > 0){
      logstream(LOG_WARNING)<<"Stopping engine because of validation RMSE increase" << std::endl;
      gcontext.set_last_iteration(gcontext.iteration);
    }
  }



/* compute predictions for test data */
void test_predictions_N(
    float (*prediction_func)(const vertex_data ** node_array, int node_array_size, const float * val_array, int val_array_size, float rating, double & predictioni, vec * sum), 
    feature_control & fc, 
    bool square = false) {
  FILE * f = NULL;
  uint Mt, Nt;
  size_t nz;   

  if (test == ""){
    logstream(LOG_INFO)<<"No test file was found, skipping test predictions " << std::endl;
    return;
  }

  if (!file_exists(test)) {
    if (test != (training + "t"))
      logstream(LOG_WARNING)<<" test predictions file was specified but not found: " << test << std::endl;
    return;
  }

  detect_matrix_size(test, f, Mt, Nt, nz);
  if (f == NULL){
    logstream(LOG_WARNING)<<"Failed to open test file. Skipping " << std::endl;
    return;
  }
  if ((M > 0 && N > 0 ) && (Mt != M || Nt != N))
    logstream(LOG_FATAL)<<"Input size of test matrix must be identical to training matrix, namely " << M << "x" << N << std::endl;

  FILE * fout = open_file((test + ".predict").c_str(),"w");

  std::vector<float> valarray; valarray.resize(fc.total_features);
  float val = 0.0f;
  double prediction;
  uint I,J;

  uint i=0;
  char linebuf_debug[1024];
  for (i=0; i<nz; i++)
  {
    int size = num_feature_bins();
    if (!read_line(f, test, i, I, J, val, valarray, TEST, linebuf_debug))
      logstream(LOG_FATAL)<<"Failed to read line: " <<i << " in file: " << test << std::endl;

    if (I == (uint)-1 || J == (uint)-1){
        /*if (cold_start == NONE){
           fprintf(fout, "N/A\n");
          new_test_users++;
        }
        else if (cold_start ==2 ||  (cold_start == 1 && I ==(uint)-1 && J==(uint)-1)){
           fprintf(fout, "%12.8g\n", inputGlobalMean);
           new_test_users++;
        }
        else if (cold_start == ITEM && I == (uint)-1 && J != (uint)-1)
           fprintf(fout, "%12.8g\n", latent_factors_inmem[fc.offsets[1]+J].avg_rating);
        else if (cold_start == ITEM && I != (uint)-1 && J == (uint)-1)
           fprintf(fout, "%12.8g\n", latent_factors_inmem[I].avg_rating);
        else if (cold_start == ITEM){
           fprintf(fout, "%12.8g\n", inputGlobalMean);
           new_test_users++;
        }
        continue;*/
        
    }
    vertex_data ** node_array = new vertex_data*[calc_feature_node_array_size(I,J)];
    vec sum;
    compute_prediction(I, J, val, prediction, &valarray[0], size, prediction_func, &sum, node_array, TEST);
    if (binary_prediction)
      prediction = (prediction > cutoff);
    fprintf(fout, "%12.8lg\n", prediction);
    delete[] node_array;
  }

  if (i != nz)
    logstream(LOG_FATAL)<<"Missing input lines in test file. Should be : " << nz << " found only " << i << std::endl;
  fclose(f);
  fclose(fout);

  logstream(LOG_INFO)<<"Finished writing " << nz << " predictions to file: " << test << ".predict" << std::endl;
}




/* This function implements equation (5) in the libFM paper:
 * http://www.csie.ntu.edu.tw/~b97053/paper/Factorization%20Machines%20with%20libFM.pdf
 * Note that in our implementation x_i are all 1 so the formula is slightly simpler */
float gensgd_predict(const vertex_data** node_array, int node_array_size, 
                     const float * val_array, int val_array_size, 
                     const float rating, double& prediction, vec* sum){

  
  vec sum_sqr = zeros(D);
  *sum = zeros(D);
  prediction = globalMean;
  assert(!std::isnan(prediction));
  assert((int)fc.feature_positions.size() > node_array_size+2);
  if (debug2)
    std::cout<<"Adding global mean: "<< globalMean << std::endl;

  for (int i=0; i< node_array_size; i++){
    if (i >= 2 && fc.real_features_indicators[fc.feature_positions[i-2]])
      prediction += node_array[i]->bias * val_array[i-2];
    else{
      prediction += node_array[i]->bias;
      if (debug2)
        std::cout<<"Adding bias: "<< i << " : " << node_array[i]->bias << std::endl;
    }
  }

  assert(!std::isnan(prediction));

  for (int j=0; j< D; j++){
    for (int i=0; i< node_array_size; i++){
      if (i >= 2 && fc.real_features_indicators[fc.feature_positions[i-2]])
        sum->operator[](j) += node_array[i]->pvec[j] * val_array[i-2];
      else {
        sum->operator[](j) += node_array[i]->pvec[j];
      }
      if (debug2)
      if (sum->operator[](j) >= 1e5)
        logstream(LOG_FATAL)<<"Got into numerical problems. Try to decrease step size" << std::endl;
      if (i >= 2 && fc.real_features_indicators[fc.feature_positions[i-2]])
         sum_sqr[j] += pow(node_array[i]->pvec[j] * val_array[i-2],2);
      else
         sum_sqr[j] += pow(node_array[i]->pvec[j], 2);
    }
    if (debug2 && j < 3)
      std::cout<<" Adding sum: " << sum_sqr[j] << std::endl;

    prediction += 0.5 * (pow(sum->operator[](j),2) - sum_sqr[j]);
    assert(!std::isnan(prediction));
  }

  //truncate prediction to allowed values
  prediction = std::min((double)prediction, maxval);
  prediction = std::max((double)prediction, minval);
  //return the squared error
  float err = rating - prediction;
  assert(!std::isnan(err));
  return err*err; 

}
/*float gensgd_predict(const vertex_data** node_array, int node_array_size, const float * val_array, int val_array_size,
    const float rating, double & prediction){
  vec sum;
  return gensgd_predict(node_array, node_array_size, val_array, val_array_size, rating, prediction, &sum);
}*/


void init_gensgd(bool load_factors_from_file){

  srand(time(NULL));
  int nodes = M+N+num_feature_bins();
  latent_factors_inmem.resize(nodes);
  int howmany = calc_feature_num();
  logstream(LOG_DEBUG)<<"Going to calculate: " << howmany << " offsets." << std::endl;
  fc.offsets.resize(howmany);
  get_offsets(fc.offsets);
  assert(D > 0);
    double factor = 0.1/sqrt(D);
#pragma omp parallel for
    for (int i=0; i< nodes; i++){
      latent_factors_inmem[i].pvec = (debug ? 0.1*ones(D) : (::randu(D)*factor));
    }
#pragma omp parallel for
    for (int i=0; i< FEATURE_WIDTH; i++)
      real_features[i].pvec = randu(D)*factor;
}


void training_rmse_N(int iteration, graphchi_context &gcontext, bool items = false){
  last_training_rmse = dtraining_rmse;
  dtraining_rmse = 0;
  size_t total_errors = 0;
#if 0 // unused 
  int start = 0;
  int end = M;
  if (items){
    start = M;
    end = M+N;
  }
#endif // 0
  dtraining_rmse = sum(rmse_vec);
  if (calc_error)
    total_errors = (size_t)sum(errors_vec);
  dtraining_rmse = sqrt(dtraining_rmse / pengine->num_edges());
  if (calc_error)
    std::cout<< std::setw(10) << mytimer.current_time() << ") Iteration: " << std::setw(3) <<iteration<<" Training RMSE: " << std::setw(10)<< dtraining_rmse << " Train err: " << std::setw(10) << (total_errors/(double)L);
  else 
    std::cout<< std::setw(10) << mytimer.current_time() << ") Iteration: " << std::setw(3) <<iteration<<" Training RMSE: " << std::setw(10)<< dtraining_rmse;
}

/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct GensgdVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

  /*
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
    if (validation_only) //no need to train
      return;

    if (is_user(vertex.id()) && vertex.num_outedges() == 0){
      if (gcontext.iteration == 0)
        vertex_with_no_edges++;
      return;
    }
  

    if (cold_start == ITEM && gcontext.iteration == 0){
       vertex_data & item = latent_factors_inmem[vertex.id()];
       item.avg_rating = 0;
       for(int e=0; e < vertex.num_edges(); e++) {
         item.avg_rating += vertex.edge(e)->get_data().weight;
       }
       item.avg_rating /= vertex.num_edges();
    }

    //go over all user nodes
    if (is_user(vertex.id())){

      //go over all observed ratings
      for(int e=0; e < vertex.num_outedges(); e++) {
        int howmany = calc_feature_node_array_size(vertex.id(), vertex.outedge(e)->vertex_id()-M);
        vertex_data ** node_array = new vertex_data*[howmany];
        for (int i=0; i< howmany; i++)
          node_array[i] = NULL;

        const edge_data & data = vertex.outedge(e)->get_data();
        float rui = data.weight;
        double pui;
        vec sum;

        //compute current prediction
        rmse_vec[omp_get_thread_num()] += compute_prediction(vertex.id(), vertex.outedge(e)->vertex_id()-M, rui ,pui, (float*)data.features, howmany, gensgd_predict, &sum, node_array, TRAINING);
        if (calc_error)
          if ((pui < cutoff && rui > cutoff) || (pui > cutoff && rui < cutoff))
            errors_vec[omp_get_thread_num()]++;
        float eui = pui - rui;

        //update global mean bias
        double step1 = gensgd_rate0 * (eui + gensgd_reg0 * globalMean);
        globalMean -= step1;
        stat1[omp_get_thread_num()] += fabs(step1);

        //update node biases and  vectors
        for (int i=0; i < calc_feature_node_array_size(vertex.id(), vertex.outedge(e)->vertex_id()-M); i++){

          double gensgd_rate = 0;    
          if (i == 0)  //user
            gensgd_rate = gensgd_rate1;
          else if (i == 1) //item
            gensgd_rate = gensgd_rate2;
          else if (i < 2+fc.total_features) //rating features
            gensgd_rate = gensgd_rate3;
          else if (i < 2+fc.total_features+fc.node_features) //user and item features
            gensgd_rate = gensgd_rate4;
          assert(gensgd_rate != 0);

          double step2 = gensgd_rate * (eui + gensgd_regw* node_array[i]->bias);
          node_array[i]->bias -= step2;
          stat2[omp_get_thread_num()] += fabs(step2);

          assert(!std::isnan(node_array[i]->bias));
          assert(node_array[i]->bias < 1e5);

          vec grad =  sum - node_array[i]->pvec;
          vec step3 = gensgd_rate * (eui*grad + gensgd_regv * node_array[i]->pvec);
          node_array[i]->pvec -= gensgd_rate * step3;
          stat3[omp_get_thread_num()] += fabs(step3[0]);
          assert(!std::isnan(node_array[i]->pvec[0]));
          assert(node_array[i]->pvec[0] < 1e5);
        }
        delete[] node_array;

      }


    }

  };


void print_step_size(){
   std::cout<<"Step size 1 " << sum(stat1)/(double)L <<
              "  Step size 2 " << sum(stat2)/(double)L <<
              "  Step size 3 " << sum(stat3)/(double)L << std::endl;

}

  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    if (iteration == 1 && vertex_with_no_edges > 0)
      logstream(LOG_WARNING)<<"There are " << vertex_with_no_edges << " users without ratings" << std::endl;
    gensgd_rate0 *= gensgd_mult_dec;
    gensgd_rate1 *= gensgd_mult_dec;
    gensgd_rate2 *= gensgd_mult_dec;
    gensgd_rate3 *= gensgd_mult_dec;
    gensgd_rate4 *= gensgd_mult_dec;

    if (!exact_training_rmse)
      training_rmse_N(iteration, gcontext);
    else {
      validation_rmse_N(training, &gensgd_predict, gcontext, fc, false, TRAINING);
    }
    validation_rmse_N(validation, &gensgd_predict, gcontext, fc, false, VALIDATION);
    if (verbose)
      print_step_size();
  };

  /**
   * Called before an iteration is started.
   */
  void before_iteration(int iteration, graphchi_context &gcontext) {
    rmse_vec = zeros(gcontext.execthreads);
    stat1 = zeros(gcontext.execthreads);
    stat2 = zeros(gcontext.execthreads);
    stat3 = zeros(gcontext.execthreads);
    if (calc_error)
      errors_vec = zeros(gcontext.execthreads);
  }
};


void output_model(std::string filename){
     MMOutputter_mat<vertex_data> user_output(filename + "_U.mm", 0, latent_factors_inmem.size(), "This file contains GENSGD output matrix U. In each row D factors of a single user node.", latent_factors_inmem);
     MMOutputter_vec<vertex_data> user_bias_vec(filename + "_U_bias.mm", 0, latent_factors_inmem.size(), BIAS_POS, "This file contains GENSGD output bias vector. In each row a single user bias.", latent_factors_inmem);
     MMOutputter_scalar gmean(filename + "_global_mean.mm", "This file contains GENSGD global mean which is required for computing predictions.", globalMean);

    logstream(LOG_INFO) << "GENSGD output files (in matrix market format): " << filename << "_U.mm" << "," << filename << "_global_mean.mm" << std::endl;
    //output mapping between string to array index of features.
    if (fc.hash_strings){
      assert(2+fc.total_features+fc.node_features == (int)fc.node_id_maps.size());
      for (int i=0; i < (int)fc.node_id_maps.size(); i++){
        char buf[256];
        sprintf(buf, "%s.map.%d", filename.c_str(), i);
        save_map_to_text_file(fc.node_id_maps[i].string2nodeid, buf, 0);
      }
    }
    FILE * outf = fopen((filename + ".gm").c_str(), "w");
    fprintf(outf, "%d\n%d\n%ld\n%d\n%12.8lg\n%d\n%d\n%d\n", M, N, L, fc.total_features, globalMean, (int)fc.node_id_maps.size(), (int)latent_factors_inmem.size(),(int)num_feature_bins());
    fclose(outf);

    if (has_header_titles)
      save_vec_to_text_file(header_titles, filename + ".header");
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
  gensgd_rate0 = get_option_float("gensgd_rate1", gensgd_rate0);
  gensgd_rate1 = get_option_float("gensgd_rate1", gensgd_rate1);
  gensgd_rate2 = get_option_float("gensgd_rate2", gensgd_rate2);
  gensgd_rate3 = get_option_float("gensgd_rate3", gensgd_rate3);
  gensgd_rate4 = get_option_float("gensgd_rate4", gensgd_rate4);
  gensgd_regw = get_option_float("gensgd_regw", gensgd_regw);
  gensgd_regv = get_option_float("gensgd_regv", gensgd_regv);
  gensgd_reg0 = get_option_float("gensgd_reg0", gensgd_reg0);
  gensgd_mult_dec = get_option_float("gensgd_mult_dec", gensgd_mult_dec);
  fc.hash_strings = get_option_int("rehash", fc.hash_strings);
  user_file = get_option_string("user_file", user_file);
  user_links = get_option_string("user_links", user_links);
  item_file = get_option_string("item_file", item_file);
  limit_rating = get_option_int("limit_rating", limit_rating);
  calc_error = get_option_int("calc_error", calc_error);
 has_user_titles = get_option_int("has_user_titles", has_user_titles);
  has_item_titles = get_option_int("has_item_titles", has_item_titles);
  fc.rehash_value = get_option_int("rehash_value", fc.rehash_value);
  cutoff = get_option_float("cutoff", cutoff);
  json_input = get_option_int("json_input", json_input);
  cold_start = get_option_int("cold_start", cold_start);
  binary_prediction = get_option_int("binary_prediction", 0);
  verbose = get_option_int("verbose",1); 
  debug2 = get_option_int("debug2", 0);
 D = get_option_int("D", D);
  if (D <=2 || D>= 300)
    logstream(LOG_FATAL)<<"Allowed range for latent factor vector D is [2,300]." << std::endl;

  parse_parser_command_line_arges(); 
  parse_command_line_args();
  parse_implicit_command_line();
  if (validation_only)
    load_factors_from_file = 1;
  vertex_dummy.pvec = ones(D);
  vertex_dummy.bias = 0;
 
    int nshards = convert_matrixmarket_N<edge_data>(training, false, limit_rating);

  init_gensgd(load_factors_from_file);
  if (user_file != "")
    read_node_features(user_file, false, fc, true, false);
  if (item_file != "")
    read_node_features(item_file, false, fc, false, false);
  if (user_links != "")
    read_node_links(user_links, false, fc, true, false);

  if (has_header_titles && header_titles.size() == 0 && ! validation_only)
    logstream(LOG_FATAL)<<"Please delete temp files (using --clean_cache=1 ) and run again" << std::endl;

    if (fc.node_features){
    int last_offset = fc.node_id_maps.size();
    int toadd = 0;
    for (int i = last_offset - fc.node_features; i < last_offset; i++){
      toadd += fc.node_id_maps[i].string2nodeid.size();
    }
    logstream(LOG_DEBUG)<<"Going to add " << toadd << std::endl;
    vertex_data data;
    for (int i=0; i < toadd; i++){
      data.pvec = zeros(D);
      for (int j=0; j < D; j++)
        data.pvec[j] = drand48();
      latent_factors_inmem.push_back(data);
    }
    fc.offsets.resize(calc_feature_num());
    get_offsets(fc.offsets);
  }

  /* load initial state from disk (optional) */
  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, D);
    load_matrix_market_vector(training + "_U_bias.mm", BIAS_POS, false, true);
    assert(num_feature_bins() == num_feature_bins_size);    
    if (has_header_titles)
      load_vec_from_txt_file(header_titles, training + ".header");
  } 
 
  logstream(LOG_INFO) <<"Total selected features: " << fc.total_features << " : " << std::endl;
  for (int i=0; i < MAX_FEATURES+3; i++)
    if (fc.feature_selection[i])
      logstream(LOG_INFO)<<"Selected feature: " << std::setw(3) << i << " : " << (has_header_titles? header_titles[i] : "") <<std::endl;
  logstream(LOG_INFO)<<"Target variable " << std::setw(3) << fc.val_pos << " : " << (has_header_titles? header_titles[fc.val_pos] : "") <<std::endl;
  logstream(LOG_INFO)<<"From            " << std::setw(3) << fc.from_pos<< " : " << (has_header_titles? header_titles[fc.from_pos] : "") <<std::endl;
  logstream(LOG_INFO)<<"To              " << std::setw(3) << fc.to_pos  << " : " << (has_header_titles? header_titles[fc.to_pos] : "") <<std::endl;



  /* Run */
  GensgdVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  if (validation_only)
    niters = 1;
  engine.run(program, niters);

  if (new_validation_users > 0)
    logstream(LOG_WARNING)<<"Found " << new_validation_users<< " new users with no information about them in training dataset!" << std::endl;
 
  if (!validation_only)
    output_model(training);
  if (train_only)
    return 0;

  test_predictions_N(&gensgd_predict, fc);    
  if (new_test_users > 0)
    std::cout<<"Found " << new_test_users<< " new test users with no information about them in training dataset!" << std::endl;

  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  return 0;
}

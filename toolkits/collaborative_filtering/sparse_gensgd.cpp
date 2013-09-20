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
#include "../parsers/common.hpp"
#include <omp.h>
#define MAX_FEATAURES 256
#define FEATURE_WIDTH 21//MAX NUMBER OF ALLOWED FEATURES IN TEXT FILE

double gensgd_rate1 = 1e-02;
double gensgd_rate2 = 1e-02;
double gensgd_rate3 = 1e-02;
double gensgd_rate4 = 1e-02;
double gensgd_rate5 = 1e-02;
double gensgd_mult_dec = 0.9;
double gensgd_regw = 1e-3;
double gensgd_regv = 1e-3;
double gensgd_reg0 = 1e-1;
bool debug = false;
std::string user_file; //optional file with user features
std::string item_file; //optional file with item features
std::string user_links; //optional file with user to user links
int limit_rating = 0;
size_t vertex_with_no_edges = 0;
int calc_error = 0;
int calc_roc = 0;
int binary = 1;
int round_float = 0;
std::vector<std::string> header_titles;
int has_header_titles = 0;
float cutoff = 0;
std::string format = "libsvm";
vec errors_vec;

struct single_map{
  std::map<float,uint> string2nodeid;                                                         
  single_map(){
  }
};

struct feature_control{
  std::vector<single_map> node_id_maps;
  single_map val_map;
  single_map index_map;
  int rehash_value;
  int feature_num;
  int node_features;
  int node_links;
  int total_features;
  const std::string default_feature_str;
  std::vector<int> offsets;
  bool hash_strings;
  int from_pos;
  int to_pos;
  int val_pos;

  feature_control(){
    rehash_value = 0;
    total_features = 0;
    node_features = 0;
    feature_num = FEATURE_WIDTH;
    hash_strings = false;
    from_pos = 0;
    to_pos = 1;
    val_pos = -1;
    node_links = 0;
  }
};

feature_control fc;

int num_feature_bins(){
  int sum = 0;
  if (fc.hash_strings){
    assert(2+fc.total_features+fc.node_features == (int)fc.node_id_maps.size());
    for (int i=2; i < 2+fc.total_features+fc.node_features; i++){
      sum+= fc.node_id_maps[i].string2nodeid.size();
    }
  }
  else assert(false);

  return sum;
}

int calc_feature_num(){
  return 2+fc.total_features+fc.node_features;
}
void get_offsets(std::vector<int> & offsets){
  assert(offsets.size() > 3);
  offsets[0] = 0;
  offsets[1] = M;
  offsets[2] = M+N;
  for (uint i=3; i< offsets.size(); i++){
    assert(fc.node_id_maps.size() > (uint)i);
    offsets[i] += offsets[i-1] + fc.node_id_maps[i].string2nodeid.size();
  }
}

bool is_user(vid_t id){ return id < M; }
bool is_item(vid_t id){ return id >= M && id < N; }
bool is_time(vid_t id){ return id >= M+N; }

#define BIAS_POS -1
struct vertex_data {
  fvec pvec;
  double bias;

  vertex_data() {
    bias = 0;
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

struct edge_data {
  uint  features[FEATURE_WIDTH];
  uint  index[FEATURE_WIDTH];
  uint size;
  float weight;
  edge_data() { 
    weight = 0; 
    size = 0;
    memset(features, 0, sizeof(uint)*FEATURE_WIDTH); 
    memset(index, 0, sizeof(uint)*FEATURE_WIDTH); 
  }

  edge_data(float weight, uint * valarray, uint * _index, uint size): size(size), weight(weight) { 
    memcpy(features, valarray, sizeof(uint)*FEATURE_WIDTH); 
    memcpy(index, _index, sizeof(uint)*FEATURE_WIDTH); 
  }
};



/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef edge_data EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;


int calc_feature_node_array_size(uint node, uint item, uint edge_size){
  assert(node <= M);
  assert(item <= N);
  assert(edge_size >= 0);
  assert(node < latent_factors_inmem.size());
  assert(fc.offsets[1]+item < latent_factors_inmem.size());
  return 2+edge_size;
}

void assign_id(single_map& dmap, unsigned int & outval, const float name){

  std::map<float,uint>::iterator it = dmap.string2nodeid.find(name);
  //if an id was already assigned, return it
  if (it != dmap.string2nodeid.end()){
    outval = it->second - 1;
    assert(outval < dmap.string2nodeid.size());
    return;
  }
  mymutex.lock();
  //assign a new id
  outval = dmap.string2nodeid[name];
  if (outval == 0){
    dmap.string2nodeid[name] = dmap.string2nodeid.size();
    outval = dmap.string2nodeid.size() - 1;
  }
  mymutex.unlock();
}


/**
 * return a numeric node ID out of the string text read from file (training, validation or test)
 */
float get_node_id(char * pch, int pos, size_t i, bool read_only = false){
  assert(pch != NULL);
  assert(pch[0] != 0);
  assert(i >= 0);

  float ret;
  //read numeric id
  if (!fc.hash_strings){
    ret = (pos < 2 ? atoi(pch) : atof(pch)); 
    if (pos < 2)
      ret--;
    if (pos == 0 && ret >= M)
      logstream(LOG_FATAL)<<"Row index larger than the matrix row size " << ret << " > " << M << " in line: " << i << std::endl;
    else if (pos == 1 && ret >= N)
      logstream(LOG_FATAL)<<"Col index larger than the matrix row size " << ret << " > " << N << " in line: " << i << std::endl;

  }
  //else read string id and assign numeric id
  else {
    uint id;
    float val = atof(pch);
    assert(!std::isnan(val));
    if (round_float)
      val = floorf(val * 10000 + 0.5) / 10000;
    if (pos >= 0)
      assert(pos < (int)fc.node_id_maps.size());
    single_map * pmap = NULL;
    if (pos == -1)
      pmap = &fc.index_map;
    else pmap = &fc.node_id_maps[pos];

    if (read_only){ // find if node was in map
      std::map<float,uint>::iterator it = pmap->string2nodeid.find(val);
      if (it != pmap->string2nodeid.end()){
        ret = it->second - 1;
        assert(ret < pmap->string2nodeid.size());
      }
      else ret = -1;
    } 
    else { //else enter node into map (in case it did not exist) and return its position 
      assign_id(*pmap, id, val);
      if (pos == -1 && fc.index_map.string2nodeid.size() == id+1 && fc.node_id_maps.size() < fc.index_map.string2nodeid.size()+2){//TODO debug
        single_map newmap;
        fc.node_id_maps.push_back(newmap);
      }
      ret = id;
    }
  }

  if (!read_only)
    assert(ret != -1);
  return ret;
}


#include "io.hpp"
#include "../parsers/common.hpp"



float get_value(char * pch, bool read_only){
  float ret;
  if (!fc.rehash_value){
    ret = atof(pch);
  }
  else {
    uint id;
    if (read_only){ // find if node was in map
      std::map<float,uint>::iterator it = fc.val_map.string2nodeid.find(atof(pch));
      if (it != fc.val_map.string2nodeid.end()){
        ret = it->second - 1;
      }
      else ret = -1;
    } 
    else { //else enter node into map (in case it did not exist) and return its position 
      assign_id(fc.val_map, id, atof(pch));
      ret = id;
    }

  }    
  if (std::isnan(ret) || std::isinf(ret))
    logstream(LOG_FATAL)<<"Failed to read value" << std::endl;
  return ret;
}

/* Read and parse one input line from file */
bool read_line(FILE * f, const std::string filename, size_t i, uint & I, uint & J, float &val, std::vector<uint>& valarray, std::vector<uint>& positions, int & index, int type, int & skipped_features){

  char * linebuf = NULL;
  size_t linesize;
  char linebuf_debug[1024];

  int token = 0;
  I = J = 0;
  index = 0;

  int rc = getline(&linebuf, &linesize, f);
  if (rc == -1)
    logstream(LOG_FATAL)<<"Failed to get line: " << i << " in file: " << filename << std::endl;
  char * linebuf_to_free = linebuf;
  strncpy(linebuf_debug, linebuf, 1024);

  while (index < FEATURE_WIDTH){

    /* READ FROM */
    if (token == fc.from_pos){
      char *pch = strsep(&linebuf,"\t,\r\n: ");
      if (pch == NULL)
        logstream(LOG_FATAL)<<"Error reading line " << i << " [ " << linebuf_debug << " ] " << std::endl;
      I = (uint)get_node_id(pch, 0, i, type != TRAINING);
      token++;
    }
    else if (token == fc.to_pos){
      /* READ TO */
      char * pch = strsep(&linebuf, "\t,\r\n: ");
      if (pch == NULL)
        logstream(LOG_FATAL)<<"Error reading line " << i << " [ " << linebuf_debug << " ] " << std::endl;
      J = (uint)get_node_id(pch, 1, i, type != TRAINING);
      token++;
    }
    else if (token == fc.val_pos){
      /* READ RATING */
      char * pch = strsep(&linebuf, "\t,\r\n ");
      if (pch == NULL)
        logstream(LOG_FATAL)<<"Error reading line " << i << " [ " << linebuf_debug << " ] " << std::endl;

      val = get_value(pch, type != TRAINING);
      token++;
    }
    else {
      /* READ FEATURES */
      char * pch = strsep(&linebuf, "\t,\r\n:; ");
      if (pch == NULL || pch[0] == 0)
        break;

      uint pos = get_node_id(pch, -1, i, type != TRAINING);
      if (type != TRAINING && pos == (uint)-1){ //this feature was not observed on training, skip
        char * pch2 = strsep(&linebuf, "\t\r\n ");
        if (pch2 == NULL || pch2[0] == 0)
          logstream(LOG_FATAL)<<"Error reading line " << i << " feature2 " << index << " [ " << linebuf_debug << " ] " << std::endl;
        skipped_features++;
        continue;

      }
      assert(pos != (uint)-1 && pos < fc.index_map.string2nodeid.size());

      char * pch2 = strsep(&linebuf, "\t\r\n ");
      if (pch2 == NULL || pch2[0] == 0)
        logstream(LOG_FATAL)<<"Error reading line " << i << " feature2 " << index << " [ " << linebuf_debug << " ] " << std::endl;

      uint second_index = get_node_id(pch2, pos, i, type != TRAINING);
      if (type != TRAINING && second_index == (uint)-1){ //this value was not observed in training, skip
        second_index = 0; //skipped_features++;
        //continue;
      }
      assert(second_index != (uint)-1);
      assert(index< (int)valarray.size());
      assert(index< (int)positions.size());
      valarray[index] = second_index; 
      positions[index] = pos;
      index++;
      token++;
    }
  }//end while
  free(linebuf_to_free);
  return true;
}//end read_line

/* compute an edge prediction based on input features */
float compute_prediction(
    uint I,
    uint J,
    const float val, 
    double & prediction, 
    uint * valarray, 
    uint * positions,
    uint edge_size,
    float (*prediction_func)(std::vector<vertex_data*>& node_array, int arraysize, float rating, double & prediction, fvec * psum), 
    fvec * psum, 
    std::vector<vertex_data*>& node_array,
    uint node_array_size){


  /* COMPUTE PREDICTION */
  /* USER NODE **/
  int index = 0;
  int loc = 0;
  node_array[index] = &latent_factors_inmem[I+fc.offsets[index]];
  assert(node_array[index]->pvec[0] < 1e5);
  index++; loc++;

  /* 1) ITEM NODE */
  assert(J+fc.offsets[index] < latent_factors_inmem.size());
  node_array[index] = &latent_factors_inmem[J+fc.offsets[index]];
  assert(node_array[index]->pvec[0] < 1e5);
  index++; loc++;

  /* 2) FEATURES GIVEN IN RATING LINE */
  for (int j=0; j< (int)edge_size; j++){
    assert(fc.offsets.size() > positions[j]);
    uint pos = fc.offsets[positions[j]] + valarray[j];
    assert(pos >= 0 && pos < latent_factors_inmem.size());
    assert(j+index < (int)node_array_size);
    node_array[j+index] = & latent_factors_inmem[pos];
    assert(node_array[j+index]->pvec[0] < 1e5);
  }
  index+= edge_size;
  loc += edge_size;

  assert(index == calc_feature_node_array_size(I,J, edge_size));
  (*prediction_func)(node_array, node_array_size, val, prediction, psum);
  return pow(val - prediction,2);
} 
#include "rmse.hpp"

/**
 * Create a bipartite graph from a matrix. Each row corresponds to vertex
 * with the same id as the row number (0-based), but vertices correponsing to columns
 * have id + num-rows.
 * Line format of the type
 * [user] [item] [feature1] [feature2] ... [featureN] [rating]
 */

/* Read input file, process it and save a binary representation for faster loading */
template <typename als_edge_type>
int convert_matrixmarket_N(std::string base_filename, bool square, feature_control & fc, int limit_rating = 0) {
  // Note, code based on: http://math.nist.gov/MatrixMarket/mmio/c/example_read.c
  FILE *f;
  size_t nz;
  /**
   * Create sharder object
   */
  int nshards;
  sharder<als_edge_type> sharderobj(base_filename);
  sharderobj.start_preprocessing();

  detect_matrix_size(base_filename, f, M, N, nz);
  /* if .info file is not present, try to find matrix market header inside the base_filename file */

  if (format == "libsvm")
    assert(!has_header_titles);

  if (has_header_titles){
    char * linebuf = NULL;
    size_t linesize;
    char linebuf_debug[1024];

    /* READ LINE */
    int rc = getline(&linebuf, &linesize, f);
    if (rc == -1)
      logstream(LOG_FATAL)<<"Error header line " << " [ " << linebuf_debug << " ] " << std::endl;

    strncpy(linebuf_debug, linebuf, 1024);


    /** READ [FROM] */
    char *pch = strtok(linebuf,"\t,\r; ");
    if (pch == NULL)
      logstream(LOG_FATAL)<<"Error header line " << " [ " << linebuf_debug << " ] " << std::endl;

    header_titles.push_back(pch);

    /** READ USER FEATURES */
    while (pch != NULL){
      pch = strtok(NULL, "\t,\r; ");
      if (pch == NULL)
        break;
      header_titles.push_back(pch);
      //update stats if needed
    }
  }

  if (M == 0 && N == 0)
    logstream(LOG_FATAL)<<"Failed to detect matrix size. Please prepare a file named: " << base_filename << ":info with matrix market header, as explained here: http://bickson.blogspot.co.il/2012/12/collaborative-filtering-3rd-generation_14.html " << std::endl;

  logstream(LOG_INFO) << "Starting to read matrix-market input. Matrix dimensions: " << M << " x " << N << ", non-zeros: " << nz << std::endl;

  uint I, J;
  std::vector<uint> valarray; valarray.resize(FEATURE_WIDTH);
  std::vector<uint> positions; positions.resize(FEATURE_WIDTH);
  float val = 0.0f;

  if (limit_rating > 0)
    nz = limit_rating;
  int skipped_features = 0;

  for (size_t i=0; i<nz; i++)
  {
    int index;
    if (!read_line(f, base_filename, i,I, J, val, valarray, positions, index, TRAINING, skipped_features))
      logstream(LOG_FATAL)<<"Failed to read line: " <<i<< " in file: " << base_filename << std::endl;

    if (index < 1)
      logstream(LOG_FATAL)<<"Failed to read line: " <<i<< " in file: " << base_filename << std::endl;

    if (nz > 1000000 && (i % 1000000) == 0)
      logstream(LOG_INFO)<< mytimer.current_time() << " Finished reading " << i << " lines " << std::endl;
    //calc stats
    L++;
    globalMean += val;
    sharderobj.preprocessing_add_edge(I, square?J:M+J, als_edge_type(val, &valarray[0], &positions[0], index));
  }

  sharderobj.end_preprocessing();

  //calc stats
  assert(L > 0);
  assert(globalMean != 0);
  globalMean /= L;
  logstream(LOG_INFO)<<"Coputed global mean is: " << globalMean << std::endl;

  fclose(f);

  logstream(LOG_INFO) << "Now creating shards." << std::endl;
  // Shard with a specified number of shards, or determine automatically if not defined
  nshards = sharderobj.execute_sharding(get_option_string("nshards", "auto"));

  return nshards;
}




static bool mySort(const std::pair<double, double> &p1,const std::pair<double, double> &p2)
{
  return p1.second > p2.second;
}


/**
  compute validation rmse
  */
  void validation_rmse_N(
      float (*prediction_func)(std::vector<vertex_data*>& array, int arraysize, float rating, double & prediction, fvec * psum)
      ,graphchi_context & gcontext, 
      feature_control & fc, 
      bool square = false) {

    if (validation == "")
      return;

    FILE * f = NULL;
    size_t nz = 0;
    detect_matrix_size(validation, f, Me, Ne, nz);
    if (f == NULL){
        logstream(LOG_WARNING)<<"Failed to open validation file: " << validation << " -  skipping."<<std::endl;
        return;
    }
    if ((M > 0 && N > 0) && (Me != M || Ne != N))
      logstream(LOG_WARNING)<<"Input size of validation matrix must be identical to training matrix, namely " << M << "x" << N << std::endl;

    Le = nz;

    last_validation_rmse = dvalidation_rmse;
    dvalidation_rmse = 0;   

    std::vector<uint> valarray; valarray.resize(FEATURE_WIDTH);
    std::vector<uint> positions; positions.resize(FEATURE_WIDTH);
    uint I, J;
    float val = 0.0f;
    int skipped_features = 0;
    int skipped_nodes = 0;
    int errors = 0; 

    //FOR ROC. ROC code thanks to Justin Yan.
    double _MM = 0;
    double _NN = 0;
    std::vector<std::pair<double, double> > realPrediction;

    for (size_t i=0; i<nz; i++)
    {
      int index;
      if (!read_line(f, validation, i, I, J, val, valarray, positions, index, VALIDATION, skipped_features))
        logstream(LOG_FATAL)<<"Failed to read line: " << i << " in file: " << validation << std::endl;

      if (I == (uint)-1 || J == (uint)-1){
        skipped_nodes++;
        continue;
      }

      double prediction;
      int howmany = calc_feature_node_array_size(I,J, index);
      std::vector<vertex_data*> node_array; node_array.resize(howmany); 
      for (int k=0; k< howmany; k++)
        node_array[k] = NULL;
      
      fvec sum;
      compute_prediction(I, J, val, prediction, &valarray[0], &positions[0], index, prediction_func, &sum, node_array, howmany);
      if (calc_roc)
        realPrediction.push_back(std::make_pair(val, prediction));
     
      double temp_pred = prediction;
      temp_pred = std::min(temp_pred, maxval);
      temp_pred = std::max(temp_pred, minval); 
      dvalidation_rmse += pow(prediction - val, 2);
      if (prediction < cutoff && val >= cutoff)
        errors++;
      else if (prediction >= cutoff && val < cutoff)
        errors++;
    }

    fclose(f);

    assert(Le > 0);
    dvalidation_rmse = sqrt(dvalidation_rmse / (double)Le);
    std::cout<<"  Validation RMSE: " << std::setw(10) << dvalidation_rmse;
    if (calc_error)
      std::cout<<" Validation Err: " << std::setw(10) << ((double)errors/(double)(nz-skipped_nodes));

    if (calc_roc){
      double roc = 0;
      double ret = 0;
      std::vector<double> L;
      std::sort(realPrediction.begin(), realPrediction.end(),mySort);
      std::vector<std::pair<double, double> >::iterator iter;
      for(iter=realPrediction.begin();iter!=realPrediction.end();iter++)
      {
        L.push_back(iter->first);
        if(iter->first > cutoff) _MM++;
        else _NN++;
      }
      std::vector<double>:: iterator iter2;
      int i=0;
      for(iter2=L.begin();iter2!=L.end();iter2++)
      {
        if(*iter2 > cutoff) ret += ((_MM+_NN) - i);
        i++;
      }
      double ret2 = _MM *(_MM+1)/2;
      roc= (ret-ret2)/(_MM*_NN);
      std::cout<<" Validation ROC: " << roc << std::endl;
    }
    else std::cout<<std::endl;



    if (halt_on_rmse_increase && dvalidation_rmse > last_validation_rmse && gcontext.iteration > 0){
      logstream(LOG_WARNING)<<"Stopping engine because of validation RMSE increase" << std::endl;
      gcontext.set_last_iteration(gcontext.iteration);
    }
    if (skipped_features > 0)
      logstream(LOG_DEBUG)<<"Skipped " << skipped_features << " when reading from file. " << std::endl;
    if (skipped_nodes > 0)
      logstream(LOG_DEBUG)<<"Skipped " << skipped_nodes << " when reading from file. " << std::endl;
  }



/* compute predictions for test data */
void test_predictions_N(
    float (*prediction_func)(std::vector<vertex_data*>& node_array, int node_array_size, float rating, double & predictioni, fvec * sum), 
    feature_control & fc, 
    bool square = false) {
  FILE *f = NULL;
  uint Me, Ne;
  size_t nz;   

  if (test == ""){
    logstream(LOG_INFO)<<"No test file was found, skipping test predictions " << std::endl;
    return;
  }

  detect_matrix_size(test, f, Me, Ne, nz);
  if (f == NULL){
    logstream(LOG_WARNING)<<"Failed to open test file " << test<< " skipping test predictions " << std::endl;
    return;
  }

  if ((M > 0 && N > 0 ) && (Me != M || Ne != N))
    logstream(LOG_FATAL)<<"Input size of test matrix must be identical to training matrix, namely " << M << "x" << N << std::endl;

  FILE * fout = open_file((test + ".predict").c_str(),"w", false);

  MM_typecode matcode;
  mm_set_array(&matcode);
  mm_write_banner(fout, matcode);
  mm_write_mtx_array_size(fout ,nz, 1); 

  std::vector<uint> valarray; valarray.resize(FEATURE_WIDTH);
  std::vector<uint> positions; positions.resize(FEATURE_WIDTH);
  float val = 0.0f;
  double prediction;
  uint I,J;
  int skipped_features = 0;
  int skipped_nodes = 0;

  for (uint i=0; i<nz; i++)
  {
    int index;
    if (!read_line(f, test, i, I, J, val, valarray, positions, index, TEST, skipped_features))
      logstream(LOG_FATAL)<<"Failed to read line: " <<i << " in file: " << test << std::endl;

    if (I == (uint)-1 || J == (uint)-1){
      skipped_nodes++;
      fprintf(fout, "%d\n", 0); //features for this node are not found in the training set, write a default value
      continue;
    }

    int howmany = calc_feature_node_array_size(I,J,index);
    std::vector<vertex_data*> node_array; node_array.resize(howmany);
    for (int k=0; k< howmany; k++)
      node_array[k] = NULL;

    fvec sum;
    compute_prediction(I, J, val, prediction, &valarray[0], &positions[0], index, prediction_func, &sum, node_array, howmany);
    fprintf(fout, "%12.8lg\n", prediction);
  }
  fclose(f);
  fclose(fout);

  logstream(LOG_INFO)<<"Finished writing " << nz << " predictions to file: " << test << ".predict" << std::endl;
  if (skipped_features > 0)
    logstream(LOG_DEBUG)<<"Skipped " << skipped_features << " when reading from file. " << std::endl;
  if (skipped_nodes > 0)
    logstream(LOG_WARNING)<<"Skipped node in test dataset: " << skipped_nodes << std::endl;
}





float gensgd_predict(std::vector<vertex_data*> & node_array, int node_array_size,
    const float rating, double& prediction, fvec* sum){

  fvec sum_sqr = fzeros(D);
  *sum = fzeros(D);
  prediction = globalMean;
  assert(!std::isnan(prediction));
  for (int i=0; i< node_array_size; i++)
    prediction += node_array[i]->bias;
  assert(!std::isnan(prediction));

  for (int j=0; j< D; j++){
    for (int i=0; i< node_array_size; i++){
      sum->operator[](j) += node_array[i]->pvec[j];
      assert(sum->operator[](j) < 1e5);
      sum_sqr[j] += pow(node_array[i]->pvec[j],2);
    }
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
float gensgd_predict(std::vector<vertex_data*>& node_array, int node_array_size,
    const float rating, double & prediction){
  fvec sum;
  return gensgd_predict(node_array, node_array_size, rating, prediction, &sum);
}


void init_gensgd(bool load_factors_from_file){

  srand(time(NULL));
  int nodes = M+N+num_feature_bins();
  latent_factors_inmem.resize(nodes);
  int howmany = calc_feature_num();
  logstream(LOG_DEBUG)<<"Going to calculate: " << howmany << " offsets." << std::endl;
  fc.offsets.resize(howmany);
  get_offsets(fc.offsets);
  assert(D > 0);
  if (!load_factors_from_file){
  double factor = 0.1/sqrt(D);
#pragma omp parallel for
  for (int i=0; i< nodes; i++){
    latent_factors_inmem[i].pvec = (debug ? 0.1*fones(D) : (::frandu(D)*factor));
  }
  }

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
  if (calc_error){
    total_errors = sum(errors_vec); 
  }
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
struct Sparse_GensgdVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

  /*
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {



    //go over all user nodes
    if (is_user(vertex.id())){
      //go over all observed ratings
      for(int e=0; e < vertex.num_outedges(); e++) {
        const edge_data & data = vertex.edge(e)->get_data();
        int howmany = calc_feature_node_array_size(vertex.id(), vertex.edge(e)->vertex_id()-M, data.size);
        std::vector<vertex_data*> node_array; node_array.resize(howmany);
        for (int i=0; i< howmany; i++)
          node_array[i] = NULL;

        float rui = data.weight;
        double pui;
        fvec sum;

        //compute current prediction
        rmse_vec[omp_get_thread_num()] += compute_prediction(vertex.id(), vertex.edge(e)->vertex_id()-M, rui ,pui, (uint*)data.features, (uint*)data.index, data.size, gensgd_predict, &sum, node_array, howmany);
        if (calc_error){
          if ((pui < cutoff && rui > cutoff) || (pui > cutoff && rui < cutoff))
            errors_vec[omp_get_thread_num()]++;
        }
        float eui = pui - rui;

        //update global mean bias
        globalMean -= gensgd_rate1 * (eui + gensgd_reg0 * globalMean);

        //update node biases and  vectors
        for (int i=0; i < howmany; i++){

          double gensgd_rate;    
          if (i == 0)  //user
            gensgd_rate = gensgd_rate1;
          else if (i == 1) //item
            gensgd_rate = gensgd_rate2;
          else if (i < (int)(data.size+2)) //rating features
            gensgd_rate = gensgd_rate3;
          else if (i < (int)(2+data.size+fc.node_features)) //user and item features
            gensgd_rate = gensgd_rate4;
          else 
            gensgd_rate = gensgd_rate5; //last item

          node_array[i]->bias -= gensgd_rate * (eui + gensgd_regw* node_array[i]->bias);
          assert(!std::isnan(node_array[i]->bias));
          assert(node_array[i]->bias < 1e3);

          fvec grad =  sum - node_array[i]->pvec;
          node_array[i]->pvec -= gensgd_rate * (eui*grad + gensgd_regv * node_array[i]->pvec);
          assert(!std::isnan(node_array[i]->pvec[0]));
          assert(node_array[i]->pvec[0] < 1e3);
        }

      }


    }

  };

  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    if (iteration == 1 && vertex_with_no_edges > 0)
      logstream(LOG_WARNING)<<"There are " << vertex_with_no_edges << " users without ratings" << std::endl;
    gensgd_rate1 *= gensgd_mult_dec;
    gensgd_rate2 *= gensgd_mult_dec;
    gensgd_rate3 *= gensgd_mult_dec;
    gensgd_rate4 *= gensgd_mult_dec;
    gensgd_rate5 *= gensgd_mult_dec;
    training_rmse_N(iteration, gcontext);
    validation_rmse_N(&gensgd_predict, gcontext, fc);
  };

  /**
   * Called before an iteration is started.
   */
  void before_iteration(int iteration, graphchi_context &gcontext) {
    rmse_vec = zeros(number_of_omp_threads());
    if (calc_error)
      errors_vec = zeros(number_of_omp_threads());
  }


};


void output_gensgd_result(std::string filename) {
  MMOutputter_mat<vertex_data> mmoutput(filename + "_U.mm", 0, M+N+num_feature_bins(), "This file contains Sparse_Gensgd output matrices. In each row D factors of a single user node, then item nodes, then features", latent_factors_inmem);
  MMOutputter_vec<vertex_data> mmoutput_bias(filename + "_U_bias.mm", 0, num_feature_bins(), BIAS_POS, "This file contains Sparse_Gensgd output bias vector. In each row a single user bias.", latent_factors_inmem);
  MMOutputter_scalar gmean(filename + "_global_mean.mm", "This file contains Sparse_Gensgd global mean which is required for computing predictions.", globalMean);

  logstream(LOG_INFO) << " GENSGD output files (in matrix market format): " << filename << "_U.mm" << ",  "<< filename <<  "_global_mean.mm, " << filename << "_U_bias.mm "  <<std::endl;
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
  gensgd_rate1 = get_option_float("gensgd_rate1", gensgd_rate1);
  gensgd_rate2 = get_option_float("gensgd_rate2", gensgd_rate2);
  gensgd_rate3 = get_option_float("gensgd_rate3", gensgd_rate3);
  gensgd_rate4 = get_option_float("gensgd_rate4", gensgd_rate4);
  gensgd_rate5 = get_option_float("gensgd_rate5", gensgd_rate5);
  gensgd_regw = get_option_float("gensgd_regw", gensgd_regw);
  gensgd_regv = get_option_float("gensgd_regv", gensgd_regv);
  gensgd_reg0 = get_option_float("gensgd_reg0", gensgd_reg0);
  gensgd_mult_dec = get_option_float("gensgd_mult_dec", gensgd_mult_dec);
  fc.hash_strings = get_option_int("rehash", fc.hash_strings);
  user_file = get_option_string("user_file", user_file);
  user_links = get_option_string("user_links", user_links);
  item_file = get_option_string("item_file", item_file);
  D = get_option_int("D", D);
  fc.from_pos = get_option_int("from_pos", fc.from_pos);
  fc.to_pos = get_option_int("to_pos", fc.to_pos);
  fc.val_pos = get_option_int("val_pos", fc.val_pos);
  limit_rating = get_option_int("limit_rating", limit_rating);
  calc_error = get_option_int("calc_error", calc_error);
  calc_roc = get_option_int("calc_roc", calc_roc);
  round_float = get_option_int("round_float", round_float);
  has_header_titles = get_option_int("has_header_titles", has_header_titles);
  fc.rehash_value = get_option_int("rehash_value", fc.rehash_value);
  cutoff = get_option_float("cutoff", cutoff);
  binary = get_option_int("binary", binary);

  parse_command_line_args();
  parse_implicit_command_line();

  fc.node_id_maps.resize(2); //initial place for from/to map
  //fc.stats_array.resize(fc.total_features);


  if (format == "libsvm"){
    fc.val_pos = 0;
    fc.to_pos = 2;
    fc.from_pos = 1;
    binary = false;
    fc.hash_strings = true;
  }

  int nshards = convert_matrixmarket_N<edge_data>(training, false, fc, limit_rating);
  fc.total_features = fc.index_map.string2nodeid.size();

  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, D);
    vec user_bias =      load_matrix_market_vector(training +"_U_bias.mm", false, true);
    assert(user_bias.size() == num_feature_bins());
    for (uint i=0; num_feature_bins(); i++){
      latent_factors_inmem[i].bias = user_bias[i];
    }
    vec gm = load_matrix_market_vector(training + "_global_mean.mm", false, true);
    globalMean = gm[0];
  }

  init_gensgd(load_factors_from_file);

  if (has_header_titles && header_titles.size() == 0)
    logstream(LOG_FATAL)<<"Please delete temp files (using : \"rm -f " << training << ".*\") and run again" << std::endl;

  logstream(LOG_INFO)<<"Target variable " << std::setw(3) << fc.val_pos << " : " << (has_header_titles? header_titles[fc.val_pos] : "") <<std::endl;
  logstream(LOG_INFO)<<"From            " << std::setw(3) << fc.from_pos<< " : " << (has_header_titles? header_titles[fc.from_pos] : "") <<std::endl;
  logstream(LOG_INFO)<<"To              " << std::setw(3) << fc.to_pos  << " : " << (has_header_titles? header_titles[fc.to_pos] : "") <<std::endl;



  /* Run */
  Sparse_GensgdVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output test predictions in matrix-market format */
  output_gensgd_result(training);
  test_predictions_N(&gensgd_predict, fc);    

  /* Report execution metrics */
  metrics_report(m);
  return 0;
}

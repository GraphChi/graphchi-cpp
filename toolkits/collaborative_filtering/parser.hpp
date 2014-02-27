#define MAX_FEATURES 256
#define FEATURE_WIDTH 68 //MAX NUMBER OF ALLOWED FEATURES IN TEXT FILE

#include "../parsers/common.hpp"

char tokens[]={"\n\r\t ;,"};
char csv_tokens[] = {",\n\r"};
char * ptokens = tokens;
int csv = 0;
int limit_rating = 0;
int has_header_titles = 0;
int file_columns = 0;
int train_only = 0;
int validation_only = 0;
std::vector<std::string> header_titles;
double inputGlobalMean = 0;
int latent_factors_inmem_size = 0;
int num_feature_bins_size = 0;
std::string real_features_string;

struct feature_control{
  std::vector<double_map> node_id_maps;
  double_map val_map;
  int rehash_value;
  int feature_num;
  int node_features;
  int node_links;
  int total_features;
  std::vector<bool> feature_selection;
  std::vector<int> real_features_indicators;
  std::vector<int> feature_positions;
  const std::string default_feature_str;
  std::vector<int> offsets;
  bool hash_strings;
  int from_pos;
  int to_pos;
  int val_pos;
  std::string string_features;

  feature_control(){
    rehash_value = 0;
    total_features = 0;
    node_features = 0;
    feature_num = FEATURE_WIDTH;
    hash_strings = true;
    from_pos = 0;
    to_pos = 1;
    val_pos = -1;
    node_links = 0;
    feature_selection.resize(MAX_FEATURES+3);
    real_features_indicators.resize(MAX_FEATURES+3);
    feature_positions.resize(MAX_FEATURES+3);
    for (uint i=0; i< feature_positions.size(); i++)
      feature_positions[i] = -1;
  }
};

feature_control fc;

/**
 * return a numeric node ID out of the string text read from file (training, validation or test)
 */
float get_node_id(char * pch, int pos, int token, size_t i, bool read_only = false){
  assert(pch != NULL);
  assert(i >= 0);

  float ret;
  //read numeric id
  if (!fc.hash_strings || fc.real_features_indicators[token]){
    ret = (pos < 2 ? atoi(pch) : atof(pch)); 
    if (pos < 2)
      ret-=input_file_offset;
    if (pos == 0 && ret >= M)
      logstream(LOG_FATAL)<<"Row index larger than the matrix row size " << ret << " > " << M << " in line: " << i << std::endl;
    else if (pos == 1 && ret >= N)
      logstream(LOG_FATAL)<<"Col index larger than the matrix row size " << ret << " > " << N << " in line: " << i << std::endl;

  }
  //else read string id and assign numeric id
  else {
    uint id;
    assert(pos < (int)fc.node_id_maps.size());
    if (read_only){ // find if node was in map
      std::map<std::string,uint>::iterator it = fc.node_id_maps[pos].string2nodeid.find(pch);
      if (it != fc.node_id_maps[pos].string2nodeid.end()){
        ret = it->second;
        assert(ret < fc.node_id_maps[pos].string2nodeid.size());
      }
      else ret = -1;
    } 
    else { //else enter node into map (in case it did not exist) and return its position 
      assign_id(fc.node_id_maps[pos], id, pch);
      assert(id < fc.node_id_maps[pos].string2nodeid.size());
      ret = id;
    }
  }

  if (!read_only)
    assert(ret != -1);
  return ret;
}

float get_value(char * pch, bool read_only, const char * linebuf_debug, int i){
  float ret;
  if (!fc.rehash_value){
    if ( pch[0] == '"' ) {
       pch++;
    }
    ret = atof(pch);
  }
  else {
    uint id;
    if (read_only){ // find if node was in map
      std::map<std::string,uint>::iterator it = fc.val_map.string2nodeid.find(pch);
      if (it != fc.val_map.string2nodeid.end()){
        ret = it->second;
      }
      else ret = -1;
    } 
    else { //else enter node into map (in case it did not exist) and return its position 
      assign_id(fc.val_map, id, pch);
      assert(id < fc.val_map.string2nodeid.size());
      ret = id;
    }

  }    
  if (std::isnan(ret) || std::isinf(ret))
    logstream(LOG_FATAL)<<"Failed to read value (inf/nan) on line: " << i << " " << 
       "[" << linebuf_debug << "]" << std::endl;
  return ret;
}


char * read_one_token(char *& linebuf, size_t i, char * linebuf_debug, int token, int type = TRAINING){
  char *pch = strsep(&linebuf,ptokens);
  if (pch == NULL && type == TRAINING)
        logstream(LOG_FATAL)<<"Error reading line " << i << " [ " << linebuf_debug << " ] " << std::endl;
  else if (pch == NULL && type == TEST)
     return NULL;
  return pch;
}
  
 
/* Read and parse one input line from file */
bool read_line(FILE * f, const std::string filename, size_t i, uint & I, uint & J, float &val, std::vector<float>& valarray, int type, char * linebuf_debug){

  char * linebuf = NULL;
  size_t linesize = 0;

  I = J = 0;
  int token = 0;
  int index = 0;
  int rc = getline(&linebuf, &linesize, f);
  if (rc == -1){
    perror("getline");
    logstream(LOG_FATAL)<<"Failed to get line: " << i << " in file: " << filename << std::endl;
  }

  char * linebuf_to_free = linebuf;
  strncpy(linebuf_debug, linebuf, 1024);

  assert(file_columns >= 2);


  char * pch = NULL;
 
  while (token < file_columns){
    /* READ FROM */
    if (token == fc.from_pos){
      pch = read_one_token(linebuf, i, linebuf_debug, token);
      I = (uint)get_node_id(pch, 0, token, i, type != TRAINING);
      if (type == TRAINING){
        assert( I >= 0 && I < M);
      }
      token++;
    }
    else if (token == fc.to_pos){
      /* READ TO */
      pch = read_one_token(linebuf, i, linebuf_debug, token);
      J = (uint)get_node_id(pch, 1, token, i, type != TRAINING);
      if (type == TRAINING)
        assert(J >= 0 && J < N);
      token++;
    }
    else if (token == fc.val_pos){
      /* READ RATING */
      pch = read_one_token(linebuf, i, linebuf_debug, token, type);
      if (pch == NULL && type == TEST)
         return true;
      val = get_value(pch, type != TRAINING, linebuf_debug, i);
      token++;
    }
    else { 
      if (token >= file_columns)
        break;

      /* READ FEATURES */
      pch = read_one_token(linebuf, i, linebuf_debug, token, type);
      if (pch == NULL && type == TEST)
        return true;
      if (!fc.feature_selection[token]){
        token++;
        continue;
      }

      assert(index < (int)valarray.size());
      valarray[index] = get_node_id(pch, index+2, token, i, type != TRAINING); 
      if (type == TRAINING)

        if (std::isnan(valarray[index]))
          logstream(LOG_FATAL)<<"Error reading line " << i << " feature " << token << " [ " << linebuf_debug << " ] " << std::endl;

      index++;
      token++;
    }
  }//end while
  free(linebuf_to_free);
  return true;
}//end read_line


void detect_matrix_size(std::string filename, FILE *&f, uint &_MM, uint &_NN, size_t & nz, uint nodes, size_t edges, int type);
void compute_matrix_size(size_t nz, int type);
bool decide_if_edge_is_active(size_t i, int type);

/**
 * Create a bipartite graph from a matrix. Each row corresponds to vertex
 * with the same id as the row number (0-based), but vertices correponsing to columns
 * have id + num-rows.
 * Line format of the type
 * [user] [item] [feature1] [feature2] ... [featureN] [rating]
 */

/* Read input file, process it and save a binary representation for faster loading */
template <typename als_edge_type>
int convert_matrixmarket_N(std::string base_filename, bool square, int limit_rating = 0) {
  // Note, code based on: http://math.nist.gov/MatrixMarket/mmio/c/example_read.c
  FILE *f;
  size_t nz;

  int nshards;
  if (validation_only && (nshards = find_shards<als_edge_type>(base_filename, get_option_string("nshards", "auto")))) {
    if (check_origfile_modification_earlier<als_edge_type>(base_filename, nshards)) {
      logstream(LOG_INFO) << "File " << base_filename << " was already preprocessed, won't do it again. " << std::endl;
      FILE * infile = fopen((base_filename + ".gm").c_str(), "r");
      int node_id_maps_size = 0;
      assert( fscanf(infile, "%d\n%d\n%ld\n%d\n%lf\n%d\n%d\n%d\n", &M, &N, &L, &fc.total_features, &globalMean, &node_id_maps_size, &latent_factors_inmem_size,&num_feature_bins_size) ==8);
      assert(node_id_maps_size >= 0);
      assert(latent_factors_inmem_size >=M+N);
      fclose(infile);
      fc.node_id_maps.resize(node_id_maps_size);
      for (int i=0; i < (int)fc.node_id_maps.size(); i++){
        char buf[256];
        sprintf(buf, "%s.map.%d", training.c_str(), i);
        load_map_from_txt_file(fc.node_id_maps[i].string2nodeid, buf, 2);
        assert(fc.node_id_maps[i].string2nodeid.size() > 0);
      }
      logstream(LOG_INFO)<<"Finished loading " << node_id_maps_size << " maps. "<<std::endl;
      return nshards;
    }
  }

  /**
   * Create sharder object
   */
  sharder<als_edge_type> sharderobj(base_filename);
  sharderobj.start_preprocessing();

  detect_matrix_size(base_filename, f, M, N, nz, 0, 0, 0);
  if (f == NULL)
    logstream(LOG_FATAL) << "Could not open file: " << base_filename << ", error: " << strerror(errno) << std::endl;
  if (M == 0 && N == 0)
    logstream(LOG_FATAL)<<"Failed to detect matrix size. Please prepare a file named: " << base_filename << ":info with matrix market header, as explained here: http://bickson.blogspot.co.il/2012/12/collaborative-filtering-3rd-generation_14.html " << std::endl;

  logstream(LOG_INFO) << "Starting to read matrix-market input. Matrix dimensions: " << M << " x " << N << ", non-zeros: " << nz << std::endl;


  if (has_header_titles){
    char * linebuf = NULL;
    size_t linesize;
    char linebuf_debug[1024];

    /* READ LINE */
    int rc = getline(&linebuf, &linesize, f);
    if (rc == -1)
      logstream(LOG_FATAL)<<"Error header line " << " [ " << linebuf_debug << " ] " << std::endl;

    strncpy(linebuf_debug, linebuf, 1024);
    char *pch = strtok(linebuf,ptokens);
    if (pch == NULL)
      logstream(LOG_FATAL)<<"Error header line " << " [ " << linebuf_debug << " ] " << std::endl;

    header_titles.push_back(std::string(pch));

    while (pch != NULL){
      pch = strtok(NULL, ptokens);
      if (pch == NULL)
        break;
      header_titles.push_back(pch);
    }
  }

  compute_matrix_size(nz, TRAINING);
  uint I, J;
  int val_array_len = std::max(1, fc.total_features);
  assert(val_array_len < FEATURE_WIDTH);
  std::vector<float> valarray; valarray.resize(val_array_len);
  float val = 0.0f;

  if (limit_rating > 0 && limit_rating < (int)nz)
    nz = limit_rating;

  char linebuf_debug[1024];
  for (size_t i=0; i<nz; i++)
  {

    if (!read_line(f, base_filename, i,I, J, val, valarray, TRAINING, linebuf_debug))
      logstream(LOG_FATAL)<<"Failed to read line: " <<i<< " in file: " << base_filename << std::endl;

    if (I>= M || J >= N || I < 0 || J < 0){
      if (i == 0)
        logstream(LOG_FATAL)<<"Failed to parse first line, there are too many tokens. Did you forget the --has_header_titles=1 flag when file has string column headers? [ " << linebuf_debug << " ] " << " I : " << I << " J: " << J << std::endl;
      else 
        logstream(LOG_FATAL)<<"Problem parsing input line number: " << i <<" in file: " << base_filename << ".  Can not add edge from " << I << " to  J " << J << 
                            " since matrix size is: " << M <<"x" <<N<< " [ original line: " << linebuf_debug << " ] . You probaably need to increase matrix size in the matrix market header." << std::endl;
    }

    bool active_edge = decide_if_edge_is_active(i, TRAINING);

    if (active_edge){
      //calc stats
      globalMean += val;
      sharderobj.preprocessing_add_edge(I, square?J:M+J, als_edge_type(val, &valarray[0], val_array_len));
    }
  }

  sharderobj.end_preprocessing();

  //calc stats
  assert(L > 0);
  //assert(globalMean != 0);
  if (globalMean == 0)
    logstream(LOG_WARNING)<<"Found global mean of the data to be zero (val_pos). Please verify this is correct." << std::endl;
  globalMean /= L;
  logstream(LOG_INFO)<<"Computed global mean is: " << globalMean << std::endl;
  inputGlobalMean = globalMean;

  fclose(f);

  if (fc.hash_strings){
    for (int i=0; i< fc.total_features+2; i++){
      if (fc.node_id_maps[i].string2nodeid.size() == 0)
        logstream(LOG_FATAL)<<"Failed sanity check for feature number : " << i << " no values find in data " << std::endl;
    }
  }

  logstream(LOG_INFO) << "Now creating shards." << std::endl;
  // Shard with a specified number of shards, or determine automatically if not defined
  nshards = sharderobj.execute_sharding(get_option_string("nshards", "auto"));

  return nshards;
}

void parse_parser_command_line_arges(){
  fc.string_features = get_option_string("features", fc.default_feature_str);
   csv = get_option_int("csv", 0);
  if (csv) 
    ptokens = csv_tokens;
  
  file_columns = get_option_int("file_columns", file_columns); //get the number of columns in the edge file
  //input sanity checks
  if (file_columns < 3)
    logstream(LOG_FATAL)<<"You must have at least 3 columns in input file: [from] [to] [value] on each line"<<std::endl;
  if (file_columns >= FEATURE_WIDTH)
    logstream(LOG_FATAL)<<"file_columns exceeds the allowed storage limit - please increase FEATURE_WIDTH and recompile." << std::endl;
  fc.from_pos = get_option_int("from_pos", fc.from_pos);
  fc.to_pos = get_option_int("to_pos", fc.to_pos);
  fc.val_pos = get_option_int("val_pos", fc.val_pos);
  if (fc.from_pos >= file_columns || fc.to_pos >= file_columns || fc.val_pos >= file_columns)
    logstream(LOG_FATAL)<<"Please note that column numbering of from_pos, to_pos and val_pos starts from zero and should be smaller than file_columns" << std::endl;
  if (fc.from_pos == fc.to_pos || fc.from_pos == fc.val_pos || fc.to_pos == fc.val_pos)
    logstream(LOG_FATAL)<<"from_pos, to_pos and val_pos should have different values" << std::endl; 
  if (fc.val_pos == -1)
    logstream(LOG_FATAL)<<"you must specify a target column using --val_pos=XXX. Colmn index starts from 0." << std::endl;
  has_header_titles = get_option_int("has_header_titles", has_header_titles);
  limit_rating= get_option_int("limit_rating", 0); 
  //parse features (optional)
  if (fc.string_features != ""){
    char * pfeatures = strdup(fc.string_features.c_str());
    char * pch = strtok(pfeatures, ptokens);
    int node = atoi(pch);
    if (node < 0 || node >= MAX_FEATURES+3)
      logstream(LOG_FATAL)<<"Feature id using the --features=XX command should be non negative, starting from zero"<<std::endl;
    if (node >= file_columns)
      logstream(LOG_FATAL)<<"Feature id using the --feature=XX command should be < file_columns (counting starts from zero)" << std::endl;
    if (node == fc.from_pos || node == fc.to_pos || node == fc.val_pos)
      logstream(LOG_FATAL)<<"Feature id " << node << " can not be equal to --from_pos, --to_pos or --val_pos " << std::endl;
    fc.feature_selection[node] = true;
    fc.total_features++;
    while ((pch = strtok(NULL, ptokens))!= NULL){
      node = atoi(pch);
      if (node < 0 || node >= MAX_FEATURES+3)
        logstream(LOG_FATAL)<<"Feature id using the --features=XX command should be non negative, starting from zero"<<std::endl;
      fc.feature_selection[node] = true;
      fc.total_features++;
    }
  }
  train_only = get_option_int("train_only", 0);
  validation_only = get_option_int("validation_only", 0);
  fc.node_id_maps.resize(2+fc.total_features);
  real_features_string = get_option_string("real_features", real_features_string);
   //parse real features (optional)
  if (real_features_string != ""){
    int i=0;
    char * pfeatures = strdup(real_features_string.c_str());
    char * pch = strtok(pfeatures, ptokens);
    int node = atoi(pch);
    if (node < 0 || node >= MAX_FEATURES+3)
      logstream(LOG_FATAL)<<"Feature id using the --real_features=XX command should be non negative, starting from zero"<<std::endl;
    if (node >= file_columns)
      logstream(LOG_FATAL)<<"Feature id using the --real_feature=XX command should be < file_columns (counting starts from zero)" << std::endl;
    if (node == fc.from_pos || node == fc.to_pos || node == fc.val_pos)
      logstream(LOG_FATAL)<<"Feature id " << node << " can not be equal to --from_pos, --to_pos or --val_pos " << std::endl;
    fc.real_features_indicators[node] = true;
    fc.feature_positions[node] = i;
    i++;
    while ((pch = strtok(NULL, ptokens))!= NULL){
      node = atoi(pch);
      if (node < 0 || node >= MAX_FEATURES+3)
        logstream(LOG_FATAL)<<"Feature id using the --real_features=XX command should be non negative, starting from zero"<<std::endl;
      fc.real_features_indicators[node] = true;
      fc.feature_positions[node] = i;
      i++;
    }
  }

 
}


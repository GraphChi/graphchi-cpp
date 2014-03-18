/**
 * @file
 * @author  Danny Bickson, CMU
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
 * This program computes top K recommendations based on the linear model computed
 * by one of: SVD++, bias-SGD, RBM.
 * 
 */




#include "common.hpp"
#include "eigen_wrapper.hpp"
#include "timer.hpp"

int debug;
int num_ratings;
double knn_sample_percent = 1.0;
const double epsilon = 1e-16;
timer mytimer;
int algo = 0;
uint users_without_ratings = 0;
uint users_no_ratings = 0;
mutex mymutex;

int    rbm_bins         = 6;
double rbm_scaling      = 1;


#define BIAS_POS -1
double dot(double * a, double * b){
  double ret = 0;
  for(int i = 0; i < D; ++i)
    ret += a[i] * b[i];
  return ret;
}

enum {
  SVDPP = 0, BIASSGD = 1, RBM = 2
};
struct vertex_data {
  vec ratings;
  ivec ids;
  vec pvec;
  vec weight;
  double bias;

  vertex_data() {
    bias = 0;
    assert(num_ratings > 0);
    ratings = zeros(num_ratings);
    ids = ivec::Ones(num_ratings)*-1;
    assert(D > 0);
    if (algo != RBM){
      pvec = zeros(D);
      weight = zeros(D);
    }
  }
  void set_val(int index, float val){
    if (index == BIAS_POS)
      bias = val;
    else if (algo == RBM || index < D)
      pvec[index] = val;
    else weight[index-D] = val;
  }
  float get_val(int index){
    if (index== BIAS_POS)
      return bias;
    else if (algo == RBM || index < D)
      return pvec[index];
    else return weight[index-D];
  }
};

/*
 * h = pvec = D * DOUBLE
 * h0 = weight = D * DOUBLE
 * h1 = weight+D = D * DOUBLE
 */
struct rbm_user{
  double * h;
  double * h0;
  double * h1;

  rbm_user(const vertex_data & vdata){
    h = (double*)&vdata.pvec[0];
    h0 = h + D;
    h1 = h0 + D;
  }

  rbm_user & operator=(vertex_data & data){
    h = &data.pvec[0];
    h0 = h + D;
    h1 = h0 + D;
    return * this;
  }   
};


/**
 * ni = bias = DOUBLE
 * bi = pvec = rbm_bins * DOUBLE 
 * w = weight = rbm_bins * D * Double
 */
struct rbm_movie{
  double * bi;
  double * w;

  rbm_movie(const vertex_data& vdata){
    bi = (double*)&vdata.pvec[0];
    w = bi + rbm_bins;
  }

  rbm_movie & operator=(vertex_data & data){
    bi = (double*)&data.pvec[0];
    w = bi + rbm_bins;
    return * this;
  }
};



struct edge_data {
  double weight;

  edge_data() { weight = 0; }

  edge_data(double weight) : weight(weight) { }
};

struct edge_data4 {
  double weight;
  double time;

  edge_data4() { weight = time = 0; }

  edge_data4(double weight, double time) : weight(weight), time(time) { }
};


/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef edge_data EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

/** compute a missing value based on SVD++ algorithm */
float svdpp_predict(const vertex_data& user, const vertex_data& movie, const float rating, double & prediction, void * extra = NULL){
  //\hat(r_ui) = \mu + 
  prediction = globalMean;
  // + b_u  +    b_i +
  prediction += user.bias + movie.bias;
  // + q_i^T   *(p_u      +sqrt(|N(u)|)\sum y_j)
  //prediction += dot_prod(movie.pvec,(user.pvec+user.weight));
  for (int j=0; j< D; j++)
    prediction += movie.pvec[j] * (user.pvec[j] + user.weight[j]);

  prediction = std::min((double)prediction, maxval);
  prediction = std::max((double)prediction, minval);
  float err = rating - prediction;
  if (std::isnan(err))
    logstream(LOG_FATAL)<<"Got into numerical errors. Try to decrease step size using the command line: svdpp_user_bias_step, svdpp_item_bias_step, svdpp_user_factor2_step, svdpp_user_factor_step, svdpp_item_step" << std::endl;
  return err*err; 
}


/** compute a missing value based on bias-SGD algorithm */
float biassgd_predict(const vertex_data& user, 
    const vertex_data& movie, 
    const float rating, 
    double & prediction, 
    void * extra = NULL){


  prediction = globalMean + user.bias + movie.bias + dot_prod(user.pvec, movie.pvec);  
  //truncate prediction to allowed values
  prediction = std::min((double)prediction, maxval);
  prediction = std::max((double)prediction, minval);
  //return the squared error
  float err = rating - prediction;
  if (std::isnan(err))
    logstream(LOG_FATAL)<<"Got into numerical errors. Try to decrease step size using bias-SGD command line arugments)" << std::endl;
  return err*err; 

}

/** compute a missing value based on RBM algorithm */
float rbm_predict(const rbm_user & usr, 
    const rbm_movie & mov, 
    const float rating, 
    double & prediction, 
    void * extra = NULL){

  float ret = 0;
  double nn = 0;
  for(int r = 0; r < rbm_bins; ++r){               
    double zz = exp(mov.bi[r] + std::min(maxval, dot(usr.h, &mov.w[r*D])));
    //if (std::isinf(zz))
    //  std::cout<<" mov.bi[r] " << mov.bi[r] << " dot: " << dot(usr.h, &mov.w[r*D]) << std::endl;
    ret += zz * (float)(r);
    assert(!std::isnan(ret));
    nn += zz;
  }
  assert(!std::isnan(ret));
  assert(std::fabs(nn) > 1e-32);
  ret /= nn;
  if(ret < minval) ret = minval;
  else if(ret > maxval) ret = maxval;
  assert(!std::isnan(ret));
  prediction = ret;
  assert(!std::isnan(prediction));
  return pow(prediction - rating,2);
}

float rbm_predict(const vertex_data & usr, 
    const vertex_data & mov, 
    const float rating, 
    double & prediction, 
    void * extra){
  return rbm_predict(rbm_user((vertex_data&)usr), rbm_movie((vertex_data&)mov), rating, prediction, NULL);
}   

void rating_stats(){

  double min=1e100, max=0, avg=0;
  int cnt = 0;
  int startv = std::max(0, start_user);
  int endv = std::min(M, (uint)end_user);

  for (int i=startv; i< endv; i++){
    vertex_data& data = latent_factors_inmem[i];
    if (data.ratings.size() > 0){
      min = std::min(min, data.ratings[0]);
      max = std::max(max, data.ratings[0]);
      if (std::isnan(data.ratings[0])){
        printf("bug: nan on %d\n", i);
        users_no_ratings++;
      }
      else {
        avg += data.ratings[0];    
        cnt++;
      }
    }
  }

  printf("Distance statistics: min %g max %g avg %g\n", min, max, avg/cnt);
}



#include "io.hpp"

void read_factors(std::string base_filename){
    if (algo == SVDPP)
      load_matrix_market_matrix(training + "_U.mm", 0, 2*D);
    else if (algo == BIASSGD)
      load_matrix_market_matrix(training + "_U.mm", 0, D);
    else if (algo == RBM){
      load_matrix_market_matrix(training + "_U.mm", 0, 3*D);

    }
    else assert(false);
    
    if (algo == SVDPP || algo == BIASSGD)
       load_matrix_market_matrix(training + "_V.mm", M, D);
    else if (algo == RBM)
       load_matrix_market_matrix(training + "_V.mm", M, rbm_bins + rbm_bins * D);
    else assert(false);

 
    if (algo == SVDPP || algo == BIASSGD){
      vec user_bias = load_matrix_market_vector(training +"_U_bias.mm", false, true);
      assert(user_bias.size() == M);

      vec item_bias = load_matrix_market_vector(training +"_V_bias.mm", false, true);
      assert(item_bias.size() == N);
      for (uint i=0; i<M+N; i++){
        latent_factors_inmem[i].bias = ((i<M)?user_bias[i] : item_bias[i-M]);
      }
      vec gm = load_matrix_market_vector(training + "_global_mean.mm", false, true);
      globalMean = gm[0];
    }
}


template<typename VertexDataType, typename EdgeDataType>
struct RatingVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {


  /**
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {

    //compute only for user nodes
    if (vertex.id() >= std::min(M,(uint)end_user) || vertex.id() < (uint)start_user)
      return;

    vertex_data & vdata = latent_factors_inmem[vertex.id()];
    int howmany = (int)(N*knn_sample_percent);
    assert(howmany > 0 );
    if (vertex.num_outedges() == 0){
       mymutex.lock();
       users_without_ratings++;
       mymutex.unlock();
    }

    vec distances = zeros(howmany);
    ivec indices = ivec::Zero(howmany);
    for (int i=0; i< howmany; i++){
      indices[i]= -1;
    }
    std::vector<bool> curratings;
    curratings.resize(N);
    for(int e=0; e < vertex.num_edges(); e++) {
      //no need to calculate this rating since it is given in the training data reference
      assert(vertex.edge(e)->vertex_id() - M >= 0 && vertex.edge(e)->vertex_id() - M < N);
      curratings[vertex.edge(e)->vertex_id() - M] = true;
    }
    if (knn_sample_percent == 1.0){
      for (uint i=M; i< M+N; i++){
        if (curratings[i-M])
          continue;
        vertex_data & other = latent_factors_inmem[i];
        double dist;
        if (algo == SVDPP)
          svdpp_predict(vdata, other, 0, dist); 
        else if (algo == BIASSGD) 
	  biassgd_predict(vdata, other, 0, dist);
        else if (algo == RBM)
          rbm_predict(vdata, other, 0, dist);
        else assert(false);
        indices[i-M] = i-M;
        distances[i-M] = dist + 1e-10;
      }
    }
    else for (int i=0; i<howmany; i++){
      int random_other = ::randi(M, M+N-1);
      vertex_data & other = latent_factors_inmem[random_other];
      double dist;
      if (algo == SVDPP)
        svdpp_predict(vdata, other, 0, dist); 
      else if (algo == BIASSGD)
        biassgd_predict(vdata, other, 0, dist);
      else if (algo == RBM)
        rbm_predict(vdata, other, 0, dist);
      else assert(false);
        
      indices[i] = random_other-M;
      distances[i] = dist;
    }

    vec out_dist(num_ratings);
    ivec indices_sorted = reverse_sort_index2(distances, indices, out_dist, num_ratings);
    assert(indices_sorted.size() <= num_ratings);
    assert(out_dist.size() <= num_ratings);
    vdata.ids = indices_sorted;
    vdata.ratings = out_dist;
    if (debug)
      printf("Closest is: %d with distance %g\n", (int)vdata.ids[0], vdata.ratings[0]);

    if (vertex.id() % 1000 == 0)
      printf("Computing recommendations for user %d at time: %g\n", vertex.id()+1, mytimer.current_time());
  }

};

struct  MMOutputter_ratings{
  MMOutputter_ratings(std::string fname, uint start, uint end, std::string comment)  {
    assert(start < end);
    MM_typecode matcode;
    set_matcode(matcode);     
    FILE * outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, end-start, num_ratings+1); 
    for (uint i=start; i < end; i++){
      fprintf(outf, "%u ", i+1);
      for(int j=0; j < latent_factors_inmem[i].ratings.size(); j++) {
        fprintf(outf, "%1.12e ", latent_factors_inmem[i].ratings[j]);
      }
      fprintf(outf, "\n");
    }
    fclose(outf);
  }
};

struct  MMOutputter_ids{
  MMOutputter_ids(std::string fname, uint start, uint end, std::string comment)  {
    assert(start < end);
    MM_typecode matcode;
    set_matcode(matcode);     
    FILE * outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, end-start, num_ratings+1); 
    for (uint i=start; i < end; i++){
      fprintf(outf, "%u ", i+1);
      for(int j=0; j < latent_factors_inmem[i].ids.size(); j++) {
        fprintf(outf, "%u ", (int)latent_factors_inmem[i].ids[j]+1);//go back to item ids starting from 1,2,3, (and not from zero as in c)
      }
      fprintf(outf, "\n");
    }
    fclose(outf);
  }

};



void output_knn_result(std::string filename) {
  MMOutputter_ratings ratings(filename + ".ratings", std::max(start_user,0),std::min((uint)end_user,M),"This file contains user scalar ratings. In each row i, num_ratings top scalar ratings of different items for user i. (First column: user id, next columns, top K ratings). Note that item 0 means no item could be recommended for this user.");
  MMOutputter_ids mmoutput_ids(filename + ".ids", std::max(start_user, 0), std::min((uint)end_user,M) ,"This file contains item ids matching the ratings. In each row i, num_ratings top item ids for user i. (First column: user id, next columns, top K ratings). Note: 0 item id means there are no more items to recommend for this user.");
 
  std::cout << "Rating output files (in matrix market format): " << filename << ".ratings" <<
                                                                    ", " << filename + ".ids " << std::endl;
}

int main(int argc, const char ** argv) {

  mytimer.start();
  print_copyright();

  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("rating2");

  knn_sample_percent = get_option_float("knn_sample_percent", 1.0);
  if (knn_sample_percent <= 0 || knn_sample_percent > 1)
    logstream(LOG_FATAL)<<"Sample percente should be in the range (0, 1] " << std::endl;

  num_ratings   = get_option_int("num_ratings", 10);
  if (num_ratings <= 0)
    logstream(LOG_FATAL)<<"num_ratings, the number of recomended items for each user, should be >=1 " << std::endl;

  debug         = get_option_int("debug", 0);
  tokens_per_row = get_option_int("tokens_per_row", tokens_per_row);
  std::string algorithm     = get_option_string("algorithm");
  /* Basic arguments for RBM algorithm */
  rbm_bins      = get_option_int("rbm_bins", rbm_bins);
  rbm_scaling   = get_option_float("rbm_scaling", rbm_scaling);

  if (algorithm == "svdpp" || algorithm == "svd++")
    algo = SVDPP;
  else if (algorithm == "biassgd")
    algo = BIASSGD;
  else if (algorithm == "rbm")
    algo = RBM;
  else logstream(LOG_FATAL)<<"--algorithm should be svd++ or biassgd or rbm"<<std::endl;

  parse_command_line_args();

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = 0;
  if (tokens_per_row == 3)
    nshards = convert_matrixmarket<edge_data>(training, 0, 0, 3, TRAINING, false);
  else if (tokens_per_row == 4)
    nshards = convert_matrixmarket4<edge_data4>(training);
  else logstream(LOG_FATAL)<<"--tokens_per_row should be either 3 or 4" << std::endl;

  assert(M > 0 && N > 0);
  latent_factors_inmem.resize(M+N); // Initialize in-memory vertices.

  //initialize data structure to hold the matrix read from file
  if (algo == RBM){
#pragma omp parallel for
    for (uint i=0; i< M+N; i++){
      if (i < M){
        latent_factors_inmem[i].pvec = zeros(D*3);
      }
      else {  
        latent_factors_inmem[i].pvec = zeros(rbm_bins + rbm_bins * D);
      }
    } 
  }
 
  read_factors(training);
  if ((uint)num_ratings > N){
    logstream(LOG_WARNING)<<"num_ratings is too big - setting it to: " << N << std::endl;
    num_ratings = N;
  }
  srand(time(NULL));

  /* Run */
  if (tokens_per_row == 3){
    RatingVerticesInMemProgram<VertexDataType, EdgeDataType> program;
    graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
    set_engine_flags(engine);
    engine.run(program, 1);
  } 
  else if (tokens_per_row == 4){
    RatingVerticesInMemProgram<VertexDataType, edge_data4> program;
    graphchi_engine<VertexDataType, edge_data4> engine(training, nshards, false, m); 
    set_engine_flags(engine);
    engine.run(program, 1);
  }
  /* Output latent factor matrices in matrix-market format */
  output_knn_result(training);

  rating_stats();

  if (users_without_ratings > 0)
    logstream(LOG_WARNING)<<"Found " << users_without_ratings << " without ratings. For those users no items are recommended (item id 0)" << std::endl;

  if (users_no_ratings > 0)
    logstream(LOG_WARNING)<<"Failed to compute ratings for " << users_no_ratings << " Users. For those users no items are recommended (item id 0)" << std::endl;


  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  return 0;
}

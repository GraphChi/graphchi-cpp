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
 * by one of: als,sparse_als,wals, sgd, nmf, climf and svd algos.
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
vec singular_values;
mutex mymutex;

enum {
  ALS = 0, SPARSE_ALS = 1, SGD = 2, NMF = 3, WALS = 4, SVD = 5, CLIMF = 6
};

struct vertex_data {
  vec ratings;
  ivec ids;
  vec pvec;

  vertex_data() {
    pvec = vec::Zero(D);
    ids = ivec::Zero(num_ratings);
    ratings = vec::Zero(num_ratings);
  }
  void set_val(int index, float val){
    pvec[index] = val;
  }
  float get_val(int index){
    return pvec[index];
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

float als_predict(const vertex_data& user, 
    const vertex_data& movie, 
    const float rating, 
    double & prediction){


  prediction = dot_prod(user.pvec, movie.pvec);
  //truncate prediction to allowed values
  prediction = std::min((double)prediction, maxval);
  prediction = std::max((double)prediction, minval);
  //return the squared error
  float err = rating - prediction;
  assert(!std::isnan(err));
  return err*err; 

}

/** compute a missing value based on SVD algorithm */
float svd_predict(const vertex_data& user, 
    const vertex_data& movie, 
    const float rating, 
    double & prediction, 
    void * extra = NULL){

  Eigen::DiagonalMatrix<double, Eigen::Dynamic> diagonal_matrix(D);      
  diagonal_matrix.diagonal() = singular_values;

  prediction = user.pvec.transpose() * diagonal_matrix * movie.pvec;
  //truncate prediction to allowed values
  prediction = std::min((double)prediction, maxval);
  prediction = std::max((double)prediction, minval);
  //return the squared error
  float err = rating - prediction;
  assert(!std::isnan(err));
  return err*err; 

}

// logistic function
double g(double x)
{
  double ret = 1.0 / (1.0 + std::exp(-x));

  if (std::isinf(ret) || std::isnan(ret))
  {
    logstream(LOG_FATAL) << "overflow in g()" << std::endl;
  }

  return ret;
}

/* compute prediction based on CLiMF algorithm */
float climf_predict(const vertex_data& user,
    const vertex_data& movie,
    const float rating,
    double & prediction,
    void * extra = NULL)
{
  prediction = g(dot(user.pvec,movie.pvec));  // this is actually a predicted reciprocal rank, not a rating
  return 0;  // as we have to return something
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
      if (std::isnan(data.ratings[0]))
        printf("bug: nan on %d\n", i);
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
    load_matrix_market_matrix(training + "_U.mm", 0, D);
    load_matrix_market_matrix(training + "_V.mm", M, D);
    if (algo == SVD)
       singular_values = load_matrix_market_vector(training + ".singular_values", false, true);
}



/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
template<typename VertexDataType, typename EdgeDataType>
struct RatingVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {


  /**
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {

    //compute only for user nodes
    if (vertex.id() >= std::min(M, (uint)end_user) || vertex.id() < (uint)start_user)
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
        if (algo != SVD && algo != CLIMF)
           als_predict(vdata, other, 0, dist); 
        else if (algo == SVD)
           svd_predict(vdata, other, 0, dist);
        else if (algo == CLIMF)
           climf_predict(vdata, other, 0, dist);
        else assert(false);
        indices[i-M] = i-M;
        distances[i-M] = dist + 1e-10;
      }
    }
    else for (int i=0; i<howmany; i++){
      int random_other = ::randi(M, M+N-1);
      vertex_data & other = latent_factors_inmem[random_other];
      double dist;
      if (algo != SVD && algo != CLIMF)
           als_predict(vdata, other, 0, dist); 
      else if (algo == CLIMF)
           climf_predict(vdata, other, 0, dist);
      if (algo != SVD)
           als_predict(vdata, other, 0, dist); 
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
  MMOutputter_ratings ratings(filename + ".ratings", std::max(start_user,0),std::min((uint)end_user, M),"This file contains user scalar ratings. In each row i, num_ratings top scalar ratings of different items for user i. (First column: user id, next columns, top K ratings)");
  MMOutputter_ids mmoutput_ids(filename + ".ids", std::max(start_user, 0), std::min((uint)end_user, M) ,"This file contains item ids matching the ratings. In each row i, num_ratings top item ids for user i. (First column: user id, next columns, top K ratings). Note: 0 item id means there are no more items to recommend for this user.");
 
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
  metrics m("rating");

  knn_sample_percent = get_option_float("knn_sample_percent", 1.0);
  if (knn_sample_percent <= 0 || knn_sample_percent > 1)
    logstream(LOG_FATAL)<<"Sample percente should be in the range (0, 1] " << std::endl;

  num_ratings   = get_option_int("num_ratings", 10);
  if (num_ratings <= 0)
    logstream(LOG_FATAL)<<"num_ratings, the number of recomended items for each user, should be >=1 " << std::endl;

  debug         = get_option_int("debug", 0);
  std::string algorithm = get_option_string("algorithm");
  if (algorithm == "als" || algorithm == "sparse_als" || algorithm == "sgd" || algorithm == "nmf" || algorithm == "svd" || algorithm == "climf")
    tokens_per_row = 3;
  else if (algorithm == "wals")
    tokens_per_row = 4;
  else logstream(LOG_FATAL)<<"--algorithm=XX should be one of: als, sparse_als, sgd, nmf, wals, svd, climf" << std::endl;

  if (algorithm == "svd")
     algo = SVD;
  else if (algorithm == "climf")
     algo = CLIMF;


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

  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  return 0;
}

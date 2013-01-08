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
 *
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
int tokens_per_row = 3;

struct vertex_data {
  vec ratings;
  ivec ids;
  vec pvec;

  vertex_data() {
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


void rating_stats(){

   double min=1e100, max=0, avg=0;
   int cnt = 0;
   int startv = 0;
   int endv = M;

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



template<typename vertex_data>
void read_factors(std::string base_filename, bool users) {
  // Note, code based on: http://math.nist.gov/MatrixMarket/mmio/c/example_read.c
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  size_t nz;
  uint _M, _N;

  f = fopen(base_filename.c_str(), "r");
  if (f == NULL)
    logstream(LOG_FATAL) << "Could not open file: " << base_filename << std::endl;

  if (mm_read_banner(f, &matcode) != 0)
    logstream(LOG_FATAL) << "Could not process Matrix Market banner. File: " << base_filename << std::endl;

  if (!mm_is_dense(matcode))
    logstream(LOG_FATAL) << "Problem reading input file: " << base_filename << " Should be in dense matrix market format"<< std::endl;
  /* find out size of sparse matrix .... */
  uint feature_width;
  if (users){
    if ((ret_code = mm_read_mtx_array_size(f, &_M, &feature_width)) !=0) {
      logstream(LOG_FATAL) << "Failed reading matrix size: error=" << ret_code << std::endl;
    }
    if (_M != M)
      logstream(LOG_FATAL) << "Wrong size of feature vector matrix. Should be " << M << " rows instead of " << _M << std::endl;
  }
  else {    
    if ((ret_code = mm_read_mtx_array_size(f, &_N, &feature_width)) !=0) {
      logstream(LOG_FATAL) << "Failed reading matrix size: error=" << ret_code << std::endl;
    }
    if (_N != N)
      logstream(LOG_FATAL) << "Wrong size of feature vector matrix. Should be " << N << " rows instead of " << _N << std::endl;
   }

  D = (int)feature_width;
  logstream(LOG_INFO) << "Starting to read matrix-market input. Matrix dimensions: " 
    << (users? M :N) << " x " << D << ", non-zeros: " << (users?M:N)*D << std::endl;

  double val;
  nz = M*D;

  for (size_t i=(users? 0 : M); i< (users? M : M+N); i++)
    for (size_t j=0; j < (size_t)D; j++)
    {
      int rc = fscanf(f, "%lg", &val);
      if (rc != 1)
        logstream(LOG_FATAL)<<"Error when reading input file at line " << i << std::endl;
      if (latent_factors_inmem[i].pvec.size() == 0){
        latent_factors_inmem[i].pvec = vec::Zero(D);
        latent_factors_inmem[i].ratings = vec::Zero(num_ratings);
        latent_factors_inmem[i].ids = ivec::Zero(num_ratings);
      }
      latent_factors_inmem[i].pvec[j] = val;
    }

  fclose(f);
}


#include "io.hpp"

/** compute a missing value based on Rating algorithm */



/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct RatingVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {


  /**
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {

  if (vertex.id() >= M)
    return;

  vertex_data & vdata = latent_factors_inmem[vertex.id()];
  int howmany = (int)(N*knn_sample_percent);
  assert(howmany > 0 );
  vec distances = vec::Zero(howmany);
  ivec indices = ivec(howmany);
  for (int i=0; i< howmany; i++){
    indices[i]= -2;
  }
  std::vector<bool> curratings;
  curratings.resize(N);
  for(int e=0; e < vertex.num_edges(); e++) {
  //no need to calculate this rating since it is given in the training data reference
    curratings[vertex.edge(e)->vertex_id() - M] = true;
  }
   if (knn_sample_percent == 1.0){
     for (uint i=M; i< M+N; i++){
        if (curratings[i-M])
          continue;
        vertex_data & other = latent_factors_inmem[i];
        double dist;
        als_predict(vdata, other, 0, dist); 
        indices[i-M] = i-M;
        distances[i-M] = dist;
     }
  }
  else for (int i=0; i<howmany; i++){
        int random_other = ::randi(M, M+N-1);
        vertex_data & other = latent_factors_inmem[random_other];
        double dist;
        als_predict(vdata, other, 0, dist); 
        indices[i-M] = i-M;
        distances[i-M] = dist;
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
  FILE * outf;
  MMOutputter_ratings(std::string fname, uint start, uint end, std::string comment)  {
    assert(start < end);
    MM_typecode matcode;
    set_matcode(matcode);     
    outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, end-start, num_ratings+1); 
    for (uint i=start; i < end; i++){
      fprintf(outf, "%u ", i+1);
      for(int j=0; j < num_ratings; j++) {
        fprintf(outf, "%1.12e ", latent_factors_inmem[i].ratings[j]);
      }
      fprintf(outf, "\n");
    }
  }

  ~MMOutputter_ratings() {
    if (outf != NULL) fclose(outf);
  }

};
struct  MMOutputter_ids{
  FILE * outf;
  MMOutputter_ids(std::string fname, uint start, uint end, std::string comment)  {
    assert(start < end);
    MM_typecode matcode;
    set_matcode(matcode);     
    outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, end-start, num_ratings+1); 
    for (uint i=start; i < end; i++){
      fprintf(outf, "%u ", i+1);
      for(int j=0; j < num_ratings; j++) {
        fprintf(outf, "%u ", (int)latent_factors_inmem[i].ids[j]+1);//go back to item ids starting from 1,2,3, (and not from zero as in c)
      }
      fprintf(outf, "\n");
    }
  }

  ~MMOutputter_ids() {
    if (outf != NULL) fclose(outf);
  }

};



void output_knn_result(std::string filename) {
  MMOutputter_ratings mmoutput_ratings(filename + ".ratings", 0, M, "This file contains user scalar ratings. In each row i, num_ratings top scalar ratings of different items for user i. (First column: user id, next columns, top K ratings)");
  MMOutputter_ids mmoutput_ids(filename + ".ids", 0, M ,"This file contains item ids matching the ratings. In each row i, num_ratings top item ids for user i. (First column: user id, next columns, top J ratings)");
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
  metrics m("nmf-inmemory-factors");

  knn_sample_percent = get_option_float("knn_sample_percent", 1.0);
  if (knn_sample_percent <= 0 || knn_sample_percent > 1)
    logstream(LOG_FATAL)<<"Sample percente should be in the range (0, 1] " << std::endl;

  num_ratings   = get_option_int("num_ratings", 10);
  if (num_ratings <= 0)
    logstream(LOG_FATAL)<<"num_ratings, the number of recomended items for each user, should be >=1 " << std::endl;

  debug         = get_option_int("debug", 0);
  tokens_per_row = get_option_int("tokens_per_row", tokens_per_row);
  parse_command_line_args();

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = 0;
  if (tokens_per_row == 3)
     nshards = convert_matrixmarket<edge_data>(training);
  else if (tokens_per_row == 4)
     nshards = convert_matrixmarket4<edge_data4>(training);
  else logstream(LOG_FATAL)<<"--tokens_per_row should be either 3 or 4" << std::endl;

  assert(M > 0 && N > 0);
  latent_factors_inmem.resize(M+N); // Initialize in-memory vertices.
  read_factors<vertex_data>(training + "_U.mm", true);
  read_factors<vertex_data>(training + "_V.mm", false);
  if ((uint)num_ratings > N){
    logstream(LOG_WARNING)<<"num_ratings is too big - setting it to: " << N << std::endl;
    num_ratings = N;
  }
  srand(time(NULL));

  /* Run */
  RatingVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, 1);

  /* Output latent factor matrices in matrix-market format */
  output_knn_result(training);

  rating_stats();
  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  return 0;
}

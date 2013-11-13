
/**
 * @file
 * @author  Danny Bickson, based on code by Aapo Kyrola <akyrola@cs.cmu.edu>
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
 * This file implements item based collaborative filtering by comparing all item pairs which
 * are connected by one or more user nodes. 
 *
 *
 * For Pearson's correlation 
 *
 * see: http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient

 Cosine Similarity

See: http://en.wikipedia.org/wiki/Cosine_similarity

Manhattan Distance

See http://en.wikipedia.org/wiki/Taxicab_geometry

Log Similarity Distance

See http://tdunning.blogspot.co.il/2008/03/surprise-and-coincidence.html

Chebychev Distance

http://en.wikipedia.org/wiki/Chebyshev_distance

Tanimoto Distance

See http://en.wikipedia.org/wiki/Jaccard_index

Slope One 

See "A prorammers guide to data mining" page 18:
http://guidetodatamining.com/guide/ch3/DataMining-ch3.pdf

*/
#define GRAPHCHI_DISABLE_COMPRESSION

#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <set>
#include <iostream>
#include "eigen_wrapper.hpp"
#include "distance.hpp"
#include "util.hpp"
#include "timer.hpp"
#include "common.hpp"
#include <libgen.h>

enum DISTANCE_METRICS{
  JACKARD = 0,
  AA = 1,
  RA = 2,
  PEARSON = 3,
  COSINE = 4,
  CHEBYCHEV = 5,
  MANHATTEN = 6,
  TANIMOTO = 7,
  LOG_LIKELIHOOD = 8,
  SLOPE_ONE = 9
};

int min_allowed_intersection = 1;
vec written_pairs;
size_t zero_dist = 0;
size_t item_pairs_compared = 0;
size_t not_enough = 0;
std::vector<FILE*> out_files;
timer mytimer;
bool * relevant_items  = NULL;
vec mean;
vec stddev;
int grabbed_edges = 0;
int distance_metric;
int debug;

bool is_item(vid_t v){ return v >= M; }
bool is_user(vid_t v){ return v < M; }

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef unsigned int VertexDataType;
typedef float  EdgeDataType;  // Edges store the "rating" of user->movie pair

struct vertex_data{ 
  vec pvec; 
  vertex_data(){ }

  void set_val(int index, float val){
    pvec[index] = val;
  }
  float get_val(int index){
    return pvec[index];
  }
};
std::vector<vertex_data> latent_factors_inmem;
#include "io.hpp"





struct dense_adj {
  sparse_vec edges;
  dense_adj() { }
  double intersect(const dense_adj & other){
    sparse_vec x1 = edges.unaryExpr(std::ptr_fun(equal_greater));
    sparse_vec x2 = other.edges.unaryExpr(std::ptr_fun(equal_greater));
    sparse_vec x3 = x1.cwiseProduct(x2);
    return sum(x3);
  }
};


// This is used for keeping in-memory
class adjlist_container {
  std::vector<dense_adj> adjs;
  //mutex m;
  public:
  vid_t pivot_st, pivot_en;

  adjlist_container() {
    if (debug)
      std::cout<<"setting pivot st and end to " << M << std::endl;
    pivot_st = M; //start pivor on item nodes (excluding user nodes)
    pivot_en = M;
  }

  void clear() {
    for(std::vector<dense_adj>::iterator it=adjs.begin(); it != adjs.end(); ++it) {
      if (nnz(it->edges)) {
        it->edges.resize(0);
      }
    }
    adjs.clear();
    if (debug)
      std::cout<<"setting pivot st to " << pivot_en << std::endl;
    pivot_st = pivot_en;
  }

  /** 
   * Extend the interval of pivot vertices to en.
   */
  void extend_pivotrange(vid_t en) {
    assert(en>pivot_en);
    pivot_en = en; 
    adjs.resize(pivot_en - pivot_st);
  }

  /**
   * Grab pivot's adjacency list into memory.
   */
  int load_edges_into_memory(graphchi_vertex<uint32_t, float> &v) {
    //assert(is_pivot(v.id()));
    //assert(is_item(v.id()));

    int num_edges = v.num_edges();
    //not enough user rated this item, we don't need to compare to it
    if (num_edges < min_allowed_intersection){
      relevant_items[v.id() - M] = false;
      return 0;
    }

    relevant_items[v.id() - M] = true;

    // Count how many neighbors have larger id than v
    dense_adj dadj;
    for(int i=0; i<num_edges; i++) 
      set_new( dadj.edges, v.edge(i)->vertex_id(), v.edge(i)->get_data());

    //std::sort(&dadj.adjlist[0], &dadj.adjlist[0] + num_edges);
    adjs[v.id() - pivot_st] = dadj;
    assert(v.id() - pivot_st < adjs.size());
    __sync_add_and_fetch(&grabbed_edges, num_edges /*edges_to_larger_id*/);
    return num_edges;
  }

  int acount(vid_t pivot) {
    return nnz(adjs[pivot - pivot_st].edges);
  }


  /** 
   * calc distance between two items.
   * Let a be all the users rated item 1
   * Let b be all the users rated item 2
   *
   * 3) Using Pearson correlation
   *      Dist_12 = (a - mean)*(b- mean)' / (std(a)*std(b))
   *
   * 4) Using cosine similarity:
   *      Dist_12 = (a*b) / sqrt(sum_sqr(a)) * sqrt(sum_sqr(b)))
   *
   * 5) Using chebychev:
   *      Dist_12 = max(abs(a-b))
   *
   * 6) Using manhatten distance:
   *      Dist_12 = sum(abs(a-b))
   *
   * 7) Using tanimoto:
   *      Dist_12 = 1.0 - [(a*b) / (sum_sqr(a) + sum_sqr(b) - a*b)]
   *
   * 8) Using log likelihood similarity
   *      Dist_12 = 1.0 - 1.0/(1.0 + loglikelihood)
   *
   * 9) Using slope one:
   *      Dist_12 = sum_(u in intersection (a,b) (r_u1-ru2 ) / size(intersection(a,b))) 
   */
  double calc_distance(graphchi_vertex<uint32_t, float> &v, vid_t pivot, int distance_metric) {
    //assert(is_pivot(pivot));
    //assert(is_item(pivot) && is_item(v.id()));
    dense_adj &pivot_edges = adjs[pivot - pivot_st];
    int num_edges = v.num_edges();
    //if there are not enough neighboring user nodes to those two items there is no need
    //to actually count the intersection
    if (num_edges < min_allowed_intersection || nnz(pivot_edges.edges) < min_allowed_intersection)
      return 0;

    dense_adj item_edges; 
    for(int i=0; i < num_edges; i++) 
      set_new(item_edges.edges, v.edge(i)->vertexid, v.edge(i)->get_data());

    double intersection_size = item_edges.intersect(pivot_edges); 

    //not enough user nodes rated both items, so the pairs of items are not compared.
    if (intersection_size < (double)min_allowed_intersection)
      return 0;

    if (distance_metric == PEARSON){
      if (debug){
        std::cout<< pivot -M+1<<" Pivot edges: " <<pivot_edges.edges << std::endl;
        std::cout<< "Minusmean:   " << minus(pivot_edges.edges,mean) << std::endl;
        std::cout<< v.id() -M+1<<"Item edges:  " <<item_edges.edges << std::endl;
        std::cout<< "Minusmean:   " << minus(item_edges.edges, mean) << std::endl;
      }
      double dist = minus(pivot_edges.edges, mean).dot(minus(item_edges.edges, mean));
      if (debug)
        std::cout<<"dist " << pivot-M+1 << ":" << v.id()-M+1 << " " << dist << std::endl;

      return dist / (stddev[pivot-M] * stddev[v.id()-M]);
    }
    else if (distance_metric == TANIMOTO){
      return calc_tanimoto_distance(pivot_edges.edges, 
          item_edges.edges,
          sum_sqr(pivot_edges.edges),
          sum_sqr(item_edges.edges));
    }
    else if (distance_metric == CHEBYCHEV){
      return calc_chebychev_distance(pivot_edges.edges, 
          item_edges.edges);
    }
    else if (distance_metric == LOG_LIKELIHOOD){
      return calc_loglikelihood_distance(pivot_edges.edges, 
          item_edges.edges,
          sum_sqr(pivot_edges.edges),
          sum_sqr(item_edges.edges));
    }
    else if (distance_metric == COSINE){
      return calc_cosine_distance(pivot_edges.edges, 
          item_edges.edges,
          sum_sqr(pivot_edges.edges),
          sum_sqr(item_edges.edges));
    }
    else if (distance_metric ==MANHATTEN){
      return calc_manhatten_distance(pivot_edges.edges, 
          item_edges.edges);
    }
    else if (distance_metric == SLOPE_ONE){
      return calc_slope_one_distance(pivot_edges.edges, item_edges.edges) / intersection_size;
    }
    return NAN;  
  }

  inline bool is_pivot(vid_t vid) {
    return vid >= pivot_st && vid < pivot_en;
  }
};


adjlist_container * adjcontainer;
struct index_val{
  uint index;
  float val;
  index_val(){
    index = -1; val = 0;
  }
  index_val(uint index, float val): index(index), val(val){ }
};
bool Greater(const index_val& a, const index_val& b)
{
      return a.val > b.val;
}

struct ItemDistanceProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {


  /**
   *  Vertex update function.
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &v, graphchi_context &gcontext) {
    if (debug)
      printf("Entered iteration %d with %d\n", gcontext.iteration, v.id());

    //in the zero iteration compute the mean
    if (gcontext.iteration == 0){
      if (is_item(v.id())){
        for(int i=0; i<v.num_edges(); i++) {
          graphchi_edge<float> * e = v.edge(i);
          vid_t user = e->vertexid;
          mean[user] += e->get_data() / (float)N;
        }
      }
    }
    //at the first iteration compute the stddev of each item from the mean
    else if (gcontext.iteration == 1){
      if (is_item(v.id())){
        dense_adj item_edges; 
        for(int i=0; i < v.num_edges(); i++) 
          set_new(item_edges.edges, v.edge(i)->vertexid, v.edge(i)->get_data());
        stddev[v.id() - M] = sum(minus(item_edges.edges, mean).array().pow(2)) / (M-1.0);
        if (debug)
          std::cout<<"item: " << v.id() - M+1 << " stddev: " << stddev[v.id() - M] << std::endl;
      }
    }

    /* even iteration numbers:
     * 1) load a subset of items into memory (pivots)
     * 2) Find which subset of items needs to compared to the users
     */
    else if (gcontext.iteration % 2 == 0) {
      if (adjcontainer->is_pivot(v.id()) && is_item(v.id())){
        adjcontainer->load_edges_into_memory(v);         
        if (debug)
          printf("Loading pivot %d intro memory\n", v.id());
      }
      else if (is_user(v.id())){
        //check if this user is connected to any pivot item
        bool has_pivot = false;
        int pivot = -1;
        for(int i=0; i<v.num_edges(); i++) {
          graphchi_edge<float> * e = v.edge(i);
          //assert(is_item(e->vertexid)); 
          if (adjcontainer->is_pivot(e->vertexid) && relevant_items[e->vertexid-M]) {
            has_pivot = true;
            pivot = e->vertexid;
            break;
          }
        }
        if (debug)
          printf("user %d is linked to pivot %d\n", v.id(), pivot);
        if (!has_pivot) //this user is not connected to any of the pivot item nodes and thus
          //it is not relevant at this point
          return; 

        //this user is connected to a pivot items, thus all connected items should be compared
        for(int i=0; i<v.num_edges(); i++) {
          graphchi_edge<float> * e = v.edge(i);
          //assert(v.id() != e->vertexid);
          relevant_items[e->vertexid - M] = true;
        }
      }//is_user 
    } //iteration % 2 =  1
    /* odd iteration number:
     * 1) For any item connected to a pivot item
     *       compute itersection
     */
    else {
      if (!relevant_items[v.id() - M]){
        return;
      }
      std::vector<index_val> heap;

      for (vid_t i=adjcontainer->pivot_st; i< adjcontainer->pivot_en; i++){
        //since metric is symmetric, compare only to pivots which are smaller than this item id
        if (i >= v.id() || (!relevant_items[i-M]))
          continue;

        double dist = adjcontainer->calc_distance(v, i, distance_metric);
        item_pairs_compared++;
        if (item_pairs_compared % 100000 == 0)
          logstream(LOG_INFO)<< std::setw(10) << mytimer.current_time() << ")  " << std::setw(10) << item_pairs_compared << " pairs compared " << std::setw(10) <<sum(written_pairs) << " written. " << std::endl;
        if (debug)
          printf("comparing %d to pivot %d distance is %lg\n", i - M + 1, v.id() - M + 1, dist);
        if (dist != 0){
          heap.push_back(index_val(i, dist)); 
        }
        else zero_dist++;

      }
      sort(heap.begin(), heap.end(), &Greater);
      int thread_num = omp_get_thread_num();
      if (heap.size() < K)
        not_enough++;
      for (uint i=0; i< std::min(heap.size(), (size_t)K); i++){
          int rc = fprintf(out_files[thread_num], "%u %u %.12lg\n", v.id()-M+1, heap[i].index-M+1, (double)heap[i].val);//write item similarity to file
          written_pairs[omp_get_thread_num()]++;
         if (rc <= 0){
            perror("Failed to write output");
            logstream(LOG_FATAL)<<"Failed to write output to: file: " << training << omp_get_thread_num() << ".out" << std::endl;  
         }
      }
    }//end of iteration % 2 == 1
  }//end of update function

  /**
   * Called before an iteration starts. 
   * On odd iteration, schedule both users and items.
   * on even iterations, schedules only item nodes
   */
  void before_iteration(int iteration, graphchi_context &gcontext) {
    gcontext.scheduler->remove_tasks(0, gcontext.nvertices - 1);
    if (gcontext.iteration == 0)
      written_pairs = zeros(gcontext.execthreads);

    if (gcontext.iteration % 2 == 0){
      memset(relevant_items, 0, sizeof(bool)*N);
      for (vid_t i=0; i < M+N; i++){
        gcontext.scheduler->add_task(i); 
      }
      if (debug)
        printf("scheduling all nodes, setting relevant_items to zero\n");
      grabbed_edges = 0;
      adjcontainer->clear();
    } else { //iteration % 2 == 1
      for (vid_t i=M; i < M+N; i++){
        gcontext.scheduler->add_task(i); 
      }
    } 
  }

  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    if (debug && gcontext.iteration == 0)
      std::cout<<"Mean : " << mean << std::endl;
  }

  /**
   * Called before an execution interval is started.
   *
   * On every even iteration, we load pivot's item connected user lists to memory. 
   * Here we manage the memory to ensure that we do not load too much
   * edges into memory.
   */
  void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        

    /* on even iterations, loads pivot items into memory base on the membudget_mb allowed memory size */
    if ((gcontext.iteration % 2 == 0) && (gcontext.iteration >= 2)) {
      if (debug){
        printf("entering iteration: %d on before_exec_interval\n", gcontext.iteration);
        printf("pivot_st is %d window_en %d\n", adjcontainer->pivot_st, window_en);
      }
      if (adjcontainer->pivot_st <= window_en) {
        size_t max_grab_edges = get_option_long("membudget_mb", 1024) * 1024 * 1024 / 8;
        if (grabbed_edges < max_grab_edges * 0.8) {
          logstream(LOG_DEBUG) << "Window init, grabbed: " << grabbed_edges << " edges" << " extending pivor_range to : " << window_en + 1 << std::endl;
          adjcontainer->extend_pivotrange(window_en + 1);
          logstream(LOG_DEBUG) << "Window en is: " << window_en << " vertices: " << gcontext.nvertices << std::endl;
          if (window_en+1 == gcontext.nvertices) {
            // every item was a pivot item, so we are done
            logstream(LOG_DEBUG)<<"Setting last iteration to: " << gcontext.iteration + 2 << std::endl;
            gcontext.set_last_iteration(gcontext.iteration + 2);                    
          }
        } else {
          logstream(LOG_DEBUG) << "Too many edges, already grabbed: " << grabbed_edges << std::endl;
        }
      }
    }
  }
};




int main(int argc, const char ** argv) {
  print_copyright();

  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("item-cf2");    
  /* Basic arguments for application */
  min_allowed_intersection = get_option_int("min_allowed_intersection", min_allowed_intersection);

  distance_metric          = get_option_int("distance", PEARSON);
  if (distance_metric != PEARSON && distance_metric != MANHATTEN && distance_metric != COSINE &&
      distance_metric != CHEBYCHEV && distance_metric != LOG_LIKELIHOOD && distance_metric != TANIMOTO && distance_metric != SLOPE_ONE)
    logstream(LOG_FATAL)<<"--distance_metrix=XX should be one of: 3=PEARSON, 4=COSINE, 5=CHEBYCHEV, 6=MANHATTEN, 7=TANIMOTO, 8=LOG_LIKELIHOOD, 9 = SLOPE_ONE" << std::endl;
  debug                    = get_option_int("debug", 0);
  parse_command_line_args();

  //if (distance_metric != JACKARD && distance_metric != AA && distance_metric != RA)
  //  logstream(LOG_FATAL)<<"Wrong distance metric. --distance_metric=XX, where XX should be either 0) JACKARD, 1) AA, 2) RA" << std::endl;  

  mytimer.start();
  int nshards          = convert_matrixmarket<EdgeDataType>(training, 0, 0, 3, TRAINING, false);
  if (nshards != 1)
    logstream(LOG_FATAL)<<"This application currently supports only 1 shard" << std::endl;
  K                        = get_option_int("K", K);
  if (K <= 0)
    logstream(LOG_FATAL)<<"Please specify the number of ratings to generate for each user using the --K command" << std::endl;

 logstream(LOG_INFO) << "M = " << M << std::endl;
 
  assert(M > 0 && N > 0);

  //initialize data structure which saves a subset of the items (pivots) in memory
  adjcontainer = new adjlist_container();
  //array for marking which items are conected to the pivot items via users.
  relevant_items = new bool[N];
  mean = vec::Zero(M);
  stddev = vec::Zero(N); 

  /* Run */
  ItemDistanceProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, 1, true, m); 
  set_engine_flags(engine);
  engine.set_maxwindow(M+N+1);

  //open output files as the number of operating threads
  out_files.resize(number_of_omp_threads());
  for (uint i=0; i< out_files.size(); i++){
    char buf[256];
    sprintf(buf, "%s.out%d", training.c_str(), i);
    out_files[i] = open_file(buf, "w");
  }

  //run the program
  engine.run(program, niters);

  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  
  std::cout<<"Total item pairs compared: " << item_pairs_compared << " total written to file: " << sum(written_pairs) << " pairs with zero distance: " << zero_dist << std::endl;
  if (not_enough)
    logstream(LOG_WARNING)<<"Items that did not have enough similar items: " << not_enough << std::endl;
 
  for (uint i=0; i< out_files.size(); i++)
    fclose(out_files[i]);

  delete[] relevant_items;

  /* write the matrix market info header to be used later */
  FILE * pmm = fopen((training + "-topk:info").c_str(), "w");
  if (pmm == NULL)
    logstream(LOG_FATAL)<<"Failed to open " << training << ":info to file" << std::endl;
  fprintf(pmm, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(pmm, "%u %u %u\n", N, N, (unsigned int)sum(written_pairs));
  fclose(pmm);

  /* sort output files */
  logstream(LOG_INFO)<<"Going to sort and merge output files " << std::endl;
  std::string dname= dirname(strdup(argv[0]));
  system(("bash " + dname + "/topk.sh " + std::string(basename(strdup(training.c_str())))).c_str()); 

  return 0;
}

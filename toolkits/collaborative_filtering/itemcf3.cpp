
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
*/

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
  JACCARD_WEIGHT = 9
};

int min_allowed_intersection = 1;
size_t written_pairs = 0;
size_t item_pairs_compared = 0;
std::vector<FILE*> out_files;
timer mytimer;
vec mean;
vec stddev;
int grabbed_edges = 0;
int distance_metric;
int debug;

bool is_item(vid_t v){ return M == N ? true : v >= M; }
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
  //mutex m;
  public:
  std::vector<dense_adj> adjs;
  vid_t pivot_st, pivot_en;

  adjlist_container() {
    if (debug)
      std::cout<<"setting pivot st and end to " << M << std::endl;
    if (distance_metric == JACCARD_WEIGHT){
      pivot_st = 0;
      pivot_en = 0;
    }
    else {
      pivot_st = M; //start pivor on item nodes (excluding user nodes)
      pivot_en = M;
    }
  }

  void clear() {
    for(std::vector<dense_adj>::iterator it=adjs.begin(); it != adjs.end(); ++it) {
      if (nnz(it->edges)) {
        it->edges.resize(0);
      }
    }
    adjs.clear();
    if (debug)
      std::cout<<"setting pivot end to " << pivot_en << std::endl;
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
  int load_edges_into_memory(graphchi_vertex<VertexDataType, EdgeDataType> &v) {
    //assert(is_pivot(v.id()));
    //assert(is_item(v.id()));

    int num_edges = v.num_edges();
    //not enough user rated this item, we don't need to compare to it
    if (num_edges < min_allowed_intersection){
      if (debug)
        logstream(LOG_DEBUG)<<"Skipping since num edges: " << num_edges << std::endl;
      return 0;
    }


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
   *      Dist_ab = (a - mean)*(b- mean)' / (std(a)*std(b))
   *
   * 4) Using cosine similarity:
   *      Dist_ab = (a*b) / sqrt(sum_sqr(a)) * sqrt(sum_sqr(b)))
   *
   *    5) Using chebychev:
   *          Dist_ab = max(abs(a-b))
   *
   * 6) Using manhatten distance:
   *      Dist_ab = sum(abs(a-b))
   *
   * 7) Using tanimoto:
   *      Dist_ab = 1.0 - [(a*b) / (sum_sqr(a) + sum_sqr(b) - a*b)]
   *
   * 8) Using log likelihood similarity
   *      Dist_ab = 1.0 - 1.0/(1.0 + loglikelihood)
   *
   * 9) Using Jaccard:
   *      Dist_ab = intersect(a,b) / (size(a) + size(b) - intersect(a,b)) 
   */
  double calc_distance(graphchi_vertex<VertexDataType, EdgeDataType> &v, vid_t pivot, int distance_metric) {
    //assert(is_pivot(pivot));
    //assert(is_item(pivot) && is_item(v.id()));
    dense_adj &pivot_edges = adjs[pivot - pivot_st];
    int num_edges = v.num_edges();

    dense_adj item_edges; 
    for(int i=0; i < num_edges; i++){ 
      set_new(item_edges.edges, v.edge(i)->vertexid, v.edge(i)->get_data());
    }

    if (distance_metric == JACCARD_WEIGHT){
      return calc_jaccard_weight_distance(pivot_edges.edges, item_edges.edges, get_val( pivot_edges.edges, v.id()), 0);
    }
    return NAN;  
  }

  inline bool is_pivot(vid_t vid) {
    return vid >= pivot_st && vid < pivot_en;
  }
};


adjlist_container * adjcontainer;

struct ItemDistanceProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {


  /**
   *  Vertex update function.
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &v, graphchi_context &gcontext) {
    if (debug)
      printf("Entered iteration %d with %d - edges %d\n", gcontext.iteration, v.id(), v.num_edges());

    /* even iteration numbers:
     * 1) load a subset of items into memory (pivots)
     * 2) Find which subset of items needs to compared to the users
     */
    if (gcontext.iteration % 2 == 0) {
      if (adjcontainer->is_pivot(v.id())){
        adjcontainer->load_edges_into_memory(v);         
        if (debug)
          printf("Loading pivot %d intro memory\n", v.id());
      }
    }
    else {

      for (vid_t i=adjcontainer->pivot_st; i< adjcontainer->pivot_en; i++){
        //since metric is symmetric, compare only to pivots which are smaller than this item id
        if (i >= v.id())
          continue;
        
        dense_adj &pivot_edges = adjcontainer->adjs[i - adjcontainer->pivot_st];
        //pivot is not connected to this item, continue
        if (get_val(pivot_edges.edges, v.id()) == 0)
            continue;

        double dist = adjcontainer->calc_distance(v, i, distance_metric);
        item_pairs_compared++;
        if (item_pairs_compared % 1000000 == 0)
          logstream(LOG_INFO)<< std::setw(10) << mytimer.current_time() << ")  " << std::setw(10) << item_pairs_compared << " pairs compared " << std::endl;
        if (debug)
          printf("comparing %d to pivot %d distance is %lg\n", i+ 1, v.id() + 1, dist);
        if (dist != 0){
          fprintf(out_files[omp_get_thread_num()], "%u %u %.12lg\n", v.id()+1, i+1, (double)dist);//write item similarity to file
          //where the output format is: 
          //[item A] [ item B ] [ distance ] 
          written_pairs++;
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
      
    if (gcontext.iteration % 2 == 0){
      for (vid_t i=0; i < M; i++){
        gcontext.scheduler->add_task(i); 
      }
      grabbed_edges = 0;
      adjcontainer->clear();
    } else { //iteration % 2 == 1
      for (vid_t i=0; i< M; i++){
        gcontext.scheduler->add_task(i); 
      }
    } 
  }

  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
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
    if ((gcontext.iteration % 2 == 0)) {
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

  distance_metric          = get_option_int("distance", JACCARD_WEIGHT);
      if (distance_metric != JACCARD_WEIGHT)
    logstream(LOG_FATAL)<<"--distance_metrix=XX should be one of:9= JACCARD_WEIGHT" << std::endl;
  debug                    = get_option_int("debug", 0);
  parse_command_line_args();

  //if (distance_metric != JACKARD && distance_metric != AA && distance_metric != RA)
  //  logstream(LOG_FATAL)<<"Wrong distance metric. --distance_metric=XX, where XX should be either 0) JACKARD, 1) AA, 2) RA" << std::endl;  

  mytimer.start();
  int nshards          = convert_matrixmarket<EdgeDataType>(training, 0, 0, 3, TRAINING, true);

  assert(M > 0 && N > 0);

  //initialize data structure which saves a subset of the items (pivots) in memory
  adjcontainer = new adjlist_container();

  /* Run */
  ItemDistanceProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, true, m); 
  set_engine_flags(engine);

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
  
  std::cout<<"Total item pairs compared: " << item_pairs_compared << " total written to file: " << written_pairs << std::endl;

  for (uint i=0; i< out_files.size(); i++)
    fclose(out_files[i]);

  std::cout<<"Created output files with the format: " << training << ".outXX, where XX is the output thread number" << std::endl; 

  return 0;
}

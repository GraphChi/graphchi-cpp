
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
 * For the Jaccard index see: http://en.wikipedia.org/wiki/Jaccard_index
 *
 * For the AA index see: http://arxiv.org/abs/0907.1728 "Role of Weak Ties in Link Prediction of Complex Networks", equation (2)
 *
 * For the RA index see the above paper, equation (3)
 */


#include <set>
#include <iomanip>
#include "common.hpp"
#include "timer.hpp"
#include "eigen_wrapper.hpp"
#include "engine/dynamic_graphs/graphchi_dynamicgraph_engine.hpp"

using namespace graphchi;

enum DISTANCE_METRICS{
  JACKARD = 0,
  AA = 1,
  RA = 2,
};

int min_allowed_intersection = 1;
size_t written_pairs = 0;
size_t item_pairs_compared = 0;
std::vector<FILE*> out_files;
timer mytimer;
bool * relevant_items  = NULL;
int grabbed_edges = 0;
int distance_metric;

bool is_item(vid_t v){ return v >= M; }
bool is_user(vid_t v){ return v < M; }

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef unsigned int VertexDataType;
typedef unsigned int  EdgeDataType;  // Edges store the "rating" of user->movie pair

struct vertex_data{ 
   vec pvec; 
   int degree; 
   vertex_data(){ degree = 0; }
};
std::vector<vertex_data> latent_factors_inmem;
#include "io.hpp"


struct dense_adj {
  int count;
  vid_t * adjlist;

  dense_adj() { adjlist = NULL; }
  dense_adj(int _count, vid_t * _adjlist) : count(_count), adjlist(_adjlist) {
  }

};


// This is used for keeping in-memory
class adjlist_container {
  std::vector<dense_adj> adjs;
  //mutex m;
  public:
  vid_t pivot_st, pivot_en;

  adjlist_container() {
    pivot_st = M; //start pivor on item nodes (excluding user nodes)
    pivot_en = M;
  }

  void clear() {
    for(std::vector<dense_adj>::iterator it=adjs.begin(); it != adjs.end(); ++it) {
      if (it->adjlist != NULL) {
        free(it->adjlist);
        it->adjlist = NULL;
      }
    }
    adjs.clear();
    pivot_st = pivot_en;
  }

  /** 
   * Extend the interval of pivot vertices to en.
   */
  void extend_pivotrange(vid_t en) {
    assert(en>=pivot_en);
    pivot_en = en; 
    adjs.resize(pivot_en - pivot_st);
  }

  /**
   * Grab pivot's adjacency list into memory.
   */
  int load_edges_into_memory(graphchi_vertex<uint32_t, uint32_t> &v) {
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
    dense_adj dadj = dense_adj(num_edges, (vid_t*) calloc(sizeof(vid_t), num_edges));
    for(int i=0; i<num_edges; i++) {
      dadj.adjlist[i] = v.edge(i)->vertex_id();
    }
    std::sort(dadj.adjlist, dadj.adjlist + num_edges);
    adjs[v.id() - pivot_st] = dadj;
    assert(v.id() - pivot_st < adjs.size());
    __sync_add_and_fetch(&grabbed_edges, num_edges /*edges_to_larger_id*/);
    return num_edges;
  }

  int acount(vid_t pivot) {
    return adjs[pivot - pivot_st].count;
  }


  /** 
   * calc distance between two items.
   * Let a be all the users rated item 1
   * Let b be all the users rated item 2
   *
   * 1) Using Jackard index:
   *      Dist_ab = intersection(a,b) / (size(a) + size(b) - size(intersection(a,b))
   *
   * 2) Using AA index:
   *      Dist_ab = sum_user k in intersection(a,b) [ 1 / log(degree(k)) ] 
   *
   * 3) Using RA index:
   *      Dist_ab = sum_user k in intersection(a,b) [ 1 / degree(k) ] 
   */
  double calc_distance(graphchi_vertex<uint32_t, uint32_t> &v, vid_t pivot, int distance_metric) {
    //assert(is_pivot(pivot));
    //assert(is_item(pivot) && is_item(v.id()));
    dense_adj &pivot_edges = adjs[pivot - pivot_st];
    int num_edges = v.num_edges();
    //if there are not enough neighboring user nodes to those two items there is no need
    //to actually count the intersection
    if (num_edges < min_allowed_intersection || pivot_edges.count < min_allowed_intersection)
      return 0;

    std::vector<vid_t> edges;
    edges.resize(num_edges);
    for(int i=0; i < num_edges; i++) {
      vid_t other_vertex = v.edge(i)->vertexid;
      edges[i] = other_vertex;
    }
    sort(edges.begin(), edges.end());
    
    std::set<vid_t> intersection;
    std::set_intersection(
        pivot_edges.adjlist, pivot_edges.adjlist + pivot_edges.count, 
        edges.begin(), edges.end(), 
        std::inserter(intersection, intersection.begin()));
      
    double intersection_size = (double)intersection.size();
    //not enough user nodes rated both items, so the pairs of items are not compared.
    if (intersection_size < (double)min_allowed_intersection)
        return 0;
  
    if (distance_metric == JACKARD){
      uint set_a_size = v.num_edges(); //number of users connected to current item
      uint set_b_size = acount(pivot); //number of users connected to current pivot
      return intersection_size / (double)(set_a_size + set_b_size - intersection_size); //compute the distance
    }
    else if (distance_metric == AA){
       double dist = 0;
       for (std::set<vid_t>::iterator i= intersection.begin() ; i != intersection.end(); i++){
         vid_t user = *i;
         assert(latent_factors_inmem.size() == M && is_user(user));
         assert(latent_factors_inmem[user].degree > 0);
         dist += 1.0 / log(latent_factors_inmem[user].degree);
       }
       return dist;
    }
    else if (distance_metric == RA){
       double dist = 0;
       for (std::set<vid_t>::iterator i= intersection.begin() ; i != intersection.end(); i++){
         vid_t user = *i;
         assert(latent_factors_inmem.size() == M && is_user(user));
         assert(latent_factors_inmem[user].degree > 0);
         dist += 1.0 / latent_factors_inmem[user].degree;
       }
       return dist;
    }
    return 0;
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
    //printf("Entered iteration %d with %d\n", gcontext.iteration, v.id());
 
    /* even iteration numbers:
     * 1) load a subset of items into memory (pivots)
     * 2) Find which subset of items needs to compared to the users
     */
    if (gcontext.iteration % 2 == 0) {
      if (adjcontainer->is_pivot(v.id()) && is_item(v.id())){
        adjcontainer->load_edges_into_memory(v);         
        //printf("Loading pivot %dintro memory\n", v.id());
      }
      else if (is_user(v.id())){

        //in the zero iteration, if using AA distance metric, initialize array
        //with node degrees 
        if (gcontext.iteration == 0 && (distance_metric == AA || distance_metric == RA)){
           latent_factors_inmem[v.id()].degree = v.num_edges();
        }

        //check if this user is connected to any pivot item
        bool has_pivot = false;
        int pivot = -1;
        for(int i=0; i<v.num_edges(); i++) {
          graphchi_edge<uint32_t> * e = v.edge(i);
          //assert(is_item(e->vertexid)); 
          if (adjcontainer->is_pivot(e->vertexid) && relevant_items[e->vertexid-M]) {
            has_pivot = true;
            pivot = e->vertexid;
            break;
          }
        }
        //printf("user %d is linked to pivot %d\n", v.id(), pivot);
        if (!has_pivot) //this user is not connected to any of the pivot item nodes and thus
          //it is not relevant at this point
          return; 

        //this user is connected to a pivot items, thus all connected items should be compared
        for(int i=0; i<v.num_edges(); i++) {
          graphchi_edge<uint32_t> * e = v.edge(i);
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

      for (vid_t i=adjcontainer->pivot_st; i< adjcontainer->pivot_en; i++){
        //since metric is symmetric, compare only to pivots which are smaller than this item id
        if (i >= v.id() || (!relevant_items[i-M]))
          continue;
        
        double dist = adjcontainer->calc_distance(v, i, distance_metric);
        item_pairs_compared++;
        if (item_pairs_compared % 1000000 == 0)
          logstream(LOG_INFO)<< std::setw(10) << mytimer.current_time() << ")  " << std::setw(10) << item_pairs_compared << " pairs compared " << std::endl;

        //printf("comparing %d to pivot %d intersection is %d\n", i - M + 1, v.id() - M + 1, intersection_size);
        if (dist != 0){
         fprintf(out_files[omp_get_thread_num()], "%u %u %lg\n", v.id()-M+1, i-M+1, (double)dist);//write item similarity to file
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
    gcontext.scheduler->remove_tasks(0, (int) gcontext.nvertices - 1);
    if (gcontext.iteration % 2 == 0){
      memset(relevant_items, 0, sizeof(bool)*N);
      for (vid_t i=0; i < M+N; i++){
        gcontext.scheduler->add_task(i); 
      }
      //printf("setting relevant_items to zero\n");
      grabbed_edges = 0;
      adjcontainer->clear();
    } else { //iteration % 2 == 1
      for (vid_t i=M; i < M+N; i++){
        gcontext.scheduler->add_task(i); 
      }
    } 
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
    if (gcontext.iteration % 2 == 0) {
      printf("entering iteration: %d on before_exec_interval\n", gcontext.iteration);
      printf("pivot_st is %d window_en %d\n", adjcontainer->pivot_st, window_en);
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
  metrics m("item-cf");    
  /* Basic arguments for application */
  min_allowed_intersection = get_option_int("min_allowed_intersection", min_allowed_intersection);
  distance_metric          = get_option_int("distance", JACKARD);
  if (distance_metric != JACKARD && distance_metric != AA && distance_metric != RA)
    logstream(LOG_FATAL)<<"Wrong distance metric. --distance_metric=XX, where XX should be either 0) JACKARD, 1) AA, 2) RA" << std::endl;  
  parse_command_line_args();

  mytimer.start();
  int nshards          = convert_matrixmarket<EdgeDataType>(training/*, orderByDegreePreprocessor*/);

  assert(M > 0 && N > 0);
  
  //initialize data structure which saves a subset of the items (pivots) in memory
  adjcontainer = new adjlist_container();
  //array for marking which items are conected to the pivot items via users.
  relevant_items = new bool[N];

  //store node degrees in an array to be used for AA distance metric
  if (distance_metric == AA || distance_metric == RA)
    latent_factors_inmem.resize(M);

  /* Run */
  ItemDistanceProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training,nshards, true, m); 
  set_engine_flags(engine);

  //open output files as the number of operating threads
  out_files.resize(number_of_omp_threads());
  for (uint i=0; i< out_files.size(); i++){
    char buf[256];
    sprintf(buf, "%s.out%d", training.c_str(), i);
    out_files[i] = fopen(buf, "w");
    if (out_files[i] == NULL)
      logstream(LOG_FATAL)<<"Failed to open out file " << training << ".out" << i << std::endl;
  }

  //run the program
  engine.run(program, niters);

  /* Report execution metrics */
  metrics_report(m);
  logstream(LOG_INFO)<<"Total item pairs compared: " << item_pairs_compared << " total written to file: " << written_pairs << std::endl;

  for (uint i=0; i< out_files.size(); i++)
    fclose(out_files[i]);

  logstream(LOG_INFO)<<"Created output files with the format: " << training << "XX.out, where XX is the output thread number" << std::endl; 

  delete[] relevant_items;
  return 0;
}

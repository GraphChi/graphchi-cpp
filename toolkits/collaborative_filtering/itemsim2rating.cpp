
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


int min_allowed_intersection = 1;
size_t written_pairs = 0;
size_t item_pairs_compared = 0;
FILE * out_file;
timer mytimer;
bool * relevant_items  = NULL;
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
  vec        ratings;
  dense_adj() { }
};


// This is used for keeping in-memory
class adjlist_container {
  public:
    std::vector<dense_adj> adjs;
    //mutex m;
  public:
    vid_t pivot_st, pivot_en;

    adjlist_container() {
      if (debug)
        std::cout<<"setting pivot st and end to " << 0 << std::endl;
      pivot_st = 0; //start pivot on user nodes (excluding item nodes)
      pivot_en = 0;
    }

    void clear() {
      for(std::vector<dense_adj>::iterator it=adjs.begin(); it != adjs.end(); ++it) {
        if (nnz(it->edges)) {
          it->edges.resize(0);
        }
        it->ratings.resize(0);
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
      assert(en > pivot_st);
      pivot_en = en; 
      adjs.resize(pivot_en - pivot_st);
      for (uint i=0; i< pivot_en - pivot_st; i++)
        adjs[i].ratings = zeros(N);
    }

    /**
     * Grab pivot's adjacency list into memory.
     */
    int load_edges_into_memory(graphchi_vertex<uint32_t, float> &v) {
      assert(is_pivot(v.id()));
      assert(is_user(v.id()));

      int num_edges = v.num_edges();

      dense_adj dadj;
      for(int i=0; i<num_edges; i++) 
        set_new( dadj.edges, v.edge(i)->vertex_id(), v.edge(i)->get_data());
      dadj.ratings = zeros(N);
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
     */
    double calc_distance(graphchi_vertex<uint32_t, EdgeDataType> &item, vid_t user_pivot, int distance_metric) {
      assert(is_pivot(user_pivot));
      //assert(is_item(pivot) && is_item(v.id()));
      dense_adj &pivot_edges = adjs[user_pivot - pivot_st];

      if (!get_val(pivot_edges.edges, item.id())){
        if (debug)
          logstream(LOG_DEBUG)<<"Skipping item pivot pair since not connected!" << std::endl;
        return 0;
      }

      int num_edges = item.num_edges();
      //if there are not enough neighboring user nodes to those two items there is no need
      //to actually count the intersection
      if (num_edges < min_allowed_intersection || nnz(pivot_edges.edges) < min_allowed_intersection)
        return 0;



      std::vector<vid_t> edges;
    edges.resize(num_edges);
    for(int i=0; i < num_edges; i++) {
      vid_t other_vertex = item.edge(i)->vertexid;
      edges[i] = other_vertex;
    }
    sort(edges.begin(), edges.end());
 
      for(int i=0; i < num_edges; i++){
        if (is_item(edges[i])){
          //skip duplicate ratings (if any)
          if (i > 0 && edges[i] == edges[i-1])
            continue;
          //skip self similarity of items (if any)
          if (edges[i] == item.id())
            continue;
          //skip items that are already rated
          if (get_val(pivot_edges.edges, edges[i]))
              continue;

          pivot_edges.ratings[edges[i]-M] += item.edge(i)->get_data();
          if (debug)
            logstream(LOG_DEBUG)<<"Adding weight: " << item.edge(i)->get_data() << " to item: " << edges[i]-M+1 << " for user: " << user_pivot+1<<std::endl;
        }
        else if (debug)
          logstream(LOG_DEBUG)<<"Skpping edge to: " << edges[i] << " connected? " << get_val(pivot_edges.edges, edges[i]) << std::endl;
      }

      //not enough user nodes rated both items, so the pairs of items are not compared.
      //if (intersection_size < (double)min_allowed_intersection)
      return 0;
      //TODO
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
      printf("Entered iteration %d with %d\n", gcontext.iteration, is_item(v.id()) ? (v.id() - M + 1): v.id());

    /* Even iteration numbers:
     * 1) load a subset of users into memory (pivots)
     * 2) Find which subset of items is connected to the users
     */
    if (gcontext.iteration % 2 == 0) {
      if (adjcontainer->is_pivot(v.id()) && is_user(v.id())){
        adjcontainer->load_edges_into_memory(v);         
        if (debug)
          printf("Loading pivot %d intro memory\n", v.id());
      }
      else if (is_item(v.id())){
        //check if this item is connected to any pivot user
        bool has_pivot = false;
        int pivot = -1;
        for(int i=0; i<v.num_edges(); i++) {
          graphchi_edge<float> * e = v.edge(i);
          if (is_user(e->vertexid) && adjcontainer->is_pivot(e->vertexid)){ //items are connected both to users and similar items
            has_pivot = true;
            pivot = e->vertexid;
            break;
          }
        }
        if (debug)
          printf("item %d is linked to pivot %d\n", v.id(), pivot);

        relevant_items[v.id() - M] = true;
        if (!has_pivot) //this item is not connected to any of the pivot users nodes and thus
          //it is not relevant at this point
          return; 

        //this item is connected to a pivot user, thus all connected items should be compared
        for(int i=0; i<v.num_edges(); i++) {
          graphchi_edge<float> * e = v.edge(i);
          assert(v.id() != e->vertexid);
          if (is_item(e->vertexid))
            relevant_items[e->vertexid - M] = true;
        }
      }//is_user 
    } //iteration % 2 =  1 */
    /* odd iteration number:
     * 1) For any item connected to a pivot item
     *       compute itersection
     */
    else {
      assert(is_item(v.id()));
      if (!relevant_items[v.id() - M]){
        if (debug)
        logstream(LOG_DEBUG)<<"Skipping item: " << v.id()-M <<  " since it is not relevant. " << std::endl;
        return;
      }

      for (int i=0; i< v.num_edges(); i++){
     // for (vid_t i=adjcontainer->pivot_st; i< adjcontainer->pivot_en; i++){
        //since metric is symmetric, compare only to pivots which are smaller than this item id
        //if (i >= v.id())
        //  continue;
        if (!is_user(v.edge(i)->vertex_id()) || !adjcontainer->is_pivot(v.edge(i)->vertex_id()))
          break;
        if (debug)
          printf("comparing user pivot %d to item %d\n", v.edge(i)->vertex_id()+1 , v.id() - M + 1);
   
        double dist = adjcontainer->calc_distance(v, v.edge(i)->vertex_id(), distance_metric);
        item_pairs_compared++;
        if (item_pairs_compared % 1000000 == 0)
          logstream(LOG_INFO)<< std::setw(10) << mytimer.current_time() << ")  " << std::setw(10) << item_pairs_compared << " pairs compared " << std::endl;
       //if (dist != 0){
        //  fprintf(out_files[omp_get_thread_num()], "%u %u %lg\n", v.id()-M+1, i-M+1, (double)dist);//write item similarity to file
        //where the output format is: 
        //[item A] [ item B ] [ distance ] 
        //  written_pairs++;
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
      if (debug)
        printf("scheduling all nodes, setting relevant_items to zero\n");
      grabbed_edges = 0;
      adjcontainer->clear();
    } else { //iteration % 2 == 1, schedule only item nodes
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
    if ((gcontext.iteration % 2 == 0)) {
      printf("entering iteration: %d on before_exec_interval\n", gcontext.iteration);
      printf("pivot_st is %d window_en %d\n", adjcontainer->pivot_st, window_en);
      if (adjcontainer->pivot_st <= window_en) {
        size_t max_grab_edges = get_option_long("membudget_mb", 1024) * 1024 * 1024 / 8;
        if (grabbed_edges < max_grab_edges * 0.8) {
          logstream(LOG_DEBUG) << "Window init, grabbed: " << grabbed_edges << " edges" << " extending pivor_range to : " << window_en + 1 << std::endl;
          adjcontainer->extend_pivotrange(window_en + 1);
          logstream(LOG_DEBUG) << "Window en is: " << window_en << " vertices: " << gcontext.nvertices << std::endl;
          if (window_en+1 == M) {
            // every user was a pivot item, so we are done
            logstream(LOG_DEBUG)<<"Setting last iteration to: " << gcontext.iteration + 2 << std::endl;
            gcontext.set_last_iteration(gcontext.iteration + 2);                    
          }
        } else {
          logstream(LOG_DEBUG) << "Too many edges, already grabbed: " << grabbed_edges << std::endl;
        }
      }
    }

  }


  /**
   * Called before an execution interval is started.
   *
   */
  void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        

    /* on even iterations, loads pivot items into memory base on the membudget_mb allowed memory size */
    if (gcontext.iteration % 2 == 1){
      printf("entering iteration: %d on after_exec_interval\n", gcontext.iteration);
      printf("pivot_st is %d window_en %d\n", adjcontainer->pivot_st, window_en);
      for (uint i=window_st; i < window_en; i++){
        if (is_user(i)){
          dense_adj user = adjcontainer->adjs[i - window_st];
          if (nnz(user.edges) == 0)
            continue;
          assert(user.ratings.size() == N);
          ivec positions = reverse_sort_index(user.ratings, K);
          assert(positions.size() > 0);
          for (int j=0; j < positions.size(); j++){
            assert(positions[j] >= 0);
            assert(positions[j] < (int)N);
            if (user.ratings[positions[j]] == 0)
              break;
            //assert(user.ratings[positions[j]]> 0);
            int rc = fprintf(out_file, "%u %u %lg\n", i+1, positions[j]+1, user.ratings[positions[j]]);//write item similarity to file
            assert(rc > 0);
            written_pairs++;
          }
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
  metrics m("triangle-counting");    
  /* Basic arguments for application */
  min_allowed_intersection = get_option_int("min_allowed_intersection", min_allowed_intersection);

  debug                    = get_option_int("debug", 0);
  parse_command_line_args();
  std::string similarity   = get_option_string("similarity", "");
  if (similarity == "")
    logstream(LOG_FATAL)<<"Missing similarity input file. Please specify one using the --similarity=filename command line flag" << std::endl;

  mytimer.start();
  int nshards          = convert_matrixmarket_and_item_similarity<EdgeDataType>(training, similarity);
  K = get_option_int("K");

  assert(M > 0 && N > 0);

  //initialize data structure which saves a subset of the items (pivots) in memory
  adjcontainer = new adjlist_container();
  //array for marking which items are conected to the pivot items via users.
  relevant_items = new bool[N];

  /* Run */
  ItemDistanceProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training/*+orderByDegreePreprocessor->getSuffix()*/  ,nshards, true, m); 
  set_engine_flags(engine);
  engine.set_maxwindow(M+N+1);

  char buf[256];
  sprintf(buf, "%s.out", training.c_str());
  out_file = open_file(buf, "w");

  //run the program
  engine.run(program, niters);

  /* Report execution metrics */
  if (quiet)
    metrics_report(m);

  std::cout<<"Total item pairs compaed: " << item_pairs_compared << " total written to file: " << written_pairs << std::endl;

  fclose(out_file);

  std::cout<<"Created output files with the format: " << training << ".out" << std::endl; 

  delete[] relevant_items;
  return 0;
}


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
 * This program takes both a rating file (user to item rasting) and a similarity
 * file (item to item similarities). 
 * The output of this program is K top recommendations for each user based using
 * the current user ratings and the item similarities.  
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
int undirected = 0;
double Q = 3; //the power of the weights added into the total score
bool is_item(vid_t v){ return v >= M; }
bool is_user(vid_t v){ return v < M; }

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef unsigned int VertexDataType;

struct edge_data{
  float up_weight;
  float down_weight;
  edge_data(){ up_weight = 0; down_weight = 0; }
  edge_data(float up_weight, float down_weight) : up_weight(up_weight), down_weight(down_weight) { };
};

typedef edge_data  EdgeDataType;  // Edges store the "rating" of user->movie pair

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
  sparse_vec        ratings;
  mutex mymutex;
  dense_adj() { }
};

bool find_twice(std::vector<vid_t>& edges, vid_t val){
  int ret = 0;
  for (int i=0; i < edges.size(); i++){
      if (edges[i] == val)
        ret++;
  }
  assert(ret >= 0 && ret <= 2);
  return (ret == 2);
}
// This is used for keeping in-memory
class adjlist_container {
  public:
    std::vector<dense_adj> adjs;
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
      //for (uint i=0; i< pivot_en - pivot_st; i++)
      //  adjs[i].ratings = zeros(N);
    }

    /**
     * Grab pivot's adjacency list into memory.
     */
    int load_edges_into_memory(graphchi_vertex<uint32_t, EdgeDataType> &v) {
      assert(is_pivot(v.id()));
      assert(is_user(v.id()));

      int num_edges = v.num_edges();

      dense_adj dadj;
      for(int i=0; i<num_edges; i++) 
        set_new( dadj.edges, v.edge(i)->vertex_id(), v.edge(i)->get_data().up_weight);
      //dadj.ratings = zeros(N);
      adjs[v.id() - pivot_st] = dadj;
      assert(v.id() - pivot_st < adjs.size());
      __sync_add_and_fetch(&grabbed_edges, num_edges /*edges_to_larger_id*/);
      return num_edges;
    }


    /** 
     * add weighted ratings for each linked item
     *
     */
    double compute_ratings(graphchi_vertex<uint32_t, EdgeDataType> &item, vid_t user_pivot, int distance_metric) {
      assert(is_pivot(user_pivot));
      //assert(is_item(pivot) && is_item(v.id()));
      dense_adj &pivot_edges = adjs[user_pivot - pivot_st];

      if (!get_val(pivot_edges.edges, item.id())){
        if (debug)
          logstream(LOG_DEBUG)<<"Skipping item pivot pair since not connected!" << item.id() << std::endl;
        return 0;
      }

      int num_edges = item.num_edges();
      if (debug)
        logstream(LOG_DEBUG)<<"Found " << num_edges << " edges from item : " << item.id() << std::endl;
      //if there are not enough neighboring user nodes to those two items there is no need
      //to actually count the intersection
      if (num_edges < min_allowed_intersection || nnz(pivot_edges.edges) < min_allowed_intersection){
        if (debug)
          logstream(LOG_DEBUG)<<"skipping item pivot pair since < min_allowed_intersection" << std::endl;
        return 0;
      }

      std::vector<vid_t> edges;
      for(int i=0; i < num_edges; i++){
        if (is_item(item.edge(i)->vertex_id()))
        edges.push_back(item.edge(i)->vertex_id());
      }
      std::sort(edges.data(), edges.data()+edges.size());


      for(int i=0; i < num_edges; i++){
        vid_t other_item = item.edge(i)->vertex_id();
        bool up = item.id() < other_item;
        if (debug)
          logstream(LOG_DEBUG)<<"Checking now edge: " << other_item << std::endl;

        if (is_user(other_item)){
          if (debug)
              logstream(LOG_DEBUG)<<"skipping edge to user " << other_item << std::endl;
          continue;
        }

          if (!undirected && ((!up && item.edge(i)->get_data().up_weight == 0) ||
              (up && item.edge(i)->get_data().down_weight == 0))){
            if (debug)
              logstream(LOG_DEBUG)<<"skipping edge with wrong direction to " << other_item << std::endl;
            continue;
          }

          if (get_val(pivot_edges.edges, other_item)){
            if (debug)
              logstream(LOG_DEBUG)<<"skipping edge to " << other_item << " because alrteady connected to pivot" << std::endl;
            continue;
          }

	  assert(get_val(pivot_edges.edges, item.id()) != 0);
          float weight = std::max(item.edge(i)->get_data().down_weight, item.edge(i)->get_data().up_weight);
          assert(weight != 0);

          if (find_twice(edges, other_item)){
          //pivot_edges.ratings[edges[i]-M] += item.edge(i)->get_data() * get_val(pivot_edges.edges, item.id());
          pivot_edges.mymutex.lock();
          set_val(pivot_edges.ratings, other_item-M, get_val(pivot_edges.ratings, other_item-M) + pow(weight,Q) /* * get_val(pivot_edges.edges, item.id())*/);
          pivot_edges.mymutex.unlock();
          if (debug)
            logstream(LOG_DEBUG)<<"Adding weight: " << weight << " to item: " << other_item-M+1 << " for user: " << user_pivot+1<<std::endl;
          }
      }

      if (debug)
        logstream(LOG_DEBUG)<<"Finished user pivot " << user_pivot << std::endl;
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
    }
      /* odd iteration number:
     * 1) For any item connected to a pivot item
     *       compute itersection
     */
    else {
      assert(is_item(v.id()));

      for (int i=0; i< v.num_edges(); i++){
        if (!is_user(v.edge(i)->vertex_id()) || !adjcontainer->is_pivot(v.edge(i)->vertex_id()))
          continue;
        if (debug)
          printf("comparing user pivot %d to item %d\n", v.edge(i)->vertex_id()+1 , v.id() - M + 1);
   
        adjcontainer->compute_ratings(v, v.edge(i)->vertex_id(), distance_metric);
        item_pairs_compared++;
        if (item_pairs_compared % 1000000 == 0)
          logstream(LOG_INFO)<< std::setw(10) << mytimer.current_time() << ")  " << std::setw(10) << item_pairs_compared << " pairs compared " << std::endl;
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
      for (vid_t i=0; i < M; i++){
        gcontext.scheduler->add_task(i); 
      }
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
      //if (debug){
        printf("entering iteration: %d on before_exec_interval\n", gcontext.iteration);
        printf("pivot_st is %d window_St %d, window_en %d\n", adjcontainer->pivot_st, window_st, window_en);
      //}
      //if (adjcontainer->pivot_st <= window_en) {
      if (adjcontainer->pivot_st < M){
         // if (grabbed_edges == 0) {
          logstream(LOG_DEBUG) << "Window init, grabbed: " << grabbed_edges << " edges" << " extending pivor_range to : " << window_en + 1 << std::endl;
          adjcontainer->extend_pivotrange(M/*window_en + 1*/);
          logstream(LOG_DEBUG) << "Window en is: " << window_en << " vertices: " << gcontext.nvertices << std::endl;
          if (window_en == M+N) {
            // every user was a pivot item, so we are done
            logstream(LOG_DEBUG)<<"Setting last iteration to: " << gcontext.iteration + 2 << std::endl;
            gcontext.set_last_iteration(gcontext.iteration + 2);                    
          }
       // } else {
       //   logstream(LOG_DEBUG) << "Too many edges, already grabbed: " << grabbed_edges << std::endl;
       // }
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
     // if (debug){
        printf("entering iteration: %d on after_exec_interval\n", gcontext.iteration);
        printf("pivot_st is %d window_st %d, window_en %d\n", adjcontainer->pivot_st, window_st, window_en);
     // }

      for (uint i=window_st; i < window_en; i++){
        if (is_user(i)){
          if (debug)
            logstream(LOG_DEBUG)<<"Going over user" << i << std::endl;
          dense_adj &user = adjcontainer->adjs[i - window_st];
          if (nnz(user.edges) == 0 || nnz(user.ratings) == 0){
            if (debug)
              logstream(LOG_DEBUG)<<"User with no edges" << std::endl;
            continue;
          }
          //assert(user.ratings.size() == N);
          ivec positions = reverse_sort_index(user.ratings, K);
          assert(positions.size() > 0);
          for (int j=0; j < positions.size(); j++){
            assert(positions[j] >= 0);
            assert(positions[j] < (int)N);

	    //skip zero entries
            if (get_val(user.ratings, positions[j])== 0){
              if (debug)
                logstream(LOG_DEBUG)<<"Found zero in position " << j << std::endl;
              break;
            }
            int rc = fprintf(out_file, "%u %u %lg\n", i+1, positions[j]+1, get_val(user.ratings, positions[j]));//write item similarity to file
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
  metrics m("itemsim2rating");    
  /* Basic arguments for application */
  min_allowed_intersection = get_option_int("min_allowed_intersection", min_allowed_intersection);

  debug                    = get_option_int("debug", 0);
  parse_command_line_args();
  std::string similarity   = get_option_string("similarity", "");
  if (similarity == "")
    logstream(LOG_FATAL)<<"Missing similarity input file. Please specify one using the --similarity=filename command line flag" << std::endl;
  undirected               = get_option_int("undirected", 0);
  Q                        = get_option_float("Q", Q);
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
  graphchi_engine<VertexDataType, EdgeDataType> engine(training,nshards, true, m); 
  set_engine_flags(engine);
  //engine.set_maxwindow(M+N+1);

  out_file = open_file((training + "-rec").c_str(), "w");

  //run the program
  engine.run(program, niters);

  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);

  std::cout<<"Total item pairs compared: " << item_pairs_compared << " total written to file: " << written_pairs << std::endl;
  std::cout<<"Created output files with the format: " << training << "-rec" << std::endl; 

  delete[] relevant_items;
  fclose(out_file);
  return 0;
}

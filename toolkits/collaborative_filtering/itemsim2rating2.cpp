
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
//#define GRAPHCHI_DISABLE_COMPRESSION
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


int min_allowed_intersection = 1;
vec written_pairs;
size_t item_pairs_compared = 0;
std::vector<FILE*> out_files;
timer mytimer;
int grabbed_edges = 0;
int debug;
int undirected = 1;
double prob_sim_normalization_constant = 0;
bool is_item(vid_t v){ return v >= M; }
bool is_user(vid_t v){ return v < M; }
vec degrees;
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


struct vertex_data{ 
  //vec pvec; 
  vertex_data(){}

  void set_val(int index, float val){
    //pvec[index] = val;
  }
  float get_val(int index){
    //return pvec[index];
    return 0;
  }
};
std::vector<vertex_data> latent_factors_inmem;
#include "io.hpp"

struct dense_adj {
  sparse_vec edges;
  sparse_vec ratings;
  mutex mymutex;
  vid_t vid;
  dense_adj() { 
     vid = -1;
  }
};

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
      pivot_st = start_user;
      pivot_en = std::min(end_user, (int)en); 
      adjs.resize(pivot_en - pivot_st);
      //for (uint i=0; i< pivot_en - pivot_st; i++)
      //  adjs[i].ratings = zeros(N);
    }

    /**
     * Grab pivot's adjacency list into memory.
     */
    int load_edges_into_memory(graphchi_vertex<uint32_t, edge_data> &v) {
      assert(is_pivot(v.id()));
      assert(is_user(v.id()));

      int num_edges = v.num_edges();

      dense_adj dadj;
      for(int i=0; i<num_edges; i++) 
        set_new( dadj.edges, v.edge(i)->vertex_id(), v.edge(i)->get_data().up_weight);
      //dadj.ratings = zeros(N);
      dadj.vid = v.id();
      adjs[v.id() - pivot_st] = dadj;
      assert(v.id() - pivot_st < adjs.size());
      __sync_add_and_fetch(&grabbed_edges, num_edges /*edges_to_larger_id*/);
      return num_edges;
    }


    /** 
     * add weighted ratings for each linked item
     *
     */
    double compute_ratings(graphchi_vertex<uint32_t, edge_data> &item, vid_t user_pivot, float user_item_edge_weight) {
      assert(is_pivot(user_pivot));

      if (!allow_zeros)
        assert(user_item_edge_weight != 0);
      else {
         if (user_item_edge_weight == 0)
           return 0;
      }
      dense_adj &pivot_edges = adjs[user_pivot - pivot_st];

      if (!get_val(pivot_edges.edges, item.id())){
        if (debug)
          std::cout<<"Skipping item pivot pair since not connected!" << item.id() << std::endl;
        return 0;
      }

      int num_edges = item.num_edges();
      if (debug)
        std::cout<<"Found " << num_edges << " edges from item : " << item.id()-M+1 << std::endl;
      
      //if there are not enough neighboring user nodes to those two items there is no need
      //to actually count the intersection
      if (num_edges < min_allowed_intersection || nnz(pivot_edges.edges) < min_allowed_intersection){
        if (debug)
          std::cout<<"skipping item pivot pair since < min_allowed_intersection" << std::endl;
        return 0;
      }


      for(int i=0; i < num_edges; i++){
        vid_t other_item = item.edge(i)->vertex_id();
        //user node, continue
        if (other_item < M ){
          if (debug)
            std::cout<<"skipping an edge to item " << other_item << std::endl;
          continue;
        }

        //other item node
        assert(is_item(other_item));
        assert(other_item != item.id());
        bool up = item.id() < other_item;
        if (debug)
          std::cout<<"Checking now edge number " << i << "  " << item.id()-M+1 << " -> " << other_item-M+1 << " weight: " << 
item.edge(i)->get_data().up_weight + item.edge(i)->get_data().down_weight << std::endl;

        if ((up && item.edge(i)->get_data().up_weight == 0) ||
              (!up && item.edge(i)->get_data().down_weight == 0)){
            if (debug)
              std::cout<<"skipping edge with wrong direction to " << other_item-M+1 << std::endl;
            continue;
        }

        if (get_val(pivot_edges.edges, other_item)){
            if (debug)
              std::cout<<"skipping edge to " << other_item << " because alrteady connected to pivot" << std::endl;
            continue;
        }

       	assert(get_val(pivot_edges.edges, item.id()) != 0);
        double weight = item.edge(i)->get_data().up_weight+ item.edge(i)->get_data().down_weight;
        if (weight == 0)
           logstream(LOG_FATAL)<<"Bug: found zero edge weight between: " << item.id()-M+input_file_offset << " -> " << other_item-M+input_file_offset <<std::endl;

        if (weight <= 1){
           if (debug)
              std::cout<<"skipping edge to " << item.id()-M+1 << " -> " << other_item-M+1 << " because of similarity is smaller or equal to one: " << weight << std::endl;
           continue;
        }

        pivot_edges.mymutex.lock();
        //add weight according to equation (15) in the probabalistic item similarity paper
        set_val(pivot_edges.ratings, other_item-M, get_val(pivot_edges.ratings, other_item-M) + ((user_item_edge_weight-0.5)/0.5)* (weight- 1));
        if (debug){
           std::cout<<"Adding weight: " << (((user_item_edge_weight-0.5)/0.5)* (weight- 1)) << " to item: " << other_item-M+1 << " for user: " << user_pivot+1<< " weight-1: " << weight-1<<std::endl;
           std::cout<<pivot_edges.ratings<<std::endl;
         }
         pivot_edges.mymutex.unlock();

     }

      if (debug)
        std::cout<<"Finished user pivot " << user_pivot << std::endl;
      return 0;
    }

    inline bool is_pivot(vid_t vid) {
      return vid >= pivot_st && vid < pivot_en;
    }
};


adjlist_container * adjcontainer;

struct ItemDistanceProgram : public GraphChiProgram<VertexDataType, edge_data> {

  /**
   *  Vertex update function.
   */
  void update(graphchi_vertex<VertexDataType, edge_data> &v, graphchi_context &gcontext) {
    if (debug)
      printf("Entered iteration %d with %d\n", gcontext.iteration, is_item(v.id()) ? (v.id() - M + 1): v.id()+1);
       

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
        if (!adjcontainer->is_pivot(v.edge(i)->vertex_id()))
          continue;
        if (debug)
          printf("comparing user pivot %d to item %d\n", v.edge(i)->vertex_id()+input_file_offset , v.id() - M + 1);
   
        adjcontainer->compute_ratings(v, v.edge(i)->vertex_id(), v.edge(i)->get_data().up_weight+ v.edge(i)->get_data().down_weight);
        item_pairs_compared++;

        if (item_pairs_compared % 1000000 == 0)
          logstream(LOG_INFO)<< std::setw(10) << mytimer.current_time() << ")  " << std::setw(10) << item_pairs_compared << " pairs compared " << std::setw(10) << sum(written_pairs) << std::endl;
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
      for (vid_t i=0; i < M; i++){
        //even iterations, schedule only user nodes
        gcontext.scheduler->add_task(i); 
      }
   } else { //iteration % 2 == 1, schedule only item nodes
      for (vid_t i=M; i < M+N; i++){
        gcontext.scheduler->add_task(i); 
      }
    } 
  }


  void after_iteration(int iteration, graphchi_context &gcontext){
    if (gcontext.iteration % 2 == 1){
#ifndef __APPLE__
#pragma omp parallel for
#endif
     for (int i=0; i< (int)adjcontainer->adjs.size(); i++){
          if (debug)
            logstream(LOG_DEBUG)<<"Going over user" << adjcontainer->adjs[i].vid << std::endl;
          dense_adj &user = adjcontainer->adjs[i];
          if (nnz(user.ratings) == 0){
            if (debug)
              logstream(LOG_DEBUG)<<"User with no edges" << std::endl;
            continue;
          }
          //assert(user.ratings.size() == N);
         assert(adjcontainer->adjs[i].vid >= (uint)start_user && adjcontainer->adjs[i].vid < (uint)end_user);
         uint user_id = adjcontainer->adjs[i].vid;
         assert(user_id < M);
         int degree_k = degrees[user_id];
         assert(degree_k > 0);
             
         /* GO over the compu sum*/
         FOR_ITERATOR(j, user.ratings){
           uint item_id = j.index(); 
           assert(item_id < N);
           int degree_x = degrees[M+item_id];
           if (degree_x <= 0)
             logstream(LOG_WARNING)<<"Item degree is 0: " << user_id << " -> " << item_id << std::endl;
           assert(degree_x > 0);
           double p_k_1 = 1.0 / ( 1.0 + prob_sim_normalization_constant * ((N - degree_k)/(double)degree_k) * ((M - degree_x) / (double)degree_x));
           assert(p_k_1 > 0 && p_k_1 <= 1.0);
           set_val( user.ratings, item_id, p_k_1 * (1.0 + get_val(user.ratings, item_id)));
           //assert(get_val(user.ratings, item_id) > 0);
         }
          
          ivec positions = reverse_sort_index(user.ratings, std::min(nnz(user.ratings),(int)K));
          if (debug)
            std::cout<<positions<<std::endl;
          assert(positions.size() > 0);
          positions.conservativeResize(std::min(nnz(user.ratings),(int)K));
          for (int j=0; j < positions.size(); j++){
            
            if (positions[j] >= (int)N){
               std::cout<<"bug: user rating " << user.ratings << " pos: " << positions << std::endl;
               continue;
            }
            assert(positions[j] >= (int)0);
            assert(positions[j] < (int)N);

	    //skip zero entries
            if (get_val(user.ratings, positions[j])== 0){
              if (debug)
                logstream(LOG_DEBUG)<<"Found zero in position " << j << std::endl;
              break;
            }
            int rc = fprintf(out_files[omp_get_thread_num()], "%u %u %lg\n", user.vid+1, positions[j]+1, get_val(user.ratings, positions[j]));//write item similarity to file
            if (debug)
              logstream(LOG_DEBUG)<<"Writing rating from user" << user.vid+1 << " to item: " << positions[j]-M+1 << std::endl;
            assert(rc > 0);
            written_pairs[omp_get_thread_num()]++;
          }
        }
      grabbed_edges = 0;
      adjcontainer->clear();
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
      if ((int)adjcontainer->pivot_st < std::min(std::min((int)M,end_user), (int)window_en)){
        size_t max_grab_edges = get_option_long("membudget_mb", 1024) * 1024 * 1024 / 8;
        if (grabbed_edges < max_grab_edges * 0.8) {
          logstream(LOG_DEBUG) << "Window init, grabbed: " << grabbed_edges << " edges" << " extending pivor_range to : " << window_en + 1 << std::endl;
          adjcontainer->extend_pivotrange(std::min(std::min((int)M, end_user), (int)window_en + 1));
          logstream(LOG_DEBUG) << "Window en is: " << window_en << " vertices: " << gcontext.nvertices << std::endl;
          if (window_en+1 >= gcontext.nvertices) {
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

};




int main(int argc, const char ** argv) {
  print_copyright();

  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("itemsim2rating2");    

  /* Basic arguments for application */
  min_allowed_intersection = get_option_int("min_allowed_intersection", min_allowed_intersection);
  debug                    = get_option_int("debug", 0);
  parse_command_line_args();
  std::string similarity   = get_option_string("similarity", "");
  if (similarity == "")
    logstream(LOG_FATAL)<<"Missing similarity input file. Please specify one using the --similarity=filename command line flag" << std::endl;
  undirected               = get_option_int("undirected", 0);
  
  mytimer.start();

  int nshards          = convert_matrixmarket_and_item_similarity<edge_data>(training, similarity, 3, degrees);
  if (debug) for (int i=0; i< degrees.size(); i++)
    std::cout<<"degree of node: " << i << " is: " << degrees[i]<<std::endl;

  assert(M > 0 && N > 0);
  prob_sim_normalization_constant = (double)L / (double)(M*N-L);
  
  //initialize data structure which saves a subset of the items (pivots) in memory
  adjcontainer = new adjlist_container();


  /* Run */
  ItemDistanceProgram program;
  graphchi_engine<VertexDataType, edge_data> engine(training, nshards, true, m); 
  set_engine_flags(engine);

  //open output files as the number of operating threads
  out_files.resize(number_of_omp_threads());
  for (uint i=0; i< out_files.size(); i++){
    char buf[256];
    sprintf(buf, "%s-rec.out%d", training.c_str(), i);
    out_files[i] = open_file(buf, "w");
  }


  K 			   = get_option_int("K");
  assert(K > 0);
  //run the program
  engine.run(program, niters);

  for (uint i=0; i< out_files.size(); i++)
    fclose(out_files[i]);
  


  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);

  std::cout<<"Total item pairs compared: " << item_pairs_compared << " total written to file: " << sum(written_pairs) << std::endl;

  logstream(LOG_INFO)<<"Going to sort and merge output files " << std::endl;
  std::string dname= dirname(strdup(argv[0]));
  system(("bash " + dname + "/topk.sh " + std::string(basename(strdup((training+"-rec").c_str())))).c_str()); 


  return 0;
}


/**
 * @file
 * @author  Danny Bickson, based on code by Aapo Kyrola <akyrola@cs.cmu.edu>
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright [2012] [Aapo Kyrola, Guy Blelloch, Carlos Guestrin / Carnegie Mellon University]
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
 * Triangle counting application. Counts the number of incident (full) triangles
 * for each vertex. Edge direction is ignored.
 *
 * This algorithm is quite complicated and requires 'trickery' to work
 * well on GraphChi. The complexity stems from the need to store large number
 * of adjacency lists in memory: we cannot store the adjacency lists reasonable
 * to edges, nor can we store all of them once at memory. Therefore the problems
 * is solved in a series of phases. On each phase, the relevant adjacency lists of an interval
 * of vertices (called 'pivots') is loaded into memory, and all vertices that have id smaller than the
 * pivots are matched with them. With 'relevant adjacency list' I mean the list of neighbors
 * that have higher id then the pivots themselves. That is, we only count triangles a -> b -> c
 * where a > b > c. 
 *
 * The application involves a special preprocessing step which orders the vertices in ascending
 * order of their degree. This turns out to be a very important optimization on big graphs. 
 *
 * This algorithm also utilizes the dynamic graph engine, and deletes edges after they have been
 * accounted for. 
 */



#include <string>
#include <vector>

/**
 * Need to define prior to including GraphChi
 * headers. This enabled edge-deletion in the vertex object.
 */
#define SUPPORT_DELETIONS 1


#include "graphchi_basic_includes.hpp"
#include "engine/dynamic_graphs/graphchi_dynamicgraph_engine.hpp"
#include "../../example_apps/matrix_factorization/matrixmarket/mmio.h"
#include "../../example_apps/matrix_factorization/matrixmarket/mmio.c"
#include "api/chifilenames.hpp"
#include "api/vertex_aggregator.hpp"
#include "preprocessing/sharder.hpp"
#include "eigen_wrapper.hpp"


using namespace graphchi;
double minval = -1e100;
double maxval = 1e100;
std::string training;
std::string validation;
std::string test;
uint M, N, K;
size_t L;
uint Me, Ne, Le;
double globalMean = 0;
int min_allowed_intersection = 5;

vid_t max_left_vertex =0 ;
vid_t max_right_vertex = 0;
bool * relevant_items  = NULL;
#define NLATENT 20
struct vertex_data {
  double pvec[NLATENT];
  double rmse;

  vertex_data() {
    for(int k=0; k < NLATENT; k++) pvec[k] =  drand48(); 
    rmse = 0;
  }

  double dot(const vertex_data &oth) const {
    double x=0;
    for(int i=0; i<NLATENT; i++) x+= oth.pvec[i]*pvec[i];
    return x;
  }

};

bool is_item(vid_t v){ return v >= M; }
bool is_user(vid_t v){ return v < M; }

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef unsigned int VertexDataType;
typedef unsigned int  EdgeDataType;  // Edges store the "rating" of user->movie pair

std::vector<vertex_data> latent_factors_inmem;
#include "io.hpp"


/**
 * Code for intersection size computation and 
 * pivot management.
 */
int grabbed_edges = 0;


// Linear search
inline bool find_in_list_linear(vid_t * datachunk, size_t n, vid_t target) {
  for(int i=0; i<(int)n; i++) {
    if (datachunk[i] == target) return true;
    else if (datachunk[i] > target) return false;
  }
  return false;
}

// Binary search
inline bool find_in_list(vid_t * datachunk, size_t n, vid_t target) {
  assert(is_user(target));
  if (n<32) return find_in_list_linear(datachunk, n, target);
  register size_t lo = 0;
  register size_t hi = n;
  register size_t m = lo + (hi-lo)/2;
  while(hi>lo) {
    vid_t eto = datachunk[m];
    if (target == eto) {
      return true;
    }
    if (target > eto) {
      lo = m+1;
    } else {
      hi = m;
    }
    m = lo + (hi-lo)/2;
  }
  return false;
}



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
    pivot_st = 0;
    pivot_en = 0;
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
    //assert(en>pivot_en);
    pivot_en = en; 
    adjs.resize(pivot_en - pivot_st);
  }

  /**
   * Grab pivot's adjacency list into memory.
   */
  int load_edges_into_memory(graphchi_vertex<uint32_t, uint32_t> &v) {
    assert(is_pivot(v.id()));

    int num_edges = v.num_edges();
    // Count how many neighbors have larger id than v
    v.sort_edges_indirect();
    int edges_to_larger_id = 0;
    for(int i=0; i<num_edges; i++) {
      if (v.edge(i)->vertexid > v.id())  
        edges_to_larger_id++;  // Need to store only ids larger than me
    }

    // Allocate the in-memory adjacency list, using the
    // knowledge of the number of edges.
    dense_adj dadj = dense_adj(edges_to_larger_id, (vid_t*) calloc(sizeof(vid_t), edges_to_larger_id));
    edges_to_larger_id = 0;
    for(int i=0; i<num_edges; i++) {
      if (v.edge(i)->vertexid > v.id())
        dadj.adjlist[edges_to_larger_id++] = v.edge(i)->vertex_id();
    }
    assert(dadj.count == edges_to_larger_id);
    adjs[v.id() - pivot_st] = dadj;
    assert(v.id() - pivot_st < adjs.size());
    __sync_add_and_fetch(&grabbed_edges, edges_to_larger_id);
    return edges_to_larger_id;
  }

  int acount(vid_t pivot) {
    return adjs[pivot - pivot_st].count;
  }


  /** 
   * for every two relevant items, go over all the users which are connected to those items
   * and count how many such users exist.
   */
  int intersection_size(graphchi_vertex<uint32_t, uint32_t> &v, vid_t pivot) {
    assert(is_pivot(pivot));
    assert(is_item(pivot) && is_item(v.id()));

    int count = 0;        
    if (pivot < v.id()) {  
      dense_adj &pivot_edges = adjs[pivot - pivot_st];
      int num_edges = v.num_edges();
      for(int i=0; i < num_edges; i++) {
        vid_t other_vertex = v.edge(i)->vertexid;
        assert(is_user(other_vertex));
        //if (other_vertex > pivot) {
        int match = find_in_list(pivot_edges.adjlist, pivot_edges.count, other_vertex);
        count += match;
        /*if (match > min_allowed_intersection) {
        // Add one to edge between v and the match 
        v.edge(i)->set_data(v.edge(i)->get_data() + match); 
        }*/
        //}
      }
    }        
    assert(count >= 1);
    return count;
  }

  inline bool is_pivot(vid_t vid) {
    return vid >= pivot_st && vid < pivot_en;
  }
};

adjlist_container * adjcontainer;



/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct TriangleCountingProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {


  /**
   *  Vertex update function.
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &v, graphchi_context &gcontext) {
    printf("Entered iteration %d with %d\n", gcontext.iteration, v.id());
    if (gcontext.iteration % 2 == 0) {
      if (adjcontainer->is_pivot(v.id()) && is_item(v.id())){
        adjcontainer->load_edges_into_memory(v);         
        printf("Loading pivot %dintro memory\n", v.id());
      }
      else if (is_user(v.id())){
        
        uint32_t oldcount = v.get_data();
        uint32_t newcounts = 0;

        v.sort_edges_indirect();

        //check if this user is connected to any pivot item
        bool has_pivot = false;
        int pivot = -1;
        for(int i=0; i<v.num_edges(); i++) {
          graphchi_edge<uint32_t> * e = v.edge(i);
          if (adjcontainer->is_pivot(e->vertexid)) {
            has_pivot = true;
            pivot = e->vertexid;
          }
        }
        printf("user %d is linked to pivot %d\n", v.id(), pivot);

        if (!has_pivot)
          return; 


        //has pivot
        for(int i=0; i<v.num_edges(); i++) {
          graphchi_edge<uint32_t> * e = v.edge(i);
          //if (!adjcontainer->is_pivot(e->vertexid)) {
           assert(is_item(e->vertexid)); 
           relevant_items[e->vertexid - M] = true;
            //uint32_t pivot_triangle_count = adjcontainer->intersection_size(v, e->vertexid, i);
            /*newcounts += pivot_triangle_count;
              if (pivot_triangle_count == 0 && e->get_data() == 0) {
              v.remove_edge(i); 
              } else {
              e->set_data(e->get_data() + pivot_triangle_count);
              }
              } else {
              break;
              } */
          //e->set_data(pivot);
        }

        if (newcounts > 0) {
          v.set_data(oldcount + newcounts);
        }
      //}
    }//is_user 

  } //iteration % 2 =  1
  else {
    if (!is_item(v.id()) || (is_item(v.id()) && !relevant_items[v.id() - M])){
      printf("node %d is not relevan\n", v.id());
      return;
    }

    for (vid_t i=adjcontainer->pivot_st; i< adjcontainer->pivot_en; i++){
      //dense_adj &dadj = adjcontainer->adjs[i - adjcontainer->pivot_st];
      if (!is_item(i))
        continue;
      uint32_t pivot_triangle_count = adjcontainer->intersection_size(v, i);
      printf("comparing to pivot %d\n", i);
      if (pivot_triangle_count > (uint)min_allowed_intersection){
        //TODO save to file
      }
    }
  }//end of iteration % 2 == 1
 }//end of update function
  /**
   * Called before an iteration starts.
   */
  void before_iteration(int iteration, graphchi_context &gcontext) {
    gcontext.scheduler->remove_tasks(0, (int) gcontext.nvertices - 1);
    /*if (gcontext.iteration % 2 == 0) {
    // Schedule vertices that were pivots on last iteration, so they can
    // keep count of the triangles counted by their lower id neighbros.
    for(vid_t i=adjcontainer->pivot_st; i < adjcontainer->pivot_en; i++) {
    gcontext.scheduler->add_task(i); 
    }
    grabbed_edges = 0;
    adjcontainer->clear();
    } else {
    // Schedule everything that has id < pivot
    logstream(LOG_INFO) << "Now pivots: " << adjcontainer->pivot_st << " " << adjcontainer->pivot_en << std::endl;
    for(vid_t i=0; i < gcontext.nvertices; i++) {
    if (i < adjcontainer->pivot_en) { 
    gcontext.scheduler->add_task(i); 
    }
    }
    }*/
    for (vid_t i=0; i < M+N; i++){
        //assert(is_item(i));
        gcontext.scheduler->add_task(i); 
    }
     if (gcontext.iteration % 2 == 0){
      memset(relevant_items, 0, sizeof(bool)*N);
      // Schedule vertices that were pivots on last iteration, so they can
      // keep count of the triangles counted by their lower id neighbros.
      //for(vid_t i=adjcontainer->pivot_st; i < adjcontainer->pivot_en; i++) {
     //for (vid_t i = 0; i< M; i++)
      //  gcontext.scheduler->add_task(i);
      printf("setting relevant_items to zero\n");
      grabbed_edges = 0;
      adjcontainer->clear();
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
   * On every even iteration, we store pivot's adjacency lists to memory. 
   * Here we manage the memory to ensure that we do not load too much
   * edges into memory.
   */
  void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
    if (gcontext.iteration % 2 == 0) {
      if (adjcontainer->pivot_st <= window_en) {
        size_t max_grab_edges = get_option_long("membudget_mb", 1024) * 1024 * 1024 / 8;
        if (grabbed_edges < max_grab_edges * 0.8) {
          logstream(LOG_DEBUG) << "Window init, grabbed: " << grabbed_edges << " edges" << std::endl;
          for(vid_t vid=window_st; vid <= window_en; vid++) {
            gcontext.scheduler->add_task(vid);
          }
          adjcontainer->extend_pivotrange(window_en + 1);
          if (window_en == gcontext.nvertices) {
            // Last iteration needed for collecting last triangle counts
            gcontext.set_last_iteration(gcontext.iteration + 3);                    
          }
        } else {
          std::cout << "Too many edges, already grabbed: " << grabbed_edges << std::endl;
        }
      }
    }

  }

  /**
   * Called after an execution interval has finished.
   */
  void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
  }

};




int main(int argc, const char ** argv) {
  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("triangle-counting");    
  /* Basic arguments for application */
  training = get_option_string("training");  // Base filename
  int niters           = get_option_int("max_iter", 100000); // Automatically determined during running
  bool scheduler       = true;

  /* Preprocess the file, and order the vertices in the order of their degree.
     Mapping from original ids to new ids is saved separately. */
  OrderByDegree<EdgeDataType> * orderByDegreePreprocessor = new OrderByDegree<EdgeDataType> ();
  int nshards          = convert_matrixmarket<EdgeDataType>(training, orderByDegreePreprocessor);

  assert(M > 0 && N > 0);
  max_left_vertex = M-1;
  max_right_vertex = M+N-1;

  /* Initialize adjacency container */
  adjcontainer = new adjlist_container();

  relevant_items = new bool[N];
  // TODO: ordering by degree.

  /* Run */
  TriangleCountingProgram program;
  graphchi_dynamicgraph_engine<VertexDataType, EdgeDataType> engine(training + orderByDegreePreprocessor->getSuffix(),
      nshards, scheduler, m); 
  engine.set_enable_deterministic_parallelism(false);

  // Low memory budget is required to prevent swapping as triangle counting
  // uses more memory than standard GraphChi apps.
  engine.set_membudget_mb(std::min(get_option_int("membudget_mb", 1024), 1024)); 
  engine.run(program, niters);

  /* Report execution metrics */
  metrics_report(m);

  /* Count triangles */
  size_t ntriangles = sum_vertices<vid_t, size_t>(training + "_degord", 0, (vid_t)engine.num_vertices());
  std::cout << "Number of triangles: " << ntriangles / 3 << "(" << ntriangles << ")" << std::endl;
  return 0;
}


/**
 * @file
 * @author  Aapo Kyrola <akyrola@cs.cmu.edu>
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
#define GRAPHCHI_DISABLE_COMPRESSION


#include "graphchi_basic_includes.hpp"
#include "engine/dynamic_graphs/graphchi_dynamicgraph_engine.hpp"
#include "engine/auxdata/degree_data.hpp"
#include "preprocessing/util/orderbydegree.hpp"

using namespace graphchi;

/**
  * Type definitions. Vertex data stores the number of incident triangles.
  * Edge stores number of unaccounted  triangles that the edge participates on.
  * When vertex is updated, it updates its vertex count by summing up the
  * counts from edges (after which the edges are deleted).
  */
typedef uint32_t VertexDataType;
typedef uint32_t EdgeDataType;


/*
 * Class for writing the output number of triangles for each node
 */
class OutputVertexCallback : public VCallback<VertexDataType> {
  public:
    virtual void callback(vid_t vertex_id, VertexDataType &value) {
       if (value > 0)
        std::cout << vertex_id << " " << value << std::endl;
    }
};

/**
  * Code for intersection size computation and 
  * pivot management.
  */
int grabbed_edges = 0;


// Linear search
inline bool findadj_linear(vid_t * datachunk, size_t n, vid_t target) {
    for(int i=0; i<(int)n; i++) {
        if (datachunk[i] == target) return true;
        else if (datachunk[i] > target) return false;
    }
    return false;
}

// Binary search
inline bool findadj(vid_t * datachunk, size_t n, vid_t target) {
    if (n<32) return findadj_linear(datachunk, n, target);
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
    mutex m;
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
        assert(en>=pivot_en);
        pivot_en = en; 
        adjs.resize(pivot_en - pivot_st);
    }
    
    /**
      * Grab pivot's adjacency list into memory.
      */
    int grab_adj(graphchi_vertex<uint32_t, uint32_t> &v) {
        if(is_pivot(v.id())) {            
            int ncount = v.num_edges();
            // Count how many neighbors have larger id than v
            v.sort_edges_indirect();
     
            
            int actcount = 0;
            vid_t lastvid = 0;
            for(int i=0; i<ncount; i++) {
                if (v.edge(i)->vertexid > v.id() && v.edge(i)->vertexid != lastvid)  
                    actcount++;  // Need to store only ids larger than me
                lastvid = v.edge(i)->vertex_id();
            }
            
            // Allocate the in-memory adjacency list, using the
            // knowledge of the number of edges.
            dense_adj dadj = dense_adj(actcount, (vid_t*) calloc(sizeof(vid_t), actcount));
            actcount = 0;
            lastvid = 0;
            for(int i=0; i<ncount; i++) {
                if (v.edge(i)->vertexid > v.id() && v.edge(i)->vertexid != lastvid) {  // Need to store only ids larger than me
                    dadj.adjlist[actcount++] = v.edge(i)->vertex_id();
                }
                lastvid = v.edge(i)->vertex_id();
            }
            assert(dadj.count == actcount);
            adjs[v.id() - pivot_st] = dadj;
            assert(v.id() - pivot_st < adjs.size());
            __sync_add_and_fetch(&grabbed_edges, actcount);
            return actcount;
        }
        return 0;
    }
    
    int acount(vid_t pivot) {
        return adjs[pivot - pivot_st].count;
    }
    
    
    /** 
      * Compute size of the relevant intersection of v and a pivot
      */
    int intersection_size(graphchi_vertex<uint32_t, uint32_t> &v, vid_t pivot, int start_i) {
        assert(is_pivot(pivot));
        int count = 0;
        if (pivot > v.id()) {
            dense_adj &dadj = adjs[pivot - pivot_st];
            int vc = v.num_edges();
             
            /**
              * If the adjacency list sizes are not too different, use
              * 'merge'-type of operation to compute size intersection.
              */
            if (dadj.count < 32 * (vc - start_i)) { // TODO: do real profiling to find best cutoff value
                // Do merge-style of check
                assert(v.edge(start_i)->vertex_id() == pivot);
                int i1 = 0;
                int i2 = start_i+1;
                int nedges = v.num_edges(); 
                
                while (i1 < dadj.count && i2 < nedges) {
                    vid_t dst = v.edge(i2)->vertexid;
                    vid_t a = dadj.adjlist[i1];
                    if (a == dst) {
                        /* Add one to edge between v and the match */
                        v.edge(i2)->set_data(v.edge(i2)->get_data() + 1);
                        count++;
                        i1++; i2++;
                        
                    } else {
                        i1 += a < dst;
                        i2 += a > dst;
                    }  
                }
            } else {
                /**
                  * Otherwise, use linear/binary search.
                  */
                vid_t lastvid = 0;
                for(int i=start_i+1; i < vc; i++) {
                    vid_t nb = v.edge(i)->vertexid;
                    if (nb > pivot && nb != lastvid) {
                        int match = findadj(dadj.adjlist, dadj.count, nb);
                        count += match;
                        if (match > 0) {
                            /* Add one to edge between v and the match */
                            v.edge(i)->set_data(v.edge(i)->get_data() + 1);
                        }
                    }
                    lastvid = nb;
                }
            }
        }        
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
        
        if (gcontext.iteration % 2 == 0) {
            adjcontainer->grab_adj(v);
        } else {
            uint32_t oldcount = v.get_data();
            uint32_t newcounts = 0;

            v.sort_edges_indirect();
            
            vid_t lastvid = 0;
            
            /**
              * Iterate through the edges, and if an edge is from a 
              * pivot vertex, compute intersection of the relevant
              * adjacency lists.
              */
            for(int i=0; i<v.num_edges(); i++) {
                graphchi_edge<uint32_t> * e = v.edge(i);
                if (e->vertexid > v.id() && e->vertexid >= adjcontainer->pivot_st) {
                    assert(!is_deleted_edge_value(e->get_data()));
                    if (e->vertexid != lastvid) {  // Handles reciprocal edges (a->b, b<-a)
                        if (adjcontainer->is_pivot(e->vertexid)) {
                            uint32_t pivot_triangle_count = adjcontainer->intersection_size(v, e->vertexid, i);
                            newcounts += pivot_triangle_count;
                            
                            /* Write the number of triangles into edge between this vertex and pivot */
                            if (pivot_triangle_count == 0 && e->get_data() == 0) {
                                /* ... or remove the edge, if the count is zero. */
                                v.remove_edge(i); 
                            } else {
                                
                                e->set_data(e->get_data() + pivot_triangle_count);
                            }
                        } else {
                            break;
                        }
                    }
                    lastvid = e->vertexid;
                }  
                assert(newcounts >= 0);
            }
            
            if (newcounts > 0) {
                v.set_data(oldcount + newcounts);
            }
        }
        
        
        /* Collect triangle counts matched by vertices with id lower than
            his one, and delete */
        if (gcontext.iteration % 2 == 0) {
            int newcounts = 0;
          
            for(int i=0; i < v.num_edges(); i++) {
                graphchi_edge<uint32_t> * e = v.edge(i);
                if (e->vertexid < v.id()) {
                    newcounts += e->get_data();
                    e->set_data(0);
                    
                    // This edge can be now deleted. Is there some other situations we can delete?
                    if (v.id() < adjcontainer->pivot_st && e->vertexid < adjcontainer->pivot_st) {
                        v.remove_edge(i);
                    }
                }
            }
            v.set_data(v.get_data() + newcounts);
        }
        
     }
    
    /**
     * Called before an iteration starts.
     */
    void before_iteration(int iteration, graphchi_context &gcontext) {
        if (gcontext.iteration % 2 == 0) {
            // Schedule vertices that were pivots on last iteration, so they can
            // keep count of the triangles counted by their lower id neighbros.
            for(vid_t i=adjcontainer->pivot_st; i < adjcontainer->pivot_en; i++) {
                gcontext.scheduler->add_task(i, true);
            }
            grabbed_edges = 0;
            adjcontainer->clear();
        } else {
            // Schedule everything that has id < pivot
            logstream(LOG_INFO) << "Now pivots: " << adjcontainer->pivot_st << " " << adjcontainer->pivot_en << std::endl;
            for(vid_t i=0; i < gcontext.nvertices; i++) {
                if (i < adjcontainer->pivot_en) { 
                    gcontext.scheduler->add_task(i, true);
                }
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
                        gcontext.scheduler->add_task(vid, true);
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
    std::string filename = get_option_string("file");  // Base filename
    int niters           = 100000; // Automatically determined during running
    bool scheduler       = true;
    
    /* Preprocess the file, and order the vertices in the order of their degree.
       Mapping from original ids to new ids is saved separately. */
    int nshards          = convert_if_notexists_novalues<EdgeDataType>(filename, 
                                                                get_option_string("nshards", "auto"));
    
    if (nshards == 1) {
        logstream(LOG_FATAL) << "Triangle counting does not work in in-memory mode. Please set --nshards=2" << std::endl;
        exit(1);
    }
    assert(nshards > 1);
    
    nshards = order_by_degree<EdgeDataType>(filename, nshards, m);
    
    
    /* Initialize adjacency container */
    adjcontainer = new adjlist_container();
    
    // TODO: ordering by degree.
    
    /* Run */
    TriangleCountingProgram program;
    graphchi_dynamicgraph_engine<VertexDataType, EdgeDataType> engine(filename + "_degord",
                                                                      nshards, scheduler, m); 
    engine.set_enable_deterministic_parallelism(false);
    
    // Low memory budget is required to prevent swapping as triangle counting
    // uses more memory than standard GraphChi apps.
    engine.set_membudget_mb(std::min(get_option_int("membudget_mb", 1024), 1024)); 
    engine.run(program, niters);
    
    /* Report execution metrics */
    metrics_report(m);
    
    /* Count triangles */
    size_t ntriangles = sum_vertices<vid_t, size_t>(filename + "_degord", 0, (vid_t)engine.num_vertices());
    std::cout << "Number of triangles: " << ntriangles / 3 << "(" << ntriangles << ")" << std::endl;

    
    /* If run as a test, see the number matches */
    size_t expected = get_option_long("assertequals", 0);
    if (expected > 0) {
        std::cout << "Testing the result is as expected: " << (ntriangles / 3) << " vs. " << expected << std::endl;
        assert(expected == ntriangles / 3);
    }
    
    /* write the output */
  //  OutputVertexCallback callback;
  //  foreach_vertices<VertexDataType>(filename + "_degord", 0, engine.num_vertices(), callback);

    return 0;
}


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
 * Minimum spanning forest based on Boruvska steps. Also alternatively implementation using
 * star contraction.
 */

#define GRAPHCHI_DISABLE_COMPRESSION

#include <string>

#include "graphchi_basic_includes.hpp"

using namespace graphchi;


enum ContractionType {
    STAR = 1, BORUVSKA = 2
};

ContractionType contractionType;

#define MAX_VIDT 0xffffffff

struct bidirectional_component_weight {
    vid_t smaller_component;
    vid_t larger_component;
    vid_t orig_src, orig_dst;
    bool in_mst;
    double weight;
    
    bidirectional_component_weight() {
        smaller_component = larger_component = MAX_VIDT;
        in_mst = false;
        weight = 0.0f;
        orig_src = orig_dst = 0;
    }
    
    bidirectional_component_weight(double x) {
        smaller_component = larger_component = MAX_VIDT;
        in_mst = false;
        weight = x;
        orig_src = orig_dst = 0;
    }
    
    
    vid_t neighbor_label(vid_t myid, vid_t nbid) {
        vid_t label = (myid < nbid ? larger_component : smaller_component);
        if (label == MAX_VIDT) label = nbid;  // NOTE: important optimization (for random orders!)
        return label;
    }
    
    vid_t & my_label(vid_t myid, vid_t nbid) {
        if (myid < nbid) {
            return smaller_component;
        } else {
            return larger_component;
        }
    }
    
    bool labels_agree() {
        return smaller_component == larger_component;
    }
    
};

class AcceptMinimum : public DuplicateEdgeFilter<bidirectional_component_weight> {
    bool acceptFirst(bidirectional_component_weight &first, bidirectional_component_weight &second) {
        return (first.weight < second.weight);
    }
};

 

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program.
 */
typedef vid_t VertexDataType;
typedef bidirectional_component_weight EdgeDataType;

graphchi_engine<VertexDataType, EdgeDataType> * gengine;


size_t MST_OUTPUT;
size_t CONTRACTED_GRAPH_OUTPUT;
FILE * complog;


/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type>
 * class. The main logic is usually in the update function.
 */
struct BoruvskaStarContractionStep : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    // Hash parameters, always chosen randomly
    uint64_t a,  b;
    size_t num_edges;
    int num_contract, num_tails, num_heads, num_active_vertices;
    
    BoruvskaStarContractionStep() {
        a = (uint64_t) std::rand();
        b = (uint64_t) std::rand();
        num_edges = 0;
        num_contract = num_tails = num_heads = num_active_vertices = 0;
        logstream(LOG_INFO) << "Chose random hash function: a = " << a << " b = " << b << std::endl;
        
    }
    
    bool heads(vid_t vertex_id) {
        const long prime = 7907; // Courtesy of Wolfram alpha
        return ((a * vertex_id + b) % prime) % 2 == 0;
    }
    
    
    /**
     *  Vertex update function. Note: we assume fresh edge values.
     */
    void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
        
        if (vertex.num_edges() == 0) {
            return;
        }
        
        
        if (gcontext.iteration == 0) {
            num_active_vertices ++;
            num_edges += vertex.num_inedges();
            double min_edge_weight = 1e30f;
            int min_edge_idx = 0;
        
            // TODO: replace with reductions
            /* Get minimum edge */
            for(int i=0; i < vertex.num_edges(); i++) {
                bidirectional_component_weight edata = vertex.edge(i)->get_data();
                
                // Remember the original edge identity
                if (edata.orig_src == edata.orig_dst) {
                    edata.orig_src = vertex.id();
                    edata.orig_dst = vertex.edge(i)->vertex_id(); // Note: forgets direction
                    vertex.edge(i)->set_data(edata);
                }                
                double w = edata.weight;
                if (w < min_edge_weight || (w == min_edge_weight && edata.in_mst)) {  // Tie-breaking
                    min_edge_idx = i;
                    min_edge_weight = w;
                }
            }
            if (!vertex.edge(min_edge_idx)->get_data().in_mst) {
                bidirectional_component_weight edata = vertex.edge(min_edge_idx)->get_data();
                edata.in_mst = true;
                vertex.edge(min_edge_idx)->set_data(edata);
            }
        }
        if (contractionType == STAR && gcontext.iteration > 0) {
            // Communicate label and check if want to contract
           
            
            // Tails collapse into Heads
            bool meTails = !heads(vertex.id());
            vid_t my_id = vertex.id();
            if (meTails) {
                num_tails++;
                // Find minimum edge with a heads
                for(int i=0; i < vertex.num_edges(); i++) {
                        graphchi_edge<EdgeDataType> * e = vertex.edge(i);
                        if (heads(e->vertex_id())) {
                            if (e->get_data().in_mst) {
                                my_id = e->vertex_id();
                                num_contract++;
                                break;
                            }
                        }
                }
            } else {
                num_heads++;
            }
            
            // Write my label
            for(int i=0; i < vertex.num_edges(); i++) {
                graphchi_edge<EdgeDataType> * e = vertex.edge(i);
                bidirectional_component_weight edata = e->get_data();
                edata.my_label(vertex.id(), e->vertex_id()) = my_id;
                e->set_data(edata);
                
            }
        }
        
        if (contractionType == BORUVSKA) {
            /* Get my component id. It is the minimum label of a neighbor via a mst edge (or my own id) */
            vid_t min_component_id = vertex.id();
            for(int i=0; i < vertex.num_edges(); i++) {
                graphchi_edge<EdgeDataType> * e = vertex.edge(i);
                if (e->get_data().in_mst) {
                    min_component_id = std::min(e->get_data().neighbor_label(vertex.id(), e->vertex_id()), min_component_id);
                }
            }
            
            /* Set component ids and schedule neighbors */
            for(int i=0; i < vertex.num_edges(); i++) {
                graphchi_edge<EdgeDataType> * e = vertex.edge(i);
                bidirectional_component_weight edata = e->get_data();
                
                if (edata.my_label(vertex.id(), e->vertex_id()) != min_component_id) {
                    edata.my_label(vertex.id(), e->vertex_id()) = min_component_id;
                    e->set_data(edata);
                   
                }
            }
        }
    }
    
    /**
     * Called before an iteration starts.
     */
    void before_iteration(int iteration, graphchi_context &gcontext) {
        logstream(LOG_INFO) << "Start iteration " << iteration << ", scheduled tasks=" << gcontext.scheduler->num_tasks() << std::endl;
    }
    
    void after_iteration(int iteration, graphchi_context &gcontext) {
        logstream(LOG_INFO) << "To contract: " << num_contract << ", tails=" << num_tails << " heads=" << num_heads <<
            " active=" << num_active_vertices << std::endl;
        if (iteration == 1) {
            fprintf(complog, "%d,%d,%ld\n", num_contract, num_active_vertices, num_edges);
        }
    }
    void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {}
    void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {}
    
};


double totalMST = 0.0;
mutex lock;

/**
  * Update function that writes the contracted graph for next iteration and
  * outputs also the minimum spanning edges.
  */
struct ContractionStep : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    bool new_edges;
    
    ContractionStep() {
        new_edges = false;
    }
    
    /**
     *  Vertex update function. Note: we assume fresh edge values.
     */
    void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
        
        if (vertex.num_inedges() == 0) {
            return;
        }
        // Loop over only in-edges
        for(int i=0; i < vertex.num_inedges(); i++) {
            graphchi_edge<EdgeDataType> * e = vertex.inedge(i);
            
            bidirectional_component_weight edata = e->get_data();
            
            if (e->get_data().in_mst) {
                lock.lock();
                if (edata.weight >= 0.0) totalMST += edata.weight;
                lock.unlock();
            }
            
            if (e->get_data().in_mst && edata.labels_agree()) {
                if (edata.weight >= 0.0) {
                    gengine->output(MST_OUTPUT)->output_edge(edata.orig_src, edata.orig_dst, edata.weight);
                }
            } else if (!edata.labels_agree()) {
                // Output the contracted edge
                
                vid_t a = edata.my_label(vertex.id(), e->vertex_id());
                vid_t b = edata.neighbor_label(vertex.id(), e->vertex_id());
                
                // NOTE: If in MST, we need to emit it but will set it to -1 ("invalid") so it will be
                // picked up on next round for sure and the component is kept in-tact, but will
                // not affect the MST as it is zero weight.
                if (edata.in_mst) {
                    if (edata.weight >= 0.0)
                        gengine->output(MST_OUTPUT)->output_edge(edata.orig_src, edata.orig_dst, edata.weight);
                    edata.weight = -1.0;
                }
                
                edata.smaller_component = MAX_VIDT;
                edata.larger_component = MAX_VIDT;
                
                new_edges = true;
                gengine->output(CONTRACTED_GRAPH_OUTPUT)->output_edgeval(std::min(a, b),
                                                                         std::max(a, b),
                                                                         edata);
            } else {
                // Otherwise: discard the edge
            }
        }
    }
    
    /**
     * Called before an iteration starts.
     */
    void before_iteration(int iteration, graphchi_context &gcontext) {
        logstream(LOG_INFO) << "Contraction: Start iteration " << iteration << std::endl;
    }
    
    void after_iteration(int iteration, graphchi_context &gcontext) {
    }
    void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {}
    void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {}
    
};

 


int main(int argc, const char ** argv) {
    /* GraphChi initialization will read the command line
     arguments and the configuration file. */
    graphchi_init(argc, argv);
    
    /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
    metrics m("minimum-spanning-forest");
    m.start_time("msf-total-runtime");
    
    /* Basic arguments for application */
    std::string filename = get_option_string("file");  // Base filename
    bool scheduler       = false; // Whether to use selective scheduling
    
    /* Detect the number of shards or preprocess an input to create them */
    int nshards          = get_option_int("nshards", 0);
    delete_shards<EdgeDataType>(filename, nshards);
    
    convert_if_notexists<double, EdgeDataType>(filename, get_option_string("nshards", "10"));
    
    contractionType = get_option_string("algo", "boruvska") == "boruvska" ? BORUVSKA : STAR;
    
    if (contractionType == BORUVSKA) {
        complog = fopen("msflog_boruvska.txt", "w");
    } else {
        complog = fopen("msflog_star.txt", "w");

    }

    for(int MSF_iteration=0; MSF_iteration < 100; MSF_iteration++) {
        logstream(LOG_INFO) << "MSF ITERATION " << MSF_iteration << " contraction: " << contractionType << std::endl;
        /* Step 1: Run boruvska step */
        BoruvskaStarContractionStep boruvska_starcontraction;
        graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, scheduler, m);
        engine.set_disable_vertexdata_storage();
        gengine = &engine;
        engine.set_save_edgesfiles_after_inmemmode(true);
        engine.set_modifies_inedges(true);
        engine.set_modifies_outedges(true);
        engine.set_disable_outedges(false);

        engine.run(boruvska_starcontraction, 2);
        
        /* Step 2: Run contraction */
        /* Initialize output */
        basic_text_output<VertexDataType, EdgeDataType> mstout(filename + ".mst", "\t");
        
        int orig_numshards = (int) engine.get_intervals().size();
        std::string contractedname = filename + "C";
        sharded_graph_output<VertexDataType, EdgeDataType> shardedout(contractedname, new AcceptMinimum());
        MST_OUTPUT = engine.add_output(&mstout);
        CONTRACTED_GRAPH_OUTPUT = engine.add_output(&shardedout);
        
        ContractionStep contraction;
        engine.set_disable_vertexdata_storage();
        engine.set_modifies_inedges(true);
        engine.set_modifies_outedges(false);
        engine.set_disable_outedges(true);
        engine.set_save_edgesfiles_after_inmemmode(true);
        engine.run(contraction, 1);

        // Clean up
        delete_shards<EdgeDataType>(filename, orig_numshards);

        std::cout << "Total MST now: " << totalMST << std::endl;
        
        if (contraction.new_edges == false) {
            logstream(LOG_INFO) << "MSF ready!" << std::endl;
            break;
        }
        
        nshards = (int)shardedout.finish_sharding();
        filename = contractedname;
        
    }
    m.stop_time("msf-total-runtime");
    /* Report execution metrics */
    metrics_report(m);
    fclose(complog);
    return 0;
}
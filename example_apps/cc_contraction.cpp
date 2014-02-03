
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
 * star contraction. Unfortunately, this code is quite optimized and hard to read.
 *
 * This application demonstrates how graph contraction algorithms can be implemented efficiently
 * with GraphChi.
 */

#define GRAPHCHI_DISABLE_COMPRESSION

#include <string>

#include "graphchi_basic_includes.hpp"
#include "util/labelanalysis.hpp"

using namespace graphchi;


enum ContractionType {
    STAR = 1, BORUVSKA = 2
};

ContractionType contractionType;

#define MAX_VIDT 0xffffffff

struct bidirectional_label {
    vid_t smaller_component;
    vid_t larger_component;
    
    bidirectional_label() {
        smaller_component = larger_component = MAX_VIDT;
    }
    
    bidirectional_label(int x) {
        smaller_component = larger_component = MAX_VIDT;
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


class AcceptMinimum : public DuplicateEdgeFilter<bidirectional_label> {
    bool acceptFirst(bidirectional_label &first, bidirectional_label &second) {
        return (first.smaller_component < second.smaller_component);
    }
};


// Temporary solution
std::vector<vid_t> componentids;


/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program.
 */
typedef vid_t VertexDataType;
typedef bidirectional_label EdgeDataType;

void * gengine;

size_t CONTRACTED_GRAPH_OUTPUT;
FILE * complog;



template <typename EdgeDataType>
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

        }
        
        if (contractionType == STAR) {
            // Communicate label and check if want to contract
            
            // Tails collapse into Heads
            bool meTails = !heads(vertex.id());
            vid_t my_id = vertex.id();
            if (meTails) {
                assert(false); // not done properly yet
                num_tails++;
                // Find minimum edge with a heads
                for(int i=0; i < vertex.num_edges(); i++) {
                    graphchi_edge<EdgeDataType> * e = vertex.edge(i);
                    if (heads(e->vertex_id())) {
                        my_id = e->vertex_id();
                        num_contract++;
                        break;
                    }
                }
            } else {
                num_heads++;
            }
            
            // Write my label
            for(int i=0; i < vertex.num_edges(); i++) {
                graphchi_edge<EdgeDataType> * e = vertex.edge(i);
                EdgeDataType edata = e->get_data();
                edata.my_label(vertex.id(), e->vertex_id()) = my_id;
                e->set_data(edata);
            }
        }
        
        if (contractionType == BORUVSKA) {
            /* Get my component id. It is the minimum label of a neighbor via a mst edge (or my own id) */
            vid_t min_component_id = vertex.id();
            for(int i=0; i < vertex.num_edges(); i++) {
                graphchi_edge<EdgeDataType> * e = vertex.edge(i);
                min_component_id = std::min(
                     std::min(e->get_data().neighbor_label(vertex.id(), e->vertex_id()), e->vertex_id()), min_component_id);
            }
            
            
            componentids[vertex.id()] = min_component_id;
            
            /* Set component ids and schedule neighbors */
            for(int i=0; i < vertex.num_edges(); i++) {
                graphchi_edge<EdgeDataType> * e = vertex.edge(i);
                EdgeDataType edata = e->get_data();
                
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
        if (iteration == 0) {
            fprintf(complog, "%d,%d,%ld\n", num_contract, num_active_vertices, num_edges);
        }
    }
    void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {}
    void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {}
    
};


 mutex lock;

/**
 * Update function that writes the contracted graph for next iteration and
 * outputs also the minimum spanning edges.
 */
template <typename EdgeDataType>
struct ContractionStep : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    bool new_edges;
    
    ContractionStep() {
        new_edges = false;
    }
    
    
    void emit(vid_t from, vid_t to, vid_t a, vid_t b, bidirectional_label &edata) {
        // terrible...
        sharded_graph_output<VertexDataType, EdgeDataType, bidirectional_label> * out = (sharded_graph_output<VertexDataType, EdgeDataType, bidirectional_label> *)((graphchi_engine<VertexDataType, EdgeDataType> *)gengine)->output(CONTRACTED_GRAPH_OUTPUT);
        out->output_edgeval(a, b, edata);
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
            
            EdgeDataType edata = e->get_data();
       
            
            vid_t a = edata.my_label(vertex.id(), e->vertex_id());
            vid_t b = edata.neighbor_label(vertex.id(), e->vertex_id());
            
            if (edata.labels_agree()) {
                // Do nothing
            } else if (!edata.labels_agree()) {
                // Output the contracted edge
        
                
                edata.smaller_component = MAX_VIDT;
                edata.larger_component = MAX_VIDT;
                
                new_edges = true;
                emit(vertex.id(), e->vertex_id(), std::min(a, b), std::max(a, b),
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
    metrics m("cc-contraction");
    m.start_time("cc-contraction-total-runtime");
    
    /* Basic arguments for application */
    std::string filename = get_option_string("file");  // Base filename
    std::string origfilename = filename;
    bool scheduler       = false; // Whether to use selective scheduling
    
    /* Detect the number of shards or preprocess an input to create them */
    int nshards          = get_option_int("nshards", 0);
    delete_shards<EdgeDataType>(filename, nshards);
    
    convert_if_notexists<int, EdgeDataType>(filename, get_option_string("nshards", "0"));
    
    contractionType = get_option_string("algo", "boruvska") == "boruvska" ? BORUVSKA : STAR;
    
    if (contractionType == BORUVSKA) {
        complog = fopen("cclog_boruvska.txt", "w");
    } else {
        complog = fopen("ccflog_star.txt", "w");
        
    }
    
    
    /* NOTE: because of optimizing the first iteration data size, this is a terrible mess */
    for(int super_iteration=0; super_iteration < 100; super_iteration++) {
        logstream(LOG_INFO) << "CC ITERATION " << super_iteration << " contraction: " << contractionType << std::endl;
        
        BoruvskaStarContractionStep<EdgeDataType> boruvska_starcontraction;
        graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, scheduler, m);
        engine.set_disable_vertexdata_storage();
        gengine = &engine;
        engine.set_save_edgesfiles_after_inmemmode(true);
        engine.set_modifies_inedges(true);
        engine.set_modifies_outedges(true);
        engine.set_disable_outedges(false);
        
        if (super_iteration == 0) {
            componentids.resize(engine.num_vertices());
        }
        
        engine.run(boruvska_starcontraction, 1);  
        
        /* Step 2: Run contraction */
        /* Initialize output */
    
        int orig_numshards = (int) engine.get_intervals().size();
        std::string contractedname = filename + "C";
        sharded_graph_output<VertexDataType, EdgeDataType> shardedout(contractedname, new AcceptMinimum());
        CONTRACTED_GRAPH_OUTPUT = engine.add_output(&shardedout);
        
        ContractionStep<EdgeDataType> contraction;
        engine.set_disable_vertexdata_storage();
        engine.set_modifies_inedges(false);
        engine.set_modifies_outedges(false);
        engine.set_disable_outedges(true);
        engine.set_save_edgesfiles_after_inmemmode(true);
        engine.run(contraction, 1);
        
        // Clean up
        if (super_iteration > 0)
            delete_shards<EdgeDataType>(filename, orig_numshards);
        
    
        if (contraction.new_edges == false) {
            logstream(LOG_INFO) << "CC ready!" << std::endl;
            break;
        }
        
        nshards = (int)shardedout.finish_sharding();
        filename = contractedname;
    }
    m.stop_time("cc-contraction-total-runtime");
    
    logstream(LOG_INFO) << "Final component labeling..." << std::endl;
    
    bool changes = true;
    while(changes) {
        changes = false;
        std::cout << "..." << std::endl;
        for(int j=0; j<componentids.size(); j++) {
            vid_t x = componentids[j];
            if (x != j) {
                if (x != componentids[x]) {
                    changes = true;
                    componentids[j] = componentids[x];
                }
            }
        }
    }
    logstream(LOG_INFO) << "Done final component labeling..." << std::endl;

    
    /* Write vertex data */
    std::string outputfile = filename_vertex_data<VertexDataType>(origfilename);
    
    
    FILE * f = fopen(outputfile.c_str(), "w");
    fwrite(&componentids[0], sizeof(vid_t), componentids.size(), f);
    fclose(f);
    
    /* Analyze */
    analyze_labels<vid_t>(origfilename);
    
    
    /* Report execution metrics */
    metrics_report(m);
    fclose(complog);
    return 0;
}
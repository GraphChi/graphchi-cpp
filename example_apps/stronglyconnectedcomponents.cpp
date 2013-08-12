
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
 * Strongly Connected Components. Based on technical report (2012):
 @article{salihoglucomputing,
 title={Computing Strongly Connected Components in Pregel-like Systems},
 author={Salihoglu, Semih and Widom, Jennifer},
 publisher={Stanford InfoLab}
 }
 */


#define SUPPORT_DELETIONS 1

#include <string>

#include "graphchi_basic_includes.hpp"


using namespace graphchi;


/**
 * Unlike in weakly connected components, we need
 * to ensure that neighbors do not overwrite each
 * others values. This is achieved by keeping two values
 * in an edge. In this struct, smaller_one is the id of the
 * vertex that has smaller id, and larger_one the others.
 * This complexity is due to us ignoring the direction of an edge.
 */
struct bidirectional_label {
    vid_t smaller_one;
    vid_t larger_one;
    
    
    vid_t & neighbor_label(vid_t myid, vid_t nbid) {
        if (myid < nbid) {
            return larger_one;
        } else {
            return smaller_one;
        }
    }
    
    vid_t & my_label(vid_t myid, vid_t nbid) {
        if (myid < nbid) {
            return smaller_one;
        } else {
            return larger_one;
        }
    }
    
};

struct SCCinfo {
    vid_t color;
    bool confirmed;
    SCCinfo() : color(0), confirmed(false) {}
    SCCinfo(vid_t color) : color(color), confirmed(false) {}
    SCCinfo(vid_t color, bool confirmed) : color(color), confirmed(confirmed) {}

};

typedef SCCinfo VertexDataType;
typedef bidirectional_label EdgeDataType;

static inline bool VARIABLE_IS_NOT_USED is_deleted_edge_value(bidirectional_label val);
static inline bool VARIABLE_IS_NOT_USED is_deleted_edge_value(bidirectional_label val) {
    return 0xffffffffu == val.smaller_one;
}

static void VARIABLE_IS_NOT_USED remove_edgev(graphchi_edge<bidirectional_label> * e);
static void VARIABLE_IS_NOT_USED remove_edgev(graphchi_edge<bidirectional_label> * e) {
    bidirectional_label deletedlabel;
    deletedlabel.smaller_one = deletedlabel.larger_one = 0xffffffffu;
    e->set_data(deletedlabel);
}

bool first_iteration = true;
bool remainingvertices = true;
/**
 * FORWARD-PHASE
 */
struct SCCForward : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    /**
     *  Vertex update function.
     */
    void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {        
        if (first_iteration) {
            vertex.set_data(SCCinfo(vertex.id()));
        }
        
        if (vertex.get_data().confirmed) {
            return;
        }   
        
        /* Vertices with only in or out edges cannot be part of a SCC (Trimming) */
        if (vertex.num_inedges() == 0 || vertex.num_outedges() == 0) {
            if (vertex.num_edges() > 0) {
                // TODO: check this logic!
                vertex.set_data(SCCinfo(vertex.id()));
            }
            vertex.remove_alledges();
            return;
        }
        remainingvertices = true;

        VertexDataType vertexdata = vertex.get_data();
        bool propagate = false;
        if (gcontext.iteration == 0) {
            vertexdata = vertex.id();
            propagate = true;
            /* Clean up in-edges. This would be nicer in the messaging abstraction... */
            for(int i=0; i < vertex.num_inedges(); i++) {
                bidirectional_label edgedata = vertex.inedge(i)->get_data();
                edgedata.my_label(vertex.id(), vertex.inedge(i)->vertexid) = vertex.id();
                vertex.inedge(i)->set_data(edgedata);
            }
        } else {
            
            /* Loop over in-edges and choose minimum color */
            vid_t minid = vertexdata.color;
            for(int i=0; i < vertex.num_inedges(); i++) {
                minid = std::min(minid, vertex.inedge(i)->get_data().neighbor_label(vertex.id(), vertex.inedge(i)->vertexid));
            }
            
            if (minid != vertexdata.color) {
                vertexdata.color = minid;
                propagate = true;
            }            
        }
        vertex.set_data(vertexdata);
        
        if (propagate) {
            for(int i=0; i < vertex.num_outedges(); i++) {
                bidirectional_label edgedata = vertex.outedge(i)->get_data();
                edgedata.my_label(vertex.id(), vertex.outedge(i)->vertexid) = vertexdata.color;
                vertex.outedge(i)->set_data(edgedata);
                gcontext.scheduler->add_task(vertex.outedge(i)->vertexid);
            }
        }
    }
    
    void before_iteration(int iteration, graphchi_context &gcontext) {
    }
    
    void after_iteration(int iteration, graphchi_context &gcontext) {
        first_iteration = false;
    }
    void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) { }
  
    void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {}
};

/**
  * BACKWARD phase
  */
struct SCCBackward : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    /**
     *  Vertex update function.
     */
    void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
    
        if (vertex.get_data().confirmed) {
            return;
        }   
        
        VertexDataType vertexdata = vertex.get_data();
        bool propagate = false;
        if (gcontext.iteration == 0) {
            /* "Leader" of the SCC */
            if (vertexdata.color == vertex.id()) {
                propagate = true;
                vertex.remove_alloutedges();
            }
            
        } else {
            
            /* Loop over in-edges and see if there is a match */
            bool match = false;
            for(int i=0; i < vertex.num_outedges(); i++) {
                if (vertex.outedge(i)->get_data().neighbor_label(vertex.id(), vertex.outedge(i)->vertexid) == vertexdata.color) {
                    match = true;
                    break;
                }
            }
            if (match) {
                propagate = true;
                std::cout << "Vertex " << vertex.id() << " is part of SCC=" << vertexdata.color << " outedges:" << vertex.num_outedges() << std::endl;
                vertex.remove_alloutedges();
                vertex.set_data(SCCinfo(vertexdata.color, true));
            } else {
                vertex.set_data(SCCinfo(vertex.id(), false));
            }
        }
        
        
        if (propagate) {
            for(int i=0; i < vertex.num_inedges(); i++) {
                bidirectional_label edgedata = vertex.inedge(i)->get_data();
                edgedata.my_label(vertex.id(), vertex.inedge(i)->vertexid) = vertexdata.color;
                vertex.inedge(i)->set_data(edgedata);
                gcontext.scheduler->add_task(vertex.inedge(i)->vertexid);
                std::cout << "Schedule : " << vertex.inedge(i)->vertexid << std::endl;
            }
        }
    }
    
    void before_iteration(int iteration, graphchi_context &gcontext) {}
    
    void after_iteration(int iteration, graphchi_context &gcontext) {}
    void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) { }
    
    void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {}
};


int main(int argc, const char ** argv) {
    /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
    graphchi_init(argc, argv);
    global_logger().set_log_level(LOG_ERROR);
    
    /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
    metrics m("strongly-connected-components");
    
    /* Basic arguments for application */
    std::string filename = get_option_string("file");  // Base filename
    bool scheduler       = true;
    
    /* Detect the number of shards or preprocess an input to create them */
    
    int nshards = find_shards<EdgeDataType>(filename);
    if (nshards > 0) {
        delete_shards<EdgeDataType>(filename, nshards);
    }
    
    nshards          = convert_if_notexists<EdgeDataType>(filename, 
                                                              get_option_string("nshards", "auto"));
    
 
    
    /* Run */
    int super_step = 0;
    while(remainingvertices) {
        std::cout  << "STARTING SUPER STEP: " << super_step << std::endl;
        super_step++;
        remainingvertices = false;

        SCCForward forwardSCC;
        graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, scheduler, m); 
        if (first_iteration) {
            engine.set_reset_vertexdata(true);
        }
        engine.run(forwardSCC, 1000);
        
        if (remainingvertices) {
            std::cout  << "STARTING BACKWARD " << std::endl;
            SCCBackward backwardSCC;
            graphchi_engine<VertexDataType, EdgeDataType> engine2(filename, nshards, scheduler, m); 
            engine2.run(backwardSCC, 1000);
        }
    }
    
    delete_shards<EdgeDataType>(filename, nshards);

    
    /* Report execution metrics */
    metrics_report(m);
    return 0;
}

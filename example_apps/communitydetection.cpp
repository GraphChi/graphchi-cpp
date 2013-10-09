
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
 * A simple community detection algorithm based on label propagation.
 * LPA-algorithm is explained in http://arxiv.org/pdf/0910.1154.pdf
 * "Advanced modularity-specialized label propagation algorithm for detecting communities in networks
 * X. Liu, T. Murata Tokyo Institute of Technology, 2-12-1 Ookayama, Meguro, Tokyo 152-8552, Japan
 *
 * @section REMARKS
 *
 * The algorithm is very similar to the connected components algorithm, but instead
 * of vertex choosing the minimum label of its neighbor, it chooses the most frequent one.
 *
 * However, because the operation (most frequent label) is not commutative,
 * we need to store both vertices labels in an edge. See comment below, above the
 * struct "bidirectional_label".
 *
 * Note, that this algorithm is not very sophisticated and is prone to local minimas.
 * If you want to use this seriously, try with different initial labeling. 
 * Also, a more sophisticated algorithm called LPAm should be doable on GraphChi.
 *
 * @author Aapo Kyrola
 */

#include <cmath>
#include <map>
#include <string>

#include "graphchi_basic_includes.hpp"
#include "util/labelanalysis.hpp"

using namespace graphchi;

#define GRAPHCHI_DISABLE_COMPRESSION


/**
 * Unlike in connected components, we need
 * to ensure that neighbors do not overwrite each
 * others values. This is achieved by keeping two values
 * in an edge. In this struct, smaller_one is the id of the
 * vertex that has smaller id, and larger_one the others.
 * This complexity is due to us ignoring the direction of an edge.
 */
struct bidirectional_label {
    vid_t smaller_one;
    vid_t larger_one;
};

vid_t & neighbor_label(bidirectional_label & bidir, vid_t myid, vid_t nbid) {
    if (myid < nbid) {
        return bidir.larger_one;
    } else {
        return bidir.smaller_one;
    }
}

vid_t & my_label(bidirectional_label & bidir, vid_t myid, vid_t nbid) {
    if (myid < nbid) {
        return bidir.smaller_one;
    } else {
        return bidir.larger_one;
    }
}

 
typedef vid_t VertexDataType;       // vid_t is the vertex id type
typedef bidirectional_label EdgeDataType;  // Note, 8-byte edge data

void parse(bidirectional_label &x, const char * s) { } // Do nothing



/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct CommunityDetectionProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    /**
     *  Vertex update function.
     */
    void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
        /* This program requires selective scheduling. */
        assert(gcontext.scheduler != NULL);
        vid_t newlabel;
        if (gcontext.iteration == 0) {
            /* On first iteration, choose label vertex id */
            vid_t firstlabel = vertex.id();
            vertex.set_data(firstlabel);
            
            newlabel = firstlabel;
            
            /* Scheduler myself for next iteration */
            gcontext.scheduler->add_task(vertex.id());
            
        } else {
            if (vertex.num_edges() == 0) return; // trivial
            
            /* The basic idea is to find the label that is most popular among
               this vertex's neighbors. This label will be chosen as the new label
               of this vertex. */
            // This part could be optimized: STL map is quite slow.
            std::map<vid_t, int> counts;
            int maxcount=0;
            vid_t maxlabel=0;
            /* Iterate over all the edges */
            for(int i=0; i < vertex.num_edges(); i++) {
                /* Extract neighbor's current label. The edge contains the labels of
                   both vertices it connects, so we need to use the right one. 
                   (See comment for bidirectional_label above) */
                bidirectional_label edgelabel = vertex.edge(i)->get_data();
                vid_t nblabel = neighbor_label(edgelabel, vertex.id(), vertex.edge(i)->vertex_id());
                
                /* Check if this label (nblabel) has been encountered before ... */
                std::map<vid_t, int>::iterator existing = counts.find(nblabel);
                int newcount = 0;
                if(existing == counts.end()) {
                    /* ... if not, we add this label with count of one to the map */
                    counts.insert(std::pair<vid_t,int>(nblabel, 1));
                    newcount = 1;
                } else {
                    /* ... if yes, we increment the counter for this label by 1 */
                    existing->second++;
                    newcount = existing->second;
                }
                
                /* Finally, we keep track of the most frequent label */
                if (newcount > maxcount || (maxcount == newcount && nblabel > maxlabel)) {
                    maxlabel = nblabel;
                    maxcount = newcount;
                }
            }
            newlabel = maxlabel;
        }
        /**
         * Write my label to my neighbors.
         */
        if (newlabel != vertex.get_data() || gcontext.iteration == 0) {
            vertex.set_data(newlabel);
            for(int i=0; i<vertex.num_edges(); i++) {
                bidirectional_label labels_on_edge = vertex.edge(i)->get_data();
                my_label(labels_on_edge, vertex.id(), vertex.edge(i)->vertex_id()) = newlabel;
                vertex.edge(i)->set_data(labels_on_edge);
                
                // On first iteration, everyone schedules themselves.
                if (gcontext.iteration > 0)
                    gcontext.scheduler->add_task(vertex.edge(i)->vertex_id());                
            }
        }
        
    }
    
    
    /**
     * Called before an iteration starts.
     */
    void before_iteration(int iteration, graphchi_context &info) {
    }
    
    /**
     * Called after an iteration has finished.
     */
    void after_iteration(int iteration, graphchi_context &ginfo) {
    }
    
    /**
     * Called before an execution interval is started.
     */
    void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &ginfo) {        
    }
    
    /**
     * Called after an execution interval has finished.
     */
    void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &ginfo) {        
    }
    
};

int main(int argc, const char ** argv) {
    /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
    graphchi_init(argc, argv);
    
    /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
    metrics m("community-detection");
    
    /* Basic arguments for application */
    std::string filename = get_option_string("file");  // Base filename
    int niters           = get_option_int("niters", 10); // Number of iterations (max)
    bool scheduler       = true;    // Always run with scheduler
        
    /* Process input file - if not already preprocessed */
    int nshards             = convert_if_notexists<EdgeDataType>(filename, get_option_string("nshards", "auto"));

    if (get_option_int("onlyresult", 0) == 0) {
        /* Run */
        CommunityDetectionProgram program;
        graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, scheduler, m); 
        engine.run(program, niters);
    }
    
    /* Run analysis of the communities (output is written to a file) */
    m.start_time("label-analysis");
    
    analyze_labels<vid_t>(filename);
    
    m.stop_time("label-analysis");
    
    /* Report execution metrics */
    metrics_report(m);
    return 0;
}



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
 * Template for GraphChi applications. To create a new application, duplicate
 * this template.
 */



#include <string>

#include "graphchi_basic_includes.hpp"

using namespace graphchi;

typedef bool VertexDataType;
typedef bool EdgeDataType;

FILE * logff;
 
struct Contractor : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    std::vector<vid_t> vertex_labels;
    std::vector<vid_t> vertex_labels_last;
    bool synchronous;
    
    Contractor(bool synchronous) : synchronous(synchronous) {}
    
    /**
     *  Vertex update function.
     */
    void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
        
        vid_t min_label = vertex_labels[vertex.id()];
        
        std::vector<vid_t> & nbr_labels = (synchronous ? vertex_labels_last : vertex_labels);
        
        /* Loop over all edges (ignore direction) */
        for(int i=0; i < vertex.num_edges(); i++) {
            min_label = std::min(min_label, nbr_labels[vertex.edge(i)->vertex_id()]);
        }
        vertex_labels[vertex.id()] = min_label;
    }
    
    void print() {
        for(size_t i=0; i<vertex_labels.size(); i++) {
            std::cout << i << " = " << vertex_labels[i] << std::endl;
        }
    }
    
    /**
     * Called before an iteration starts.
     */
    void before_iteration(int iteration, graphchi_context &gcontext) {
        if (iteration == 0) {
            // Initialize labels
            vertex_labels.resize(gcontext.nvertices);
            for(vid_t v=0; v<gcontext.nvertices; v++) {
                vertex_labels[v] = v;
            }
            // Shuffle
            std::cout << "Shuffling labels" << std::endl;
            std::random_shuffle(vertex_labels.begin(), vertex_labels.end());
            if (synchronous) vertex_labels_last = vertex_labels;
            
            
            if (get_option_int("print", 0) == 1) {
                std::cout << "Initial: " << std::endl;
                print();
            }
            
            
        }
        
        if (synchronous) {
            vertex_labels_last = vertex_labels;
        }
    }
    
    /**
     * Called after an iteration has finished.
     */
    void after_iteration(int iteration, graphchi_context &gcontext) {
        
    }
    
    size_t unique_labels() {
        std::vector<vid_t> labs = vertex_labels;
        sort(labs.begin(), labs.end(), std::less<vid_t>());
        size_t n = 1;
        for(size_t i=1; i<labs.size(); i++) {
            if (labs[i-1] != labs[i]) n++;
            assert(labs[i-1] <= labs[i]);
        }
        
        
        return n;
    }
   
    
    
};

int main(int argc, const char ** argv) {
    /* GraphChi initialization will read the command line
     arguments and the configuration file. */
    graphchi_init(argc, argv);
    global_logger().set_log_level(LOG_ERROR);
    
    // Seed
    timeval t;
    gettimeofday(&t, NULL);
    srand(t.tv_usec);

    /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
    metrics m("contraction-research");
    
    /* Basic arguments for application */
    std::string filename = get_option_string("file");  // Base filename
    int niters           = get_option_int("niters", 1); // Number of iterations
    bool scheduler       = get_option_int("scheduler", 0); // Whether to use selective scheduling
    
    /* Detect the number of shards or preprocess an input to create them */
    int nshards          = convert_if_notexists<EdgeDataType>(filename,
                                                              get_option_string("nshards", "auto"));
    logff = fopen("contraction_log.txt", "a");

    Contractor program(get_option_int("sync"));

    
    /* Run */
    graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, scheduler, m);
    engine.set_modifies_inedges(false);
    engine.set_modifies_outedges(false);
    engine.set_disable_vertexdata_storage();
    engine.run(program, niters);
    
    if (get_option_int("print", 0) == 1) {
        std::cout << "After: " << std::endl;

        program.print();
    }
    
    fprintf(logff, "%s,random-initlabels,%s,%s,%u,%u,%lu,%lf\n", filename.c_str(), program.synchronous ? "synchronous" : "gauss-seidel",
            get_option_int("randomization", 0) ? "random-schedule" : "nonrandom-schedule", niters, engine.num_vertices(), program.unique_labels(),
            double(engine.num_vertices() - program.unique_labels()) / engine.num_vertices());
    fclose(logff);
    printf("%s,%s,%s,%u,%u,%lu,%lf\n", filename.c_str(), program.synchronous ? "synchronous" : "gauss-seidel",
            get_option_int("randomization", 0) ? "random-schedule" : "nonrandom-schedule", niters, engine.num_vertices(), program.unique_labels(),
            double(engine.num_vertices() - program.unique_labels()) / engine.num_vertices());
    
    return 0;
}

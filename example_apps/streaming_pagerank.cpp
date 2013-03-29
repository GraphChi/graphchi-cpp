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
 * Demonstration for streaming graph updates. This application reads from a file
 * list of edges and adds them into the graph continuously. Simultaneously, pagerank
 * is computed for the evolving graph.
 *
 * This code includes a fair amount of code for demo purposes. To be cleaned
 * eventually.
 */

#include <string>
#include <fstream>
#include <cmath>

#define GRAPHCHI_DISABLE_COMPRESSION


#include "graphchi_basic_includes.hpp"
#include "engine/dynamic_graphs/graphchi_dynamicgraph_engine.hpp"
#include "util/toplist.hpp"

/* HTTP admin tool */
#include "httpadmin/chi_httpadmin.hpp"
#include "httpadmin/plotter.hpp"

using namespace graphchi;

#define THRESHOLD 1e-1f    
#define RANDOMRESETPROB 0.15f

#define DEMO 1

typedef float VertexDataType;
typedef float EdgeDataType;

graphchi_dynamicgraph_engine<float, float> * dyngraph_engine;
std::string streaming_graph_file;

std::string getname(vid_t v);
std::string getname(vid_t userid) {
#ifdef DEMO
    // Temporary code for demo purposes!
    int f = open("/Users/akyrola/graphs/twitter_names.dat", O_RDONLY);
    if (f < 0) return "n/a";
    char s[16];
    
    size_t idx = userid * 16;
    preada(f, s, 16, idx);
    close(f);
    s[15] = '\0';
    
    return std::string(s);
#else
    return "":
#endif
}


struct PagerankProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
    /**
     * Called before an iteration starts.
     */
    void before_iteration(int iteration, graphchi_context &gcontext) {
    }
    
    /**
     * Called after an iteration has finished.
     */
    void after_iteration(int iteration, graphchi_context &gcontext) {
#ifdef DEMO
        std::vector< vertex_value<float> > top = get_top_vertices<float>(gcontext.filename, 20);
        
        for(int i=0; i < (int) top.size(); i++) {
            vertex_value<float> vv = top[i];
            std::cout << (i+1) << ". " << vv.vertex << " " << getname(vv.vertex) << ": " << vv.value << std::endl; 
        }
        
        /* Keep top 20 available for http admin */
        for(int i=0; i < (int) top.size(); i++) {
            vertex_value<float> vv = top[i];
            std::stringstream ss;
            ss << "rank" << i;
            std::stringstream sv;
            sv << vv.vertex << ":" << getname(vv.vertex) << ":" << vv.value<< "";
            dyngraph_engine->set_json(ss.str(), sv.str());
        }         
#endif
    }
    
    /**
     * Called before an execution interval is started.
     */
    void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &ginfo) {        
#ifdef DEMO
        update_plotdata(dyngraph_engine);
#endif
    }
    
    
    /**
     * Pagerank update function.
     */
    void update(graphchi_vertex<VertexDataType, EdgeDataType> &v, graphchi_context &ginfo) {
        float sum=0;
        if (ginfo.iteration == 0) {
            /* On first iteration, initialize vertex */
            for(int i=0; i < v.num_outedges(); i++) {
                graphchi_edge<float> * edge = v.outedge(i);
                edge->set_data(1.0f / v.num_outedges());
                if (ginfo.scheduler != NULL)  
                    ginfo.scheduler->add_task(edge->vertex_id());
            }
            v.set_data(RANDOMRESETPROB); 
            
            /* If using selective scheduling, schedule myself for next iteration */
            if (ginfo.scheduler != NULL) 
                ginfo.scheduler->add_task(v.id());
        } else {
            /* Compute the sum of neighbors' weighted pageranks */
            for(int i=0; i < v.num_inedges(); i++) {
                float val = v.inedge(i)->get_data();
                sum += val;                    
            }
            
            /* Compute my pagerank */
            float pagerank = RANDOMRESETPROB + (1 - RANDOMRESETPROB) * sum;
            float oldvalue = v.get_data();
            float delta = (float) fabs(oldvalue - pagerank);
            bool significant_change = (delta >= THRESHOLD);
            
            if (v.num_outedges() > 0) {
                float pagerankcont = pagerank/v.num_outedges();
                for(int i=0; i < v.num_outedges(); i++) {
                    graphchi_edge<float> * edge = v.outedge(i);
                    
                    /* If using selective scheduling, and the change was larger than
                     a threshold, add neighbor to schedule. */
                    if (ginfo.scheduler != NULL) {
                        if (significant_change) {
                            ginfo.scheduler->add_task(edge->vertex_id());
                        }
                    }
                    edge->set_data(pagerankcont);
                }
            }
            v.set_data(pagerank); 
            
            /* Keep track of the progression of the computation */
            ginfo.log_change(delta);
        }
    }
    
};

/* Demo stuff. */
class IntervalTopRequest : public custom_request_handler {
public:
    virtual std::string handle(const char * req) {
        const char * shardnum_str = &req[strlen("/ajax/shardpagerank")];
        int shardnum = atoi(shardnum_str);
        logstream(LOG_DEBUG) << "Requested shard pagerank: " << shardnum_str << std::endl;
        if (shardnum >= 0 && shardnum < dyngraph_engine->get_nshards()) {
            vid_t fromvid = dyngraph_engine->get_interval_start(shardnum);
            vid_t tovid = dyngraph_engine->get_interval_end(shardnum);
            
            std::vector< vertex_value<float> > top = 
            get_top_vertices<float>(dyngraph_engine->get_context().filename, 10,
                        fromvid, tovid + 1);
          
            
            std::stringstream ss;
            ss << "{";
            for(int i=0; i < (int) top.size(); i++) {
                vertex_value<float> vv = top[i];
                if (i > 0) ss << ",";
                ss << "\"rank" << i << "\": \"" <<  vv.vertex << ":" << getname(vv.vertex) << ":" << vv.value<< "\"";
            }         
            ss << "}"; 
            std::cout << ss.str();
            return ss.str();
        }
        return "error";
    }
    virtual bool responds_to(const char * req) {
        return (strncmp(req, "/ajax/shardpagerank", 19) == 0);
    }
    
};

bool running = true;

void * plotter_thread(void * info);
void * plotter_thread(void * info) {

    usleep(1000000 * 10);
    init_plots(dyngraph_engine);

    while(running) {
       /* Update plots */
       drawplots(); 
       usleep(1000000 * 10);
    }
    return NULL;
}

/**
  * Function executed by a separate thread that streams
  * graph from a file.
  */
void * dynamic_graph_reader(void * info);
void * dynamic_graph_reader(void * info) {
    std::cout << "Start sleeping..." << std::endl;
    usleep(50000);
    std::cout << "End sleeping..." << std::endl;
    
    int edges_per_sec = get_option_int("edges_per_sec", 100000);
    
    logstream(LOG_INFO) << "Going to stream from: " << streaming_graph_file << std::endl;   
    FILE * f = fopen(streaming_graph_file.c_str(), "r");
    if (f == NULL) {
        logstream(LOG_ERROR) << "Could not open file for streaming: " << streaming_graph_file << 
        " error: " << strerror(errno) << std::endl;
    }
    assert(f != NULL);
    
    logstream(LOG_INFO) << "Streaming speed capped at: " << edges_per_sec << " edges/sec." << std::endl;
    
    size_t c = 0;
    size_t ingested = 0;
    // Used for flow control
    timeval last, now;
    gettimeofday(&last, NULL);
    
    vid_t from;
    vid_t to;
    char s[1024];
    
    while(fgets(s, 1024, f) != NULL) {
        FIXLINE(s);
        /* Read next line */
        char delims[] = "\t ";	
        char * t;
        t = strtok(s, delims);
        from = atoi(t);
        t = strtok(NULL, delims);
        to = atoi(t);
        
        if (from == to) {
            // logstream(LOG_WARNING) << "Self-edge in stream: " << from << " <-> " << to << std::endl;
            continue;
        }
        
        bool success=false;
        while (!success) {
            success = dyngraph_engine->add_edge(from, to, 0.0f);
        }
        dyngraph_engine->add_task(from);
        ingested++;
        
        if (++c % edges_per_sec == 0) {
            std::cout << "Stream speed check...." << std::endl;
            double sincelast;
            double speed;
            
            // Throttling - keeps average speed of edges/sec in control
            do {
                gettimeofday(&now, NULL);
                sincelast =  now.tv_sec-last.tv_sec+ ((double)(now.tv_usec-last.tv_usec))/1.0E6;
                usleep(20000);
                speed = (c / sincelast);
            } while (speed > edges_per_sec);
            dyngraph_engine->set_json("ingestspeed", speed);
            logstream(LOG_INFO) << "Stream speed check ended.... Speed now:" << speed << " edges/sec" << std::endl;
            dyngraph_engine->set_json("ingestedges", ingested);
        }
        if (c % 1000 == 0) {
            set_ingested_edges(ingested);
        }
                
        
    } 
    fclose(f);
    dyngraph_engine->finish_after_iters(10);
    return NULL;
}


int main(int argc, const char ** argv) {
    graphchi_init(argc, argv);
    metrics m("streaming-pagerank");
    
    /* Parameters */
    std::string filename    = get_option_string("file"); // Base filename
    int niters              = 100000;                    // End of computation to be determined programmatically
    // Pagerank can be run with or without selective scheduling
    bool scheduler          = true;
    int ntop                = get_option_int("top", 20);
    
    /* Process input file (the base graph) - if not already preprocessed */
    int nshards             = convert_if_notexists<EdgeDataType>(filename, get_option_string("nshards", "auto"));
    
    /* Streaming input graph - must be in edge-list format */
    streaming_graph_file = get_option_string_interactive("streaming_graph_file", 
                                                         "Pathname to graph file to stream edges from");
    
    /* Create the engine object */
    dyngraph_engine = new graphchi_dynamicgraph_engine<float, float>(filename, nshards, scheduler, m); 
    dyngraph_engine->set_modifies_inedges(false); // Improves I/O performance.
    
    /* Start streaming thread */
    pthread_t strthread;
    int ret = pthread_create(&strthread, NULL, dynamic_graph_reader, NULL);
    assert(ret>=0);
    
    /* Start HTTP admin */
    start_httpadmin< graphchi_dynamicgraph_engine<float, float> >(dyngraph_engine);
    register_http_request_handler(new IntervalTopRequest());
    
    pthread_t plotterthr;
    ret = pthread_create(&plotterthr, NULL, plotter_thread, NULL);
    assert(ret>=0);
    
    /* Run the engine */
    PagerankProgram program;
    dyngraph_engine->run(program, niters);
    
    
    running = false;
    
    /* Output top ranked vertices */
    std::vector< vertex_value<float> > top = get_top_vertices<float>(filename, ntop);
    std::cout << "Print top " << ntop << " vertices:" << std::endl;
    for(int i=0; i < (int)top.size(); i++) {
        std::cout << (i+1) << ". " << top[i].vertex << "\t" << top[i].value << std::endl;
    }
    
    metrics_report(m);    
    return 0;
}







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
 * Engine for the alternative "functional" API for GraphChi.
 * The functional engine first processes in-edges, then executes "updates",
 * and then loads and updates out-edges.
 */


#ifndef GRAPHCHI_FUNCTIONALENGINE_DEF
#define GRAPHCHI_FUNCTIONALENGINE_DEF

#include "engine/graphchi_engine.hpp"
#include "logger/logger.hpp"

namespace graphchi {
    
    template <typename VertexDataType, typename EdgeDataType, typename fvertex_t>
    class functional_engine : public graphchi_engine<VertexDataType, EdgeDataType,  fvertex_t> {
    public:
        functional_engine(std::string base_filename, int nshards, bool selective_scheduling, metrics &_m) :
        graphchi_engine<VertexDataType, EdgeDataType, fvertex_t>(base_filename, nshards, selective_scheduling, _m){
            _m.set("engine", "functional");
        }
        
    protected:
        /* Override - load only memory shard (i.e inedges) */
        virtual void load_before_updates(std::vector<fvertex_t> &vertices) {
            logstream(LOG_DEBUG) << "Processing in-edges." << std::endl;
            /* Load memory shard */
            if (!this->memoryshard->loaded()) {
                this->memoryshard->load();
            }
            
            /* Load vertex edges from memory shard */
            this->memoryshard->load_vertices(this->sub_interval_st, this->sub_interval_en, vertices, true, false);
            
            /* Load vertices */ 
            this->vertex_data_handler->load(this->sub_interval_st, this->sub_interval_en);
            
            /* Wait for all reads to complete */
            this->iomgr->wait_for_reads();
        }
        
        
        
        virtual bool is_inmemory_mode() {
            return false;
        }

        /* Override - do not allocate edge data */
        virtual void init_vertices(std::vector<fvertex_t> &vertices, graphchi_edge<EdgeDataType> * &e) {
            size_t nvertices = vertices.size();
            
            /* Compute number of edges */
            size_t num_edges = this->num_edges_subinterval(this->sub_interval_st, this->sub_interval_en);
            
             /* Assign vertex edge array pointers */
            size_t ecounter = 0;
            for(int i=0; i < (int)nvertices; i++) {
                degree d = this->degree_handler->get_degree(this->sub_interval_st + i);
                int inc = d.indegree;
                int outc = d.outdegree;
                vertices[i] = fvertex_t(this->chicontext, this->sub_interval_st + i, inc, outc);

                if (this->scheduler != NULL) {
                    bool is_sched = this->scheduler->is_scheduled(this->sub_interval_st + i);
                    if (is_sched) {
                        vertices[i].scheduled =  true;
                        this->nupdates++;
                        ecounter += inc + outc;
                    }
                } else {
                    this->nupdates++;
                    vertices[i].scheduled =  true;
                    ecounter += inc + outc;
                }
            }
            this->work += num_edges;
        }        

        
        /* Override - now load sliding shards, to write (broadcast) to out vertices */
        virtual void load_after_updates(std::vector<fvertex_t> &vertices) {
            logstream(LOG_DEBUG) << "Processing out-edges (broadcast)." << std::endl;
            omp_set_num_threads(this->load_threads);
#pragma omp parallel for schedule(dynamic, 1)
            for(int p=0; p < this->nshards; p++)  {
                /* Stream forward other than the window partition */
                if (p != this->exec_interval) {
                    this->sliding_shards[p]->read_next_vertices(vertices.size(), this->sub_interval_st, vertices,
                                                         this->scheduler != NULL && this->iter == 0);
                    
                } else {
                    this->memoryshard->load_vertices(this->sub_interval_st, this->sub_interval_en, vertices, false, true); // Inedges=false, outedges=true
                }
                
            }
        }   
        
    }; // End class
    
}; // End namespace


#endif



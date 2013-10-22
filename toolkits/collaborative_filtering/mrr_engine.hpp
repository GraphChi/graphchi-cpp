#ifndef __GRAPHCHI_MRR_ENGINE
#define __GRAPHCHI_MRR_ENGINE
/**
 * @file
 * @author  Mark Levy
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
 * File for aggregating and displaying error mesasures and algorithm progress
 */

#include <set>
#include <sstream>

#include "climf.hpp"

vec mrr_vec;                     // cumulative sum of MRR per thread
vec users_vec;                   // user count per thread
int num_threads = 1;
int cur_iteration = 0;

/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct ValidationMRRProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

  /**
   *  compute MRR for a single user
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {

    if (vertex.id() < M)
    {
      // we're at a user node
      const vec & U = latent_factors_inmem[vertex.id()].pvec;

      std::set<int> known_likes;
      {
        for(int j = 0; j < vertex.num_edges(); j++)
        {
          if (is_relevant(vertex.edge(j)))
          {
            known_likes.insert(vertex.edge(j)->vertex_id() - M);
          }
        }
      }

      if (!known_likes.empty())
      {
        // make predictions
        ivec indices = ivec::Zero(N);
        vec distances = zeros(N);
        for (uint i = M; i < M+N; i++)
        {
          const vec & V = latent_factors_inmem[i].pvec;
          indices[i-M] = i-M;
          distances[i-M] = dot(U,V);
        }

        int num_predictions = std::min(num_ratings, static_cast<int>(N));
        vec sorted_distances(num_predictions);
        ivec sorted_indices = reverse_sort_index2(distances, indices, sorted_distances, num_predictions);

        // compute actual MRR
        double MRR = 0;
        for (uint i = 0; i < sorted_indices.size(); ++i)
        {
          if (known_likes.find(sorted_indices[i]) != known_likes.end())
          {
            if (debug && vertex.id() % 1000 == 0 ){
               std::cout<<"User: "<< vertex.id() << " item: "<< sorted_indices[i] << " in position: " << i << " prediction: " << sorted_distances[i] << std::endl;
            }
            MRR = 1.0/(i+1);
            break;
          }
        }

        assert(mrr_vec.size() > omp_get_thread_num());
        mrr_vec[omp_get_thread_num()] += MRR;

        assert(users_vec.size() > omp_get_thread_num());
        users_vec[omp_get_thread_num()]++;
      }
    }
  }

  void before_iteration(int iteration, graphchi_context & gcontext)
  {
    users_vec = zeros(num_threads);
    mrr_vec = zeros(num_threads);
  }

  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext)
  {
    double mrr = sum(mrr_vec) / sum(users_vec);
    std::cout<<"  Validation MRR:" << std::setw(10) << mrr << std::endl;
  }
};

void reset_mrr(int exec_threads)
{
  logstream(LOG_DEBUG)<<"Detected number of threads: " << exec_threads << std::endl;
  num_threads = exec_threads;
  mrr_vec = zeros(num_threads);
}

template<typename VertexDataType, typename EdgeDataType>
void init_mrr_engine(graphchi_engine<VertexDataType,EdgeDataType> *& pvalidation_engine, int nshards)
{
  if (nshards == -1)
    return;
  metrics * m = new metrics("validation_mrr_engine");
  graphchi_engine<VertexDataType, EdgeDataType> * engine = new graphchi_engine<VertexDataType, EdgeDataType>(validation, nshards, false, *m); 
  set_engine_flags(*engine);
  pvalidation_engine = engine;
}

template<typename VertexDataType, typename EdgeDataType>
void run_validation(graphchi_engine<VertexDataType, EdgeDataType> * pvalidation_engine, graphchi_context & context)
{
  //no validation data, no need to run validation engine calculations
  cur_iteration = context.iteration;
  if (pvalidation_engine == NULL)
    return;
  ValidationMRRProgram program;
  pvalidation_engine->run(program, 1);
}

#endif //__GRAPHCHI_MRR_ENGINE

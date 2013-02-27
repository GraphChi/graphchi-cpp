#ifndef __GRAPHCHI_RMSE_ENGINE
#define __GRAPHCHI_RMSE_ENGINE
/**
 * @file
 * @author  Danny Bickson
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
 * File for aggregating and siplaying error mesasures and algorithm progress
 */

float (*pprediction_func)(const vertex_data&, const vertex_data&, const float, double &, void *) = NULL;
vec validation_rmse_vec;
vec users_vec;
vec sum_ap_vec;
bool user_nodes = true;
int num_threads = 1;
bool converged_engine = false;
int cur_iteration = 0;
/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct ValidationAPProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

  /**
   *  compute validaton AP for a single user
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {

    if (user_nodes && vertex.id() >= M)
      return;
    else if (!user_nodes && vertex.id() < M)
      return;
    vertex_data & vdata = latent_factors_inmem[vertex.id()];
    vec ratings = zeros(vertex.num_outedges());
    vec real_vals = zeros(vertex.num_outedges());
    if (ratings.size() > 0){
      users_vec[omp_get_thread_num()]++;
      int j=0;
      int real_click_count = 0;
      for(int e=0; e < vertex.num_outedges(); e++) {
        const EdgeDataType & observation = vertex.edge(e)->get_data();                
        vertex_data & pdata = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        double prediction;
        (*pprediction_func)(vdata, pdata, observation, prediction, NULL);
        ratings[j] = prediction;
        real_vals[j] = observation;
        if (observation > 0)
          real_click_count++;
        j++;
      }
      int count = 0;
      double ap = 0;
      ivec pos = sort_index(ratings);
      for (int j=0; j< std::min(ap_number, (int)ratings.size()); j++){
        if (real_vals[pos[ratings.size() - j - 1]] > 0)
          ap += (++count * 1.0/(j+1));    
      }
      if (real_click_count > 0 )
        ap /= real_click_count;
      else ap = 0;
      sum_ap_vec[omp_get_thread_num()] += ap;
    }
  }
  void before_iteration(int iteration, graphchi_context & gcontext){
    last_validation_rmse = dvalidation_rmse;
    users_vec = zeros(num_threads);
    sum_ap_vec = zeros(num_threads);
  }
  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    assert(Le > 0);
    dvalidation_rmse = finalize_rmse(sum(sum_ap_vec) , (double)sum(users_vec));
    std::cout<<"  Validation  " << error_names[loss_type] << ":" << std::setw(10) << dvalidation_rmse << std::endl;
    if (halt_on_rmse_increase > 0 && halt_on_rmse_increase < cur_iteration && dvalidation_rmse > last_validation_rmse){
      logstream(LOG_WARNING)<<"Stopping engine because of validation " << error_names[loss_type] <<  " increase" << std::endl;
      //gcontext.set_last_iteration(gcontext.iteration);
      converged_engine = true;
    }
  }
};



/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct ValidationRMSEProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

  /**
   *  compute validaton RMSE for a single user
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
    if (user_nodes && vertex.id() >= M)
      return;
    else if (!user_nodes && vertex.id() < M)
      return;
    vertex_data & vdata = latent_factors_inmem[vertex.id()];
    for(int e=0; e < vertex.num_outedges(); e++) {
      const EdgeDataType & observation = vertex.edge(e)->get_data();                
      vertex_data & nbr_latent = latent_factors_inmem[vertex.edge(e)->vertex_id()];
      double prediction;
      double rmse = (*pprediction_func)(vdata, nbr_latent, observation, prediction, NULL);
      assert(rmse <= pow(maxval - minval, 2));
      assert(validation_rmse_vec.size() > omp_get_thread_num());
      validation_rmse_vec[omp_get_thread_num()] += rmse;
    }
  }

  void before_iteration(int iteration, graphchi_context & gcontext){
    last_validation_rmse = dvalidation_rmse;
    validation_rmse_vec = zeros(num_threads);
  }
  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    assert(Le > 0);
    dvalidation_rmse = finalize_rmse(sum(validation_rmse_vec) , (double)Le);
    std::cout<<"  Validation  " << error_names[loss_type] << ":" << std::setw(10) << dvalidation_rmse << std::endl;
    if (halt_on_rmse_increase > 0 && halt_on_rmse_increase < cur_iteration && dvalidation_rmse > last_validation_rmse){
      logstream(LOG_WARNING)<<"Stopping engine because of validation RMSE increase" << std::endl;
       converged_engine = true;
    }
  }
};

void reset_rmse(int exec_threads){
  logstream(LOG_DEBUG)<<"Detected number of threads: " << exec_threads << std::endl;
  num_threads = exec_threads;
  rmse_vec = zeros(exec_threads);
}

template<typename VertexDataType, typename EdgeDataType>
void init_validation_rmse_engine(graphchi_engine<VertexDataType,EdgeDataType> *& pvalidation_engine, int nshards,float (*prediction_func)(const vertex_data & user, const vertex_data & movie, float rating, double & prediction, void * extra)){
  if (nshards == -1)
    return;
  metrics * m = new metrics("validation_rmse_engine");
  graphchi_engine<VertexDataType, EdgeDataType> * engine = new graphchi_engine<VertexDataType, EdgeDataType>(validation, nshards, false, *m); 
  set_engine_flags(*engine);
  pvalidation_engine = engine;
  pprediction_func = prediction_func;
}

template<typename VertexDataType, typename EdgeDataType>
void run_validation(graphchi_engine<VertexDataType, EdgeDataType> * pvalidation_engine, graphchi_context & context){
  //no validation data, no need to run validation engine calculations
  cur_iteration = context.iteration;
  if (pvalidation_engine == NULL){
    std::cout << std::endl;
    return;
  }
  if (calc_ap){ //AP
    ValidationAPProgram program;
    pvalidation_engine->run(program, 1);
  }
  else { //RMSE
    ValidationRMSEProgram program;
    pvalidation_engine->run(program, 1);
  }
  if (converged_engine)
    context.set_last_iteration(context.iteration);
}

#endif //__GRAPHCHI_RMSE_ENGINE

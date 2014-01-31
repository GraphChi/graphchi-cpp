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
 * @section DESCRIPTION
 *
 * CLiMF Collaborative Less-is-More Filtering, a variant of latent factor CF
 * which optimises a lower bound of the smoothed reciprocal rank of "relevant"
 * items in ranked recommendation lists.  The intention is to promote diversity
 * as well as accuracy in the recommendations.  The method assumes binary
 * relevance data, as for example in friendship or follow relationships.
 *
 * CLiMF: Learning to Maximize Reciprocal Rank with Collaborative Less-is-More Filtering
 * Yue Shi, Martha Larson, Alexandros Karatzoglou, Nuria Oliver, Linas Baltrunas, Alan Hanjalic
 * ACM RecSys 2012
 *
 */

#include <string>
#include <algorithm>

#include "util.hpp"
#include "eigen_wrapper.hpp"
#include "common.hpp"
#include "climf.hpp"
#include "io.hpp"
#include "rmse.hpp"  // just for test_predictions()
#include "mrr_engine.hpp"

int node_without_edges=0;

/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type>
 * class. The main logic is usually in the update function.
 */
struct SGDVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

  /**
   * Called before an iteration is started.
   */
  void before_iteration(int iteration, graphchi_context &gcontext)
  {
    logstream(LOG_DEBUG) << "before_iteration: resetting MRR" << std::endl;
    reset_mrr(gcontext.execthreads);
    last_training_objective = training_objective;
    objective_vec = zeros(gcontext.execthreads);
    stat_vec = zeros(gcontext.execthreads);
    node_without_edges = 0;
    if (gcontext.iteration == 0)
      run_validation(pvalidation_engine, gcontext);
  }

  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext)
  {
    training_objective = sum(objective_vec);
    std::cout<<"  Training objective:" << std::setw(10) << training_objective << std::endl;
    if (halt_on_mrr_decrease > 0 && halt_on_mrr_decrease < cur_iteration && training_objective < last_training_objective)
    {
      logstream(LOG_WARNING) << "Stopping engine because of validation objective decrease" << std::endl;
      gcontext.set_last_iteration(gcontext.iteration);
    }
    logstream(LOG_DEBUG) << "after_iteration: running validation engine" << std::endl;
    run_validation(pvalidation_engine, gcontext);
    if (verbose)
       std::cout<<"Average step size: " << sum(stat_vec)/(double)M << "Node without edges: " << node_without_edges << std::endl;
    sgd_gamma *= sgd_step_dec;
  }

  /**
   *  Vertex update function.
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext)
  {
    double objective = -0.5*sgd_lambda*latent_factors_inmem[vertex.id()].pvec.squaredNorm();

    // go over all user nodes
    if (vertex.num_outedges() > 1)   // can't compute with CLiMF if we have only 1 out edge!
    {
      vec & U = latent_factors_inmem[vertex.id()].pvec;
      int Ni = vertex.num_edges();

      // precompute f_{ij} = <U_i,V_j> for j = 1..N_i
      std::vector<double> f(Ni);
      int num_relevant = 0;

      for (int j = 0; j < Ni; ++j)
      {
         if (is_relevant(vertex.edge(j)))
         {
           const vec & Vj = latent_factors_inmem[vertex.edge(j)->vertex_id()].pvec;
           f[j] = dot(U, Vj);
           ++num_relevant;
         }
      }

      if (num_relevant < 2)
      {
        return;  // need at least 2 edges to compute updates with CLiMF!
        node_without_edges++;
      }

      // compute gradients
      vec dU = -sgd_lambda*U;

      for (int j = 0; j < Ni; ++j)
      {
         if (is_relevant(vertex.edge(j)))
         {
           vec & Vj = latent_factors_inmem[vertex.edge(j)->vertex_id()].pvec;
           vec dVj = g(-f[j])*ones(D) - sgd_lambda*Vj;

           for (int k = 0; k < Ni; ++k)
           {
              if (k != j && is_relevant(vertex.edge(k)))
              {
                 dVj += dg(f[j]-f[k])*(1.0/(1.0-g(f[k]-f[j]))-1.0/(1.0-g(f[j]-f[k])))*U;
              }
           }

           Vj += sgd_gamma*dVj;  // not thread-safe
           dU += g(-f[j])*Vj;

           for (int k = 0; k < Ni; ++k)
           {
              if (k != j && is_relevant(vertex.edge(k)))
              {
                 const vec & Vk = latent_factors_inmem[vertex.edge(k)->vertex_id()].pvec;
                 dU += (Vj-Vk)*dg(f[k]-f[j])/(1.0-g(f[k]-f[j]));
              }
           }
         }
      }

      U += sgd_gamma*dU;  // not thread-safe
      stat_vec[omp_get_thread_num()] += fabs(sgd_gamma*dU[0]);

      // compute smoothed MRR
      for(int j = 0; j < Ni; j++)
      {
        if (is_relevant(vertex.edge(j)))
        {
          objective += std::log(g(f[j]));
          for(int k = 0; k < Ni; k++)
          {
            if (is_relevant(vertex.edge(k)))
            {
              objective += std::log(1.0-g(f[k]-f[j]));
            }
          }
        }
      }
    }

    objective_vec[omp_get_thread_num()] += objective;
  }
};

//dump output to file
void output_sgd_result(std::string filename) {
  MMOutputter_mat<vertex_data> user_mat(filename + "_U.mm", 0, M, "This file contains SGD output matrix U. In each row D factors of a single user node.", latent_factors_inmem);
  MMOutputter_mat<vertex_data> item_mat(filename + "_V.mm", M, M+N,  "This file contains SGD  output matrix V. In each row D factors of a single item node.", latent_factors_inmem);

  logstream(LOG_INFO) << "CLiMF output files (in matrix market format): " << filename << "_U.mm" <<
                                                                           ", " << filename + "_V.mm " << std::endl;
}

// compute test prediction
float climf_predict(const vertex_data& user,
    const vertex_data& movie,
    const float rating,
    double & prediction,
    void * extra = NULL)
{
  prediction = g(dot(user.pvec,movie.pvec));  // this is actually a predicted reciprocal rank, not a rating
  return 0;  // as we have to return something
}

int main(int argc, const char ** argv) {
  //* GraphChi initialization will read the command line arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("climf-inmemory-factors");

  /* Basic arguments for application. NOTE: File will be automatically 'sharded'. */
  sgd_lambda    = get_option_float("sgd_lambda", 1e-3);
  sgd_gamma     = get_option_float("sgd_gamma", 1e-4);
  sgd_step_dec  = get_option_float("sgd_step_dec", 1.0);
  binary_relevance_thresh = get_option_float("binary_relevance_thresh", 0);
  halt_on_mrr_decrease = get_option_int("halt_on_mrr_decrease", 0);
  num_ratings = get_option_int("num_ratings", 10000); //number of top predictions over which we compute actual MRR
  verbose     = get_option_int("verbose", 0);
  debug       = get_option_int("debug", 0);

  parse_command_line_args();
  parse_implicit_command_line();

  /* Preprocess data if needed, or discover preprocess files */
  bool allow_square = false;
  int nshards = convert_matrixmarket<EdgeDataType>(training, 0, 0, 3, TRAINING, allow_square);
  init_feature_vectors<std::vector<vertex_data> >(M+N, latent_factors_inmem, !load_factors_from_file, 0.01);

  if (validation != ""){
    int vshards = convert_matrixmarket<EdgeDataType>(validation, 0, 0, 3, VALIDATION);
    init_mrr_engine<VertexDataType, EdgeDataType>(pvalidation_engine, vshards);
  }

  if (load_factors_from_file)
  {
    load_matrix_market_matrix(training + "_U.mm", 0, D);
    load_matrix_market_matrix(training + "_V.mm", M, D);
  }

  print_config();

  /* Run */
  SGDVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m);
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output latent factor matrices in matrix-market format */
  output_sgd_result(training);
  test_predictions(&climf_predict);

  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  
  return 0;
}

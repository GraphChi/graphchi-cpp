#ifndef _CLIMF_HPP__
#define _CLIMF_HPP__

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


struct vertex_data {
  vec pvec;     //storing the feature vector

  vertex_data()
  {
    pvec = zeros(D);
  }

  void set_val(int index, float val)
  {
    pvec[index] = val;
  }

  float get_val(int index) const
  {
    return pvec[index];
  }
};

typedef vertex_data VertexDataType;  // Vertices store the low-dimensional factorized feature vector
typedef float EdgeDataType;          // Edges store the rating/observed count for a user->item pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL;
graphchi_engine<VertexDataType, EdgeDataType> * pvalidation_engine = NULL;

std::vector<vertex_data> latent_factors_inmem;

double sgd_gamma = 1e-3;             // sgd step size
double sgd_step_dec = 0.9;           // sgd step decrement
double sgd_lambda = 1e-3;            // sgd regularization
double binary_relevance_thresh = 0;  // min rating for binary relevance
int halt_on_mrr_decrease = 0;        // whether to halt if smoothed MRR increases
int num_ratings = 10000;             // number of top predictions over which we compute actual MRR
vec objective_vec;                   // cumulative sum of smoothed MRR per thread
vec stat_vec;                        // verbose info about step size
int verbose;                         // additional output about step size
double training_objective;
double last_training_objective;
int debug;                           // printout more debug information

/* other relevant global args defined in common.hpp:
uint M;                    // number of users
uint N;                    // number of items
uint Me;                   // number of users (validation file)
uint Ne;                   // number of items (validation file)
uint Le;                   // number of ratings (validation file)
size_t L;                  // number of ratings (training file)
int D = 20;                // feature vector width
*/

// logistic function
double g(double x)
{
  double ret = 1.0 / (1.0 + std::exp(-x));

  if (std::isinf(ret) || std::isnan(ret))
  {
    logstream(LOG_FATAL) << "overflow in g()" << std::endl;
  }

  return ret;
}

// derivative of logistic function
double dg(double x)
{
  double ret = std::exp(x) / ((1.0 + std::exp(x)) * (1.0 + std::exp(x)));

  if (std::isinf(ret) || std::isnan(ret))
  {
    logstream(LOG_FATAL) << "overflow in dg()" << std::endl;
  }

  return ret;
}

bool is_relevant(graphchi_edge<EdgeDataType> * e)
{
  return e->get_data() >= binary_relevance_thresh;  // for some reason get_data() is non const :(
}

#endif


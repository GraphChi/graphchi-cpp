/*
 * Copyright (c) 2009 Carnegie Mellon University.
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://www.graphlab.ml.cmu.edu
 *
 */


/**
 * \file cgs_lda.cpp
 *
 * \brief This file contains a GraphLab based implementation of the
 * Collapsed Gibbs Sampler (CGS) for the Latent Dirichlet Allocation
 * (LDA) model.
 *
 * 
 *
 * \author Joseph Gonzalez, Diana Hu
 */

#include <vector>
#include <set>
#include <algorithm>

#include "util/atomic.hpp"

#include <boost/math/special_functions/gamma.hpp>
#include <vector>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/input_sequence.hpp>


// Global Types
// ============================================================================
typedef int count_type;


/**
 * \brief The factor type is used to store the counts of tokens in
 * each topic for words, documents, and assignments.
 *
 * Atomic counts are used because we violate the abstraction by
 * modifying adjacent vertex data on scatter.  As a consequence
 * multiple threads on the same machine may try to update the same
 * vertex data at the same time.  The graphlab::atomic type ensures
 * that multiple increments are serially consistent.
 */
typedef std::vector< graphchi::atomic<count_type> > factor_type;


/**
 * \brief We use the factor type in accumulators and so we define an
 * operator+=
 */
inline factor_type& operator+=(factor_type& lvalue,
                               const factor_type& rvalue) {
  if(!rvalue.empty()) {
    if(lvalue.empty()) lvalue = rvalue;
    else {
      for(size_t t = 0; t < lvalue.size(); ++t) lvalue[t] += rvalue[t];
    }
  }
  return lvalue;
} // end of operator +=







/**
 * \brief The latent topic id of a token is the smallest reasonable
 * type.
 */
typedef uint16_t topic_id_type;

// We require a null topic to represent the topic assignment for
// tokens that have not yet been assigned.
#define NULL_TOPIC (topic_id_type(-1))

#define NTOPICS 20


/**
 * \brief The assignment type is used on each edge to store the
 * assignments of each token.  There can be several occurrences of the
 * same word in a given document and so a vector is used to store the
 * assignments of each occurrence.
 */
typedef uint16_t assignment_type[NTOPICS];


// Global Variables
// ============================================================================

/**
 * \brief The alpha parameter determines the sparsity of topics for
 * each document.
 */
double ALPHA = 1;

/**
 * \brief the Beta parameter determines the sparsity of words in each
 * document.
 */
double BETA = 0.1;

/**
 * \brief the total number of topics to uses
 */

/**
 * \brief The total number of words in the dataset.
 */
size_t NWORDS = 0;

/**
 * \brief The total number of docs in the dataset.
 */
size_t NDOCS = 0;

/**
 * \brief The total number of tokens in the corpus
 */
size_t NTOKENS = 0;


/**
 * \brief The number of top words to display during execution (from
 * each topic).
 */
size_t TOPK = 5;

/**
 * \brief The interval to display topics during execution.
 */
size_t INTERVAL = 10;

/**
 * \brief The global variable storing the global topic count across
 * all machines.  This is maintained periodically using aggregation.
 */
factor_type GLOBAL_TOPIC_COUNT;

/**
 * \brief A dictionary of words used to print the top words during
 * execution.
 */
std::vector<std::string> DICTIONARY;

/**
 * \brief The maximum occurences allowed for an individual term-doc
 * pair. (edge data)
 */
size_t MAX_COUNT = 100;


/**
 * \brief The time to run until the first sample is taken.  If less
 * than zero then the sampler will run indefinitely.
 */
float BURNIN = -1;






// Graph Types
// ============================================================================

/**
 * \brief The vertex data represents each term and document in the
 * corpus and contains the counts of tokens in each topic.
 */
struct vertex_data {
  ///! The total number of updates
  uint32_t nupdates;
  ///! The total number of changes to adjacent tokens
  uint32_t nchanges;
  ///! The count of tokens in each topic
  factor_type factor;
  vertex_data() : nupdates(0), nchanges(0), factor(NTOPICS) { }
}; // end of vertex_data


/**
 * \brief The edge data represents the individual tokens (word,doc)
 * pairs and their assignment to topics.
 */
struct edge_data {
  ///! The number of changes on the last update
  uint16_t nchanges;
  ///! The assignment of all tokens
  assignment_type assignment;
  edge_data(size_t ntokens = 0) : nchanges(0) {
      for(int i=0; i<NTOPICS; i++) assignment[i] = 0;
  }
}; // end of edge_data

typedef graphlab::distributed_graph<vertex_data, edge_data> graph_type;

static void parse(edge_data &x, const char * s) {
    size_t count = atol(s);
    count = std::min(count, MAX_COUNT);
    x = (edge_data(count));
}

/**
 * \brief Edge data parser used in graph.load_json
 *
 * Make sure that the edge file list
 * has docids from -2 to -(total #docid) and wordids 0 to (total #words -1)
 */
bool eparser(edge_data& ed, const std::string& line){
  const int BASE = 10;
  char* next_char_ptr = NULL;
  size_t count = strtoul(line.c_str(), &next_char_ptr, BASE);
  if(next_char_ptr ==NULL) return false;

  //threshold count
  count = std::min(count, MAX_COUNT);
  ed = (edge_data(count));
  return true;
}

/**
 * \brief Vertex data parser used in graph.load_json
 */
bool vparser(vertex_data& vd, const std::string& line){
  vd = vertex_data();
  return true;
}








/**
 * \brief Determine if the given vertex is a word vertex or a doc
 * vertex.
 *
 * For simplicity we connect docs --> words and therefore if a vertex
 * has in edges then it is a word.
 */
inline bool is_word(const graph_type::vertex_type& vertex) {
  return vertex.num_in_edges() > 0 ? 1 : 0;
}


/**
 * \brief Determine if the given vertex is a doc vertex
 *
 * For simplicity we connect docs --> words and therefore if a vertex
 * has out edges then it is a doc
 */
inline bool is_doc(const graph_type::vertex_type& vertex) {
  return vertex.num_out_edges() > 0 ? 1 : 0;
}

/**
 * \brief return the number of tokens on a particular edge.
 */
inline size_t count_tokens(const graph_type::edge_type& edge) {
  return edge.data().assignment.size();
}


/**
 * \brief Get the other vertex in the edge.
 */
inline graph_type::vertex_type
get_other_vertex(const graph_type::edge_type& edge,
                 const graph_type::vertex_type& vertex) {
  return vertex.id() == edge.source().id()? edge.target() : edge.source();
}



// ========================================================
// The Collapsed Gibbs Sampler Function



/**
 * \brief The gather type for the collapsed Gibbs sampler is used to
 * collect the topic counts on adjacent edges so that the apply
 * function can compute the correct topic counts for the center
 * vertex.
 *
 */
struct gather_type {
  factor_type factor;
  uint32_t nchanges;
  gather_type() : nchanges(0) { };
  gather_type(uint32_t nchanges) : factor(NTOPICS), nchanges(nchanges) { };
  gather_type& operator+=(const gather_type& other) {
    factor += other.factor;
    nchanges += other.nchanges;
    return *this;
  }
}; // end of gather type







/**
 * \brief The collapsed Gibbs sampler vertex program updates the topic
 * counts for the center vertex and then draws new topic assignments
 * for each edge durring the scatter phase.
 * 
 */
class cgs_lda_vertex_program :
  public graphlab::ivertex_program<graph_type, gather_type> {
public:

  /**
   * \brief At termination we want to disable sampling to allow the
   * correct final counts to be computed.
   */
  static bool DISABLE_SAMPLING; 

  /** \brief gather on all edges */
  edge_dir_type gather_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    return graphlab::ALL_EDGES;
  } // end of gather_edges

  /**
   * \brief Collect the current topic count on each edge.
   */
  gather_type gather(icontext_type& context, const vertex_type& vertex,
                     edge_type& edge) const {
    gather_type ret(edge.data().nchanges);
    const assignment_type& assignment = edge.data().assignment;
    foreach(topic_id_type asg, assignment) {
      if(asg != NULL_TOPIC) ++ret.factor[asg];
    }
    return ret;
  } // end of gather


  /**
   * \brief Update the topic count for the center vertex.  This
   * ensures that the center vertex has the correct topic count before
   * resampling the topics for each token along each edge.
   */
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& sum) {
    const size_t num_neighbors = vertex.num_in_edges() + vertex.num_out_edges();
    ASSERT_GT(num_neighbors, 0);
    // There should be no new edge data since the vertex program has been cleared
    vertex_data& vdata = vertex.data();
    ASSERT_EQ(sum.factor.size(), NTOPICS);
    ASSERT_EQ(vdata.factor.size(), NTOPICS);
    vdata.nupdates++;
    vdata.nchanges = sum.nchanges;
    vdata.factor = sum.factor;
  } // end of apply


  /**
   * \brief Scatter on all edges if the computation is on-going.
   * Computation stops after bunrin or when disable sampling is set to
   * true.
   */
  edge_dir_type scatter_edges(icontext_type& context,
                              const vertex_type& vertex) const {
    return (DISABLE_SAMPLING || (BURNIN > 0 && context.elapsed_seconds() > BURNIN))? 
      graphlab::NO_EDGES : graphlab::ALL_EDGES;
  }; // end of scatter edges


  /**
   * \brief Draw new topic assignments for each edge token.
   *
   * Note that we exploit the GraphLab caching model here by DIRECTLY
   * modifying the topic counts of adjacent vertices.  Making the
   * changes immediately visible to any adjacent vertex programs
   * running on the same machine.  However, these changes will be
   * overwritten during the apply step and are only used to accelerate
   * sampling.  This is a potentially dangerous violation of the
   * abstraction and should be taken with caution.  In our case all
   * vertex topic counts are preallocated and atomic operations are
   * used.  In addition during the sampling phase we must be careful
   * to guard against potentially negative temporary counts.
   */
  void scatter(icontext_type& context, const vertex_type& vertex,
               edge_type& edge) const {
    factor_type& doc_topic_count =  is_doc(edge.source()) ?
      edge.source().data().factor : edge.target().data().factor;
    factor_type& word_topic_count = is_word(edge.source()) ?
      edge.source().data().factor : edge.target().data().factor;
    ASSERT_EQ(doc_topic_count.size(), NTOPICS);
    ASSERT_EQ(word_topic_count.size(), NTOPICS);
    // run the actual gibbs sampling
    std::vector<double> prob(NTOPICS);
    assignment_type& assignment = edge.data().assignment;
    edge.data().nchanges = 0;
    foreach(topic_id_type& asg, assignment) {
      const topic_id_type old_asg = asg;
      if(asg != NULL_TOPIC) { // construct the cavity
        --doc_topic_count[asg];
        --word_topic_count[asg];
        --GLOBAL_TOPIC_COUNT[asg];
      }
      for(size_t t = 0; t < NTOPICS; ++t) {
        const double n_dt =
          std::max(count_type(doc_topic_count[t]), count_type(0));
        const double n_wt =
          std::max(count_type(word_topic_count[t]), count_type(0));
        const double n_t  =
          std::max(count_type(GLOBAL_TOPIC_COUNT[t]), count_type(0));
        prob[t] = (ALPHA + n_dt) * (BETA + n_wt) / (BETA * NWORDS + n_t);
      }
      asg = graphlab::random::multinomial(prob);
      // asg = std::max_element(prob.begin(), prob.end()) - prob.begin();
      ++doc_topic_count[asg];
      ++word_topic_count[asg];
      ++GLOBAL_TOPIC_COUNT[asg];
      if(asg != old_asg) {
        ++edge.data().nchanges;
      }
    } // End of loop over each token
    // singla the other vertex
    context.signal(get_other_vertex(edge, vertex));
  } // end of scatter function

}; // end of cgs_lda_vertex_program


bool cgs_lda_vertex_program::DISABLE_SAMPLING = false;


/**
 * \brief The icontext type associated with the cgs_lda_vertex program
 * is needed for all aggregators.
 */
typedef cgs_lda_vertex_program::icontext_type icontext_type;


// ========================================================
// Aggregators


/**
 * \brief The topk aggregator is used to periodically compute and
 * display the topk most common words in each topic.
 *
 * The number of words is determined by the global variable \ref TOPK
 * and the interval is determined by the global variable \ref INTERVAL.
 *
 */
class topk_aggregator {
  typedef std::pair<float, graphlab::vertex_id_type> cw_pair_type;
private:
  std::vector< std::set<cw_pair_type> > top_words;
  size_t nchanges, nupdates;
public:
  topk_aggregator(size_t nchanges = 0, size_t nupdates = 0) :
    nchanges(nchanges), nupdates(nupdates) { }

  topk_aggregator& operator+=(const topk_aggregator& other) {
    nchanges += other.nchanges;
    nupdates += other.nupdates;
    if(other.top_words.empty()) return *this;
    if(top_words.empty()) top_words.resize(NTOPICS);
    for(size_t i = 0; i < top_words.size(); ++i) {
      // Merge the topk
      top_words[i].insert(other.top_words[i].begin(),
                          other.top_words[i].end());
      // Remove excess elements
      while(top_words[i].size() > TOPK)
        top_words[i].erase(top_words[i].begin());
    }
    return *this;
  } // end of operator +=

  static topk_aggregator map(icontext_type& context,
                             const graph_type::vertex_type& vertex) {
    topk_aggregator ret_value;
    const vertex_data& vdata = vertex.data();
    ret_value.nchanges = vdata.nchanges;
    ret_value.nupdates = vdata.nupdates;
    if(is_word(vertex)) {
      const graphlab::vertex_id_type wordid = vertex.id();
      ret_value.top_words.resize(vdata.factor.size());
      for(size_t i = 0; i < vdata.factor.size(); ++i) {
        const cw_pair_type pair(vdata.factor[i], wordid);
        ret_value.top_words[i].insert(pair);
      }
    }
    return ret_value;
  } // end of map function


  static void finalize(icontext_type& context,
                       const topk_aggregator& total) {
    if(context.procid() != 0) return;
     for(size_t i = 0; i < total.top_words.size(); ++i) {
      std::cout << "Topic " << i << ": ";
      rev_foreach(cw_pair_type pair, total.top_words[i])  {
    
        std::cout << DICTIONARY[pair.second]
                  << "(" << pair.first << ")" << ", ";
      }
      
      std::cout << std::endl;
    }
   
    std::cout << "\nNumber of token changes: " << total.nchanges << std::endl;
    std::cout << "\nNumber of updates:       " << total.nupdates << std::endl;
  } // end of finalize
}; // end of topk_aggregator struct



/**
 * \brief The global counts aggregator computes the total number of
 * tokens in each topic across all words and documents and then
 * updates the \ref GLOBAL_TOPIC_COUNT variable.
 *
 */
struct global_counts_aggregator {
  typedef graph_type::vertex_type vertex_type;
  static factor_type map(icontext_type& context, const vertex_type& vertex) {
    return vertex.data().factor;
  } // end of map function

  static void finalize(icontext_type& context, const factor_type& total) {
    size_t sum = 0;
    for(size_t t = 0; t < total.size(); ++t) {
      GLOBAL_TOPIC_COUNT[t] =
        std::max(count_type(total[t]/2), count_type(0));
      sum += GLOBAL_TOPIC_COUNT[t];
    }
    context.cout() << "Total Tokens: " << sum << std::endl;
  } // end of finalize
}; // end of global_counts_aggregator struct



/**
 * \brief The Likelihood aggregators maintains the current estimate of
 * the log-likelihood of the current token assignments.
 *
 *  llik_words_given_topics = ...
 *    ntopics * (gammaln(nwords * beta) - nwords * gammaln(beta)) - ...
 *    sum_t(gammaln( n_t + nwords * beta)) +
 *    sum_w(sum_t(gammaln(n_wt + beta)));
 *
 *  llik_topics = ...
 *    ndocs * (gammaln(ntopics * alpha) - ntopics * gammaln(alpha)) + ...
 *    sum_d(sum_t(gammaln(n_td + alpha)) - gammaln(sum_t(n_td) + ntopics * alpha));
 */
class likelihood_aggregator : public graphlab::IS_POD_TYPE {
  typedef graph_type::vertex_type vertex_type;
  double lik_words_given_topics;
  double lik_topics;
public:
  likelihood_aggregator() : lik_words_given_topics(0), lik_topics(0) { }

  likelihood_aggregator& operator+=(const likelihood_aggregator& other) {
    lik_words_given_topics += other.lik_words_given_topics;
    lik_topics += other.lik_topics;
    return *this;
  } // end of operator +=

  static likelihood_aggregator
  map(icontext_type& context, const vertex_type& vertex) {
    using boost::math::lgamma;
    const factor_type& factor = vertex.data().factor;
    ASSERT_EQ(factor.size(), NTOPICS);
   likelihood_aggregator ret;
    if(is_word(vertex)) {
      for(size_t t = 0; t < NTOPICS; ++t) {
        const double value = std::max(count_type(factor[t]), count_type(0));
        ret.lik_words_given_topics += lgamma(value + BETA);
      }
    } else {  ASSERT_TRUE(is_doc(vertex));
      double ntokens_in_doc = 0;
      for(size_t t = 0; t < NTOPICS; ++t) {
        const double value = std::max(count_type(factor[t]), count_type(0));
        ret.lik_topics += lgamma(value + ALPHA);
        ntokens_in_doc += factor[t];
      }
      ret.lik_topics -= lgamma(ntokens_in_doc + NTOPICS * ALPHA);
    }
    return ret;
  } // end of map function

  static void finalize(icontext_type& context, const likelihood_aggregator& total) {
    using boost::math::lgamma;
    // Address the global sum terms
    double denominator = 0;
    for(size_t t = 0; t < NTOPICS; ++t) {
      denominator += lgamma(GLOBAL_TOPIC_COUNT[t] + NWORDS * BETA);
    } // end of for loop

    const double lik_words_given_topics =
      NTOPICS * (lgamma(NWORDS * BETA) - NWORDS * lgamma(BETA)) -
      denominator + total.lik_words_given_topics;

    const double lik_topics =
      NDOCS * (lgamma(NTOPICS * ALPHA) - NTOPICS * lgamma(ALPHA)) +
      total.lik_topics;

    const double lik = lik_words_given_topics + lik_topics;
    context.cout() << "Likelihood: " << lik << std::endl;
  } // end of finalize
}; // end of likelihood_aggregator struct



/**
 * \brief The selective signal functions are used to signal only the
 * vertices corresponding to words or documents.  This is done by
 * using the iengine::map_reduce_vertices function.
 */
struct signal_only {
  /**
   * \brief Signal only the document vertices and skip the word
   * vertices.
   */ 
  static graphlab::empty
  docs(icontext_type& context, const graph_type::vertex_type& vertex) {
    if(is_doc(vertex)) context.signal(vertex);
    return graphlab::empty();
  } // end of signal_docs
 
 /**
  * \brief Signal only the word vertices and skip the document
  * vertices.
  */
  static graphlab::empty
  words(icontext_type& context, const graph_type::vertex_type& vertex) {
    if(is_word(vertex)) context.signal(vertex);
    return graphlab::empty();
  } // end of signal_words
}; // end of selective_only






/**
 * \brief Load the dictionary global variable from the file containing
 * the terms (one term per line).
 *
 * Note that while graphs can be loaded from multiple files the
 * dictionary must be in a single file.  The dictionary is loaded
 * entirely into memory and used to display word clouds and the top
 * terms in each topic.
 *
 * \param [in] fname the file containing the dictionary data.  The
 * data can be located on HDFS and can also be gzipped (must end in
 * ".gz").
 * 
 */
bool load_dictionary(const std::string& fname)  {
  // std::cout << "staring load on: "
  //           << graphlab::get_local_ip_as_str() << std::endl;
  const bool gzip = boost::ends_with(fname, ".gz");
  // test to see if the graph_dir is an hadoop path
 
    std::cout << "opening: " << fname << std::endl;
    std::ifstream in_file(fname.c_str(),
                          std::ios_base::in | std::ios_base::binary);
    boost::iostreams::filtering_stream<boost::iostreams::input> fin;
    fin.push(in_file);
    if(!fin.good() || !fin.good()) {
      logstream(LOG_ERROR) << "Error loading dictionary: "
                           << fname << std::endl;
      return false;
    }
    std::string term;
    std::cout << "Loooping" << std::endl;
    while(std::getline(fin, term).good()) DICTIONARY.push_back(term);
    fin.pop();
    in_file.close();
    // std::cout << "Finished load on: "
  //           << graphlab::get_local_ip_as_str() << std::endl;
  std::cout << "Dictionary Size: " << DICTIONARY.size() << std::endl;
  return true;
} // end of load dictionary




struct count_saver {
  bool save_words;
  count_saver(bool save_words) : save_words(save_words) { }
  typedef graph_type::vertex_type vertex_type;
  typedef graph_type::edge_type   edge_type;
  std::string save_vertex(const vertex_type& vertex) const {
    // Skip saving vertex data if the vertex type is not consistent
    // with the save type
    if((save_words && is_doc(vertex)) ||
       (!save_words && is_word(vertex))) return "";
    // Proceed to save
    std::stringstream strm;
    if(save_words) {
      const graphlab::vertex_id_type vid = vertex.id();
      strm << vid << '\t';
    } else { // save documents
      const graphlab::vertex_id_type vid = (-vertex.id()) - 2;
      strm << vid << '\t';
    }
    const factor_type& factor = vertex.data().factor;
    for(size_t i = 0; i < factor.size(); ++i) { 
      strm << factor[i];
      if(i+1 < factor.size()) strm << '\t';
    }
    strm << '\n';
    return strm.str();
  }
  std::string save_edge(const edge_type& edge) const {
    return ""; //nop
  }
}; // end of prediction_saver










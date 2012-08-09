//
//  cgs_lda.cpp
//  graphchi_xcode
//
//  Created by Aapo Kyrola on 8/8/12.
//
//


#include <string>
#include <algorithm>

#include "graphchi_basic_includes.hpp"
#include "api/graphlab2_1_GAS_api/graphlab.hpp"

#include "cgs_lda_vertexprogram.hpp"

using namespace graphchi;
using namespace graphlab;

int main(int argc, const char ** argv) {
    /* GraphChi initialization will read the command line
     arguments and the configuration file. */
    graphchi_init(argc, argv);
    
    /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
    metrics m("LDA-graphlab");
    
    
    /* Basic arguments for application. NOTE: File will be automatically 'sharded'. */
    std::string filename = get_option_string("file");    // Base filename
    int niters           = get_option_int("niters", 4);  // Number of iterations
    
    /* Preprocess data if needed, or discover preprocess files */
    int nshards = convert_if_notexists<edge_data>(filename, get_option_string("nshards", "auto"));
    
    /* Run */
    std::vector<vertex_data> * vertices =
    run_graphlab_vertexprogram<cgs_lda_vertex_program>(filename, nshards, niters, false, m, false, false);
     
    /* TODO: write output latent matrices */
    delete vertices;
    /* Report execution metrics */
    metrics_report(m);
    return 0;
}


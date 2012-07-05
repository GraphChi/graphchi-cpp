
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
 * Simple smoke test for the bulk synchronous functional api.
 */

#define RANDOMRESETPROB 0.15

#include <string>
#include <fstream>
#include <cmath>

#include "util/cmdopts.hpp"
#include "api/graphchi_context.hpp"
#include "api/graph_objects.hpp"
#include "api/ischeduler.hpp"
#include "api/functional/functional_api.hpp"
#include "metrics/metrics.hpp"
#include "metrics/reps/basic_reporter.hpp"
#include "util/toplist.hpp"

using namespace graphchi;

struct smoketest_program : public functional_kernel<int, int> {
    
    /* Initial value - on first iteration */
    int initial_value(graphchi_context &info, vertex_info& myvertex) {
        return 0;
    }
    
    /* Called before first "gather" */
    int reset() {
        return 0;
    }
    
    // Note: Unweighted version, edge value should also be passed
    // "Gather"
    int op_neighborval(graphchi_context &info, vertex_info& myvertex, vid_t nbid, int nbval) {
        assert(nbval == (int) info.iteration - 1);
        return nbval;
    }
    
    // "Sum"
    int plus(int curval, int toadd) {
        assert(curval == 0 || toadd == curval);
        return toadd;
    }
    
    // "Apply"
    int compute_vertexvalue(graphchi_context &ginfo, vertex_info& myvertex, int nbvalsum) {
        return ginfo.iteration;
    }
    
    // "Scatter
    int value_to_neighbor(graphchi_context &info, vertex_info& myvertex, vid_t nbid, int myval) {
        assert(myval == (int) info.iteration);
        return myval;
    }
    
}; 

int main(int argc, const char ** argv) {
    graphchi_init(argc, argv);
    metrics m("test-functional");
    
    std::string filename = get_option_string("file");
    int niters = get_option_int("niters", 5);
    std::string mode = get_option_string("mode", "semisync");
    
    logstream(LOG_INFO) << "Running bulk sync smoke test." << std::endl;
    run_functional_unweighted_synchronous<smoketest_program>(filename, niters, m);
    
    logstream(LOG_INFO) << "Smoketest passed successfully! Your system is working!" << std::endl;
    return 0;
}





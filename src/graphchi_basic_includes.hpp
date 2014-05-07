

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
 * This header includes all the main headers needed for a GraphChi
 * program.
 */


#ifndef GRAPHCHI_DEF_ALLBASIC_INCLUDES
#define GRAPHCHI_DEF_ALLBASIC_INCLUDES

#include <omp.h>
#include <sstream>

#include "api/chifilenames.hpp"
#include "api/graphchi_context.hpp"
#include "api/graphchi_program.hpp"
#include "api/graph_objects.hpp"
#include "api/ischeduler.hpp"
#include "api/vertex_aggregator.hpp"

#include "engine/graphchi_engine.hpp"

#include "logger/logger.hpp"

#include "metrics/metrics.hpp"
#include "metrics/reps/basic_reporter.hpp"
#include "metrics/reps/file_reporter.hpp"
#include "metrics/reps/html_reporter.hpp"

#include "preprocessing/conversions.hpp"

#include "util/cmdopts.hpp"


namespace graphchi {
        
    /**
      * Helper for metrics.
      */
    static VARIABLE_IS_NOT_USED void metrics_report(metrics &m);
    static VARIABLE_IS_NOT_USED void metrics_report(metrics &m) {
        std::string reporters = get_option_string("metrics.reporter", "console");
        char * creps = (char*)reporters.c_str();
        const char * delims = ",";
        char * t = strtok(creps, delims);

        while(t != NULL) {            
            std::string repname(t);
            if (repname == "basic" || repname == "console") {
                basic_reporter rep;
                m.report(rep);
            } else if (repname == "file") {
                file_reporter rep(get_option_string("metrics.reporter.filename", "metrics.txt"));
                m.report(rep);
            } else if (repname == "html") {
                html_reporter rep(get_option_string("metrics.reporter.htmlfile", "metrics.html"));
                m.report(rep);
            } else {
                logstream(LOG_WARNING) << "Could not find metrics reporter with name [" << repname << "], ignoring." << std::endl;
            }
            t = strtok(NULL, delims);
        }
        
        
    }
    
    
};

#endif

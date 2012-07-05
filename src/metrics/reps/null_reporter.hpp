
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
 * Empty metrics reporter.
 */
#ifndef GRAPHLAB_NULL_REPORTER
#define GRAPHLAB_NULL_REPORTER

#include "metrics/metrics.hpp"

/**
 * Simple metrics reporter that dumps metrics to
 * standard output.
 */

namespace graphchi {

  class null_reporter : public imetrics_reporter {

  public:
      virtual ~null_reporter();
    virtual void do_report(std::string name, std::string ident, std::map<std::string, metrics_entry> & entries) {
    }

  };

}



#endif


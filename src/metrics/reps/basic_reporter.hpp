
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
 * Simple metrics reporter that dumps metrics to
 * standard output. 
 */



#ifndef GRAPHCHI_BASIC_REPORTER
#define GRAPHCHI_BASIC_REPORTER

#include <iostream>
#include <map>

#include "metrics/metrics.hpp"

/**
 * Simple metrics reporter that dumps metrics to
 * standard output.
 */

namespace graphchi {

  class basic_reporter : public imetrics_reporter {
    
  public:
    virtual ~basic_reporter() {}
    virtual void do_report(std::string name, std::string ident, std::map<std::string, metrics_entry> & entries) {
      // TODO: use reporters
      if (ident != name) {
        std::cout << std::endl <<  " === REPORT FOR " << name << "(" << ident << ") ===" << std::endl;
      } else {
        std::cout << std::endl << " === REPORT FOR " << name << " ===" << std::endl;
      }
                
      // First write numeral, then timings, then string entries
      for(int round=0; round<4; round++) { 
        std::map<std::string, metrics_entry>::iterator it;
        int c = 0;
                    
        for(it = entries.begin(); it != entries.end(); ++it) {
          metrics_entry ent = it->second;
          switch(ent.valtype) {
          case REAL:
          case INTEGER:
            if (round == 0) {
              if (c++ == 0) std::cout << "[Numeric]" << std::endl; 
              std::cout << it->first << ":\t\t";
              if (ent.count > 1) {
                std::cout << ent.value << "\t(count: " << ent.count <<  ", min: " << ent.minvalue <<
                  ", max: " << ent.maxvalue << ", avg: " 
                          << ent.cumvalue/(double)ent.count << ")" << std::endl;
              } else {
                std::cout << ent.value << std::endl;
              }
            }
            break;
          case TIME:
            if (round == 1) {
              if (c++ == 0) std::cout << "[Timings]" << std::endl;
              std::cout << it->first << ":\t\t";
              if (ent.count>1) {
                std::cout << ent.value << "s\t (count: " << ent.count << ", min: " << ent.minvalue <<
                  "s, " << "max: " << ent.maxvalue << ", avg: " 
                          << ent.cumvalue/(double)ent.count << "s)" << std::endl;
              } else {
                std::cout << ent.value << " s" << std::endl;
              }
            }
            break;
          case STRING:
            if (round == 2) {
              if (c++ == 0) std::cout << "[Other]" << std::endl;
              std::cout << it->first << ":\t";
              std::cout << ent.stringval << std::endl;
            }
            break;
          case VECTOR:
            if (round == 3) {
              if (c++ == 0) std::cout << "[Numeric]" << std::endl; 
              std::cout << it->first << ":\t\t";
              if (ent.count > 1) {
                std::cout << ent.value << "\t(count: " << ent.count <<  ", min: " << ent.minvalue <<
                  ", max: " << ent.maxvalue << ", avg: " 
                          << ent.cumvalue/(double)ent.count << ")" << std::endl;
              } else {
                std::cout << ent.value << std::endl;
              }
              std::cout << it->first << ".values:\t\t";
              for(size_t j=0; j<ent.v.size(); j++) std::cout << ent.v[j] << ",";
              std::cout << std::endl;
            }
            break;
          }
        }
      }
    };
        
  };
    
};



#endif



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
 * File metrics reporter.
 */


#ifndef DEF_GRAPHCHI_FILE_REPORTER
#define DEF_GRAPHCHI_FILE_REPORTER

#include <fstream>
#include <cstdio>

#include "metrics/metrics.hpp"
#include "util/cmdopts.hpp"

 

namespace graphchi {

  class file_reporter : public imetrics_reporter {
  private:
    file_reporter() {}
        
    std::string filename;
    FILE * f;
  public:
            
        
    file_reporter(std::string fname) : filename(fname) {
      // Create new file
        f = fopen(fname.c_str(), "w");
        assert(f != NULL);
    }
      
      virtual ~file_reporter() {}
            
      virtual void do_report(std::string name, std::string ident, std::map<std::string, metrics_entry> & entries) {
          if (ident != name) {
              fprintf(f, "[%s:%s]\n", name.c_str(), ident.c_str());
          } else {
              fprintf(f, "[%s]\n", name.c_str());
          }
          std::map<std::string, metrics_entry>::iterator it;
          
          for(it = entries.begin(); it != entries.end(); ++it) {
              metrics_entry ent = it->second;
              switch(ent.valtype) {
                  case INTEGER:
                      
                      fprintf(f, "%s.%s=%ld\n", ident.c_str(), it->first.c_str(), (long int) (ent.value));
                      fprintf(f, "%s.%s.count=%lu\n", ident.c_str(), it->first.c_str(), ent.count);
                      fprintf(f, "%s.%s.min=%ld\n", ident.c_str(), it->first.c_str(), (long int) (ent.minvalue));
                      fprintf(f, "%s.%s.max=%ld\n", ident.c_str(), it->first.c_str(), (long int) (ent.maxvalue));
                      fprintf(f, "%s.%s.avg=%lf\n", ident.c_str(), it->first.c_str(), ent.cumvalue/ent.count);
                      break;
                  case REAL:
                  case TIME:
                      fprintf(f, "%s.%s=%lf\n", ident.c_str(), it->first.c_str(),  (ent.value));
                      fprintf(f, "%s.%s.count=%lu\n", ident.c_str(), it->first.c_str(), ent.count);
                      fprintf(f, "%s.%s.min=%lf\n", ident.c_str(), it->first.c_str(),  (ent.minvalue));
                      fprintf(f, "%s.%s.max=%lf\n", ident.c_str(), it->first.c_str(),  (ent.maxvalue));
                      fprintf(f, "%s.%s.avg=%lf\n", ident.c_str(), it->first.c_str(), ent.cumvalue/ent.count);
                      break;
                  case STRING:
                      fprintf(f, "%s.%s=%s\n", ident.c_str(), it->first.c_str(), it->second.stringval.c_str());                                
                      break;
                  case VECTOR:
                      break;
              }
          }
          
          fflush(f);        
          fclose(f);
          
        // Following code used only for research purposes.
          if (get_option_int("metrics.insert_to_db", 0) == 1) {
              std::string cmd = "python2.7 benchtodb.py " + filename;
              int err = system(cmd.c_str());
              if (err != 0) {
                std::cout << "Error running the python script." << std::endl;
              }
          }
      };
        
  };
    
};



#endif



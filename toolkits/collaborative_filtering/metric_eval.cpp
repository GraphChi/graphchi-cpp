/**  
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
 *  Written by Danny Bickson, CMU
 *  Utility for computing ranking metrics - comparing between recommendations and actual
 *  ratings in test set.
 *  */


#include <cstdio>
#include <map>
#include <iostream>
#include <omp.h>
#include <assert.h>
#include "graphchi_basic_includes.hpp"
#include "timer.hpp"
#include "util.hpp"
#include "eigen_wrapper.hpp"
#include "metrics.hpp"

using namespace std;
using namespace graphchi;

timer mytime;
int K = 10;
int max_per_row = 1000;
std::string training, test;
double avg_train_size = 0;
double avg_test_size = 0;
//non word tokens that will be removed in the parsing
//it is possible to add additional special characters or remove ones you want to keep
const char spaces[] = {" \r\n\t;:"};

void get_one_line(FILE * pfile, int & index, vec & values, int & pos){ 
  
    char * saveptr = NULL, * linebuf = NULL;
    size_t linesize = 0;

    int rc = getline(&linebuf, &linesize, pfile);
    if (rc < 1){
       index = -1;
       return;
    }

    pos = 0;
    bool first_time = true; 
    while(true){
      //find from
      char *pch = strtok_r(first_time ? linebuf : NULL, spaces, &saveptr);
      if (!pch){
        return;
      }
      float val = atof(pch);
      if (first_time){
        index = (int)val;
        first_time = false;
      }
      else {
        assert(pos < values.size());
        values[pos] = val;
        pos++;
      }
    }
}

void eval_metrics(){    
  in_file trf(training);
  in_file testt(test);

  vec train_vec = zeros(max_per_row);
  vec test_vec = zeros(max_per_row);
  size_t line = 0;
  int train_index = 0, test_index = 0;
  double ap = 0;
  int train_size =0, test_size = 0;
  while(true){
    get_one_line(trf.outf, train_index, train_vec, train_size);
    get_one_line(testt.outf, test_index, test_vec, test_size);
    if (train_index == -1 || test_index == -1)
       break;
    while (test_index < train_index && test_index != -1){
      logstream(LOG_WARNING)<<"Skipping over test index: " << test_index << " train: " << train_index << std::endl;
      get_one_line(testt.outf, test_index, test_vec, test_size);
    }
    while (train_index < test_index && train_index != -1){
      logstream(LOG_WARNING)<<"Skipping over train. test index: " << test_index << " train: " << train_index << std::endl;
      get_one_line(trf.outf, train_index, train_vec, train_size);
    }
 
    if (train_index == test_index){
      avg_train_size += train_size;
      avg_test_size += test_size;
      ap+= average_precision_at_k(train_vec, train_size, test_vec, test_size, K);
      line++;
    }
    else {
       logstream(LOG_WARNING)<<"Problem parsing file, got to train index: " << train_index << " test_index: " << test_index << std::endl;
       break;
    }
    if (line % 100000 == 0)
      logstream(LOG_INFO)<<mytime.current_time() <<" Finished evaluating " << line << " instances. " << std::endl;
  }

  logstream(LOG_INFO)<<"Computed AP@" << K << " metric: " << ap/(double)line << std::endl;
  logstream(LOG_INFO)<<"Total compared: " << line << std::endl;
  logstream(LOG_INFO)<<"Avg test length: " << avg_test_size / line << std::endl;
  logstream(LOG_INFO)<<"Avg train length: " << avg_train_size / line << std::endl;
}


int main(int argc,  const char *argv[]) {

  logstream(LOG_WARNING)<<"GraphChi parsers library is written by Danny Bickson (c). Send any "
    " comments or bug reports to danny.bickson@gmail.com " << std::endl;
  global_logger().set_log_level(LOG_INFO);
  global_logger().set_log_to_console(true);

  graphchi_init(argc, argv);

  K = get_option_int("K", K);
  if (K < 1)
    logstream(LOG_FATAL)<<"Number of top elements (--K=) should be >= 1"<<std::endl;

  omp_set_num_threads(get_option_int("ncpus", 1));
  mytime.start();

  training = get_option_string("training");
  test = get_option_string("test");

  eval_metrics();
  std::cout << "Finished in " << mytime.current_time() << std::endl;

  return 0;
}




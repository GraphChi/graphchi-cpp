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
 *
 *  This program takes a file in the following graph format format:
 *  1 61 0.9
 *  1 8 0.6
 *  2 1 1.2
 *  2 2 0.9
 *  2 3 0.4 
 *  2 18 0.13
 *  2 21 0.11
 *
 *  Namely, a text file where the first field is a "from" field (uint >=1), second field is a "to" field (uint > = 1)
 *  and a numeric value. Note that the file is assumed to be sorted by the "from" field and then the "val" field.
 *`
 *  The output file is top K values for each entries (or less if there where fewer in the input file)
 *  In the abive example, assuming K = 2 the ouput will be:
 *  1 61 8
 *  2 1 2
 *  Namely, in each line the first is the "from" frield and the rest are the "to" top fields.
 *
 *  A second output is 
 *  1 0.9 0.6
 *  2 1.2 0.9
 *  */


#include <cstdio>
#include <map>
#include <iostream>
#include <map>
#include <omp.h>
#include <assert.h>
#include "graphchi_basic_includes.hpp"
#include "../collaborative_filtering/timer.hpp"
#include "../collaborative_filtering/util.hpp"
#include "../collaborative_filtering/eigen_wrapper.hpp"

using namespace std;
using namespace graphchi;

bool debug = false;
timer mytime;
size_t lines;
unsigned long long total_lines = 0;
string dir;
string outdir;
std::vector<std::string> in_files;
int col = 3;
int K = 10;
int has_value = 1;
//non word tokens that will be removed in the parsing
//it is possible to add additional special characters or remove ones you want to keep
const char spaces[] = {" \r\n\t;:"};

void dump_entry(uint from, vec & to_vec, vec & vals_vec, int pos, FILE * out_ids, FILE * out_vals){
  assert(pos <= K);
  fprintf(out_ids, "%u ", from);
  if (has_value)
    fprintf(out_vals, "%u ", from);
  for (int i=0; i< pos; i++){
    fprintf(out_ids, "%u ", (uint)to_vec[i]);
    if (has_value)
      fprintf(out_vals, "%12.6g ", vals_vec[i]);
  }
  fprintf(out_ids, "\n");
  if (has_value) 
     fprintf(out_vals, "\n");
}
void parse(int i){    
  in_file fin(in_files[i]);
  out_file ids_out((outdir + in_files[i] + ".ids"));
  out_file val_out((outdir + in_files[i] + ".vals"));


  vec to_vec = zeros(K);
  vec val_vec = zeros(K);
  int pos = 0;
  size_t linesize = 0;
  char * saveptr = NULL, * linebuf = NULL;
  size_t line = 1;
  uint from = 0, to = 0;
  uint last_from = 0;
  while(true){
    int rc = getline(&linebuf, &linesize, fin.outf);
    if (rc < 1)
      break;

    //char * line_to_free = linebuf;

    //find from
    char *pch = strtok_r(linebuf, spaces, &saveptr);
    if (!pch || atoi(pch)<= 0) 
      logstream(LOG_FATAL) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; 
    from = atoi(pch);

    //find to
    pch = strtok_r(NULL, spaces ,&saveptr);
    if (!pch || atoi(pch)<=0)
      logstream(LOG_FATAL) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl;
    to = atoi(pch);

    //find val
    if (has_value){
    pch = strtok_r(NULL, spaces ,&saveptr);
    if (!pch) 
      logstream(LOG_FATAL) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; 
    }

    if (from != last_from){
      // fprintf(fout.outf, "%u %u %g\n", last_from, last_to, total);
      if (last_from != 0){
        dump_entry(last_from, to_vec, val_vec, pos, ids_out.outf, val_out.outf);
        pos = 0;
      }   
    }
    if (pos>= K)
      continue;   
    val_vec[pos] = has_value?atof(pch):1.0;
    to_vec[pos] = to;
    pos++;

    //free(line_to_free);     
    last_from = from; 
    total_lines++;
    line++;
    if (lines && line>=lines)
      break;

    if (debug && (line % 50000 == 0))
      logstream(LOG_INFO) << "Parsed line: " << line << std::endl;
  }
  if (last_from != 0)
    dump_entry(last_from, to_vec, val_vec, pos, ids_out.outf, val_out.outf); 

  logstream(LOG_INFO) <<"Finished parsing total of " << line << " lines in file " << in_files[i] << endl;
}


int main(int argc,  const char *argv[]) {

  logstream(LOG_WARNING)<<"GraphChi parsers library is written by Danny Bickson (c). Send any "
    " comments or bug reports to danny.bickson@gmail.com " << std::endl;
  global_logger().set_log_level(LOG_INFO);
  global_logger().set_log_to_console(true);

  graphchi_init(argc, argv);

  debug = get_option_int("debug", 0);
  lines = get_option_int("lines", 0);
  has_value = get_option_int("has_value", has_value);
  K = get_option_int("K", K);

  if (K < 1)
    logstream(LOG_FATAL)<<"Number of top elements (--K=) should be >= 1"<<std::endl;

  omp_set_num_threads(get_option_int("ncpus", 1));
  mytime.start();

  std::string training = get_option_string("training");
  in_files.push_back(training);

  if (in_files.size() == 0)
    logstream(LOG_FATAL)<<"Failed to read any file names from the list file: " << dir << std::endl;

  parse(0);
  std::cout << "Finished in " << mytime.current_time() << std::endl;

  return 0;
}




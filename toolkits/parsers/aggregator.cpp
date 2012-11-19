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
 *  1 1 050803 156
 *  1 1 050803 12
 *  2 2 050803 143
 *  3 3 050803 0
 *  4 4 050803 0
 *  5 5 050803 1
 *  6 6 050803 68
 *
 *  Namely, a text file where the first field is a "from" field (uint >=1), second field is a "to" field (uint > = 1)
 *  and then there is a list of fields seperated by spaces (either strings or numbers) which characterize this edge.
 *
 *  The input file is sorted by the from and to fields.
 *
 *  The output of this program is a sorted graph, where edges values are aggregated together, s.t. each edge
 *  appears only once:
 *  1 1 050803 168
 *  2 2 050803 143
 *  3 3 050803 0
 *  4 4 050803 0
 *  5 5 050803 1
 *  6 6 050803 68
 *  
 *  Namely, an aggregation of the value in column defined by --col=XX command line flag,
 *  in the above example 168 = 156+12
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

//non word tokens that will be removed in the parsing
//it is possible to add additional special characters or remove ones you want to keep
const char spaces[] = {" \r\n\t!?@#$%^&*()-+.,~`'\";:"};



 
void parse(int i){    
  in_file fin(in_files[i]);
  out_file fout((outdir + in_files[i] + ".out"));

  size_t linesize = 0;
  char * saveptr = NULL, * linebuf = NULL;
  size_t line = 1;
  double total = 0;
  uint from = 0, to = 0;
  uint last_from = 0, last_to = 0;
  while(true){
    int rc = getline(&linebuf, &linesize, fin.outf);
    if (rc < 1)
      break;
    if (strlen(linebuf) <= 1) //skip empty lines
      continue; 

    //identify from and to fields
    char *pch = strtok_r(linebuf, spaces, &saveptr);
    if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; continue; }
    from = atoi(pch);

    pch = strtok_r(NULL, spaces ,&saveptr);
    if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; continue; }
    to = atoi(pch);

    int col_num = 3;
    //go over the rest of the line
    while(true){
      pch = strtok_r(NULL, spaces ,&saveptr);
      if (pch == NULL && col_num > col)
        break;
      if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; continue; }
    
     //aggregate the requested column 
      if (col == col_num){ 
        if (from != last_from || to != last_to){
           if (last_from != 0 && last_to != 0)
              fprintf(fout.outf, "%u %u %g\n", last_from, last_to, total);
           total = atof(pch);
        }
        else
          total += atof(pch);
      }
      col_num++; 
    }

         
    last_from = from; last_to = to;
    total_lines++;
    line++;
    if (lines && line>=lines)
      break;

    if (debug && (line % 50000 == 0))
      logstream(LOG_INFO) << "Parsed line: " << line << std::endl;

  } 

  if (last_from != 0 && last_to != 0)
     fprintf(fout.outf, "%u %u %g\n", last_from, last_to, total);
 

  logstream(LOG_INFO) <<"Finished parsing total of " << line << " lines in file " << in_files[i] << endl;
}


int main(int argc,  const char *argv[]) {

  logstream(LOG_WARNING)<<"GraphChi parsers library is written by Danny Bickson (c). Send any "
    " comments or bug reports to danny.bickson@gmail.com " << std::endl;
  global_logger().set_log_level(LOG_INFO);
  global_logger().set_log_to_console(true);

  graphchi_init(argc, argv);

  debug = get_option_int("debug", 0);
  dir = get_option_string("file_list");
  lines = get_option_int("lines", 0);
  col = get_option_int("col", 3);
  omp_set_num_threads(get_option_int("ncpus", 1));
  mytime.start();

  FILE * f = fopen(dir.c_str(), "r");
  if (f == NULL)
    logstream(LOG_FATAL)<<"Failed to open file list!"<<std::endl;

  while(true){
    char buf[256];
    int rc = fscanf(f, "%s\n", buf);
    if (rc < 1)
      break;
    in_files.push_back(buf);
  }

  if (in_files.size() == 0)
    logstream(LOG_FATAL)<<"Failed to read any file names from the list file: " << dir << std::endl;

#pragma omp parallel for
  for (int i=0; i< (int)in_files.size(); i++)
    parse(i);

  std::cout << "Finished in " << mytime.current_time() << std::endl;

  return 0;
}




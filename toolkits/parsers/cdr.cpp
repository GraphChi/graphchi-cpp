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
 *
 */


#include <cstdio>
#include <iostream>
#include <omp.h>
#include <assert.h>
#include <algorithm>
#include "graphchi_basic_includes.hpp"
#include "../collaborative_filtering/timer.hpp"
#include "../collaborative_filtering/util.hpp"


using namespace std;
using namespace graphchi;

bool debug = false;
timer mytime;
size_t lines = 0;
unsigned long long total_lines = 0;
string dir;
string outdir;
std::vector<std::string> in_files;
const char user_chars_tokens[] = {" \r\n\t,.\"!?#%^&*()|-\'+$/:"};
uint maxfrom = 0;
uint maxto = 0;



/* 2011-12-05 00:00:00  */

bool convert_string_to_time(char * linebuf, size_t line, int i, long int & outtime){
  char * saveptr = NULL;
  struct tm ptm;
  memset(&ptm, 0, sizeof(ptm));

  char *year = strtok_r(linebuf," \r\n\t:/-",&saveptr);
  if (!year){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return false; }
  ptm.tm_year = atoi(year) - 1900;

  char *month = strtok_r(NULL," \r\n\t:/-",&saveptr);
  if (!month){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return false; }
  ptm.tm_mon = atoi(month) - 1;

  char *day = strtok_r(NULL," \r\n\t:/-",&saveptr);
  if (!day){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return false; }
  ptm.tm_mday = atoi(day);

  char *hour = strtok_r(NULL," \r\n\t:/-",&saveptr);
  if (!hour){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return false; }
  ptm.tm_hour = atoi(hour);

  char *minute = strtok_r(NULL," \r\n\t:/-",&saveptr);
  if (!minute){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return false; }
  ptm.tm_min = atoi(minute);

  char *second = strtok_r(NULL," \r\n\t:/-",&saveptr);
  if (!second){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return false; }
  ptm.tm_sec = atoi(second);

  outtime = mktime(&ptm);
  return true; 

} 


/*
 * The CDR line format is:
 *
 *
 * 2011-12-05 00:00:00     15      15      9       591
 * 2011-12-05 00:00:00     15      22      1       39
 * 2011-12-05 00:00:00     15      134     1       482
 * 2011-12-05 00:00:00     15      180     2       1686
 * 2011-12-05 00:00:00     15      355     1       119
 * 2011-12-05 00:00:00     15      815     1       63
 */

void parse(int i){    
  in_file fin(in_files[i]);
  out_file fout((outdir + in_files[i] + ".out"));

  size_t linesize = 0;
  char * saveptr2 = NULL, * linebuf = NULL, linebuf_debug[1024];
  size_t line = 1;
  uint from, to, duration1, duration2;
  long int ptime;

  while(true){
    int rc = getline(&linebuf, &linesize, fin.outf);
    strncpy(linebuf_debug, linebuf, 1024);
    total_lines++;
    if (rc < 1)
      break;

    bool ok = convert_string_to_time(linebuf_debug, total_lines, i, ptime);
    if (!ok)
       return;

    char *pch = strtok_r(linebuf,"\t", &saveptr2); //skip the date
    if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return; }

    pch = strtok_r(NULL," \r\n\t:/-", &saveptr2);
    if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return; }
    from = atoi(pch);
    maxfrom = std::max(from, maxfrom);

    pch = strtok_r(NULL," \r\n\t:/-", &saveptr2);
    if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return; }
    to = atoi(pch);
    maxto = std::max(to, maxto);

    pch = strtok_r(NULL," \r\n\t:/-", &saveptr2);
    if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return; }
    duration1 = atoi(pch);

    pch = strtok_r(NULL," \r\n\t:/-", &saveptr2);
    if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return; }
    duration2 = atoi(pch);


    line++;
    if (lines && line>=lines)
      break;

    if (debug && (line % 100000 == 0))
      logstream(LOG_INFO)<<"Parsed line: " << line << endl;

    fprintf(fout.outf, "%u %u %lu %u\n", from, to, ptime, duration2); 
  } 

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

//#pragma omp parallel for
  for (uint i=0; i< in_files.size(); i++)
    parse(i);

  std::cout << "Finished in " << mytime.current_time() << std::endl << 
    "\t total lines in input file : " << total_lines <<  "\t max from: " << maxfrom << "\t max to: " <<maxto << std::endl;

  return 0;
}




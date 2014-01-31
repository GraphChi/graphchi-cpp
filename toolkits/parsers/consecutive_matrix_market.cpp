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
 */


#include <cstdio>
#include <map>
#include <iostream>
#include <map>
#include <omp.h>
#include <assert.h>
#include "graphchi_basic_includes.hpp"
#include "../collaborative_filtering/timer.hpp"
#include "../collaborative_filtering/util.hpp"
#include "../../example_apps/matrix_factorization/matrixmarket/mmio.h"
#include "../../example_apps/matrix_factorization/matrixmarket/mmio.c"
#include "common.hpp"

using namespace std;
using namespace graphchi;

bool debug = false;
map<string,uint> string2nodeid;
//map<uint,string> nodeid2hash;
map<string,uint> string2nodeid2;
//map<uint,string> nodeid2hash2;
uint conseq_id;
uint conseq_id2;
timer mytime;
size_t lines;
unsigned long long total_lines = 0;
string dir;
string outdir;
std::vector<std::string> in_files;
uint M,N;
size_t nnz = 0;
const char * string_to_tokenize;
int csv = 0;
int tsv = 0;
int binary = 0; //edges are binary, contain no weights
int single_domain = 0; //both user and movies ids are from the same id space:w
const char * spaces = " \r\n\t";
const char * tsv_spaces = "\t\n";
const char * csv_spaces = "\",\n";
timer mytimer;
int has_header_titles = 0;
int ignore_rest_of_line = 0;
/*
 * assign a consecutive id from either the [from] or [to] ids.
 */
void assign_id(map<string,uint> & string2nodeid, uint & outval, const string &name, bool from){

  map<string,uint>::iterator it = string2nodeid.find(name);
  //if an id was already assigned, return it
  if (it != string2nodeid.end()){
    outval = it->second;
    return;
  }
  mymutex.lock();
  //assign a new id
  outval = string2nodeid[name];
  if (outval == 0){
    //update the mapping between string to the id
    string2nodeid[name] = ((from || single_domain)? ++conseq_id : ++conseq_id2);
    //return the id
    outval = ((from || single_domain)? conseq_id : conseq_id2);
  }
  mymutex.unlock();
}



void parse(int i){    
  in_file fin(in_files[i]);
  out_file fout((outdir + in_files[i] + ".out"));

  size_t linesize = 0;
  char * saveptr = NULL, * linebuf = NULL;
  size_t line = 1;
  uint from,to;
  bool matrix_market = false;

  while(true){
    int rc = getline(&linebuf, &linesize, fin.outf);
    if (rc < 1)
      break;
    if (strlen(linebuf) <= 1){ //skip empty lines
      line++;
      continue;
    }

    if (has_header_titles && line == 1){
       line++;
       continue;
    } 
    //skipping over matrix market header (if any) 
    if (!strncmp(linebuf, "%%MatrixMarket", 14)){
      matrix_market = true;
      continue;
    }
    if (matrix_market && linebuf[0] == '%'){
      continue;
    }
    if (matrix_market && linebuf[0] != '%'){
      matrix_market = false;
      continue;
    }

    //read [FROM]
    char *pch = strtok_r(linebuf,string_to_tokenize, &saveptr);
    if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return; }
    assign_id(string2nodeid, from, pch, true);

    //read [TO]
    pch = strtok_r(NULL,string_to_tokenize, &saveptr);
    if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return; }
    assign_id(single_domain ? string2nodeid:string2nodeid2, to, pch, single_domain ? true : false);

    //read the rest of the line
    if (!binary){
      if (ignore_rest_of_line)
        pch = strtok_r(NULL, string_to_tokenize, &saveptr);
      else
        pch = strtok_r(NULL, "\n", &saveptr);
      if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return; }
    }
    if (tsv)
      fprintf(fout.outf, "%u\t%u\t%s\n", from, to, binary? "": pch);
    else if (csv)
      fprintf(fout.outf, "%u %u %s\n", from, to, binary? "" : pch);
    else 
      fprintf(fout.outf, "%u %u %s\n", from, to, binary? "" : pch);
    nnz++;

    line++;
    total_lines++;
    if (lines && line>=lines)
      break;

    if (debug && (line % 50000 == 0))
      logstream(LOG_INFO) << mytimer.current_time() << ") Parsed line: " << line << " map size is: " << string2nodeid.size() << std::endl;
    if (string2nodeid.size() % 500000 == 0)
      logstream(LOG_INFO) << mytimer.current_time() << ") Hash map size: " << string2nodeid.size() << " at time: " << mytime.current_time() << " edges: " << total_lines << std::endl;
  } 

  logstream(LOG_INFO) <<"Finished parsing total of " << line << " lines in file " << in_files[i] << endl <<
    "total map size: " << string2nodeid.size() << endl;

}


int main(int argc,  const char *argv[]) {
  logstream(LOG_WARNING)<<"GraphChi parsers library is written by Danny Bickson (c). Send any "
    " comments or bug reports to danny.bickson@gmail.com " << std::endl;
  global_logger().set_log_level(LOG_INFO);
  global_logger().set_log_to_console(true);

  graphchi_init(argc, argv);
  mytimer.start();

  debug = get_option_int("debug", 0);
  dir = get_option_string("file_list");
  lines = get_option_int("lines", 0);
  omp_set_num_threads(get_option_int("ncpus", 1));
  tsv = get_option_int("tsv", 0); //is this tab seperated file?
  csv = get_option_int("csv", 0); // is the comma seperated file?
  binary = get_option_int("binary", 0);
  single_domain = get_option_int("single_domain", 0);
  has_header_titles = get_option_int("has_header_titles", has_header_titles);
  ignore_rest_of_line = get_option_int("ignore_rest_of_line", ignore_rest_of_line);
  mytime.start();


  string_to_tokenize = spaces;
  if (tsv)
    string_to_tokenize = tsv_spaces;
  else if (csv)
    string_to_tokenize = csv_spaces;

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
  for (uint i=0; i< in_files.size(); i++)
    parse(i);

  std::cout << "Finished in " << mytime.current_time() << std::endl;
  M = string2nodeid.size();
  if (single_domain)
    N = M;
  else N = string2nodeid2.size();

  save_map_to_text_file(string2nodeid, outdir + dir + "user.map.text");
  if (!single_domain){
    save_map_to_text_file(string2nodeid2, outdir + dir + "movie.map.text");
  }
  std::string filename = "matrix_market.info";
  if (in_files.size() == 1)
     filename = in_files[0] + ".out:info";
  logstream(LOG_INFO)<<"Writing matrix market header into file: " << filename << std::endl;
  out_file fout(filename.c_str());
  MM_typecode out_typecode;
  mm_clear_typecode(&out_typecode);
  mm_set_integer(&out_typecode); 
  mm_set_sparse(&out_typecode); 
  mm_set_matrix(&out_typecode);
  mm_write_banner(fout.outf, out_typecode);
  mm_write_mtx_crd_size(fout.outf, M, N, nnz);
  return 0;
}




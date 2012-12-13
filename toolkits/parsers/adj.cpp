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
*  File for converting graph in SNAP format to GraphChi CF toolkit format.
*  See SNAP format explanation here: http://docs.graphlab.org/graph_formats.html
* See GraphChi matrix market format here: http://bickson.blogspot.co.il/2012/02/matrix-market-format.html
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

using namespace std;
using namespace graphchi;

bool debug = false;
map<string,uint> string2nodeid;
map<uint,string> nodeid2hash;
map<string,uint> string2nodeid2;
map<uint,string> nodeid2hash2;
uint conseq_id;
uint conseq_id2;
mutex mymutex;
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
const char * csv_spaces = ",\n";


void save_map_to_text_file(const std::map<std::string,uint> & map, const std::string filename){
    std::map<std::string,uint>::const_iterator it;
    out_file fout(filename);
    unsigned int total = 0;
    for (it = map.begin(); it != map.end(); it++){ 
      fprintf(fout.outf, "%s %u\n", it->first.c_str(), it->second);
     total++;
    } 
    logstream(LOG_INFO)<<"Wrote a total of " << total << " map entries to text file: " << filename << std::endl;
}


void save_map_to_text_file(const std::map<uint,std::string> & map, const std::string filename){
    std::map<uint,std::string>::const_iterator it;
    out_file fout(filename);
    unsigned int total = 0;
    for (it = map.begin(); it != map.end(); it++){ 
      fprintf(fout.outf, "%u %s\n", it->first, it->second.c_str());
     total++;
    } 
    logstream(LOG_INFO)<<"Wrote a total of " << total << " map entries to text file: " << filename << std::endl;
}

/*
 * assign a consecutive id from either the [from] or [to] ids.
 */
void assign_id(map<string,uint> & string2nodeid, map<uint,string> & nodeid2hash, uint & outval, const string &name, bool from){

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
    string2nodeid[name] = (from? ++conseq_id : ++conseq_id2);
    //return the id
    outval = (from? conseq_id : conseq_id2);
    //store the reverse mapping between id to string
    nodeid2hash[outval] = name;
    if (from)
      M = std::max(M, conseq_id);
    else
      N = std::max(N, conseq_id2);
  }
  mymutex.unlock();
}


/* example file format:
 * 2884424247 11 1210682095 1789803763 1879013170 1910216645 2014570227
 * 2109318072 2268277425 2289674265 2340794623 2513611825 2770280793
 * 2884596247 31 1191220232 1191258123 1225281292 1240067740
 * 2885009247 16 1420862042 1641392180 1642909335 1775498871 1781379945
 * 1784537661 1846581167 1934183965 2011304655 2016713117 2017390697
 * 2128869911 2132021133 2645747085 2684129850 2866009832
 */ 
void parse(int i){    
  in_file fin(in_files[i]);
  out_file fout((outdir + in_files[i] + ".out"));

  size_t linesize = 0;
  char * saveptr = NULL, * linebuf = NULL;
  char linebuf_debug[1024];
  size_t line = 1;
  uint from,to;
  bool matrix_market = false;

  while(true){
    int rc = getline(&linebuf, &linesize, fin.outf);
    strncpy(linebuf_debug, linebuf, 1024);
    if (rc < 1)
      break;
    if (strlen(linebuf) <= 1) //skip empty lines
      continue;
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
    if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf_debug << "]" << std::endl; return; }
    assign_id(string2nodeid,nodeid2hash, from, pch, true);

    //read [NUMBER OF EDGES]
    pch = strtok_r(NULL,string_to_tokenize, &saveptr);
    if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf_debug << "]" << std::endl; return; }
    int num_edges = atoi(pch);
    if (num_edges < 0)
    { logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf_debug << "] - number of edges < 0" << std::endl; return;   }
    
    for (int k=0; k< num_edges; k++){
      pch = strtok_r(NULL, "\n\t\r, ", &saveptr);
      if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf_debug << "]" << std::endl; return; }
 
    assign_id(single_domain ? string2nodeid:string2nodeid2,
        single_domain ? nodeid2hash : nodeid2hash2, to, pch, single_domain ? true : false);
   if (tsv)
      fprintf(fout.outf, "%u\t%u\n", from, to);
    else if (csv)
      fprintf(fout.outf, "%u,%un", from, to);
    else 
      fprintf(fout.outf, "%u %u\n", from, to);
    nnz++;
  }

      line++;
    total_lines++;
    if (lines && line>=lines)
      break;

    if (debug && (line % 50000 == 0))
      logstream(LOG_INFO) << "Parsed line: " << line << " map size is: " << string2nodeid.size() << std::endl;
    if (string2nodeid.size() % 500000 == 0)
      logstream(LOG_INFO) << "Hash map size: " << string2nodeid.size() << " at time: " << mytime.current_time() << " edges: " << total_lines << std::endl;
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

  debug = get_option_int("debug", 0);
  dir = get_option_string("file_list");
  lines = get_option_int("lines", 0);
  omp_set_num_threads(get_option_int("ncpus", 1));
  tsv = get_option_int("tsv", 0); //is this tab seperated file?
  csv = get_option_int("csv", 0); // is the comma seperated file?
  binary = get_option_int("binary", 0);
  single_domain = get_option_int("single_domain", 0);
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

//#pragma omp parallel for
  for (uint i=0; i< in_files.size(); i++)
    parse(i);

  std::cout << "Finished in " << mytime.current_time() << std::endl;

  save_map_to_text_file(string2nodeid, outdir + dir + "user.map.text");
  save_map_to_text_file(nodeid2hash, outdir + dir + "user.reverse.map.text");
  if (!single_domain){
    save_map_to_text_file(string2nodeid2, outdir + dir + "movie.map.text");
    save_map_to_text_file(nodeid2hash2, outdir + dir + "movie.reverse.map.text");
  }
  logstream(LOG_INFO)<<"Writing matrix market header into file: matrix_market.info" << std::endl;
  out_file fout("matrix_market.info");
  MM_typecode out_typecode;
  mm_clear_typecode(&out_typecode);
  mm_set_integer(&out_typecode); 
  mm_set_sparse(&out_typecode); 
  mm_set_matrix(&out_typecode);
  mm_write_banner(fout.outf, out_typecode);
  mm_write_mtx_crd_size(fout.outf, M, single_domain ? M : N, nnz);
  return 0;
}




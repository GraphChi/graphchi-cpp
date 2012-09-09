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
#include <map>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <omp.h>
#include <assert.h>
#include "graphchi_basic_includes.hpp"
#include "../collaborative_filtering/timer.hpp"
#include "../collaborative_filtering/util.hpp"

using namespace std;
using namespace graphchi;

bool debug = false;
unordered_map<string,uint> string2nodeid;
unordered_map<uint,string> nodeid2hash;
uint conseq_id;
std::mutex mymutex;
timer mytime;
size_t lines;
unsigned long long total_lines = 0;
string dir;
string outdir;
std::vector<std::string> in_files;


void save_map_to_text_file(const std::unordered_map<std::string,uint> & map, const std::string filename){
    std::unordered_map<std::string,uint>::const_iterator it;
    out_file fout(filename);
    unsigned int total = 0;
    for (it = map.begin(); it != map.end(); it++){ 
      fprintf(fout.outf, "%s %u\n", it->first.c_str(), it->second);
     total++;
    } 
    logstream(LOG_INFO)<<"Wrote a total of " << total << " map entries to text file: " << filename << std::endl;
}


void save_map_to_text_file(const std::unordered_map<uint,std::string> & map, const std::string filename){
    std::unordered_map<uint,std::string>::const_iterator it;
    out_file fout(filename);
    unsigned int total = 0;
    for (it = map.begin(); it != map.end(); it++){ 
      fprintf(fout.outf, "%u %s\n", it->first, it->second.c_str());
     total++;
    } 
    logstream(LOG_INFO)<<"Wrote a total of " << total << " map entries to text file: " << filename << std::endl;
}


void assign_id(uint & outval, const string &name, const int line, const string &filename){

  unordered_map<string,uint>::iterator it = string2nodeid.find(name);
  if (it != string2nodeid.end()){
    outval = it->second;
    return;
  }
  mymutex.lock();
  outval = string2nodeid[name];
  if (outval == 0){
    string2nodeid[name] = ++conseq_id;
    outval = conseq_id;
    nodeid2hash[outval] = name;
  }
  mymutex.unlock();
}



void find_ids(uint & to, const string& buf2, const int line, const string &filename){
  assign_id(to, buf2, line, filename);
}

bool extract_user_name(const char * linebuf, size_t line, int i, const char * saveptr, char * userid){
  char *pch = strtok_r(NULL," \r\n\t:/.",(char**)&saveptr);
  if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return false; }
  pch = strtok_r(NULL," \r\n\t:/.",(char**)&saveptr);
  if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return false; }
  pch = strtok_r(NULL," \r\n\t:/.",(char**)&saveptr);
  if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return false; }
  userid = strtok_r(NULL," \r\n\t:/.",(char**)&saveptr); //USERNAME
  if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return false; }
  return true;
}


bool convert_string_to_time(const char * linebuf, size_t line, int i, const char * saveptr, struct tm & ptm){
  char *year = strtok_r(NULL," \r\n\t:/-",(char**)&saveptr);
  if (!year){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return false; }
  ptm.tm_year = atoi(year) - 1900;

  char *month = strtok_r(NULL," \r\n\t:/-",(char**)&saveptr);
  if (!month){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return false; }
  ptm.tm_mon = atoi(month) - 1;

  char *day = strtok_r(NULL," \r\n\t:/-",(char**)&saveptr);
  if (!day){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return false; }
  ptm.tm_mday = atoi(day);

  char *hour = strtok_r(NULL," \r\n\t:/-",(char**)&saveptr);
  if (!hour){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return false; }
  ptm.tm_hour = atoi(hour);

  char *minute = strtok_r(NULL," \r\n\t:/-",(char**)&saveptr);
  if (!minute){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return false; }
  ptm.tm_min = atoi(minute);

  char *second = strtok_r(NULL," \r\n\t:/-",(char**)&saveptr);
  if (!second){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return false; }
  ptm.tm_sec = atoi(second);

  return true; 

} 

/*
 * Twitter input format is:
 *
 * T  2009-06-11 16:56:42
 * U  http://twitter.com/tiffnic85
 * W  Bus is pulling out now. We gotta be in LA by 8 to check into the Paragon.
 *
 * T  2009-06-11 16:56:42
 * U  http://twitter.com/xlamp
 * W  灰を灰皿に落とそうとすると高確率でヘッドセットの線を根性焼きする形になるんだが
 *
 * T  2009-06-11 16:56:43
 * U  http://twitter.com/danilaselva
 * W  @carolinesweatt There are no orphans...of God! :) Miss your face!
 *
 */

void parse(int i){    
  in_file fin(in_files[i]);
  out_file fout((outdir + in_files[i] + ".out"));

  char linebuf[24000], buf1[256];
  char saveptr[1024];
  size_t line = 1;
  uint id;
  struct tm ptm;
  bool ok;

  while(true){
    int rc = fscanf(fin.outf, "%s\n", linebuf);
    if (rc < 1)
      break;
    if (strlen(linebuf) == 0) //skip empty lines
      continue;

    char *pch = strtok_r(linebuf," \r\n\t:/",(char**)&saveptr);
    if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return; }

    switch(*pch){
      case 'T':
        ok = convert_string_to_time(linebuf, line, i, saveptr, ptm);
        if (!ok)
          return;
        break;

      case 'U':
        ok = extract_user_name(linebuf, line, i, saveptr, buf1);
        if (!ok)
          return;
        assign_id(id, buf1, line, in_files[i]);
        break;

      case 'W':
        //TODO
        break;

      default:
        logstream(LOG_ERROR)<<"Error: expecting with T U or W first character" << std::endl;
        return;

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


int main(int argc,  char *argv[]) {

  logstream(LOG_WARNING)<<"GraphChi parsers library is written by Danny Bickson (c). Send any "
    " comments or bug reports to danny.bickson@gmail.com " << std::endl;
  global_logger().set_log_level(LOG_INFO);
  global_logger().set_log_to_console(true);

  // int unittest = 0;
  uint lines = 0;
  string filter = "";

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

#pragma omp parallel for
  for (uint i=0; i< in_files.size(); i++)
    parse(i);

  std::cout << "Finished in " << mytime.current_time() << std::endl;

  save_map_to_text_file(string2nodeid, outdir + ".map.text.gz");
  save_map_to_text_file(nodeid2hash, outdir + ".reverse.map.text.gz");
  return 0;
}




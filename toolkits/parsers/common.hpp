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
 *  Written by Danny Bickson, CMU */


#ifndef _GRAPHCHI_PARSERS_COMMON
#define _GRAPHCHI_PARSERS_COMMON

#include <map>
#include <string>
#include "graphchi_basic_includes.hpp"
using namespace graphchi;
mutex mymutex;

struct double_map{
  std::map<std::string,uint> string2nodeid;                                                         
  std::map<uint,std::string> nodeid2hash;  
  uint maxid;
  double_map(){
    maxid = 0;
  }
};
double_map frommap;
double_map tomap;

template<typename T1>
void load_map_from_txt_file(T1 & map, const std::string filename, int fields){
  logstream(LOG_INFO)<<"loading map from txt file: " << filename << std::endl;
  FILE * f = fopen(filename.c_str(), "r");
  if (f == NULL)
    logstream(LOG_FATAL)<<"Failed to open file: " << filename << std::endl;

  char * linebuf = NULL;
  size_t linesize;
  int line = 0;
  while (true){
    int rc = getline(&linebuf, &linesize, f);
    if (rc == -1)
      break;
  
    char * line2 = strdup(linebuf);
    if (fields == 1){
       char *pch = strsep(&line2,"\r\n\t");
       if (!pch)
         logstream(LOG_FATAL) << "Error when parsing file: " << filename << ":" << line <<std::endl;
        map[pch] = ++line;
    }
    else {
      char *pch = strsep(&line2,"\r\n\t");
      if (!pch){
        logstream(LOG_FATAL) << "Error when parsing file: " << filename << ":" << line <<std::endl;
      }
      char * pch2 = strsep(&line2,"\r\t\n");
      if (!pch2)
        logstream(LOG_FATAL) << "Error when parsing file: " << filename << ":" << line <<std::endl;
      map[pch] = atoi(pch2);
      line++;
    }
    //free(to_free);
  }
  logstream(LOG_INFO)<<"Map size is: " << map.size() << std::endl;
  fclose(f);
}

void load_vec_from_txt_file(std::vector<std::string> & vec, const std::string filename){
  logstream(LOG_INFO)<<"loading vec from txt file: " << filename << std::endl;
  FILE * f = fopen(filename.c_str(), "r");
  if (f == NULL)
    logstream(LOG_FATAL)<<"Failed to open file: " << filename << std::endl;

  char * linebuf = NULL;
  size_t linesize;
  while (true){
    int rc = getline(&linebuf, &linesize, f);
    if (rc == -1)
      break;
    vec.push_back(linebuf);
 }
  logstream(LOG_INFO)<<"Loaded total of  " << vec.size() << " vec entries. " << std::endl;
  fclose(f);
}


void save_map_to_text_file(const std::map<std::string,uint> & map, const std::string filename, int optional_offset = 0){
    std::map<std::string,uint>::const_iterator it;
    out_file fout(filename);
    unsigned int total = 0;
    for (it = map.begin(); it != map.end(); it++){ 
      fprintf(fout.outf, "%s\t%u\n", it->first.c_str(), it->second + optional_offset);
     total++;
    } 
    logstream(LOG_INFO)<<"Wrote a total of " << total << " map entries to text file: " << filename << std::endl;
}
void save_vec_to_text_file(const std::vector<std::string> & vec, const std::string filename){
    out_file fout(filename);
    unsigned int total = 0;
    for (int i = 0; i< (int)vec.size(); i++){ 
      fprintf(fout.outf, "%s\n", vec[i].c_str());
    } 
    logstream(LOG_INFO)<<"Wrote a total of " << vec.size() << " vec entries to text file: " << filename << std::endl;
}


void save_map_to_text_file(const std::map<unsigned long long,uint> & map, const std::string filename, int optional_offset = 0){
    std::map<unsigned long long,uint>::const_iterator it;
    out_file fout(filename);
    unsigned int total = 0;
    for (it = map.begin(); it != map.end(); it++){ 
      fprintf(fout.outf, "%llu\t%u\n", it->first , it->second + optional_offset);
     total++;
    } 
    logstream(LOG_INFO)<<"Wrote a total of " << total << " map entries to text file: " << filename << std::endl;
}


void save_map_to_text_file(const std::map<uint,std::string> & map, const std::string filename){
    std::map<uint,std::string>::const_iterator it;
    out_file fout(filename);
    unsigned int total = 0;
    for (it = map.begin(); it != map.end(); it++){ 
      fprintf(fout.outf, "%u\t%s\n", it->first, it->second.c_str());
     total++;
    } 
    logstream(LOG_INFO)<<"Wrote a total of " << total << " map entries to text file: " << filename << std::endl;
}

/*
 * assign a consecutive id from either the [from] or [to] ids.
 */
void assign_id(double_map& dmap, unsigned int & outval, const std::string &name){

  std::map<std::string,uint>::iterator it = dmap.string2nodeid.find(name);
  //if an id was already assigned, return it
  if (it != dmap.string2nodeid.end()){
    outval = it->second;
    return;
  }
  mymutex.lock();
  //assign a new id
  outval = dmap.string2nodeid[name];
  if (outval == 0){
    dmap.string2nodeid[name] = dmap.maxid;
    dmap.nodeid2hash[dmap.maxid] = name;
    outval = dmap.maxid;
    dmap.maxid++;
  }
  mymutex.unlock();
}
#endif //_GRAPHCHI_PARSERS_COMMON

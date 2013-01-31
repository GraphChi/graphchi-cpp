#ifndef _IMPLICIT_HPP__
#define _IMPLICIT_HPP__
/**
 * @file
 * @author  Danny Bickson
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright [2012] [Carnegie Mellon University]
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
 */

#include "eigen_wrapper.hpp"

enum{
IMPLICIT_RATING_DISABLED = 0,
IMPLICIT_RATING_RANDOM = 1
};

double implicitratingweight;
double implicitratingvalue = -1;
double implicitratingpercentage;
int    implicitratingtype;

template<typename als_edge_type>
uint add_implicit_edges4(int type, sharder<als_edge_type>& shrd){

  switch(type){
    case IMPLICIT_RATING_DISABLED: return 0;
    case IMPLICIT_RATING_RANDOM: break;
    default: assert(false);
  };

  uint added = 0;
  uint toadd  = (uint)(implicitratingpercentage*N*M);
  logstream(LOG_INFO)<<"Going to add: " << toadd << " implicit edges. " << std::endl;
  assert(toadd >= 1);
  for (uint j=0; j< toadd; j++){
    ivec item = ::randi(1,0,N-1);
    ivec user = ::randi(1,0,M-1);
    shrd.preprocessing_add_edge(user[0], item[0], als_edge_type(implicitratingvalue, implicitratingweight));
    added++;
  } 
  logstream(LOG_INFO)<<"Finished adding " << toadd << " implicit edges. " << std::endl;
  return added;
}

template<typename als_edge_type>
uint add_implicit_edges(int type, sharder<als_edge_type>& shrd ){

  switch(type){
    case IMPLICIT_RATING_DISABLED: return 0;
    case IMPLICIT_RATING_RANDOM: break;
    default: assert(false);
  };

  uint added = 0;
  uint toadd  = (uint)(implicitratingpercentage*N*M);
  logstream(LOG_INFO)<<"Going to add: " << toadd << " implicit edges. " << std::endl;
  assert(toadd >= 1);
  for (uint j=0; j< toadd; j++){
    ivec item = ::randi(1,0,N-1);
    ivec user = ::randi(1,0,M-1);
    shrd.preprocessing_add_edge(user[0], item[0], als_edge_type(implicitratingvalue));
    added++;
  } 
  logstream(LOG_INFO)<<"Finished adding " << toadd << " implicit edges. " << std::endl;
  return added;
}

void parse_implicit_command_line(){
   implicitratingweight = get_option_float("implicitratingweight", implicitratingweight);
   implicitratingvalue = get_option_float("implicitratingvalue", implicitratingvalue);
   implicitratingtype = get_option_int("implicitratingtype", implicitratingtype);
   if (implicitratingtype != IMPLICIT_RATING_RANDOM && implicitratingtype != IMPLICIT_RATING_DISABLED)
     logstream(LOG_FATAL)<<"Implicit rating type should be either 0 (IMPLICIT_RATING_DISABLED) or 1 (IMPLICIT_RATING_RANDOM)" << std::endl;
   implicitratingpercentage = get_option_float("implicitratingpercentage", implicitratingpercentage);
   if (implicitratingpercentage < 1e-8 && implicitratingpercentage > 0.8)
     logstream(LOG_FATAL)<<"Implicit rating percentage should be (1e-8, 0.8)" << std::endl;
   if (implicitratingtype != IMPLICIT_RATING_DISABLED && implicitratingvalue == 0)
     logstream(LOG_FATAL)<<"You are not allowed to use --implicitratingvalue=0. Please select a non zero value, for example -1" << std::endl;
}
#endif //_IMPLICIT_HPP__

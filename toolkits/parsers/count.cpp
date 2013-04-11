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
 *  Code by Yucheng Low, UW
 *  Adapted by Danny Bickson, CMU
 *
 *  File for counting the number of unsigned int occurances in a text file
 *  Where each line has one value
 *
 *  For example, input file named "in" has the following lines:
 *
 *  1
 *  2
 *  2
 *  3
 *  3
 *  3
 *  3
 *  3
 *  3
 *  3
 *  3
 *
 *  The output of "./count in"
 *  1 1
 *  2 2
 *  3 8
 */

#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <stdint.h>
#include <stdlib.h>
using namespace std;

int main(int argc, char** argv) {

  if (argc <= 1){
    std::cerr<<"Usage: counter <input file name>" << std::endl;
    exit(1);
  }


  std::ifstream fin(argv[1]);
  map<uint32_t, uint32_t> count;
  size_t lines = 0;
  while(fin.good()) {
    std::string b;
    ++lines;
    getline(fin, b);
    if (fin.eof())
      break;
    int32_t bid = atol(b.c_str());
    if (lines >= 3){
    ++count[bid];
    //std::cout<<"adding: " << bid << std::endl;
    if (lines % 5000000 == 0) {
      std::cerr << lines << " lines\n";
    }
    }
  }

  fin.close();
  map<uint32_t, uint32_t>::iterator itr;

  for(itr = count.begin(); itr != count.end(); ++itr){
    if ((*itr).second > 0 )
      std::cout<< (*itr).first << " " <<(*itr).second<<std::endl;
  }

}

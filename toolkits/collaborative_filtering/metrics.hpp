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
 *  Code for computing ranking metrics
 *
 *  */

#include <algorithm>
#include "eigen_wrapper.hpp"

/*  average_precision_at_k code based on Ben Hamer's Kaggle code:
 *  https://github.com/benhamner/Metrics/blob/master/MATLAB/metrics/averagePrecisionAtK.m
 */
double average_precision_at_k(vec & predictions, int prediction_size, vec & actual, int actual_size, int k){
  double score = 0;
  int num_hits = 0;

  vec sorted_actual = actual;
  actual_size = std::min(k, actual_size);
  std::sort(sorted_actual.data(), sorted_actual.data()+ actual_size);
  for (int i=0; i < std::min((int)predictions.size(), k); i++){
    if (std::binary_search(sorted_actual.data(), sorted_actual.data()+ actual_size, predictions[i])){
      num_hits++;
      score += num_hits / (i+1.0);
    }
  }
  score /= (double)std::min(actual_size, k);
  return score;
}




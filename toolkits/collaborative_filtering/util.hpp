#ifndef __CF_UTILS__
#define __CF_UTILS__
int number_of_omp_threads(){
  int num_threads = 0;
  int id;
#pragma omp parallel private(id)
      {
        id = omp_get_thread_num();
        if (id == 0)
           num_threads = omp_get_num_threads();
      }
      return num_threads;
} 

#endif


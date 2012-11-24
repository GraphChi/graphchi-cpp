#ifndef __CF_UTILS__
#define __CF_UTILS__

#include <omp.h>
#include <stdio.h>
#include <iostream>

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

struct  in_file{
  FILE * outf;
  in_file(std::string fname)  {
    outf = fopen(fname.c_str(), "r");
    if (outf == NULL){
      std::cerr<<"Failed to open file: " << fname << std::endl;
      exit(1);
    }
  }

  ~in_file() {
    if (outf != NULL) fclose(outf);
  }

};

struct out_file{

  FILE * outf;
  out_file(const std::string fname){
    outf = fopen(fname.c_str(), "w");
  }

  ~out_file(){
    if (outf != NULL) fclose(outf);
   }
};



/*
template<typename T1>
void load_map_from_txt_file(T1 & map, const std::string filename, bool gzip, int fields){
   logstream(LOG_INFO)<<"loading map from txt file: " << filename << std::endl;
   gzip_in_file fin(filename, gzip);
   char linebuf[1024]; 
   char saveptr[1024];
   bool mm_header = false;
   int line = 0;
   char * pch2 = NULL;
   while (!fin.get_sp().eof() && fin.get_sp().good()){ 
     fin.get_sp().getline(linebuf, 10000);
      if (fin.get_sp().eof())
        break;

      if (linebuf[0] == '%'){
        logstream(LOG_INFO)<<"Detected matrix market header: " << linebuf << " skipping" << std::endl;
        mm_header = true;
        continue;
      }
      if (mm_header){
        mm_header = false;
        continue; 
      }
         
      char *pch = strtok_r(linebuf," \r\n\t",(char**)&saveptr);
      if (!pch){
        logstream(LOG_FATAL) << "Error when parsing file: " << filename << ":" << line <<std::endl;
       }
       if (fields == 2){
         pch2 = strtok_r(NULL,"\n",(char**)&saveptr);
         if (!pch2)
            logstream(LOG_FATAL) << "Error when parsing file: " << filename << ":" << line <<std::endl;
       }
      if (fields == 1)
        map[boost::lexical_cast<std::string>(line)] = pch;
      else map[pch] = pch2;
      line++;
   }
   logstream(LOG_INFO)<<"Map size is: " << map.size() << std::endl;
 }*/





#endif


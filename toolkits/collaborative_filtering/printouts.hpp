#ifndef PRINTOUTS
#define PRINTOUTS
#define MAX_PRINTOUT_LEN 25

bool absolute_value = true;

inline double fabs2(double val){
  if (absolute_value)
    return fabs(val);
  else return val;
}

void print_vec(const char * name, const DistVec & vec, bool high = false){
  if (!debug)
    return;
  int i;
 printf("%s[%d]\n", name, vec.offset);
 for (i=vec.start; i< std::min(vec.end, MAX_PRINTOUT_LEN); i++){
  if (high)
   printf("%15.15lg ", fabs2(latent_factors_inmem[i].pvec[vec.offset]));
  else
   printf("%.5lg ", fabs2(latent_factors_inmem[i].pvec[vec.offset]));
  }
 printf("\n");
}
void print_vec(const char * name, const vec & pvec, bool high = false){
  if (!debug)
    return;
  printf("%s\n", name);
 for (int i= 0; i< std::min((int)pvec.size(), MAX_PRINTOUT_LEN); i++){
  if (high)
   printf("%15.15lg ", fabs2(pvec[i]));
  else
   printf("%.5lg ", fabs2(pvec[i]));
  }
 printf("\n");
}
void print_mat(const char * name, const mat & pmat, bool high = false){
  if (!debug)
    return;
  printf("%s\n", name);
 mat pmat2 = transpose((mat&)pmat);
 if (pmat2.cols() == 1)
    pmat2 = pmat2.transpose();
 for (int i= 0; i< std::min((int)pmat2.rows(), MAX_PRINTOUT_LEN); i++){
  for (int j=0; j< std::min((int)pmat2.cols(), MAX_PRINTOUT_LEN); j++){
    if (high)
      printf("%15.15lg ", fabs2(get_val(pmat2, i, j)));
    else
     printf("%.5lg ", fabs2(get_val(pmat2, i, j)));
  }
  printf("\n");
  
  }
}

void print_vec_pos(std::string name, vec & v, int i){
  if (!debug)
    return;
   if (i == -1)
    printf("%s\n", name.c_str());
  else {
    printf("%s[%d]: %.5lg\n", name.c_str(), i, fabs(v[i]));
    return;
  }
  for (int j=0; j< std::min((int)v.size(),MAX_PRINTOUT_LEN); j++){
   printf("%.5lg", fabs2(v(j)));
   if (v.size() > 1)
    printf(" ");
  }
  printf("\n");
}


#define PRINT_VEC(a) print_vec(#a,a,0)
#define PRINT_VEC2(a,b) print_vec(a,b,0)
#define PRINT_VEC3(a,b,c) print_vec_pos(a,b,c)
#define PRINT_VEC2_HIGH(a,i) print_vec(#a,a[i],1)
#define PRINT_INT(a) if (debug) printf("%s: %d\n", #a, a);
#define PRINT_NAMED_INT(a,b) if (debug) printf("%s: %d\n",a, b);
#define PRINT_DBL(a) if (debug) printf("%s: %.5lg\n", #a, a);
#define PRINT_NAMED_DBL(a,b) if (debug) printf("%s: %.5lg\n", a, b);
#define PRINT_MAT(a) print_mat(#a, a, 0);
#define PRINT_MAT2(a,b) print_mat(a,b,0);
#endif

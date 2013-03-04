#ifndef _DISTANCE_HPP__
#define _DISTANCE_HPP__
#include "graphchi_basic_includes.hpp"


typedef double flt_dbl;
typedef sparse_vec sparse_flt_dbl_vec;
typedef vec flt_dbl_vec;
extern int debug;

double safeLog(double d) {
    return d <= 0.0 ? 0.0 : log(d);
}
double logL(double p, double k, double n) {
    return k * safeLog(p) + (n - k) * safeLog(1.0 - p);
}

double twoLogLambda(double k1, double k2, double n1, double n2) {
    double p = (k1 + k2) / (n1 + n2);
    return 2.0 * (logL(k1 / n1, k1, n1) + logL(k2 / n2, k2, n2) - logL(p, k1, n1) - logL(p, k2, n2));
}

flt_dbl calc_loglikelihood_distance( sparse_flt_dbl_vec & datapoint, sparse_flt_dbl_vec & cluster, flt_dbl sqr_sum, flt_dbl sqr_sum_datapoint){ 
   flt_dbl intersection = dot_prod(datapoint , cluster);
   flt_dbl logLikelihood = twoLogLambda(intersection,
                                        sqr_sum - intersection,
                                        sqr_sum_datapoint,
                                        datapoint.size() - sqr_sum_datapoint);
    return 1.0 - 1.0 / (1.0 + logLikelihood);
}

flt_dbl calc_loglikelihood_distance( sparse_flt_dbl_vec & datapoint,  flt_dbl_vec &cluster, flt_dbl sqr_sum, flt_dbl sqr_sum_datapoint){
  flt_dbl intersection = dot_prod(datapoint, cluster);
  flt_dbl logLikelihood = twoLogLambda(intersection,
                                        sqr_sum - intersection,
                                        sqr_sum_datapoint,
                                        datapoint.size() - sqr_sum_datapoint);
   return 1.0 - 1.0 / (1.0 + logLikelihood);
}



flt_dbl calc_tanimoto_distance( sparse_flt_dbl_vec & datapoint, sparse_flt_dbl_vec & cluster, flt_dbl sqr_sum, flt_dbl sqr_sum_datapoint){ 
  flt_dbl a_mult_b = dot_prod(datapoint , cluster);
  flt_dbl div = (sqr_sum + sqr_sum_datapoint - a_mult_b);
  if (debug && (div == 0 || a_mult_b/div < 0)){
     logstream(LOG_ERROR) << "divisor is zeo: " << sqr_sum<< " " << sqr_sum_datapoint << " " << a_mult_b << " " << std::endl;
     print(datapoint);
     print(cluster);
     exit(1);
  }
  return 1.0 - a_mult_b/div;
}

flt_dbl calc_tanimoto_distance( sparse_flt_dbl_vec & datapoint,  flt_dbl_vec &cluster, flt_dbl sqr_sum, flt_dbl sqr_sum_datapoint){
  flt_dbl a_mult_b = dot_prod(datapoint, cluster);
  flt_dbl div = (sqr_sum + sqr_sum_datapoint - a_mult_b);
  if (debug && (div == 0 || a_mult_b/div < 0)){
     logstream(LOG_ERROR) << "divisor is zeo: " << sqr_sum << " " << sqr_sum_datapoint << " " << a_mult_b << " " << std::endl;
     print(datapoint);
     debug_print_vec("cluster", cluster, cluster.size());
     exit(1);
  }
  return 1.0 - a_mult_b/div;
}

flt_dbl calc_jaccard_weight_distance( sparse_flt_dbl_vec & datapoint, sparse_flt_dbl_vec & cluster, flt_dbl sqr_sum, flt_dbl sqr_sum_datapoint){ 
  flt_dbl a_size = 0;
  FOR_ITERATOR(i, datapoint){
   a_size+= i.value();
  }
  flt_dbl b_size = 0;
  FOR_ITERATOR(i, cluster){
   b_size+= i.value();
  }
  flt_dbl intersection_size = sqr_sum;
  assert(intersection_size != 0);
  return intersection_size / (a_size+b_size-intersection_size);
}



flt_dbl calc_euclidian_distance( sparse_flt_dbl_vec & datapoint,  sparse_flt_dbl_vec &cluster, flt_dbl sqr_sum, flt_dbl sqr_sum_datapoint){
  //sparse_flt_dbl_vec diff = minus(datapoint , cluster);
  //return sqrt(sum_sqr(diff));
  sparse_flt_dbl_vec mult = elem_mult(datapoint, cluster);
  flt_dbl diff = (sqr_sum + sqr_sum_datapoint - 2*sum(mult));
  return sqrt(fabs(diff)); //because of numerical errors, diff may be negative
}

/*
flt_dbl calc_euclidian_distance( sparse_flt_dbl_vec & datapoint,  flt_dbl_vec &cluster, flt_dbl sqr_sum, flt_dbl sqr_sum_datapoint){
  flt_dbl dist = sqr_sum + sqr_sum_datapoint;
  //for (int i=0; i< datapoint.nnz(); i++){
  FOR_ITERATOR_(i, datapoint){
      flt_dbl val = get_nz_data(datapoint, i);
      int pos = get_nz_index(datapoint, i);
      dist -= 2*val*cluster[pos];
   }
  if (debug && dist < 0 && fabs(dist) > 1e-8){
     logstream(LOG_WARNING)<<"Found a negative distance: " << dist << " initial sum: " << sqr_sum_datapoint + sqr_sum << std::endl;
     logstream(LOG_WARNING)<<"sqr sum: " << sqr_sum << " sqr_sum_datapoint: " <<sqr_sum_datapoint<<std::endl;
     FOR_ITERATOR_(i, datapoint){
        int pos = get_nz_index(datapoint, i);
        logstream(LOG_WARNING)<<"Data: " << get_nz_data(datapoint, i) << " Pos: " << get_nz_index(datapoint, i) <<" cluster valu: " << cluster[pos] 
            << "reduction: " << 2*get_nz_data(datapoint,i)*cluster[pos] << std::endl;
     } 
     dist = 0;
  
  }
  return sqrt(fabs(dist)); //should not happen, but distance is sometime negative because of the shortcut we make to calculate it..
}
*/

flt_dbl calc_chebychev_distance( sparse_flt_dbl_vec & datapoint,  sparse_flt_dbl_vec &cluster){
   sparse_flt_dbl_vec diff = minus(datapoint , cluster);
   flt_dbl ret = 0;
   FOR_ITERATOR(i, diff){
      ret = std::max(ret, (flt_dbl)fabs(get_nz_data(diff, i)));
   }
   return ret;

}
flt_dbl calc_chebychev_distance( sparse_flt_dbl_vec & datapoint,  flt_dbl_vec &cluster){
   flt_dbl_vec diff = minus(datapoint , cluster);
   flt_dbl ret = 0;
   for (int i=0; i< diff.size(); i++)
      ret = std::max(ret, (flt_dbl)fabs(diff[i]));

   return ret;

}

flt_dbl calc_manhatten_distance( sparse_flt_dbl_vec & datapoint,  sparse_flt_dbl_vec &cluster){
   sparse_flt_dbl_vec diff = minus(datapoint , cluster);
   sparse_flt_dbl_vec absvec = fabs(diff);
   flt_dbl ret = sum(absvec);
   return ret;

}
flt_dbl calc_manhatten_distance( sparse_flt_dbl_vec & datapoint,  flt_dbl_vec &cluster){
   flt_dbl_vec diff = minus(datapoint , cluster);
   flt_dbl ret = sum(fabs(diff));
   return ret;

}

/* note that distance should be divided by intersection size */
flt_dbl calc_slope_one_distance( sparse_flt_dbl_vec & datapoint,  sparse_flt_dbl_vec &cluster){
   sparse_flt_dbl_vec diff = minus(datapoint , cluster);
   flt_dbl ret = sum(diff);
   return ret;

}


flt_dbl calc_cosine_distance( sparse_flt_dbl_vec & datapoint,  sparse_flt_dbl_vec & cluster, flt_dbl sum_sqr, flt_dbl sum_sqr0){
   flt_dbl dotprod = dot_prod(datapoint,cluster);
   flt_dbl denominator = sqrt(sum_sqr0)*sqrt(sum_sqr);
   return 1.0 - dotprod / denominator; 
}

flt_dbl calc_cosine_distance( sparse_flt_dbl_vec & datapoint,  flt_dbl_vec & cluster, flt_dbl sum_sqr, flt_dbl sum_sqr0){
   flt_dbl dotprod = dot_prod(datapoint,cluster);
   flt_dbl denominator = sqrt(sum_sqr0)*sqrt(sum_sqr);
   return 1.0 - dotprod / denominator; 
}

flt_dbl calc_dot_product_distance( sparse_flt_dbl_vec & datapoint,  flt_dbl_vec & cluster){
         return dot_prod(datapoint, cluster);
}
flt_dbl calc_dot_product_distance( sparse_flt_dbl_vec & datapoint,  sparse_flt_dbl_vec & cluster){
         return dot_prod(datapoint, cluster);
}

#endif //_DISTANCE_HPP__

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
 * For more about this software visit:
 *
 *      http://graphlab.org
 *
 */

/**
 * Code by Danny Bickson, CMU
 */
#ifndef EIGEN_WRAPPER
#define EIGEN_WRAPPER
#ifdef EIGEN_NDEBUG
#define NDEBUG 
#endif

/**
 * SET OF WRAPPER FUNCTIONS FOR EIGEN
 *
 *
 */

#include <iostream>
#include <fstream>
#include <ostream>


#include "Eigen/Dense"
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include "Eigen/Sparse"
#include "Eigen/Cholesky"
#include "Eigen/Eigenvalues"
#include "Eigen/SVD"


#define EIGEN_DONT_PARALLELIZE //eigen parallel for loop interfers with ours.
using namespace Eigen;

typedef MatrixXd mat;
typedef VectorXd vec;
typedef VectorXf fvec;
typedef VectorXi ivec;
typedef MatrixXi imat;
typedef Matrix<size_t, Dynamic, Dynamic> matst;
typedef SparseVector<double> sparse_vec;

inline void debug_print_vec(const char * name,const vec& _vec, int len){
  printf("%s ) ", name);
  for (int i=0; i< len; i++)
    if (_vec[i] == 0)
      printf("      0    ");
    else printf("%12.4g    ", _vec[i]);
  printf("\n");
}
inline void debug_print_vec(const char * name,const double* _vec, int len){
  printf("%s ) ", name);
  for (int i=0; i< len; i++)
    if (_vec[i] == 0)
      printf("      0    ");
    else printf("%12.4g    ", _vec[i]);
  printf("\n");
}
mat randn1(int dx, int dy, int col);

template<typename mat, typename data>
inline void set_val(mat &A, int row, int col, data val){
  A(row, col) = val;
}
inline double get_val(const mat &A, int row, int col){
  return A(row, col);
}
inline int get_val(const imat &A, int row, int col){
  return A(row, col);
}
inline vec get_col(const mat& A, int col){
  return A.col(col);
}
inline vec get_row(const mat& A, int row){
  return A.row(row);
}
inline void set_col(mat& A, int col, const vec & val){
  A.col(col) = val;
}
inline void set_row(mat& A, int row, const vec & val){
  A.row(row) = val;
}

inline mat eye(int size){
  return mat::Identity(size, size);
}
inline vec ones(int size){
  return vec::Ones(size);
}
inline fvec fones(int size){
  return fvec::Ones(size);
}
inline vec init_vec(const double * array, int size){
  vec ret(size);
  memcpy(ret.data(), array, size*sizeof(double));
  return ret;
}
inline mat init_mat(const char * string, int row, int col){
  mat out(row, col);
  char buf[2056];
  strcpy(buf, string);
  char *pch = strtok(buf," \r\n\t;");
  for (int i=0; i< row; i++){
    for (int j=0; j< col; j++){
      out(i,j) = atof(pch);
      pch = strtok (NULL, " \r\n\t;");
    }
  }
  return out;
}
inline imat init_imat(const char * string, int row, int col){
  imat out(row, col);
  char buf[2056];
  strcpy(buf, string);
  char *pch = strtok(buf," \r\n\t;");
  for (int i=0; i< row; i++){
    for (int j=0; j< col; j++){
      out(i,j) = atol(pch);
      pch = strtok (NULL, " \r\n\t;");
    }
  }
  return out;
}
inline vec init_vec(const char * string, int size){
  vec out(size);
  char buf[2056];
  strcpy(buf, string);
  char *pch = strtok (buf," \r\n\t;");
  int i=0;
  while (pch != NULL)
  {
    out(i) =atof(pch);
    pch = strtok (NULL, " \r\n\t;");
    i++;
  }
  assert(i == size);
  return out;
}
inline vec init_dbl_vec(const char * string, int size){
  return init_vec(string, size);
}

inline vec zeros(int size){
  return vec::Zero(size);
}
inline fvec fzeros(int size){
  return fvec::Zero(size);
}
inline mat zeros(int rows, int cols){
  return mat::Zero(rows, cols);
}
inline vec head(const vec& v, int num){
  return v.head(num);
}
inline vec mid(const vec&v, int start, int num){
  return v.segment(start, std::min(num, (int)(v.size()-start)));
}
inline vec tail(const vec&v,  int num){
  return v.segment(v.size() - num, num);
}
inline ivec head(const ivec& v, int num){
  return v.head(num);
}
inline void sort(ivec &a){
  std::sort(a.data(), a.data()+a.size());
}
inline void sort(vec & a){
  std::sort(a.data(), a.data()+a.size());
}
inline ivec sort_index(const vec&a){
  ivec ret(a.size()); 
  std::vector<std::pair<double,int> > D;
  // 	
  D.reserve(a.size());
  for (int i=0;i<a.size();i++)
    D.push_back(std::make_pair<double,int>(a.coeff(i),i));
  std::sort(D.begin(),D.end());
  for (int i=0;i<a.size();i++)
  { 
    ret[i]=D[i].second;
  } 
  return ret;
}
inline void dot2(const vec&  x1, const vec& x3, mat & Q, int j, int len){
  for (int i=0; i< len; i++){
    Q(i,j) = (x1(i) * x3(i));
  }
}

inline bool ls_solve_chol(const mat &A, const vec &b, vec &result){
  //result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
  result = A.ldlt().solve(b);
  return true;
}
inline bool ls_solve(const mat &A, const vec &b, vec &result){
  //result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
  result = A.ldlt().solve(b);
  return true;
}
inline bool chol(mat& sigma, mat& out){
  out = sigma.llt().matrixLLT();
  return true;
}
inline bool backslash(const mat& A, const vec & b, vec & x){
  x = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
  return true;
} 
inline mat transpose(mat & A){
  return A.transpose();
}
inline mat randn(int dx, int dy){
  return randn1(dx,dy,-1);
}
inline void set_diag(mat &A, vec & v){
  A.diagonal()=v;
}
inline mat diag(vec & v){
  return v.asDiagonal();
}

template<typename mat>
inline double sumsum(const mat & A){
  return A.sum();
}
inline double norm(const mat &A, int pow=2){
  return A.squaredNorm();
}
inline mat inv(const mat&A){
  return A.inverse();
}
inline bool inv(const mat&A, mat &out){
  out = A.inverse();
  return true;
}
inline mat outer_product(const vec&a, const vec&b){
  return a*b.transpose();
}
//Eigen does not sort eigenvalues, as done in matlab
inline bool eig_sym(const mat & T, vec & eigenvalues, mat & eigenvectors){
  //
  //Column  of the returned matrix is an eigenvector corresponding to eigenvalue number  as returned by eigenvalues(). The eigenvectors are normalized to have (Euclidean) norm equal to one.
  SelfAdjointEigenSolver<mat> solver(T);
  eigenvectors = solver.eigenvectors();
  eigenvalues = solver.eigenvalues(); 
  ivec index = sort_index(eigenvalues);
  sort(eigenvalues);
  vec eigenvalues2 = eigenvalues.reverse();
  mat T2 = zeros(eigenvectors.rows(), eigenvectors.cols());
  for (int i=0; i< eigenvectors.cols(); i++){
    set_col(T2, index[i], get_col(eigenvectors, i));
  }   
  eigenvectors = T2;
  eigenvalues = eigenvalues2;
  return true;
}

inline vec elem_mult(const vec&a, const vec&b){
  vec ret = a;
  for (int i=0; i<b.size(); i++)
    ret(i) *= b(i);
  return ret;
}
inline sparse_vec elem_mult(const sparse_vec&a, const sparse_vec&b){
  return a.cwiseProduct(b);
}
inline double sum(const vec & a){
  return a.sum();
}
inline double min(const vec &a){
  return a.minCoeff();
}
inline double max(const vec & a){
  return a.maxCoeff();
}
inline vec randu(int size){
  return vec::Random(size);
}
inline fvec frandu(int size){
  return fvec::Random(size);
}
inline double randu(){
  return vec::Random(1)(0);
}
inline ivec randi(int size, int from, int to){
  ivec ret(size);
  for (int i=0; i<size; i++)
    ret[i]= internal::random<int>(from,to);
  return ret;
}
inline int randi(int from, int to){
  return internal::random<int>(from,to);
}
inline ivec concat(const ivec&a, const ivec&b){ 
  ivec ret(a.size()+b.size());
  ret << a,b;
  return ret;
}
inline void del(ivec&a, int i){
  memcpy(a.data()+i, a.data() + i+1, (a.size() - i - 1)*sizeof(int)); 
  a.conservativeResize(a.size() - 1); //resize without deleting values!
}
inline mat get_cols(const mat&A, ivec & cols){
  mat a(A.rows(), cols.size());
  for (int i=0; i< cols.size(); i++)
    set_col(a, i, get_col(A, cols[i]));
  return a;
}
inline mat get_cols(const mat&A, int start_col, int end_col){
  assert(end_col > start_col);
  assert(end_col <= A.cols());
  assert(start_col >= 0);
  mat a(A.rows(), end_col-start_col);
  for (int i=0; i< end_col-start_col; i++)
    set_col(a, i, get_col(A, i));
  return a;
}
inline void set_val(vec & v, int pos, double val){
  v(pos) = val;
}
inline void set_val(sparse_vec & v, int pos, double val){
  v.coeffRef(pos) = val;
}
inline double dot(const vec&a, const vec& b){
  return a.dot(b);
}
inline vec reverse(vec& a){
  return a.reverse();
}
inline ivec reverse(ivec& a){
  return a.reverse();
}
inline const double * data(const mat &A){
  return A.data();
}
inline const int * data(const imat &A){
  return A.data();
}
inline const double * data(const vec &v){
  return v.data();
}

class it_file{
  std::fstream fb;

  public:
  it_file(const char * name){
    fb.open(name, std::fstream::in);
    fb.close();

    if (fb.fail()){
      fb.clear(std::fstream::failbit);
      fb.open(name, std::fstream::out | std::fstream::trunc );
    }
    else {
      fb.open(name, std::fstream::in);
    }

    if (!fb.is_open()){
      perror("Failed opening file ");
      printf("filename is: %s\n", name);
      assert(false);
    }

  };
  std::fstream & operator<<(const std::string str){
    int size = str.size();
    fb.write((char*)&size, sizeof(int));
    assert(!fb.fail());
    fb.write(str.c_str(), size);
    return fb;
  }
  std::fstream &operator<<(mat & A){
    int rows = A.rows(), cols = A.cols();
    fb.write( (const char*)&rows, sizeof(int));
    fb.write( (const char *)&cols, sizeof(int));
    for (int i=0; i< A.rows(); i++)
      for (int j=0; j< A. cols(); j++){
        double val = A(i,j);
        fb.write( (const char *)&val, sizeof(double));
        assert(!fb.fail());
      }
    return fb;
  }
  std::fstream &operator<<(const vec & v){
    int size = v.size();
    fb.write( (const char*)&size, sizeof(int));
    assert(!fb.fail());
    for (int i=0; i< v.size(); i++){
      double val = v(i);
      fb.write( (const char *)&val, sizeof(double));
      assert(!fb.fail());
    }
    return fb;
  }
  std::fstream & operator<<(const double &v){
    fb.write((const char*)&v, sizeof(double));
    return fb;
  }
  std::fstream & operator>>(std::string  str){
    int size = -1;
    fb.read((char*)&size, sizeof(int));
    if (fb.fail() || fb.eof()){
      perror("Failed reading file");
      assert(false);
    }

    char buf[256];
    fb.read(buf, std::min(256,size));
    assert(!fb.fail());
    assert(!strncmp(str.c_str(), buf, std::min(256,size)));
    return fb;
  }

  std::fstream &operator>>(mat & A){
    int rows, cols;
    fb.read( (char *)&rows, sizeof(int));
    assert(!fb.fail());
    fb.read( (char *)&cols, sizeof(int));
    assert(!fb.fail());
    A = mat(rows, cols);
    double val;
    for (int i=0; i< A.rows(); i++)
      for (int j=0; j< A. cols(); j++){
        fb.read((char*)&val, sizeof(double));
        assert(!fb.fail());
        A(i,j) = val;
      }
    return fb;
  }
  std::fstream &operator>>(vec & v){
    int size;
    fb.read((char*)&size, sizeof(int));
    assert(!fb.fail());
    assert(size >0);
    v = vec(size);
    double val;
    for (int i=0; i< v.size(); i++){
      fb.read((char*)& val, sizeof(double));
      assert(!fb.fail());
      v(i) = val;
    }
    return fb;
  }

  std::fstream &operator>>(double &v){
    fb.read((char*)&v, sizeof(double));
    assert(!fb.fail());
    return fb;
  }

  void close(){
    fb.close();
  }
};

#define Name(a) std::string(a)
inline void set_size(sparse_vec &v, int size){
  //did not find a way to declare vector dimension, yet
}
inline void set_new(sparse_vec&v, int ind, double val){
  v.insert(ind) = val;
} 
inline int nnz(sparse_vec& v){
  return v.nonZeros();
}
inline int get_nz_index(sparse_vec &v, sparse_vec::InnerIterator& i){
  return i.index();
}
inline double get_nz_data(sparse_vec &v, sparse_vec::InnerIterator& i){
  return i.value();
}
#define FOR_ITERATOR(i,v)                       \
  for (sparse_vec::InnerIterator i(v); i; ++i)

template<typename T>
inline double sum_sqr(const T& a);

template<>
inline double sum_sqr<vec>(const vec & a){
  vec ret = a.array().pow(2);
  return ret.sum();
}
template<>
inline double sum_sqr<sparse_vec>(const sparse_vec & a){
  double sum=0;
  FOR_ITERATOR(i,a){
    sum+= powf(i.value(),2);
  }
  return sum;
}

inline double trace(const mat & a){
  return a.trace();
}
inline double get_nz_data(sparse_vec &v, int i){
  assert(nnz(v) > i);
  int cnt=0;
  FOR_ITERATOR(j, v){
    if (cnt == i){
      return j.value();
    }
    cnt++;
  }
  return 0.0;
}
inline void print(sparse_vec & vec){
  int cnt = 0;
  FOR_ITERATOR(i, vec){
    std::cout<<get_nz_index(vec, i)<<":"<< get_nz_data(vec, i) << " ";
    cnt++;
    if (cnt >= 20)
      break;
  }
  std::cout<<std::endl;
}
inline vec pow(const vec&v, int exponent){
  vec ret = vec(v.size());
  for (int i=0; i< v.size(); i++)
    ret[i] = powf(v[i], exponent);
  return ret;
}
inline double dot_prod(sparse_vec &v1, sparse_vec & v2){
  return v1.dot(v2);
}
inline double dot_prod(const vec &v1, const vec & v2){
  return v1.dot(v2);
}
inline double dot3(const vec &v1, const vec & v2, const vec & v3){
  double ret = 0;
  for (int i=0; i < v1.size(); i++)
    ret+= v1[i]*v2[i]*v3[i];
  return ret;
}
inline double dot_prod(sparse_vec &v1, const vec & v2){
  double sum = 0;
  for (int i=0; i< v2.size(); i++){
    sum+= v2[i] * v1.coeffRef(i);
  }
  return sum;
}
inline vec cumsum(vec& v){
  vec ret = v;
  for (int i=1; i< v.size(); i++)
    for (int j=0; j< i; j++)
      ret(i) += v(j);
  return ret;
}
inline double get_val(sparse_vec & v1, int i){ //TODO optimize performance
  for (sparse_vec::InnerIterator it(v1); it; ++it)
    if (it.index() == i)
      return it.value();

  return 0;
} 
inline double get_val(vec & v1, int i){
  return v1(i);
}
inline void set_div(sparse_vec&v, sparse_vec::InnerIterator i, double val){
  v.coeffRef(i.index()) /= val;
}
inline sparse_vec minus(sparse_vec &v1,sparse_vec &v2){
  return v1-v2;
}
inline vec minus( sparse_vec &v1,  vec &v2){
  vec ret = -v2;
  FOR_ITERATOR(i, v1){
    ret[i.index()] += i.value();
  }
  return ret;
}
inline void plus( vec &v1,  sparse_vec &v2){
  FOR_ITERATOR(i, v2){
    v1[i.index()] += i.value();
  }
}
inline void minus( vec &v1, sparse_vec &v2){
  FOR_ITERATOR(i, v2){
    v1[i.index()] -= i.value();
  }
}
inline sparse_vec fabs( sparse_vec & dvec1){
  sparse_vec ret = dvec1;
  FOR_ITERATOR(i, ret){
    ret.coeffRef(i.index()) = fabs(i.value()); 
  }	
  return ret;
};

inline vec fabs( const vec & dvec1){
  vec ret(dvec1.size());
  for (int i=0; i< dvec1.size(); i++){
    ret(i) = fabs(dvec1(i));
  }	
  return ret;
};
inline double abs_sum(const mat& A){
  double sum =0;
  for (int i=0; i< A.rows(); i++)
    for (int j=0; j< A.cols(); j++)
      sum += fabs(A(i,j));
  return sum;
}
inline double abs_sum(const vec &v){
  double sum =0;
  for (int i=0; i< v.size(); i++)
    sum += fabs(v(i));
  return sum;
}
inline double sum(const sparse_vec &v){
  double sum =0;
  FOR_ITERATOR(i, v){
    sum += i.value();
  }
  return sum;
}
inline vec sqrt(const vec & v){
  vec ret(v.size());
  for (int i=0; i< v.size(); i++){
    ret[i] = std::sqrt(v(i));
  }
  return ret;
}
inline void svd(const mat & A, mat & U, mat & V, vec & singular_values){
  Eigen::JacobiSVD<mat> svdEigen(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  U= svdEigen.matrixU();
  V= svdEigen.matrixV();
  singular_values =svdEigen.singularValues(); 
}

inline bool pair_compare (std::pair<double,int> &x1, std::pair<double,int> & x2) { return (x1.first>x2.first); }

inline ivec reverse_sort_index2(const vec&a, const ivec&indices, vec & out, int K){
  assert(a.size() == indices.size());
  assert(K > 0);
  int size = std::min((unsigned int)a.size(), (unsigned int)K);
  ivec ret(size); 
  std::vector<std::pair<double,int> > D;

  D.reserve(a.size());
  for (int i=0;i<a.size();i++)
    D.push_back(std::make_pair<double,int>(a[i],indices[i]));
  std::partial_sort(D.begin(),D.begin() + size, D.end(), pair_compare);
  for (int i=0;i< size;i++)
  { 
    ret[i]=D[i].second;
    out[i] = D[i].first;
  } 
  return ret;
}
inline ivec reverse_sort_index(const vec& a, int K){
  assert(K > 0);
  int size = std::min((unsigned int)a.size(), (unsigned int)K);
  ivec ret(size); 
  std::vector<std::pair<double,int> > D;

  D.reserve(a.size());
  for (int i=0;i<a.size();i++)
    D.push_back(std::make_pair<double,int>(a[i],i));
  std::partial_sort(D.begin(),D.begin() + size, D.end(), pair_compare);
  for (int i=0;i< size;i++)
  { 
    ret[i]=D[i].second;
  } 
  return ret;
}
inline ivec reverse_sort_index(sparse_vec& a, int K){
  assert(K > 0);
  int size = std::min((unsigned int)nnz(a), (unsigned int)K);
  ivec ret(size); 
  std::vector<std::pair<double,int> > D;

  D.reserve(nnz(a));
  FOR_ITERATOR(i, a){  
    D.push_back(std::make_pair<double,int>(i.value(),i.index()));
  }
  std::partial_sort(D.begin(),D.begin() + size, D.end(), pair_compare);
  for (int i=0;i< size;i++)
  { 
    ret[i]=D[i].second;
  } 
  return ret;
}
//define function to be applied coefficient-wise
  double equal_greater(double x){
    if (x != 0)
      return 1;
    else 
      return 0;
  }    
//sort(edges.begin(), edges.end());
#undef NDEBUG
#endif

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
 * @section DESCRIPTION
 *
 * time-SVD++ algorithm implementation. As described in the paper:
 * Yehuda Koren. 2009. Collaborative filtering with temporal dynamics. In Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '09). ACM, New York, NY, USA, 447-456. DOI=10.1145/1557019.1557072
 * 
 */



#include "common.hpp"
#include "eigen_wrapper.hpp"


struct timesvdpp_params{
  double lrate;
  double beta; 
  double gamma; 
  double lrate_mult_dec;

  timesvdpp_params(){
    lrate =0.0001;
    beta = 0.00001; 
    gamma = 0.0001;   
    lrate_mult_dec = 0.9;
  }
};

timesvdpp_params tsp;

bool is_user(vid_t id){ return id < M; }
bool is_item(vid_t id){ return id >= M && id < N; }
bool is_time(vid_t id){ return id >= M+N; }


inline double sum(double * pvec){
  double tsum = 0;
  for (int j=0; j< D; j++)
    tsum += pvec[j];
  return tsum;
}


struct vertex_data {
  vec pvec;
  double rmse;
  double bias;
  vertex_data() {
    rmse = 0;
    bias = 0;
  }

};

struct edge_data {
  double weight;
  double time;

  edge_data() { weight = time = 0; }

  edge_data(double weight, double time) : weight(weight), time(time) { }
};

struct time_svdpp_usr{
  double * bu;
  double * p;
  double * pu;
  double * x;
  double * ptemp;
  double * rmse;

  time_svdpp_usr(vertex_data & vdata){
    bu = &vdata.bias;
    assert(vdata.pvec.size() == D*4); //TO REMOVE
    p = &vdata.pvec[0];
    pu = p+D;
    x = pu+D;
    ptemp = x+D;
    rmse = &vdata.rmse;
  }
  time_svdpp_usr & operator = (vertex_data & vdata){
    bu = &vdata.bias;
    assert(vdata.pvec.size() == D*4); //TO REMOVE
    p = &vdata.pvec[0];
    pu = p+D;
    x = pu+D;
    ptemp = x+D;
    rmse = &vdata.rmse;
    return *this;
  }
};

struct time_svdpp_movie{
  double * bi;
  double * q;
  double * y;

  time_svdpp_movie(vertex_data& vdata){
    assert(vdata.pvec.size() == D*2);
    bi = &vdata.bias;
    q = &vdata.pvec[0];
    y = q+D;
  }
  time_svdpp_movie & operator=(const vertex_data& vdata){
    assert(vdata.pvec.size() == D*2);
    bi = (double*)&vdata.bias;
    q = (double*)&vdata.pvec[0];
    y = (double*)(q+D);
    return *this;
  }
};

struct time_svdpp_time{
  double * bt;
  double * z;
  double * pt;

  time_svdpp_time(vertex_data& vdata){
    bt = &vdata.bias;
    z = &vdata.pvec[0];
    pt = z+D;
    assert(vdata.pvec.size() == D*2);
  }
  time_svdpp_time & operator=(vertex_data & vdata){
    bt = &vdata.bias;
    z = &vdata.pvec[0];
    pt = z+D;
    assert(vdata.pvec.size() == D*2);
    return *this;

  }
};


float time_svdpp_predict(const time_svdpp_usr & usr, 
    const time_svdpp_movie & mov, 
    const time_svdpp_time & ptime,
    const float rating, 
    double & prediction){


  double pui  = globalMean + *usr.bu + *mov.bi;// + bt[t];
  int dim = D;
  for(int k=0;k<dim;k++){
    pui += (usr.ptemp[k] * mov.q[k]);
    pui += usr.x[k] * ptime.z[k];
    pui += usr.pu[k] * ptime.pt[k] * mov.q[k];
  }
  pui = std::min(pui,maxval);
  pui = std::max(pui,minval);
  prediction = pui;
  assert(!std::isnan(prediction));
  float err = rating - prediction;
  return err*err;
}



float time_svdpp_predict(const vertex_data& user, 
    const vertex_data& movie, 
    const float rating, 
    double & prediction, 
    void * extra){
  return time_svdpp_predict(time_svdpp_usr((vertex_data&)user), time_svdpp_movie((vertex_data&)movie), time_svdpp_time(*(vertex_data*)extra), rating, prediction);
}

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef edge_data EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
graphchi_engine<VertexDataType, EdgeDataType> * pvalidation_engine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

void init_time_svdpp_node_data(){
  int k = D;
#pragma omp parallel for
  for (int u = 0; u < (int)M; u++) {
    vertex_data & data = latent_factors_inmem[u];
    data.pvec = zeros(4*k);
    time_svdpp_usr usr(data);
    *usr.bu = 0;
    for (int m=0; m< k; m++){
      usr.p[m] = 0.01*drand48() / (double) (k);
      usr.pu[m] = 0.001 * drand48() / (double) (k);
      usr.x[m] = 0.001 * drand48() / (double) (k);
      usr.ptemp[m] = usr.p[m];
    }
  }

#pragma omp parallel for
  for (int i = M; i < (int)(N+M); i++) {
    vertex_data & data = latent_factors_inmem[i];
    data.pvec = zeros(2*k);
    time_svdpp_movie movie(data);
    *movie.bi = 0;
    for (int m = 0; m < k; m++){
      movie.q[m] = 0.01 * drand48() / (double) (k);
      movie.y[m] = 0.001 * drand48() / (double) (k);
    }
  }
}

void init_time_svdpp(){
  fprintf(stderr, "time-SVD++ %d factors\n", D);

  int k = D;

  latent_factors_inmem.resize(M+N+K);

  init_time_svdpp_node_data();

#pragma omp parallel for
  for (int i = M+N; i < (int)(M+N+K); i++) {
    vertex_data & data = latent_factors_inmem[i];
    data.pvec = zeros(2*k);
    time_svdpp_time timenode(data);
    *timenode.bt = 0;
    for (int m = 0; m < k; m++){
      timenode.z[m] = 0.001 * drand48() / (double) (k);
      timenode.pt[m] = 0.001 * drand48() / (double) (k);
    }
  }
}


#include "io.hpp"
#include "rmse.hpp"
#include "rmse_engine4.hpp"

/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct TIMESVDPPVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

  /*
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {

    //go over all user nodes
    if (is_user(vertex.id())){
      vertex_data & user = latent_factors_inmem[vertex.id()]; 
      user.rmse = 0; 
      time_svdpp_usr usr(user);

      unsigned int userRatings = vertex.num_outedges();
      double rRuNum = 1/sqrt(userRatings+10);
      int dim = D;
      double sumY = 0.0;

      //go over all ratings
      for(int e=0; e < vertex.num_outedges(); e++) {
        uint pos = vertex.edge(e)->vertex_id();
        assert(pos >= M && pos < M+N);
        vertex_data & data = latent_factors_inmem[pos];
        time_svdpp_movie movie(data);
        Map<vec> y(movie.y, D);
        sumY += sum((const vec&)y); //y
      }

      for( int k=0; k<dim; ++k) {
        usr.ptemp[k] = usr.pu[k] + rRuNum * sumY; // pTemp = pu + rRuNum*sumY
      }
      vec sum = zeros(dim);
      for(int e=0; e < vertex.num_edges(); e++) {  
        //edge_data & edge = scope.edge_data(oedgeid);
        //float rui = edge.weight;
        float rui = vertex.edge(e)->get_data().weight; 
        uint t = (uint)(vertex.edge(e)->get_data().time - 1); // we assume time bins start from 1
        assert(t < M+N+K);
        vertex_data & data = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        time_svdpp_movie mov(data);
        time_svdpp_time time(latent_factors_inmem[t]);
        double pui = 0; 
        time_svdpp_predict(usr, mov, time, rui, pui);
        double eui = rui - pui;
        *usr.bu += tsp.lrate*(eui - tsp.beta* *usr.bu);
        *mov.bi += tsp.lrate * (eui - tsp.beta* *mov.bi);

        for (int k = 0; k < dim; k++) {
          double oldValue = mov.q[k];
          double userValue = usr.ptemp[k] + usr.pu[k] * time.pt[k];
          sum[k] += eui * mov.q[k];
          mov.q[k] += tsp.lrate * (eui * userValue - tsp.gamma*mov.q[k]);
          usr.ptemp[k] += tsp.lrate * ( eui * oldValue - tsp.gamma * usr.ptemp[k]);
          usr.p[k] += tsp.lrate * ( eui * oldValue - tsp.gamma*usr.p[k] );
          usr.pu[k] += tsp.lrate * (eui * oldValue  * time.pt[k] - tsp.gamma * usr.pu[k]);
          time.pt[k] += tsp.lrate * (eui * oldValue * usr.pu[k] - tsp.gamma * time.pt[k]);
          double xOldValue = usr.x[k];
          double zOldValue = time.z[k];
          usr.x[k] += tsp.lrate * (eui * zOldValue - tsp.gamma * xOldValue);
          time.z[k] += tsp.lrate * (eui * xOldValue - tsp.gamma * zOldValue);
        }

        *usr.rmse += eui*eui;
      }

      for(int e=0; e < vertex.num_edges(); e++) {  
        time_svdpp_movie mov = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        for(int k=0;k<dim;k++){
          mov.y[k] += tsp.lrate * (rRuNum * sum[k]- tsp.gamma*mov.y[k]);
        }
      }

    }

  };

  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    tsp.lrate *= tsp.lrate_mult_dec;
    training_rmse(iteration, gcontext);
    run_validation4(pvalidation_engine, gcontext);
  };


};

struct  MMOutputter_bias{
  FILE * outf;
  MMOutputter_bias(std::string fname, uint start, uint end, std::string comment)  {
    MM_typecode matcode;
    set_matcode(matcode);
    outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, end-start, 1); 
    for (uint i=start; i< end; i++)
      fprintf(outf, "%1.12e\n", latent_factors_inmem[i].bias);
  }


  ~MMOutputter_bias() {
    if (outf != NULL) fclose(outf);
  }

};

struct  MMOutputter_global_mean {
  FILE * outf;
  MMOutputter_global_mean(std::string fname, std::string comment)  {
    MM_typecode matcode;
    set_matcode(matcode);
    outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, 1, 1); 
    fprintf(outf, "%1.12e\n", globalMean);
  }

  ~MMOutputter_global_mean() {
    if (outf != NULL) fclose(outf);
  }

};


struct  MMOutputter{
  FILE * outf;
  MMOutputter(std::string fname, uint start, uint end, std::string comment)  {
    assert(start < end);
    MM_typecode matcode;
    set_matcode(matcode);     
    outf = fopen(fname.c_str(), "w");
    assert(outf != NULL);
    mm_write_banner(outf, matcode);
    if (comment != "")
      fprintf(outf, "%%%s\n", comment.c_str());
    mm_write_mtx_array_size(outf, end-start, latent_factors_inmem[start].pvec.size()); 
    for (uint i=start; i < end; i++)
      for(int j=0; j < latent_factors_inmem[i].pvec.size(); j++) {
        fprintf(outf, "%1.12e\n", latent_factors_inmem[i].pvec[j]);
      }
  }

  ~MMOutputter() {
    if (outf != NULL) fclose(outf);
  }

};


void output_timesvdpp_result(std::string filename) {
  MMOutputter mmoutput_left(filename + "_U.mm", 0, M, "This file contains TIMESVDPP output matrix U. In each row 4xD factors of a single user node. The vectors are [p pu x ptemp] (");
  MMOutputter mmoutput_right(filename + "_V.mm", M ,M+N, "This file contains -TIMESVDPP  output matrix V. In each row 2xD factors of a single item node. The vectors are [q y]");
  MMOutputter mmoutput_time(filename + "_T.mm", M+N ,M+N+K, "This file contains -TIMESVDPP  output matrix T. In each row 2xD factors of a single time node. The vectors are [z pt]");
  MMOutputter_bias mmoutput_bias_left(filename + "_U_bias.mm", 0, M, "This file contains time-svd++ output bias vector. In each row a single user bias.");
  MMOutputter_bias mmoutput_bias_right(filename + "_V_bias.mm",M ,M+N , "This file contains time-svd++ output bias vector. In each row a single item bias.");
  MMOutputter_global_mean gmean(filename + "_global_mean.mm", "This file contains time-svd++ global mean which is required for computing predictions.");

  logstream(LOG_INFO) << " time-svd++ output files (in matrix market format): " << filename << "_U.mm" << ", " << filename + "_V.mm " << filename + "_T.mm, " << filename << " _global_mean.mm, " << filename << "_U_bias.mm " << filename << "_V_bias.mm " << std::endl;
}

int main(int argc, const char ** argv) {


  print_copyright();  

  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("time-svdpp-inmemory-factors");

  //specific command line parameters for time-svd++
  tsp.lrate =   get_option_float("lrate", tsp.lrate);
  tsp.beta =    get_option_float("beta", tsp.beta);
  tsp.gamma =   get_option_float("gamma", tsp.gamma);
  tsp.lrate_mult_dec = get_option_float("lrate_mult_dec", tsp.lrate_mult_dec);

  parse_command_line_args();
  parse_implicit_command_line();

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket4<edge_data>(training, false);
  init_time_svdpp();
  if (validation != ""){
    int vshards = convert_matrixmarket4<EdgeDataType>(validation, false, M==N, VALIDATION);
    init_validation_rmse_engine<VertexDataType, EdgeDataType>(pvalidation_engine, vshards, &time_svdpp_predict, false, true, 0);
   }


  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, 4*D);
    load_matrix_market_matrix(training + "_V.mm", M, 2*D);
    load_matrix_market_matrix(training + "_T.mm", M+N, 2*D);
    vec user_bias = load_matrix_market_vector(training +"_U_bias.mm", false, true);
    vec item_bias = load_matrix_market_vector(training +"_V_bias.mm", false, true);
    vec time_bias = load_matrix_market_vector(training+ "_T_bias.mm", false, true);
    for (uint i=0; i<M+N+K; i++){
      if (i < M)
        latent_factors_inmem[i].bias = user_bias[i];
      else if (i <M+N)
        latent_factors_inmem[i].bias = item_bias[i-M];
      else 
        latent_factors_inmem[i].bias = time_bias[i-M-N];
    }
    vec gm = load_matrix_market_vector(training + "_global_mean.mm", false, true);
    globalMean = gm[0];
 }


  /* Run */
  TIMESVDPPVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output test predictions in matrix-market format */
  output_timesvdpp_result(training);
  test_predictions3(&time_svdpp_predict, 1);    

  /* Report execution metrics */
  if (!quiet) 
    metrics_report(m);
  return 0;
}

/**
 * @file
 * @author  Danny Bickson
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright [2013] [GraphLab Inc.]
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
 * Implementation of the adpredictor algorithm as given in the paper:
 * Web-Scale Bayesian Click-Through Rate Prediction for Sponsored Search Advertising in Microsoft’s Bing Search Engine
 * Thore Graepel, Joaquin Quinonero Candela, Thomas Borchert, and Ralf Herbrich
 * ICML 2010
 */

#include "cdf.hpp"
#include "../collaborative_filtering/eigen_wrapper.hpp"
#include "../collaborative_filtering/common.hpp"

double beta = 1;
vec liklihood_vec;
vec err_vec; 
vec mu_ij;
vec sigma_ij;
vec validation_targets;
int debug = 0;
int vshards = 0;
const double pi = 3.14159265;
const double gaussian_normalization = 1/sqrt(2 * pi);

enum{
	Y_POS = 0,
        X_POS = 1,
        PREDICT_POS = 2
};

struct vertex_data {
	vec pvec;
	int y;
	float xT_mu; 
        float sigma;
        float predict;

	vertex_data() {
		xT_mu = 0;
		y = 0;
                sigma  = 1;
		predict = 0;
	}
	void set_val(int index, float val){
		if (index == Y_POS){
			assert(val == -1 || val == 1);
			y = val;
		} 
                else if (index == X_POS)
			xT_mu = val;
 		else if (index == PREDICT_POS)
                        predict = val; 
 		else assert(false);
	}
	float get_val(int index){
		if (index == Y_POS)
			return y;
		else if (index == X_POS)
			return xT_mu;
                else if (index == PREDICT_POS)
                        return predict;
                else assert(false);
	}

};

struct edge_data{
	float x_ij;
	edge_data(float x_ij): x_ij(x_ij) {  };
	edge_data(){ x_ij = 1; };
};


#include "../collaborative_filtering/util.hpp"

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef edge_data EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;

#include "../collaborative_filtering/rmse.hpp"
#include "../collaborative_filtering/io.hpp"

/** compute probability for click as given in equation (2) */
float ctr_predict(const vertex_data& user, 
		const vertex_data& movie, 
		const float rating, 
		double & prediction, 
		void * extra = NULL){

	assert(beta > 0);
        prediction = movie.xT_mu;
	double prob = phi(movie.xT_mu * movie.y / beta);
	if (debug)
		//std::cout<<"prediction: " << prediction << " y: " << movie.y << std::endl;
		printf("prediction %12.8lf y: %d \n", prediction, movie.y);
	return prob; 
}

/* compute v(t) according to equation (9) left */
double v(double t){
	return gaussian_normalization * exp(-t*t/2) / phi(t);
}

/* compute w(t) according to equation (9) right */
double w(double t){
	double vt = v(t);
	return vt * (vt+t);
}

/**
 * program for computing the validation error (optional)
 */
struct AdPredictorValidationProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

	/**
	 * Called before an iteration is started.
	 */
	void before_iteration(int iteration, graphchi_context &gcontext) {
		err_vec = zeros(gcontext.execthreads);
	}

	/**
	 * Called after an iteration has finished.
	 */
	void after_iteration(int iteration, graphchi_context &gcontext) {
		std::cout<< gcontext.iteration  << " Avg validation error: " << std::setw(10)<<  sum(err_vec)/(double)Me << std::endl;
	}

	/**
	 *  Vertex update function.
	 */
	void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
                if (vertex.num_outedges() > 0){
                        assert(vertex.id() < Me);
                        assert(validation_targets[vertex.id()] == -1 || validation_targets[vertex.id()] == 1);
			double sum = 0;
                        for(int e=0; e < vertex.num_outedges(); e++) {
                             uint other = vertex.edge(e)->vertex_id();
                             assert(other >= M);
                             sum += mu_ij[other];
			}
                        double p0 = phi(-1 * sum / sqrt(beta));
                        double p1 = phi(1 * sum / sqrt(beta));
                        double predict = sum > 0 ? 1 : -1;                       
                        latent_factors_inmem[vertex.id()].predict = sum; 
 
                        if (predict != validation_targets[vertex.id()])
			   err_vec[omp_get_thread_num()]++;
                        if (debug)
                            std::cout<<"node: " << vertex.id() << " sum is: " << sum << " p0: " << p0 << " p1: " << p1 << " target: " << validation_targets[vertex.id()] << std::endl;
                }
	}


};



/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct AdPredictorVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

	/**
	 * Called before an iteration is started.
	 */
	void before_iteration(int iteration, graphchi_context &gcontext) {
		liklihood_vec = zeros(gcontext.execthreads);
		err_vec = zeros(gcontext.execthreads);
	}



	/**
	 * Called after an iteration has finished.
	 */
	void after_iteration(int iteration, graphchi_context &gcontext) {
		std::cout<< gcontext.iteration << ") Log likelihood: " << std::setw(10) << log(sum(liklihood_vec)) << " Avg error: " << std::setw(10) << sum(err_vec)/(double)M; 
	        if (validation != ""){
   	           AdPredictorValidationProgram vprogram;
		   metrics m("adpredictor_validation");
	           graphchi_engine<VertexDataType, EdgeDataType> vengine(validation, vshards, false, m); 
                   set_engine_flags(vengine);
	           vengine.run(vprogram, 1);
	        }
		else std::cout<<std::endl;

	}

	/**
	 *  Vertex update function.
	 */
	void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
		//go over all samples (rows)
		if ( vertex.num_outedges() > 0){

			assert(vertex.id() < M);
			vertex_data & row = latent_factors_inmem[vertex.id()]; 
                        assert(row.y == -1 || row.y == 1);

			if (debug)
				std::cout<<"Entered item " << vertex.id() << " y: " << row.y << std::endl;
			row.sigma = beta*beta;
			row.xT_mu = 0;

			//go over all features
			for(int e=0; e < vertex.num_outedges(); e++) {
                                uint feature_id = vertex.edge(e)->vertex_id();
				edge_data edge = vertex.edge(e)->get_data();                

				assert(sigma_ij[feature_id] > 0);
                                assert(edge.x_ij  == 1);

                                /* compute equation (6) */
				row.sigma += edge.x_ij * sigma_ij[feature_id];
                                /* compute the sum xT*w as needed in equations (7) and (8) */
				row.xT_mu += edge.x_ij * mu_ij[feature_id];
                                
			}
			double prediction;
			double ret = ctr_predict(row, row, row.y, prediction);
                        double predicted_target = prediction < 0 ? -1: 1;
			if (predicted_target == -1  && row.y == 1 || predicted_target == 1 && row.y == -1)
				err_vec[omp_get_thread_num()] += 1.0;  
                        if (debug)
                                std::cout<<"Prediction was: " << prediction << " real value: " << row.y << std::endl;
			liklihood_vec[omp_get_thread_num()] += ret;

			assert(row.sigma > 0);

			//go over all features
			for(int e=0; e < vertex.num_outedges(); e++) {
				edge_data edge = vertex.edge(e)->get_data();                
                                uint feature_id = vertex.edge(e)->vertex_id();
				assert(row.sigma > 0);
				double product = row.y * row.xT_mu / sqrt(row.sigma);
				mu_ij[feature_id] +=  (row.y * edge.x_ij *  sigma_ij[feature_id]  / sqrt(row.sigma)) * v(product);
				//if (debug)
				//    std::cout<<"Added to edge: "<< vertex.edge(e)->vertex_id() << " product: " << product << " v(product): " << v(product) << " value: " <<(row.y * edge.x_ij *  edge.sigma_ij * edge.sigma_ij / sqrt(row.sigma)) * v(product) << std::endl; 
				double factor = 1.0 - (edge.x_ij * sigma_ij[feature_id] / row.sigma)*w(product); 
				//if (debug)
				//    std::cout<<"Added to edge: "<< vertex.edge(e)->vertex_id() << " product: " << product << " w(product): " << w(product) << " factor: " << (1.0 - (edge.x_ij * edge.sigma_ij / row.sigma)*w(product)) << " sigma_ij " << edge.sigma_ij << "  product: " << edge.sigma_ij * factor << std::endl; 

				assert(factor > 0);
				sigma_ij[feature_id] *= factor;
                                assert(sigma_ij[feature_id] > 0);
			}

		}
	}


};



//dump output to file
void output_adpredictor_result(std::string filename) {
        MMOutputter_vec<vertex_data> predict_Vec(training + ".predict" ,0 ,Me, PREDICT_POS, "This file contains adpredictor output prediction vector. In each row a single prediction.", latent_factors_inmem);
        MMOutputter_vec<vertex_data> weight_Vec(training + ".mu_ij" ,M ,M+N, X_POS, "This file contains adpredictor output weight vector. In each row a single weight.", latent_factors_inmem);
	logstream(LOG_INFO) << "Adpredict output files (in matrix market format): " << filename << ".mu_ij" << std::endl;
}


int main(int argc, const char ** argv) {

	print_copyright();

	//* GraphChi initialization will read the command line arguments and the configuration file. */
	graphchi_init(argc, argv);

	/* Metrics object for keeping track of performance counters
	   and other information. Currently required. */

	/* Basic arguments for application. NOTE: File will be automatically 'sharded'. */
	beta       = get_option_float("beta", 1);
	debug      = get_option_int("debug", 0);

	parse_command_line_args();
	parse_implicit_command_line();
	D          = 0; //no feature vector is needed
	binary_relevance_threshold = 0; //treat all edge values as binary

	/* Preprocess data if needed, or discover preprocess files */
	int nshards = convert_matrixmarket<EdgeDataType>(training, 0, 0, 3, TRAINING, false);
	init_feature_vectors<std::vector<vertex_data> >(M+N, latent_factors_inmem, !load_factors_from_file);
       
	//read initial vector from file 
	std::cout << "Load CTR vector from file" << training << ":vec" << std::endl;
	load_matrix_market_vector(training + ":vec", Y_POS, false, false);

        mu_ij    = zeros(M+N);
        sigma_ij = ones(M+N);

	if (validation != ""){
                //read validation data (optional)
		vshards = convert_matrixmarket<EdgeDataType>(validation, 0, 0, 3, VALIDATION, false);
		validation_targets = load_matrix_market_vector(validation + ":vec", false, false);                
                Me = validation_targets.size();
	}


	print_config();

	/* Run */
	AdPredictorVerticesInMemProgram program;
	metrics m("adpredictor");
	graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
        set_engine_flags(engine);
	pengine = &engine;
	engine.run(program, niters);

	/* Output latent factor matrices in matrix-market format */
	output_adpredictor_result(training);

	/* Report execution metrics */
	if (!quiet)
		metrics_report(m);

	return 0;
}

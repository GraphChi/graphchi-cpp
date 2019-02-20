
/**
 * @file
 * @author  Aapo Kyrola <akyrola@cs.cmu.edu>
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright [2012] [Aapo Kyrola, Guy Blelloch, Carlos Guestrin / Carnegie Mellon University]
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
 * Template for GraphChi applications. To create a new application, duplicate
 * this template.
 */

//Modified by Feiyu Yu

#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <iostream>

#include "graphchi_basic_includes.hpp"
#include "util/toplistBP.hpp"
using namespace graphchi;


#define GRAPHCHI_DISABLE_COMPRESSION
#define THRESHOLD 1e-1    
#define RANDOMRESETPROB 0.15
#define Init 1





/**
  * Type definitions. Remember to create suitable graph shards using the
  * Sharder-program. 
  */
typedef VertextValue VertexDataType;
typedef EdgeValue EdgeDataType;


void parse(VertextValue &x, const char * s) { } // Do nothing
/**
  * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
  * class. The main logic is usually in the update function.
  */
struct MyGraphChiProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
 	float normP;
	float normN;
	int flag;
    /**
     *  Vertex update function.
     */
    void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {

        if (gcontext.iteration == 0) {
            /* On first iteration, initialize vertex (and its edges). This is usually required, because
               on each run, GraphChi will modify the data files. To start from scratch, it is easiest
               do initialize the program in code. Alternatively, you can keep a copy of initial data files. */
            // vertex.set_data(init_value);
        //(std::rand()%100+1)/100
		for(int i=0; i < vertex.num_inedges(); i++) {
			VertextValue veValue;
		
			veValue.belifP =0;
			veValue.belifN =0;


			vertex.set_data(veValue);

			EdgeValue EdValue;
			EdValue.PhiP =  (float)(std::rand()%100)/100;
			//if(EdValue.PhiP<0.1)
			//	EdValue.PhiP = EdValue.PhiP +0.1;
			EdValue.PhiN =  (float)(std::rand()%100)/100;
			//if(EdValue.PhiN<0.1)
			//	EdValue.PhiN = EdValue.PhiN +0.1;
			//std::cout<<EdValue.PhiN<<std::endl;
			EdValue.PsiPP = (float)(std::rand()%100)/100;
			EdValue.PsiPN = (float)(std::rand()%100)/100;
			EdValue.PsiNP = (float)(std::rand()%100)/100;
			EdValue.PsiNN = (float)(std::rand()%100)/100;
			EdValue.mP = 1;
			EdValue.mN = 1;

			vertex.inedge(i) -> set_data(EdValue);
			 }

        } 

		else{

		if(flag ==1){

			 for(int i=0; i < vertex.num_inedges(); i++) {
				  EdgeValue value =  vertex.inedge(i)->get_data();
				  
				   value.mP =  value.mP/normP;
				   value.mN =  value.mN/normN;

				   vertex.inedge(i)->set_data(value);
			   }
			

			}

		else if(flag == 0) {

			float sumP=0;
			float sumN=0;

            /* Do computation */ 

            ///* Loop over in-edges (example) */
            for(int i=0; i < vertex.num_inedges(); i++) {

				EdgeValue EdValue;
				EdValue = vertex.inedge(i) -> get_data();
				float mp1 =  EdValue.PhiP * EdValue.PsiPP;
				float mp2 =  EdValue.PhiP * EdValue.PsiPN;
				float mp3 =  EdValue.PhiN * EdValue.PsiNP;
				float mp4 =  EdValue.PhiN * EdValue.PsiNN;
				
				float PiMP = 1;
				float PiMN = 1;
				for(int j=0; j < vertex.num_inedges(); j++){
					if(j != i)
					{
						EdgeValue EdValueS;
						EdValueS = vertex.inedge(i) -> get_data();
						PiMP = PiMP * EdValueS.mP;
						PiMN = PiMN * EdValueS.mN;
					}
				}

					
			EdValue.mP = (mp1+mp2) * PiMP;
			EdValue.mN = (mp3+mp4) * PiMN;
			sumP = sumP + EdValue.mP;
			sumN = sumN + EdValue.mN;
			//vertex.inedge(i) -> set_data(EdValue);
      
			}

			normP = normP + sumP * sumP;
			normN = normN + sumN * sumN;
			
		}

		else if(flag == 2) {
			float PiMjiP = 1;
			float PiMjiN = 1;
			VertextValue VeValue;
			VeValue = vertex.get_data();
			for(int i=0; i < vertex.num_inedges(); i++) {
				EdgeValue EdValueS;
				EdValueS = vertex.inedge(i) -> get_data();
				PiMjiP = PiMjiP * EdValueS.mP;
				
				PiMjiN = PiMjiN * EdValueS.mN;
				//std::cout<<PiMjiP<<std::endl;
				//std::cout<<PiMjiN<<std::endl;
			  }
			for(int i=0; i < vertex.num_inedges(); i++) {
				EdgeValue EdValueS;
				EdValueS = vertex.inedge(i) -> get_data();
			VeValue.belifP = EdValueS.PhiP * PiMjiP;
			VeValue.belifN = EdValueS.PhiN * PiMjiN;
			vertex.set_data(VeValue);
			}
		}
        }
	}
    
    /**
     * Called before an iteration starts.
     */
    void before_iteration(int iteration, graphchi_context &gcontext) {

		if(iteration == 0)
		{
			flag = 1;
			 normP=0;
			 normN=0;
		}
		if(flag == 0)
		{
			 normP=0;
			 normN=0;
		}
		else if (flag == 1)
		{
			//std::cout<<"iterBefore:"<<iteration<<"normH"<<normH<<std::endl;
				normP = sqrt(normP);
				normN = sqrt(normN);
		}

    }
    
    /**
     * Called after an iteration has finished.
     */
    void after_iteration(int iteration, graphchi_context &gcontext) {


		if(iteration == 4){
		flag = 2;
		}
		else{
		if(flag == 1){
				flag = 0;
				//std::cout<<"Flag0iter:"<<iteration<<"normH"<<normH<<std::endl;
		}
		else if(flag == 0){
				flag = 1;
				//std::cout<<"iterAfter:"<<iteration<<"normH"<<normH<<std::endl;
			}
		}


    }
    
    /**
     * Called before an execution interval is started.
     */
    void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
    }
    
    /**
     * Called after an execution interval has finished.
     */
    void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {        
    }
    
};

int main(int argc, const char ** argv) {
    /* GraphChi initialization will read the command line 
       arguments and the configuration file. */
    graphchi_init(argc, argv);
    
    /* Metrics object for keeping track of performance counters
       and other information. Currently required. */
    metrics m("BP");
    
    /* Basic arguments for application */
    std::string filename = get_option_string("file");  // Base filename
    int niters           = get_option_int("niters", 6); // Number of iterations
    //bool scheduler       = get_option_int("scheduler", 0); // Whether to use selective scheduling
     bool scheduler          = false;                    // Non-dynamic version of pagerank.
	 int ntop                = get_option_int("top", 20);
    /* Detect the number of shards or preprocess an input to create them */
    int nshards          = convert_if_notexists<EdgeDataType>(filename, 
                                                            get_option_string("nshards", "auto"));
    
    /* Run */
    MyGraphChiProgram program;
    graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, scheduler, m); 
    engine.run(program, niters);
    
		/* Output top 20 authorities*/
    std::vector< vertex_value<VertextValue> > topA = get_top_vertices<VertextValue>(filename, ntop,1);
    std::cout << "Print top " << ntop << " Negative" << std::endl;
    for(int i=0; i < (int)topA.size(); i++) {
        std::cout << (i+1) << ". " << topA[i].vertex << "\t" << topA[i].value.belifN << std::endl;
    }

	/* Output top 20 authorities*/
    std::vector< vertex_value<VertextValue> > topH = get_top_vertices<VertextValue>(filename, ntop,0);
    std::cout << "Print top " << ntop << " Positive:" << std::endl;
    for(int i=0; i < (int)topH.size(); i++) {
        std::cout << (i+1) << ". " << topH[i].vertex << "\t" << topH[i].value.belifP << std::endl;
    }
    /* Report execution metrics */
    metrics_report(m);
    return 0;
}

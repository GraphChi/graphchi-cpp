
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
///#define DYNAMICEDATA 1
//#define DYNAMICVERTEXDATA 1  

#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <iostream>



#include "graphchi_basic_includes.hpp"
//#include "api/dynamicdata/chivector.hpp"

//#include "util/toplist2.hpp"
#include "util/toplistHITS.hpp"
/* ALS-related classes are contained in als.hpp *///
//#include "als.hpp"

using namespace graphchi;


#define GRAPHCHI_DISABLE_COMPRESSION
#define THRESHOLD 1e-1    
#define RANDOMRESETPROB 0.15
#define Init 1
/**
  * Type definitions. Remember to create suitable graph shards using the
  * Sharder-program. 
  */
//typedef my_vertex_type VertexDataType;
//typedef my_edge_type EdgeDataType;

//struct HA_label {
//    float hub;
//    float auth;
//};



typedef HA_label VertexDataType;
//typedef float VertexDataType;
typedef HA_label EdgeDataType;
//typedef chivector<vid_t> VertexDataType;
//typedef int VertexDataType;
/**
  * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
  * class. The main logic is usually in the update function.
  */
void parse(HA_label &x, const char * s) { } // Do nothing



struct HITSProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {
    
	float normH;
	float normA;
	int flag;
	
    void update(graphchi_vertex<VertexDataType, EdgeDataType> &v, graphchi_context &ginfo) {
		
		
		

		 if (ginfo.iteration == 0) {
			   for(int i=0; i < v.num_edges(); i++) {
				   HA_label havalue;
				   havalue.hub = 1;
				   havalue.auth = 1;
				   v.edge(i)->set_data(havalue);
			
				   HA_label havalueVex;
				   havalueVex.hub = 0;
				   havalueVex.auth = 0;
					v.set_data(havalueVex); 
				   //v.set_data(0);
			   }

		 }
		 else{
			
			  if(flag == 1){


				  for(int i=0; i < v.num_outedges(); i++) {
				  HA_label havalue =  v.outedge(i)->get_data();
				  //std::cout<<"normH:  "<<normH <<std::endl;
				   havalue.hub =  havalue.hub/normH;
				 //  if(havalue.hub > 1)
				  // {std::cout<<">1: "<<havalue.hub*normH<<"nuormH: "<<normH<<std::endl;}
				   havalue.auth =  havalue.auth/normA;
				  // std::cout<<"qurt:  "<<havalue.hub <<std::endl;
				   v.outedge(i)->set_data(havalue);

				 //  HA_label Vexhavalue = v.get_data();
				 //  Vexhavalue.hub =  Vexhavalue.hub ;
				 //  Vexhavalue.auth = havalue.auth;
					//v.set_data(Vexhavalue); 
				//v.set_data(havalue.auth); 
			   }
			
				/*  for(int i=0; i < v.num_outedges(); i++) {
					HA_label havalue =  v.outedge(0)->get_data();
					HA_label Vexhavalue = v.get_data();
					Vexhavalue.hub = havalue.hub;
					Vexhavalue.auth = Vexhavalue.auth;
					v.set_data(Vexhavalue); 
				 }*/
			 }
			  else if(flag == 2){
				  
				  for(int i=0; i < v.num_inedges(); i++) {
				  HA_label havalue =  v.inedge(i)->get_data(); 
				  
				   HA_label Vexhavalue = v.get_data();
				   Vexhavalue.hub =  Vexhavalue.hub ;
				   //std::cout<<"qurt:  "<<havalue.hub <<std::endl;
				   Vexhavalue.auth = havalue.auth;
					v.set_data(Vexhavalue); 
				
			   }
				   for(int i=0; i < v.num_outedges(); i++) {
					HA_label havalue =  v.outedge(i)->get_data();
					HA_label Vexhavalue = v.get_data();
					Vexhavalue.hub = havalue.hub;
					 
					Vexhavalue.auth = Vexhavalue.auth;
					v.set_data(Vexhavalue); 
				 }
			  }
			  else if(flag == 0){
					float sumA=0;
					float sumH=0;
				//std::cout<<"pow:  "<<ginfo.iteration <<std::endl;
				 
			 ///H(x)
            for(int i=0; i < v.num_outedges(); i++) {

				//graphchi_edge<chivector<vid_t> > * outedge = v.outedge(i);
				HA_label havalue =  v.outedge(i)->get_data();
				float val = havalue.auth;
				//std::cout<<normH<<std::endl;
					sumH = sumH + val;               
				
            }
			normH = normH + sumH * sumH;
			
			
			//std::cout<<"iterlalala:"<<ginfo.iteration<<"normH"<<normH<<std::endl;
			//std::cout<<normH<<std::endl;
			for(int i=0; i < v.num_outedges(); i++) {
				//graphchi_edge<chivector<vid_t> > * outedge = v.outedge(i);
				  HA_label havalue =  v.outedge(i)->get_data();				  
				  havalue.hub = sumH; 
				  havalue.auth = havalue.auth;
				  v.outedge(i)->set_data(havalue);
            }
			//for(int i=0; i < v.num_outedges(); i++) {
			//	//graphchi_edge<chivector<vid_t> > * outedge = v.outedge(i);
			//	  HA_label havalue =  v.outedge(i)->get_data();				  
			//	 
   //         }
			
			 ///A(x)
            for(int i=0; i < v.num_inedges(); i++) {

				//graphchi_edge<chivector<vid_t> > * inedge = v.inedge(i);
				HA_label havalue =  v.inedge(i)->get_data();
				float val = havalue.hub;
					sumA = sumA + val;   

            }
			normA = normA + sumA * sumA;

			for(int i=0; i < v.num_inedges(); i++) {
				  HA_label havalue =  v.inedge(i)->get_data();
				  havalue.hub = havalue.hub;
				  havalue.auth = sumA;
				  v.inedge(i)->set_data(havalue);                     
            }
			 
		 //v.set_data(sumA); 
		 
		 
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
			 normH=0;
			 normA=0;
		}
		if(flag == 0)
		{
			 normH=0;
			 normA=0;
		}
		else if (flag == 1)
		{
			//std::cout<<"iterBefore:"<<iteration<<"normH"<<normH<<std::endl;
				normA = sqrt(normA);
				normH = sqrt(normH);
		}
	} 
    /**
     * Called after an iteration has finished.
     */
    void after_iteration(int iteration, graphchi_context &gcontext) {
		if(iteration == 8){
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

	graphchi_init(argc, argv);
    metrics m("HITS");
    global_logger().set_log_level(LOG_DEBUG);
	//std::vector<int> AValue;

	 /* Parameters */
    std::string filename    = get_option_string("file"); // Base filename
    int niters              = get_option_int("niters", 10);
    bool scheduler          = false;                    // Non-dynamic version of pagerank.
    int ntop                = get_option_int("top", 20);

	//bool preexisting_shards;
    //int nshards          = convert_if_notexists<vid_t>(filename, get_option_string("nshards", "auto"), preexisting_shards);
	 /* Process input file - if not already preprocessed */
    int nshards             = convert_if_notexists<EdgeDataType>(filename, get_option_string("nshards", "auto"));
	
   

 /* Run */
    graphchi_engine< VertexDataType, EdgeDataType > engine(filename, nshards, scheduler, m); 
   
        HITSProgram program;
        engine.run(program, niters);
 /*if (preexisting_shards) {
        engine.reinitialize_edge_data(0);
    }*/
		/* Output top 20 authorities*/
    std::vector< vertex_value<HA_label> > topA = get_top_vertices<HA_label>(filename, ntop,1);
    std::cout << "Print top " << ntop << " Authorities:" << std::endl;
    for(int i=0; i < (int)topA.size(); i++) {
        std::cout << (i+1) << ". " << topA[i].vertex << "\t" << topA[i].value.auth << std::endl;
    }

	/* Output top 20 authorities*/
    std::vector< vertex_value<HA_label> > topH = get_top_vertices<HA_label>(filename, ntop,0);
    std::cout << "Print top " << ntop << " Hub:" << std::endl;
    for(int i=0; i < (int)topH.size(); i++) {
        std::cout << (i+1) << ". " << topH[i].vertex << "\t" << topH[i].value.hub << std::endl;
    }
    /* Report execution metrics */
    metrics_report(m);
    return 0;
}

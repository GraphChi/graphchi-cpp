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
 *
 *
 *  This program reads a text input file, where each line 
 *  is taken from another document. The program counts the number of word
 *  occurances for each line (document) and outputs a document word count to be used
 *  in LDA.
 */


#include <cstdio>
#include <iostream>
#include <map>
#include <omp.h>
#include <assert.h>
#include "graphchi_basic_includes.hpp"
#include "../collaborative_filtering/timer.hpp"
#include "../collaborative_filtering/util.hpp"
#include "common.hpp"
#include <math.h>
#include <iomanip>
using namespace std;
using namespace graphchi;

bool debug = false;
timer mytime;
size_t lines;
unsigned long long total_lines = 0;
string dir;
string outdir;
std::vector<std::string> in_files;
//non word tokens that will be removed in the parsing
//it is possible to add additional special characters or remove ones you want to keep
const char spaces[] = {" \r\n\t!?@#$%^&*()-+.,~`\";:/"};
const char qoute[] = {",\""};
const char comma[] = {","};
int has_header_titles = 1;
std::map<std::string, int> p_x;
std::map<std::string, int> p_y;
int n = 0;
std::vector<std::string> header_titles;
int from_val = -1; int to_val = -1;

void parse(int i){    
	in_file fin(in_files[i]);

	size_t linesize = 0;
	char * saveptr = NULL, * saveptr2 = NULL,* linebuf = NULL;
	size_t line = 1;
	uint id;
	if (has_header_titles){
		char * linebuf = NULL;
		size_t linesize;
		char linebuf_debug[1024];

		/* READ LINE */
		int rc = getline(&linebuf, &linesize, fin.outf);
		if (rc == -1)
			logstream(LOG_FATAL)<<"Error header line " << " [ " << linebuf_debug << " ] " << std::endl;

		strncpy(linebuf_debug, linebuf, 1024);
		char *pch = strtok(linebuf,"\t,\r;\"");
		if (pch == NULL)
			logstream(LOG_FATAL)<<"Error header line " << " [ " << linebuf_debug << " ] " << std::endl;
		for (int j=0; j < strlen(pch); j++) if (pch[j] == ' ') pch[j] = '_';
		header_titles.push_back(pch);
		if (debug) printf("Found title: %s\n", pch);
		while (pch != NULL){
			pch = strtok(NULL, "\t,\r;\"");
			if (pch == NULL || pch[0] == '\0')
				break;
			for (int j=0; j < strlen(pch); j++) if (pch[j] == ' ') pch[j] = '_';
			header_titles.push_back(pch);
			if (debug) printf("Found title: %s\n", pch);
		}
	}


	while(true){

		int rc = getline(&linebuf, &linesize, fin.outf);
		if (rc < 1)
			return;

		int index = 0;
		char frombuf[256];
		char tobuf[256];
		char *pch = strtok_r(linebuf, ",\"", &saveptr);
		if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return; }
		if (debug) printf("Found token 1 %s\n", pch);
		if (pch[0] == '"')
			pch++;

		index++;
		bool found_from = false, found_to = false;
		int from,to;

		if (index == from_val){
			strncpy(frombuf, pch, 256);
			found_from = true;
		}
		if (index == to_val){
			strncpy(tobuf, pch, 256);
			found_to = true;
		}

		while(true){
			pch = strtok_r(NULL, ",\"", &saveptr);
			if (pch == NULL)
				break;
			index++;
			if (debug) printf("Found token %d %s\n", index, pch);

			if (pch[0] == '"')
				pch++;

			if (index > from_val && index > to_val)
				break;

			if (index == from_val){
				strncpy(frombuf, pch, 256);
				found_from = true;
			}
			if (index == to_val){
				strncpy(tobuf, pch, 256);
				found_to = true;
			}
		}
		char totalbuf[512];
		assert(found_from && found_to);
		sprintf(totalbuf, "%s_%s", frombuf, tobuf);

		if (debug) printf("Incrementing map: %s\n", totalbuf);
		frommap.string2nodeid[totalbuf]++;
		p_x[frombuf]++;
		p_y[tobuf]++;
		n++;
	}
}


int main(int argc,  const char *argv[]) {

	logstream(LOG_WARNING)<<"GraphChi parsers library is written by Danny Bickson (c). Send any "
		" comments or bug reports to danny.bickson@gmail.com " << std::endl;
	global_logger().set_log_level(LOG_INFO);
	global_logger().set_log_to_console(true);

	graphchi_init(argc, argv);

	debug = get_option_int("debug", 0);
	dir = get_option_string("file_list");
	lines = get_option_int("lines", 0);
	omp_set_num_threads(get_option_int("ncpus", 1));
	from_val = get_option_int("from_val", from_val);
	to_val = get_option_int("to_val", to_val);
	if (from_val == -1)
		logstream(LOG_FATAL)<<"Must set from/to " << std::endl;
	mytime.start();

	FILE * f = fopen(dir.c_str(), "r");
	if (f == NULL)
		logstream(LOG_FATAL)<<"Failed to open file list!"<<std::endl;

	while(true){
		char buf[256];
		int rc = fscanf(f, "%s\n", buf);
		if (rc < 1)
			break;
		in_files.push_back(buf);
	}

	if (in_files.size() == 0)
		logstream(LOG_FATAL)<<"Failed to read any file frommap from the list file: " << dir << std::endl;

#pragma omp parallel for
	for (int i=0; i< (int)in_files.size(); i++)
		parse(i);

	std::cout << "Finished in " << mytime.current_time() << std::endl;

	int total_x =0 , total_y = 0;
	std::map<std::string, int>::iterator it;
	double h = 0;
	for (it = p_x.begin(); it != p_x.end(); it++){
		total_x+= it->second;
		h-= (it->second / (double)n)*log2(it->second / (double)n);
	}
	for (it = p_y.begin(); it != p_y.end(); it++)
		total_y+= it->second;
	assert(total_x == n);
	assert(total_y == n);


	double mi = 0;
	std::map<std::string, uint>::iterator iter;
	assert(n != 0);

	int total_p_xy = 0;
	for (iter = frommap.string2nodeid.begin() ; iter != frommap.string2nodeid.end(); iter++){
		double p_xy = iter->second / (double)n;
		assert(p_xy > 0);
		char buf[256];
		strncpy(buf, iter->first.c_str(), 256);
		char * first = strtok(buf, "_");
		char * second = strtok(NULL, "\n\r ");
		assert(first && second);
		double px = p_x[first] / (double)n;
		double py = p_y[second] / (double)n;
		assert(px > 0 && py > 0);
		mi += p_xy * log2(p_xy / (px * py));
		total_p_xy += iter->second;
	}
	assert(total_p_xy == n);
	logstream(LOG_INFO)<<"Total examples: " <<n << std::endl;

	logstream(LOG_INFO)<<"Unique p(x) " << p_x.size() << std::endl;
	logstream(LOG_INFO)<<"Unique p(y) " << p_y.size() << std::endl;
	logstream(LOG_INFO)<<"Average F(x) " << total_x / (double)p_x.size() << std::endl;
	logstream(LOG_INFO)<<"Average F(y) " << total_y / (double)p_y.size() << std::endl;

	std::cout<<"Mutual information of " << from_val << " [" << header_titles[from_val-1] << "] <-> " << to_val << " [" << header_titles[to_val-1] << "] is: " ;
	if (mi/h > 1e-3) 
		std::cout<<std::setprecision(3) << mi << std::endl;
	else std::cout<<"-"<<std::endl;
	save_map_to_text_file(frommap.string2nodeid, outdir + dir + "map.text");
	logstream(LOG_INFO)<<"Saving map file " << outdir << dir << "map.text" << std::endl;
	return 0;
}




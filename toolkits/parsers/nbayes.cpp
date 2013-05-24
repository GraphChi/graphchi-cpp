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
std::vector<std::string> header_titles;
int from_val = -1; int to_val = -1;
int mid_val = -1;

void parse(int i){    
	in_file fin(in_files[i]);
	out_file fout((outdir + in_files[i] + ".out"));

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
		char *pch = strtok(linebuf,"\t,\r; ");
		if (pch == NULL)
			logstream(LOG_FATAL)<<"Error header line " << " [ " << linebuf_debug << " ] " << std::endl;

		header_titles.push_back(pch);
		if (debug) printf("Found title: %s\n", pch);
		while (pch != NULL){
			pch = strtok(NULL, "\t,\r; ");
			if (pch == NULL)
				break;
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
		char *pch = strtok_r(linebuf, ",", &saveptr);
		if (!pch){ logstream(LOG_ERROR) << "Error when parsing file: " << in_files[i] << ":" << line << "[" << linebuf << "]" << std::endl; return; }
		fprintf(fout.outf,"\"%s\",", pch);
		if (debug) printf("Found token 1 %s\n", pch);
		if (pch[0] == '"')
			pch++;

		index++;
		int from,to,mid;

		if (index == from_val)
			from = atoi(pch);

		if (index == to_val)
			to = atoi(pch);

		if (index == mid_val)
			mid = atoi(pch);

		while(true){
			pch = strtok_r(NULL, ",", &saveptr);
			if (pch == NULL)
				break;
			index++;
			if (debug) printf("Found token %d %s\n", index, pch);

			if (pch[0] == '"')
				pch++;

			if (index == from_val)
				from = atoi(pch);

			if (index == to_val)
				to = atoi(pch);

			if (index == mid_val)
				mid = atoi(pch);
		}
		char totalbuf[512];
		if (mid_val == -1 && to_val == -1)
			sprintf(totalbuf, "%d\t", from);
		else if (mid_val != -1)
			sprintf(totalbuf, "%d\t%d\t%d\t", from, mid, to);
		else 
			sprintf(totalbuf, "%d\t%d\t", from, to);
		if (debug) printf("Incrementing map: %s\n", totalbuf);
		frommap.string2nodeid[totalbuf]++;
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
	mid_val = get_option_int("mid_val", mid_val);
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

	save_map_to_text_file(frommap.string2nodeid, outdir + dir + "map.text");
	return 0;
}




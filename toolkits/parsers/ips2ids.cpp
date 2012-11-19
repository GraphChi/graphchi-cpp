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
 *  Written by Stefan Weigert, TUD based on the
 *  "consecutive_matrix_market.cpp" parser
 */


#include <cstdio>
#include <map>
#include <omp.h>
#include <cassert>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "graphchi_basic_includes.hpp"
#include "../collaborative_filtering/util.hpp"
using namespace graphchi;


// global options
int debug = 0;
int csv = 0;
int tsv = 0;
const char* token;
std::vector<std::string> in_files;

// global data-structs
mutex ids_mutex;
std::map<unsigned int, unsigned int> ids;

/*
 * The expected format is one of the following:
 *
 *
 * <source-ip>\t<dst-ip>\t<some attribute>
 * <source-ip>,<dst-ip>,<other stuff>
 * <source-ip> <dst-ip> <other stuff>
 */
void parse(int i) {
    uint64_t max_id = 1;

    in_file fin(in_files[i].c_str());
    out_file fout(std::string(in_files[i] + ".out").c_str());

    uint64_t lines = 0;
    uint64_t hits = 0;
    while(true) {
        struct in_addr caller;
        struct in_addr callee;
        char line[256];
        
        char* res = fgets(line, 256, fin.outf);
        // end of file
        if (res == NULL) {
            break;
        }
        
        char* caller_str = strtok(line, token);
        assert(caller_str != NULL);
        char* callee_str = strtok(NULL, token);
        assert(callee_str != NULL);
        char* attribute = strtok(NULL, token);
        assert(attribute != NULL);
        
        // try to convert caller - if it goes wrong just continue
        if (inet_aton(caller_str, &caller) == 0) {
            if (debug) logstream(LOG_WARNING) << "could not convert caller-ip:" << caller_str << std::endl;
            continue;
        }
        // try to convert caller - if it goes wrong just continue
        if (inet_aton(callee_str, &callee) == 0) {
            if (debug) logstream(LOG_WARNING) << "could not convert callee-ip:" << caller_str << std::endl;
            continue;
        }

        // check if we know that ip already
        if (ids.find(caller.s_addr) == ids.end()) {
            // we don't - IDs are global to all threads and files, so
            // lock
            ids_mutex.lock();
            // another thread might have added that IP
            // meanwhile. non-existing keys have a value of 0
            // initially
            if (ids[caller.s_addr] == 0) {
                // do the actual update
                ids[caller.s_addr] = max_id;
                ++max_id;
            }
            ids_mutex.unlock();
        } else {
            ++hits;
        }

        // repeat the same for the callee
        if (ids.find(callee.s_addr) == ids.end()) {
            ids_mutex.lock();
            if (ids[callee.s_addr] == 0) {
                ids[callee.s_addr] = max_id;
                ++max_id;
            }
            ids_mutex.unlock();
        } else {
            ++hits;
        }

        // attribute already contains the '\n' since it's the last
        // string on the line
        if (tsv) {
            fprintf(fout.outf, "%u\t%u\t%s", ids[caller.s_addr], ids[callee.s_addr], attribute);
        } else if (csv) {
            fprintf(fout.outf, "%u,%u,%s", ids[caller.s_addr], ids[callee.s_addr], attribute);
        } else {
            fprintf(fout.outf, "%u %u %s", ids[caller.s_addr], ids[callee.s_addr], attribute);
        }
        if (++lines % 100000 == 0) {
            logstream(LOG_INFO) << "Edges: " << lines
                                << ", Ids: " << ids.size()
                                << ", Hits: " << hits << "\n";
        }
    } 

    logstream(LOG_INFO) << "Finished parsing " << in_files[i] << std::endl;
}


int main(int argc,  const char *argv[]) {
    global_logger().set_log_level(LOG_INFO);
    global_logger().set_log_to_console(true);

    graphchi_init(argc, argv);

    // get options
    debug = get_option_int("debug", 0);
    std::string dir = get_option_string("file");
    omp_set_num_threads(get_option_int("ncpus", 1));
    tsv = get_option_int("tsv", 0); // is this tab seperated file?
    csv = get_option_int("csv", 0); // is the comma seperated file?
    if (tsv) token = "\t";
    else if (csv) token = ",";
    else token = " ";

    
    // read list-of files
    FILE * f = fopen(dir.c_str(), "r");
    if (f == NULL)
        logstream(LOG_FATAL) << "Failed to open file list!" << std::endl;

    while(true){
        char buf[256];
        int rc = fscanf(f, "%s\n", buf);
        if (rc < 1)
            break;
        in_files.push_back(buf);
    }

    // process each file (possibly in parallel)
#pragma omp parallel
    for(unsigned i = 0; i < in_files.size(); ++i) {
        parse(i);
    }

    // serialize the id-map
    logstream(LOG_INFO) << "serializing ids..." << std::endl;
    out_file fout(std::string(dir + ".map").c_str());
    for (std::map<unsigned int, unsigned int>::const_iterator it = ids.begin();
         it != ids.end();
         ++it)
    {
        if (tsv) fprintf(fout.outf, "%u\t%u\n", it->first, it->second);
        else if (csv) fprintf(fout.outf, "%u,%u\n", it->first, it->second);
        else fprintf(fout.outf, "%u %u\n", it->first, it->second);
    }

    return 0;
}




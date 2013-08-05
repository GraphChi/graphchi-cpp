;/**
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
  * Graph conversion and parsing routines.
  */

#ifndef GRAPHCHI_CONVERSIONS_DEF
#define GRAPHCHI_CONVERSIONS_DEF

#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>



#include "graphchi_types.hpp"
#include "logger/logger.hpp"
#include "preprocessing/sharder.hpp"
#include "preprocessing/formats/binary_adjacency_list.hpp"

/**
 * GNU COMPILER HACK TO PREVENT WARNINGS "Unused variable", if
 * the particular app being compiled does not use a function.
 */
#ifdef __GNUC__
#define VARIABLE_IS_NOT_USED __attribute__ ((unused))
#else
#define VARIABLE_IS_NOT_USED
#endif

namespace graphchi {
    
    struct dummy {};

    
    /* Simple string to number parsers */
    static void VARIABLE_IS_NOT_USED parse(int &x, const char * s);
    static void VARIABLE_IS_NOT_USED parse(unsigned int &x, const char * s);
    static void VARIABLE_IS_NOT_USED parse(float &x, const char * s);
    static void VARIABLE_IS_NOT_USED parse(long &x, const char * s);
    static void VARIABLE_IS_NOT_USED parse(char &x, const char * s);
    static void VARIABLE_IS_NOT_USED parse(bool &x, const char * s);
    static void VARIABLE_IS_NOT_USED parse(double &x, const char * s);
    static void VARIABLE_IS_NOT_USED parse(short &x, const char * s);
    static void FIXLINE(char * s);
    
    static void parse(int &x, const char * s) {
        x = atoi(s);
    }
    
    static void parse(unsigned int &x, const char * s) {
        x = (unsigned int) strtoul(s, NULL, 10);
    }
    
    static void parse(float &x, const char * s) {
        x = (float) atof(s);
    }
    
    
    /**
     * Special templated parser for PairContainers.
     */
    template <typename T>
    void parse(PairContainer<T> &x, const char * s) {
        parse(x.left, s);
        parse(x.right, s);
    }
    
    static void parse(long &x, const char * s) {
        x = atol(s);
    }
    
    static void parse(char &x, const char * s) {
        x = s[0];
    }
    
    static void parse(bool &x, const char * s) {
        x = atoi(s) == 1;
    }
    
    static  void parse(double &x, const char * s) {
        x = atof(s);
    }
    
    static void parse(short &x, const char * s) {
        x = (short) atoi(s);
    }
    
#ifdef DYNAMICEDATA
    static void VARIABLE_IS_NOT_USED parse_multiple(std::vector<dummy> &values, char * s);

    void parse_multiple(std::vector<dummy> & values, char * s) {
        assert(false);
    }
    
    /**
     * Parse ':' -delimited values into a vector.
     */
    template <typename T>
    static void parse_multiple(typename std::vector<T> & values, char * s) {
        char delims[] = ":";
        char * t;
        t = strtok(s, delims);
        T x;
        parse(x, (const char*) t);
        values.push_back(x);
        while((t = strtok(NULL, delims)) != NULL) {
            parse(x, (const char*) t);
            values.push_back(x);
        }
    }
    
#endif
    
    
    // Catch all
    template <typename T>
    void parse(T &x, const char * s) {
        logstream(LOG_FATAL) << "You need to define parse<your-type>(your-type &x, const char *s) function"
        << " to support parsing the edge value." << std::endl;
        assert(false);
    }
    
    
    
    // Removes \n from the end of line
    void FIXLINE(char * s) {
        int len = (int) strlen(s)-1;
        if(s[len] == '\n') s[len] = 0;
    }
    
    
    // http://www.linuxquestions.org/questions/programming-9/c-list-files-in-directory-379323/
    int getdir (std::string dir, std::vector<std::string> &files);
    int getdir (std::string dir, std::vector<std::string> &files)
    {
        DIR *dp;
        struct dirent *dirp;
        if((dp  = opendir(dir.c_str())) == NULL) {
            std::cout << "Error(" << errno << ") opening " << dir << std::endl;
            return errno;
        }
        
        while ((dirp = readdir(dp)) != NULL) {
            files.push_back(std::string(dirp->d_name));
        }
        closedir(dp);
        return 0;
    }
    
    std::string get_dirname(std::string arg);
    std::string get_dirname(std::string arg) {
        size_t a = arg.find_last_of("/");
        if (a != arg.npos) {
            std::string dir = arg.substr(0, a);
            return dir;
        } else {
            assert(false);
        }
    }
    
    std::string get_filename(std::string arg);
    std::string get_filename(std::string arg) {
        size_t a = arg.find_last_of("/");
        if (a != arg.npos) {
            std::string f = arg.substr(a + 1);
            return f;
        } else {
            assert(false);
        }
    }
    
    /**
     * Converts graph from an edge list format. Input may contain
     * value for the edges. Self-edges are ignored.
     */
    template <typename EdgeDataType>
    void convert_edgelist(std::string inputfile, sharder<EdgeDataType> &sharderobj, bool multivalue_edges=false) {
        
        FILE * inf = fopen(inputfile.c_str(), "r");
        size_t bytesread = 0;
        size_t linenum = 0;
        if (inf == NULL) {
            logstream(LOG_FATAL) << "Could not load :" << inputfile << " error: " << strerror(errno) << std::endl;
        }
        assert(inf != NULL);
        
        logstream(LOG_INFO) << "Reading in edge list format!" << std::endl;
        char s[1024];
        while(fgets(s, 1024, inf) != NULL) {
            linenum++;
            if (linenum % 10000000 == 0) {
                logstream(LOG_DEBUG) << "Read " << linenum << " lines, " << bytesread / 1024 / 1024.  << " MB" << std::endl;
            }
            FIXLINE(s);
            bytesread += strlen(s);
            if (s[0] == '#') continue; // Comment
            if (s[0] == '%') continue; // Comment
            
            char delims[] = "\t, ";
            char * t;
            t = strtok(s, delims);
            if (t == NULL) {
                logstream(LOG_ERROR) << "Input file is not in right format. "
                << "Expecting \"<from>\t<to>\". "
                << "Current line: \"" << s << "\"\n";
                assert(false);
            }
            vid_t from = atoi(t);
            t = strtok(NULL, delims);
            if (t == NULL) {
                logstream(LOG_ERROR) << "Input file is not in right format. "
                << "Expecting \"<from>\t<to>\". "
                << "Current line: \"" << s << "\"\n";
                assert(false);
            }
            vid_t to = atoi(t);
            
            /* Check if has value */
            t = strtok(NULL, delims);
            
            if (!multivalue_edges) {
                EdgeDataType val;
                if (t != NULL) {
                    parse(val, (const char*) t);
                }
                if (from != to) {
                    if (t != NULL) {
                        sharderobj.preprocessing_add_edge(from, to, val);
                    } else {
                        sharderobj.preprocessing_add_edge(from, to);
                    }
                }
            } else {
#ifdef DYNAMICEDATA
                std::vector<EdgeDataType> vals;
                
                parse_multiple(vals, (char*) t);
                if (from != to) {
                    if (vals.size() == 0) {
                        // TODO: go around this problem
                        logstream(LOG_FATAL) << "Each edge needs at least one value." << std::endl;
                        assert(vals.size() > 0);
                    }
                    sharderobj.preprocessing_add_edge_multival(from, to, vals);
                }
                
#else
                logstream(LOG_FATAL) << "To support multivalue-edges, dynamic edge data needs to be used." << std::endl;
                assert(false);
#endif
            }
        }
        fclose(inf);
    }
    
    
    
    /**
     * Converts a graph from adjacency list format. Edge values are not supported,
     * and each edge gets the default value for the type. Self-edges are ignored.
     */
    template <typename EdgeDataType>
    void convert_adjlist(std::string inputfile, sharder<EdgeDataType> &sharderobj) {
        FILE * inf = fopen(inputfile.c_str(), "r");
        if (inf == NULL) {
            logstream(LOG_FATAL) << "Could not load :" << inputfile << " error: " << strerror(errno) << std::endl;
        }
        assert(inf != NULL);
        logstream(LOG_INFO) << "Reading in adjacency list format!" << std::endl;
        
        int maxlen = 100000000;
        char * s = (char*) malloc(maxlen);
        
        size_t bytesread = 0;
        
        char delims[] = " \t";
        size_t linenum = 0;
        size_t lastlog = 0;
        /*** PHASE 1 - count ***/
        while(fgets(s, maxlen, inf) != NULL) {
            linenum++;
            if (bytesread - lastlog >= 500000000) {
                logstream(LOG_DEBUG) << "Read " << linenum << " lines, " << bytesread / 1024 / 1024.  << " MB" << std::endl;
                lastlog = bytesread;
            }
            FIXLINE(s);
            bytesread += strlen(s);
            
            if (s[0] == '#') continue; // Comment
            if (s[0] == '%') continue; // Comment
            char * t = strtok(s, delims);
            vid_t from = atoi(t);
            t = strtok(NULL,delims);
            if (t != NULL) {
                vid_t num = atoi(t);
                vid_t i = 0;
                while((t = strtok(NULL,delims)) != NULL) {
                    vid_t to = atoi(t);
                    if (from != to) {
                        sharderobj.preprocessing_add_edge(from, to, EdgeDataType());
                    }
                    i++;
                }
                if (num != i)
                    logstream(LOG_ERROR) << "Mismatch when reading adjacency list: " << num << " != " << i << " s: " << std::string(s)
                    << " on line: " << linenum << std::endl;
                assert(num == i);
            }
        }
        free(s);
        fclose(inf);
    }


    /**
     * Extract a vector of node indices from a line in the file.
     *
     * @param[in]   line        line from input file containing node indices
     * @param[out]  adjacencies     node indices extracted from line
     */
    static std::vector<vid_t> parseLine(std::string line) {

        std::stringstream stream(line);
        std::string token;
        char delim = ' ';
        std::vector<vid_t> adjacencies;

        // split string and push adjacent nodes
        while (std::getline(stream, token, delim)) {
            if (token.size() != 0) {
                vid_t v = atoi(token.c_str());
                adjacencies.push_back(v);
            }
        }

        return adjacencies;
    }

    /**
     * Converts a graph from the METIS adjacency format.
     * See http://people.sc.fsu.edu/~jburkardt/data/metis_graph/metis_graph.html for format documentation.
     */
    template <typename EdgeDataType>
    void convert_metis(std::string inputPath, sharder<EdgeDataType> &sharderobj) {

        std::cout << "[INFO] reading METIS graph file" << std::endl;
        
        std::ifstream graphFile(inputPath.c_str());

        if (! graphFile.good()) {
            logstream(LOG_FATAL) << "Could not load :" << inputPath << " error: " << strerror(errno) << std::endl;
        }
        
        std::string line; // current line

        // handle header line
        int n;  // number of nodes
        int m;  // number of edges
        int weighted; // indicates weight scheme: 

        if (std::getline(graphFile, line)) {
            while (line[0] == '%') { // skip comments
                std::getline(graphFile, line);
            }

            std::vector<uint> tokens = parseLine(line);
            n = tokens[0];
            m = tokens[1];
            if (tokens.size() == 2) {
                weighted = 0;
            } if (tokens.size() == 3) {
                weighted = tokens[2];
                if (weighted != 0) {
                    logstream(LOG_FATAL) << "node and edge weights currently not supported by parser" << std::endl;
                }
            }
        } else {
            logstream(LOG_FATAL) << "getting METIS file header failed" << std::endl;
        }

        logstream(LOG_INFO) << "reading graph with n=" << n << ", m=" << m << std::endl;

        vid_t u = 0; // starting node index

        // handle content lines
        while (graphFile.good()) {
            do {
                std::getline(graphFile, line);
            } while (line[0] == '%'); // skip comments

            // parse adjacency line
            std::vector<vid_t> adjacencies = parseLine(line);
            for (std::vector<vid_t>::iterator it=adjacencies.begin(); it != adjacencies.end(); ++it) {
                vid_t v = *it;
                if (u <= v) { // add edge only once; self-loops are allowed
                    sharderobj.preprocessing_add_edge(u, v, EdgeDataType());
                }
            }
            
            u += 1;
        }



    }
    
    /**
     * Converts a graph from cassovary's (Twitter) format. Edge values are not supported,
     * and each edge gets the default value for the type. Self-edges are ignored.
     */
    template <typename EdgeDataType>
    void convert_cassovary(std::string basefilename, sharder<EdgeDataType> &sharderobj) {
        std::vector<std::string> parts;
        std::string dirname = get_dirname(basefilename);
        std::string prefix =  get_filename(basefilename);
        
        std::cout << "dir=[" << dirname << "] prefix=[" << prefix << "]" << std::endl;
        getdir(dirname, parts);
        
        for(std::vector<std::string>::iterator it=parts.begin(); it != parts.end(); ++it) {
            std::string inputfile = *it;
            if (inputfile.find(prefix) == 0 && inputfile.find("tmp") == inputfile.npos) {
                std::cout << "Going to process: " << inputfile << std::endl;
            }
        }
        
        for(std::vector<std::string>::iterator it=parts.begin(); it != parts.end(); ++it) {
            std::string inputfile = *it;
            if (inputfile.find(prefix) == 0 && inputfile.find(".tmp") == inputfile.npos) {
                inputfile = dirname + "/" + inputfile;
                std::cout << "Process: " << inputfile << std::endl;
                FILE * inf = fopen(inputfile.c_str(), "r");
                if (inf == NULL) {
                    logstream(LOG_FATAL) << "Could not load :" << inputfile << " error: " << strerror(errno) << std::endl;
                }
                assert(inf != NULL);
                logstream(LOG_INFO) << "Reading in cassovary format!" << std::endl;
                
                int maxlen = 100000000;
                char * s = (char*) malloc(maxlen);
                
                size_t bytesread = 0;
                
                char delims[] = " \t";
                size_t linenum = 0;
                size_t lastlog = 0;
                while(fgets(s, maxlen, inf) != NULL) {
                    linenum++;
                    if (bytesread - lastlog >= 500000000) {
                        logstream(LOG_DEBUG) << "Read " << linenum << " lines, " << bytesread / 1024 / 1024.  << " MB" << std::endl;
                        lastlog = bytesread;
                    }
                    FIXLINE(s);
                    bytesread += strlen(s);
                    
                    if (s[0] == '#') continue; // Comment
                    if (s[0] == '%') continue; // Comment
                    char * t = strtok(s, delims);
                    vid_t from = atoi(t);
                    t = strtok(NULL,delims);
                    if (t != NULL) {
                        vid_t num = atoi(t);
                        
                        // Read next line
                        linenum += num + 1;
                        for(vid_t i=0; i < num; i++) {
                            s = fgets(s, maxlen, inf);
                            FIXLINE(s);
                            vid_t to = atoi(s);
                            if (from != to) {
                                sharderobj.preprocessing_add_edge(from, to, EdgeDataType());
                            }
                        }
                        
                    }
                }
                free(s);
                fclose(inf);
            }
        }
    }
    
    
    /**
     * Converts a set of files in the binedgelist format (binary edge list)
     */
    template <typename EdgeDataType>
    void convert_binedgelist(std::string basefilename, sharder<EdgeDataType> &sharderobj) {
        std::vector<std::string> parts;
        std::string dirname = get_dirname(basefilename);
        std::string prefix =  get_filename(basefilename);
        
        std::cout << "dir=[" << dirname << "] prefix=[" << prefix << "]" << std::endl;
        getdir(dirname, parts);
        
        for(std::vector<std::string>::iterator it=parts.begin(); it != parts.end(); ++it) {
            std::string inputfile = *it;
            if (inputfile.find(prefix) == 0 && inputfile.find("tmp") == inputfile.npos) {
                std::cout << "Going to process: " << inputfile << std::endl;
            }
        }
        
        for(std::vector<std::string>::iterator it=parts.begin(); it != parts.end(); ++it) {
            std::string inputfile = *it;
            if (inputfile.find(prefix) == 0 && inputfile.find(".tmp") == inputfile.npos) {
                inputfile = dirname + "/" + inputfile;
                std::cout << "Process: " << inputfile << std::endl;
                FILE * inf = fopen(inputfile.c_str(), "r");
                
                while(!feof(inf)) {
                    vid_t from;
                    vid_t to;
                    
                    size_t res1 = fread(&from, sizeof(vid_t), 1, inf);
                    size_t res2 = fread(&to, sizeof(vid_t), 1, inf);
                    
                    assert(res1 > 0 && res2 > 0);
                    if (from != to) {
                        sharderobj.preprocessing_add_edge(from, to, EdgeDataType());
                    }
                }
                fclose(inf);
            }
        }
    }
    
    // TODO: remove code duplication.
    template <typename EdgeDataType>
    void convert_binedgelistval(std::string basefilename, sharder<EdgeDataType> &sharderobj) {
        std::vector<std::string> parts;
        std::string dirname = get_dirname(basefilename);
        std::string prefix =  get_filename(basefilename);
        
        std::cout << "dir=[" << dirname << "] prefix=[" << prefix << "]" << std::endl;
        getdir(dirname, parts);
        
        for(std::vector<std::string>::iterator it=parts.begin(); it != parts.end(); ++it) {
            std::string inputfile = *it;
            if (inputfile.find(prefix) == 0 && inputfile.find("tmp") == inputfile.npos) {
                std::cout << "Going to process: " << inputfile << std::endl;
            }
        }
        
        for(std::vector<std::string>::iterator it=parts.begin(); it != parts.end(); ++it) {
            std::string inputfile = *it;
            if (inputfile.find(prefix) == 0 && inputfile.find(".tmp") == inputfile.npos) {
                inputfile = dirname + "/" + inputfile;
                std::cout << "Process: " << inputfile << std::endl;
                FILE * inf = fopen(inputfile.c_str(), "r");
                
                while(!feof(inf)) {
                    vid_t from;
                    vid_t to;
                    EdgeDataType edgeval;
                    
                    size_t res1 = fread(&from, sizeof(vid_t), 1, inf);
                    size_t res2 = fread(&to, sizeof(vid_t), 1, inf);
                    size_t res3 = fread(&edgeval, sizeof(EdgeDataType), 1, inf);
                    assert(res1 > 0 && res2 > 0 && res3 > 0);
                    if (from != to) {
                        sharderobj.preprocessing_add_edge(from, to, edgeval);
                    }
                }
                fclose(inf);
            }
        }
    }
    
    
    
    
    /**
     * An abstract class for defining preprocessor objects
     * that modify the preprocessed binary input prior
     * to sharding.
     */
    template <typename EdgeDataType>
    class SharderPreprocessor {
    public:
        virtual ~SharderPreprocessor() {}
        virtual std::string getSuffix() = 0;
        virtual void reprocess(std::string preprocFilename, std::string basefileName) = 0;
    };
    
    /**
     * Converts a graph input to shards. Preprocessing has several steps,
     * see sharder.hpp for more information.
     */
    template <typename EdgeDataType>
    int convert(std::string basefilename, std::string nshards_string, SharderPreprocessor<EdgeDataType> * preprocessor = NULL) {
        //
        std::cout << "Calling convert" << std::endl;
        // 
        std::string suffix = "";
        if (preprocessor != NULL) {
            suffix = preprocessor->getSuffix();
        }
        sharder<EdgeDataType> sharderobj(basefilename + suffix);
        
        if (!sharderobj.preprocessed_file_exists()) {
            //
            std::cout << "preprocessed file does not exist" << std::endl;
            // 
            std::string file_type_str = get_option_string_interactive("filetype", "edgelist, adjlist, metis");
            if (file_type_str != "adjlist" && file_type_str != "edgelist"  && file_type_str != "binedgelist" &&
                file_type_str != "multivalueedgelist" && file_type_str != "metis") {

                //
                std::cout << "file type string: " << file_type_str << std::endl;
                //

                logstream(LOG_ERROR) << "You need to specify filetype: 'edgelist' or 'adjlist'." << std::endl;
                assert(false);
            }
            
            /* Start preprocessing */
            sharderobj.start_preprocessing();
            
            if (file_type_str == "adjlist") {
                convert_adjlist<EdgeDataType>(basefilename, sharderobj);
            } else if (file_type_str == "edgelist") {
                convert_edgelist<EdgeDataType>(basefilename, sharderobj);
#ifdef DYNAMICEDATA
            } else if (file_type_str == "multivalueedgelist" ) {
                convert_edgelist<EdgeDataType>(basefilename, sharderobj, true);
#endif
            } else if (file_type_str == "binedgelist") {
                convert_binedgelistval<EdgeDataType>(basefilename, sharderobj);
            } else if (file_type_str == "metis") {
                convert_metis<EdgeDataType>(basefilename, sharderobj);
            } else {
                assert(false);
            }
            
            /* Finish preprocessing */
            sharderobj.end_preprocessing();
            
            if (preprocessor != NULL) {
                preprocessor->reprocess(sharderobj.preprocessed_name(), basefilename);
            }
            
        }
        
        vid_t max_vertex_id = get_option_int("maxvertex", 0);
        if (max_vertex_id > 0) {
            sharderobj.set_max_vertex_id(max_vertex_id);
        }
        
        int nshards = sharderobj.execute_sharding(nshards_string);
        logstream(LOG_INFO) << "Successfully finished sharding for " << basefilename + suffix << std::endl;
        logstream(LOG_INFO) << "Created " << nshards << " shards." << std::endl;
        return nshards;
    }
    
    
    /**
     * Converts a graph input to shards with no edge values. Preprocessing has several steps,
     * see sharder.hpp for more information.
     */
    int convert_none(std::string basefilename, std::string nshards_string);
    int convert_none(std::string basefilename, std::string nshards_string) {
        std::string suffix = "";
        sharder<dummy> sharderobj(basefilename + suffix);
        sharderobj.set_no_edgevalues();
        
        if (!sharderobj.preprocessed_file_exists()) {
            std::string file_type_str = get_option_string_interactive("filetype", "edgelist, adjlist, cassovary, binedgelist");
            if (file_type_str != "adjlist" && file_type_str != "edgelist" && file_type_str != "cassovary"  && file_type_str != "binedgelist") {
                logstream(LOG_ERROR) << "You need to specify filetype: 'edgelist' or 'adjlist'." << std::endl;
                assert(false);
            }
            
            /* Start preprocessing */
            sharderobj.start_preprocessing();
            
            if (file_type_str == "adjlist") {
                convert_adjlist<dummy>(basefilename, sharderobj);
            } else if (file_type_str == "edgelist") {
                convert_edgelist<dummy>(basefilename, sharderobj);
            } else if (file_type_str == "cassovary") {
                convert_cassovary<dummy>(basefilename, sharderobj);
            } else if (file_type_str == "binedgelist") {
                convert_binedgelist<dummy>(basefilename, sharderobj);
            }
            
            /* Finish preprocessing */
            sharderobj.end_preprocessing();
        }
        
        if (get_option_int("skipsharding", 0) == 1) {
            std::cout << "Skip sharding..." << std::endl;
            exit(0);
        }
        
        vid_t max_vertex_id = get_option_int("maxvertex", 0);
        if (max_vertex_id > 0) {
            sharderobj.set_max_vertex_id(max_vertex_id);
        }
        
        int nshards = sharderobj.execute_sharding(nshards_string);
        logstream(LOG_INFO) << "Successfully finished sharding for " << basefilename + suffix << std::endl;
        logstream(LOG_INFO) << "Created " << nshards << " shards." << std::endl;
        return nshards;
    }
    
    
    
    template <typename EdgeDataType>
    int convert_if_notexists(std::string basefilename, std::string nshards_string, bool &didexist,
                             SharderPreprocessor<EdgeDataType> * preprocessor = NULL) {
        int nshards;
        std::string suffix = "";
        if (preprocessor != NULL) {
            suffix = preprocessor->getSuffix();
        }
        
        /* Check if input file is already sharded */
        if ((nshards = find_shards<EdgeDataType>(basefilename + suffix, nshards_string))) {
            logstream(LOG_INFO) << "Found preprocessed files for " << basefilename << ", num shards=" << nshards << std::endl;
            didexist = true;
            if (check_origfile_modification_earlier<EdgeDataType>(basefilename + suffix, nshards)) {
                return nshards;
            }
            
        }
        didexist = false;
        
        logstream(LOG_INFO) << "Did not find preprocessed shards for " << basefilename + suffix << std::endl;
        
        logstream(LOG_INFO) << "(Edge-value size: " << sizeof(EdgeDataType) << ")" << std::endl;
        logstream(LOG_INFO) << "Will try create them now..." << std::endl;
        nshards = convert<EdgeDataType>(basefilename, nshards_string, preprocessor);
        return nshards;
    }
    
    template <typename EdgeDataType>
    int convert_if_notexists(std::string basefilename, std::string nshards_string, SharderPreprocessor<EdgeDataType> * preprocessor = NULL) {
        bool b;
        return convert_if_notexists<EdgeDataType>(basefilename, nshards_string, b, preprocessor);
    }
    
    
    
    struct vertex_degree {
        int deg;
        vid_t id;
        vertex_degree() {}
        vertex_degree(int deg, vid_t id) : deg(deg), id(id) {}
    };
    
    static bool vertex_degree_less(const vertex_degree &a, const vertex_degree &b);
    static bool vertex_degree_less(const vertex_degree &a, const vertex_degree &b) {
        return a.deg < b.deg || (a.deg == b.deg && a.id < b.id);
    }
    
    /**
     * Special preprocessor which relabels vertices in ascending order
     * of their degree.
     */
    template <typename EdgeDataType>
    class OrderByDegree : public SharderPreprocessor<EdgeDataType> {
        int phase;
        
    public:
        typedef edge_with_value<EdgeDataType> edge_t;
        vid_t * translate_table;
        vid_t max_vertex_id;
        vertex_degree * degarray;
        binary_adjacency_list_writer<EdgeDataType> * writer;
        OrderByDegree() {
            degarray = NULL;
            writer = NULL;
        }
        
        ~OrderByDegree() {
            if (degarray != NULL) free(degarray);
            degarray = NULL;
            if (writer != NULL) delete writer;
            writer = NULL;
        }
        
        std::string getSuffix() {
            return "_degord";
        }
        
        vid_t translate(vid_t vid) {
            if (vid > max_vertex_id) return vid;
            return translate_table[vid];
        }
        
        /**
         * Callback function that binary_adjacency_list_reader
         * invokes. In first phase, the degrees of vertice sare collected.
         * In the next face, they are written out to the degree-ordered data.
         * Note: this version does not preserve edge values!
         */
        void receive_edge(vid_t from, vid_t to, EdgeDataType value, bool is_value) {
            if (phase == 0) {
                degarray[from].deg++;
                degarray[to].deg++;
            } else {
                writer->add_edge(translate(from), translate(to)); // Value is ignored
            }
        }
        void reprocess(std::string preprocessedFile, std::string baseFilename) {
            
            binary_adjacency_list_reader<EdgeDataType> reader(preprocessedFile);
            max_vertex_id = (vid_t) reader.get_max_vertex_id();
            
            degarray = (vertex_degree *) calloc(max_vertex_id + 1, sizeof(vertex_degree));
            vid_t nverts = max_vertex_id + 1;
            for(vid_t i=0; i < nverts; i++) {
                degarray[i].id = i;
            }
            
            phase = 0;
            /* Reader will invoke receive_edge() above */
            reader.read_edges(this);
            
            /* Now sort */
            quickSort(degarray, nverts, vertex_degree_less);
            
            /* Create translation table */
            translate_table = (vid_t*) calloc(sizeof(vid_t), nverts);
            for(vid_t i=0; i<nverts; i++) {
                translate_table[degarray[i].id] = i;
            }
            delete degarray;
            
            /* Write translate table */
            std::string translate_table_file = baseFilename + ".vertexmap";
            int df = open(translate_table_file.c_str(), O_RDWR | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
            if (df < 0) logstream(LOG_ERROR) << "Could not write vertex map: " << translate_table_file <<
                " error: " << strerror(errno) << std::endl;
            assert(df >= 0);
            pwritea(df, translate_table, nverts * sizeof(vid_t), 0);
            close(df);
            
            /* Now recreate the processed file */
            std::string tmpfilename = preprocessedFile + ".old";
            rename(preprocessedFile.c_str(), tmpfilename.c_str());
            
            writer = new binary_adjacency_list_writer<EdgeDataType>(preprocessedFile);
            binary_adjacency_list_reader<EdgeDataType> reader2(tmpfilename);
            
            phase = 1;
            reader2.read_edges(this);
            
            writer->finish();
            delete writer;
            writer = NULL;
            
            delete translate_table;
        }
        
    };
    
} // end namespace

#endif


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
 * Graph conversion and parsing routines.
 */

#ifndef GRAPHCHI_CONVERSIONS_DEF
#define GRAPHCHI_CONVERSIONS_DEF

#include <fcntl.h>
#include <unistd.h>

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
    
    
    /**
     * Converts graph from an edge list format. Input may contain
     * value for the edges. Self-edges are ignored.
     */
    template <typename EdgeDataType>
    void convert_edgelist(std::string inputfile, sharder<EdgeDataType> &sharderobj) {
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
            
            char delims[] = "\t ";
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
            EdgeDataType val;
            if (t != NULL) {
                parse(val, (const char*) t);
            } else {
                val = EdgeDataType();
            }
            if (from != to) {
                sharderobj.preprocessing_add_edge(from, to, val);
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
    }
    
    
    /**
     * A abstract class for defining preprocessor objects
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
        std::string suffix = "";
        if (preprocessor != NULL) {
            suffix = preprocessor->getSuffix();
        }
        sharder<EdgeDataType> sharderobj(basefilename + suffix);
        
        if (!sharderobj.preprocessed_file_exists()) {
            std::string file_type_str = get_option_string_interactive("filetype", "edgelist, adjlist");
            if (file_type_str != "adjlist" && file_type_str != "edgelist") {
                logstream(LOG_ERROR) << "You need to specify filetype: 'edgelist' or 'adjlist'." << std::endl;
                assert(false);
            }
            
            /* Start preprocessing */
            sharderobj.start_preprocessing();
            
            if (file_type_str == "adjlist") {
                convert_adjlist<EdgeDataType>(basefilename, sharderobj);
            } else if (file_type_str == "edgelist") {
                convert_edgelist<EdgeDataType>(basefilename, sharderobj);
            }
            
            /* Finish preprocessing */
            sharderobj.end_preprocessing();
            
            if (preprocessor != NULL) {
                preprocessor->reprocess(sharderobj.preprocessed_name(), basefilename);
            }
            
        }
        
        int nshards = sharderobj.execute_sharding(nshards_string);
        logstream(LOG_INFO) << "Successfully finished sharding for " << basefilename + suffix << std::endl;
        logstream(LOG_INFO) << "Created " << nshards << " shards." << std::endl;
        return nshards;
    }
    
    
    template <typename EdgeDataType>
    int convert_if_notexists(std::string basefilename, std::string nshards_string, SharderPreprocessor<EdgeDataType> * preprocessor = NULL) {
        int nshards;
        std::string suffix = "";
        if (preprocessor != NULL) {
            suffix = preprocessor->getSuffix();
        }
        
        /* Check if input file is already sharded */
        if ((nshards = find_shards<EdgeDataType>(basefilename + suffix, nshards_string))) {
            logstream(LOG_INFO) << "Found preprocessed files for " << basefilename << ", num shards=" << nshards << std::endl;
            return nshards;
        }
        logstream(LOG_INFO) << "Did not find preprocessed shards for " << basefilename + suffix << std::endl;
        logstream(LOG_INFO) << "Will try create them now..." << std::endl;
        nshards = convert<EdgeDataType>(basefilename, nshards_string, preprocessor);
        return nshards;
    }
    
    /**
     * Special machinery for reordering vertices by
     * their degree, i.e setting their vertex ids to be in the order
     * of vertex degree. This is used at least by the Triangle counting application.
     */
    
    struct vertex_degree {
        int deg;
        vid_t id;
        vertex_degree() {}
        vertex_degree(int deg, vid_t id) : deg(deg), id(id) {}
    };
    
    bool vertex_degree_less(const vertex_degree &a, const vertex_degree &b);
    bool vertex_degree_less(const vertex_degree &a, const vertex_degree &b) {
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
        void receive_edge(vid_t from, vid_t to, EdgeDataType value) {
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


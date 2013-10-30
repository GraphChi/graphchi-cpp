

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
 * Returns standard filenames for all the data files used by GraphChi.
 * All functions expect a "basefilename".
 * You can specify environment variable "GRAPHCHI_ROOT", which is the
 * root directory for the GraphChi configuration and source directories.
 */

#ifndef GRAPHCHI_FILENAMES_DEF
#define GRAPHCHI_FILENAMES_DEF

#include <fstream>
#include <fcntl.h>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <vector>
#include <sys/stat.h>

#include "graphchi_types.hpp"
#include "logger/logger.hpp"

#ifdef DYNAMICEDATA
#include "shards/dynamicdata/dynamicblock.hpp"
#endif

namespace graphchi {
    
#ifdef __GNUC__
#define VARIABLE_IS_NOT_USED __attribute__ ((unused))
#else
#define VARIABLE_IS_NOT_USED
#endif
    static int VARIABLE_IS_NOT_USED get_option_int(const char *option_name, int default_value);
    
    /**
     * Vertex data file
     */
    template <typename VertexDataType>
    static std::string filename_vertex_data(std::string basefilename) {
        std::stringstream ss;
        ss << basefilename;
        ss << "." << sizeof(VertexDataType) << "B.vout";
        return ss.str();
    }
    
    static std::string filename_degree_data(std::string basefilename)  {
        return basefilename + "_degs.bin";
    }
    
    static std::string filename_intervals(std::string basefilename, int nshards) {
        std::stringstream ss;
        ss << basefilename;
        ss << "." << nshards << ".intervals";
        return ss.str();
    }
    
    
    static std::string VARIABLE_IS_NOT_USED get_part_str(int p, int nshards) {
        char partstr[32];
        sprintf(partstr, ".%d_%d", p, nshards);
        return std::string(partstr);
    }
    
    template <typename EdgeDataType>
    static std::string filename_shard_edata(std::string basefilename, int p, int nshards) {
        std::stringstream ss;
        ss << basefilename;
#ifdef DYNAMICEDATA
        ss << ".dynamic.";
#else
        ss << ".edata.";
#endif
#ifndef GRAPHCHI_DISABLE_COMPRESSION
        ss << ".Z.";
#endif
        ss << "e" << sizeof(EdgeDataType) << "B.";
        ss << p << "_" << nshards;
        
        return ss.str();
    }
    
    
    
    
    static std::string dirname_shard_edata_block(std::string edata_shardname, size_t blocksize) {
        std::stringstream ss;
        ss << edata_shardname;
        ss << "_blockdir_" << blocksize;
        return ss.str();
    }
    
    template <typename EdgeDataType>
    static size_t get_shard_edata_filesize(std::string edata_shardname) {
        size_t fsize;
        std::string fname = edata_shardname + ".size";
        std::ifstream ifs(fname.c_str());
        if (!ifs.good()) {
            logstream(LOG_FATAL) << "Could not load " << fname << ". Preprocessing forgotten?" << std::endl;
            assert(ifs.good());
        }
        ifs >> fsize;
        ifs.close();
        return fsize;
    }
    
    
    static std::string filename_shard_edata_block(std::string edata_shardname, int blockid, size_t blocksize) {
        std::stringstream ss;
        ss << dirname_shard_edata_block(edata_shardname, blocksize);
        ss << "/";
        ss << blockid;
        return ss.str();
    }
    
    
    static std::string filename_shard_adj(std::string basefilename, int p, int nshards) {
        std::stringstream ss;
        ss << basefilename;
        ss << ".edata_azv.";
        ss << p << "_" << nshards << ".adj";
        return ss.str();
    }
    
    static std::string filename_shard_adjidx(std::string adjfilename) {
        return adjfilename + "idx";
    }
    
    /**
     * Configuration file name
     */
    static std::string filename_config();
    static std::string filename_config() {
        char * chi_root = getenv("GRAPHCHI_ROOT");
        if (chi_root != NULL) {
            return std::string(chi_root) + "/conf/graphchi.cnf";
        } else {
            return "conf/graphchi.cnf";
        }
    }
    
    /**
     * Configuration file name - local version which can
     * override the version in the version control.
     */
    static std::string filename_config_local();
    static std::string filename_config_local() {
        char * chi_root = getenv("GRAPHCHI_ROOT");
        if (chi_root != NULL) {
            return std::string(chi_root) + "/conf/graphchi.local.cnf";
        } else {
            return "conf/graphchi.local.cnf";
        }
    }
    
    
    static bool file_exists(std::string sname);
    static bool file_exists(std::string sname) {
        int tryf = open(sname.c_str(), O_RDONLY);
        if (tryf < 0) {
            return false;
        } else {
            close(tryf);
            return true;
        }
    }
    
    
    
    /**
     * Returns the number of shards if a file has been already
     * sharded or 0 if not found.
     */
    template<typename EdgeDataType>
    static int find_shards(std::string base_filename, std::string shard_string="auto") {
        int try_shard_num;
        int start_num = 0;
        int last_shard_num = 2400;
        if (shard_string == "auto") {
            start_num = 0;
        } else {
            start_num = atoi(shard_string.c_str());
        }
        
        if (start_num > 0) {
            last_shard_num = start_num;
        }
        size_t blocksize = 1024 * 1024;
        while (blocksize % sizeof(EdgeDataType) != 0) blocksize++;
        
        for(try_shard_num=start_num; try_shard_num <= last_shard_num; try_shard_num++) {
            std::string last_shard_name = filename_shard_edata<EdgeDataType>(base_filename, try_shard_num - 1, try_shard_num);
            std::string last_block_name = filename_shard_edata_block(last_shard_name, 0, blocksize);
            int tryf = open(last_block_name.c_str(), O_RDONLY);
            if (tryf >= 0) {
                // Found!
                close(tryf);
                
                int nshards_candidate = try_shard_num;
                bool success = true;
                
                // Validate all relevant files exists
                for(int p=0; p < nshards_candidate; p++) {
                    std::string sname = filename_shard_edata_block(
                                                                   filename_shard_edata<EdgeDataType>(base_filename, p, nshards_candidate), 0, blocksize);
                    if (!file_exists(sname)) {
                        logstream(LOG_DEBUG) << "Missing directory file: " << sname << std::endl;
                        success = false;
                        break;
                    }
                    
                    sname = filename_shard_adj(base_filename, p, nshards_candidate);
                    if (!file_exists(sname)) {
                        logstream(LOG_DEBUG) << "Missing shard file: " << sname << std::endl;
                        success = false;
                        break;
                    }
                }
                
                // Check degree file
                std::string degreefname = filename_degree_data(base_filename);
                if (!file_exists(degreefname)) {
                    logstream(LOG_ERROR) << "Missing degree file: " << degreefname << std::endl;
                    logstream(LOG_ERROR) << "You need to preprocess (sharder) your file again!" << std::endl;
                    return 0;
                }
                
                std::string intervalfname = filename_intervals(base_filename, nshards_candidate);
                if (!file_exists(intervalfname)) {
                    logstream(LOG_ERROR) << "Missing intervals file: " << intervalfname << std::endl;
                    logstream(LOG_ERROR) << "You need to preprocess (sharder) your file again!" << std::endl;
                    return 0;
                }
                
                if (!success) {
                    continue;
                }
                
                logstream(LOG_INFO) << "Detected number of shards: " << nshards_candidate << std::endl;
                logstream(LOG_INFO) << "To specify a different number of shards, use command-line parameter 'nshards'" << std::endl;
                return nshards_candidate;
            }
        }
        if (last_shard_num == start_num) {
            logstream(LOG_WARNING) << "Could not find shards with nshards = " << start_num << std::endl;
            logstream(LOG_WARNING) << "Please define 'nshards 0' or 'nshards auto' to automatically detect." << std::endl;
        }
        return 0;
    }
    
    
    /**
     * Delete the shard files
     */
    template<typename EdgeDataType_>
    static void delete_shards(std::string base_filename, int nshards) {
#ifdef DYNAMICEDATA
        typedef int EdgeDataType;
#else
        typedef EdgeDataType_ EdgeDataType;
#endif
        logstream(LOG_DEBUG) << "Deleting files for " << base_filename << " shards=" << nshards << std::endl;
        std::string intervalfname = filename_intervals(base_filename, nshards);
        if (file_exists(intervalfname)) {
            int err = remove(intervalfname.c_str());
            if (err != 0) logstream(LOG_ERROR) << "Error removing file " << intervalfname
                << ", " << strerror(errno) << std::endl;
            
        }
        /* Note: degree file is not removed, because same graph with different number
         of shards share the file. This should be probably change.
         std::string degreefname = filename_degree_data(base_filename);
         if (file_exists(degreefname)) {
         remove(degreefname.c_str());
         } */
        
        size_t blocksize = 1024 * 1024;
        while (blocksize % sizeof(EdgeDataType) != 0) blocksize++;
        
        for(int p=0; p < nshards; p++) {
            int blockid = 0;
            std::string filename_edata = filename_shard_edata<EdgeDataType>(base_filename, p, nshards);
            std::string fsizename = filename_edata + ".size";
            if (file_exists(fsizename)) {
                int err = remove(fsizename.c_str());
                if (err != 0) logstream(LOG_ERROR) << "Error removing file " << fsizename
                    << ", " << strerror(errno) << std::endl;
            }
            while(true) {
                std::string block_filename = filename_shard_edata_block(filename_edata, blockid, blocksize);
                if (file_exists(block_filename)) {
                    int err = remove(block_filename.c_str());
                    if (err != 0) logstream(LOG_ERROR) << "Error removing file " << block_filename
                        << ", " << strerror(errno) << std::endl;
                    
                } else {
                    
                    break;
                }
#ifdef DYNAMICEDATA
                delete_block_uncompressed_sizefile(block_filename);
#endif
                blockid++;
            }
            std::string dirname = dirname_shard_edata_block(filename_edata, blocksize);
            if (file_exists(dirname)) {
                int err = remove(dirname.c_str());
                if (err != 0) logstream(LOG_ERROR) << "Error removing directory " << dirname
                    << ", " << strerror(errno) << std::endl;
                
            }
            
            std::string adjname = filename_shard_adj(base_filename, p, nshards);
            logstream(LOG_DEBUG) << "Deleting " << adjname << " exists: " << file_exists(adjname) << std::endl;
            
            if (file_exists(adjname)) {
                int err = remove(adjname.c_str());
                if (err != 0) logstream(LOG_ERROR) << "Error removing file " << adjname
                    << ", " << strerror(errno) << std::endl;
            }
            
            std::string idxname = filename_shard_adjidx(adjname);
            logstream(LOG_DEBUG) << "Deleting " << idxname << " exists: " << file_exists(idxname) << std::endl;
            
            if (file_exists(idxname)) {
                int err = remove(idxname.c_str());
                if (err != 0) logstream(LOG_ERROR) << "Error removing file " << idxname
                    << ", " << strerror(errno) << std::endl;
            }

        }
        
        std::string numv_filename = base_filename + ".numvertices";
        if (file_exists(numv_filename)) {
            int err = remove(numv_filename.c_str());
            if (err != 0) logstream(LOG_ERROR) << "Error removing file " << numv_filename
                << ", " << strerror(errno) << std::endl;
        }
        
        /* Degree file */
        std::string deg_filename = filename_degree_data(base_filename);
        if (file_exists(deg_filename)) {
            int err = remove(deg_filename.c_str());
            if (err != 0) logstream(LOG_ERROR) << "Error removing file " << deg_filename
                << ", " << strerror(errno) << std::endl;
        }
    }
    
    
    /**
     * Loads vertex intervals.
     */
    static VARIABLE_IS_NOT_USED void load_vertex_intervals(std::string base_filename, int nshards, std::vector<std::pair<vid_t, vid_t> > & intervals, bool allowfail);
    static VARIABLE_IS_NOT_USED void load_vertex_intervals(std::string base_filename, int nshards, std::vector<std::pair<vid_t, vid_t> > & intervals, bool allowfail=false) {
        std::string intervalsFilename = filename_intervals(base_filename, nshards);
        std::ifstream intervalsF(intervalsFilename.c_str());
        
        if (!intervalsF.good()) {
            if (allowfail) return; // Hack
            logstream(LOG_ERROR) << "Could not load intervals-file: " << intervalsFilename << std::endl;
        }
        assert(intervalsF.good());
        
        intervals.clear();
        
        vid_t st=0, en;
        for(int i=0; i < nshards; i++) {
            assert(!intervalsF.eof());
            intervalsF >> en;
            intervals.push_back(std::pair<vid_t,vid_t>(st, en));
            st = en + 1;
        }
        for(int i=0; i < nshards; i++) {
            logstream(LOG_INFO) << "shard: " << intervals[i].first << " - " << intervals[i].second << std::endl;
        }
        intervalsF.close();
    }
    
    /**
     * Returns the number of vertices in a graph. The value is stored in a separate file <graphname>.numvertices
     */
    static VARIABLE_IS_NOT_USED size_t get_num_vertices(std::string basefilename);
    static VARIABLE_IS_NOT_USED size_t get_num_vertices(std::string basefilename) {
        std::string numv_filename = basefilename + ".numvertices";
        std::ifstream vfileF(numv_filename.c_str());
        if (!vfileF.good()) {
            logstream(LOG_ERROR) << "Could not find file " << numv_filename << std::endl;
            logstream(LOG_ERROR) << "Maybe you have old shards - please recreate." << std::endl;
            assert(false);
        }
        size_t n;
        vfileF >> n;
        vfileF.close();
        return n;
    }
    
    template <typename EdgeDataType>
    std::string preprocess_filename(std::string basefilename) {
        std::stringstream ss;
        ss << basefilename;
        ss << "." <<  sizeof(EdgeDataType) << "B.bin";
        return ss.str();
    }
    
    
    
    /**
      * Checks if original file has more recent modification date
      * than the shards. If it has, deletes the shards and returns false.
      * Otherwise return true.
      */
    template <typename EdgeDataType>
    bool check_origfile_modification_earlier(std::string basefilename, int nshards) {
        /* Compare last modified dates of the original graph and the shards */
        if (file_exists(basefilename) && get_option_int("disable-modtime-check", 0) == 0) {
            struct stat origstat, shardstat;
            int err1 = stat(basefilename.c_str(), &origstat);
            
            std::string adjfname = filename_shard_adj(basefilename, 0, nshards);
            int err2 = stat(adjfname.c_str(), &shardstat);
            
            if (err1 != 0 || err2 != 0) {
                logstream(LOG_ERROR) << "Error when checking file modification times:  " << strerror(errno) << std::endl;
                return nshards;
            }
            
            if (origstat.st_mtime > shardstat.st_mtime) {
                logstream(LOG_INFO) << "The input graph modification date was newer than of the shards." << std::endl;
                logstream(LOG_INFO) << "Going to delete old shards and recreate new ones. To disable " << std::endl;
                logstream(LOG_INFO) << "functionality, specify --disable-modtime-check=1" << std::endl;
                
                // Delete shards
                delete_shards<EdgeDataType>(basefilename, nshards);
                
                // Delete the bin-file
                std::string preprocfile = preprocess_filename<EdgeDataType>(basefilename);
                if (file_exists(preprocfile)) {
                    logstream(LOG_DEBUG) << "Deleting: " << preprocfile << std::endl;
                    int err = remove(preprocfile.c_str());
                    if (err != 0) {
                        logstream(LOG_ERROR) << "Error deleting file: " << preprocfile << ", " <<
                        strerror(errno) << std::endl;
                    }
                }
                return false;
            } else {
                return true;
            }
        
            
        }
        return true;
    }
    
}

#endif


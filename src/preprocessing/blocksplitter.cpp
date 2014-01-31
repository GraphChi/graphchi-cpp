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
 * Splits shards into blocks. Experimental.
 */

#include <iostream>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <unistd.h>
#include <fstream>
#include <sys/stat.h>

#include "api/chifilenames.hpp"
#include "io/stripedio.hpp"
#include "logger/logger.hpp"
#include "util/ioutil.hpp"
#include "util/cmdopts.hpp"
#include "preprocessing/conversions.hpp"
#include "preprocessing/sharder.hpp"

using namespace graphchi;


typedef float EdgeDataType;

int main(int argc, const char ** argv) {
    graphchi_init(argc, argv);
    
    global_logger().set_log_level(LOG_DEBUG);
    

    std::string filename = get_option_string("file");
    int nshards             = convert_if_notexists<EdgeDataType>(filename, get_option_string("nshards", "auto"));
    size_t blocksize= get_option_long("blocksize", 1024 * 1024);
    
    char * buf = (char *) malloc(blocksize);
    for(int p=0; p < nshards; p++) {
        std::string shard_filename = filename_shard_edata<EdgeDataType>(filename, p, nshards);
        int f = open(shard_filename.c_str(), O_RDONLY);
        size_t fsize = get_filesize(shard_filename);
        
        size_t nblocks = fsize / blocksize + (fsize % blocksize != 0);
        size_t idx = 0;
        std::string block_dirname = dirname_shard_edata_block(shard_filename, blocksize);
        logstream(LOG_INFO) << "Going to create: " << block_dirname << std::endl;
        int err = mkdir(block_dirname.c_str(), 0777);
        if (err != 0) {
            logstream(LOG_ERROR) << strerror(errno) << std::endl;
        }
        
        for(int i=0; i < nblocks; i++) {
            size_t len = std::min(blocksize, fsize - idx);
            preada(f, buf, len, idx);
           
            std::string block_filename = filename_shard_edata_block(shard_filename, i, blocksize);
            int bf = open(block_filename.c_str(), O_RDWR | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
            write_compressed(bf, buf, len);
            close(bf);
            
            idx += blocksize;
        }
        close(f);
        
        std::string sizefilename = shard_filename + ".size";
        std::ofstream ofs(sizefilename.c_str());
        ofs << fsize;
        ofs.close();
    }
}
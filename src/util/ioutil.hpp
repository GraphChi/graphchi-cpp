
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
 * I/O Utils.
 */
#ifndef DEF_IOUTIL_HPP
#define DEF_IOUTIL_HPP

#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include <errno.h>
 

// Reads given number of bytes to a buffer
template <typename T>
void preada(int f, T * tbuf, size_t nbytes, size_t off) {
    size_t nread = 0;
    char * buf = (char*)tbuf;
    while(nread<nbytes) {
        size_t a = pread(f, buf, nbytes - nread, off + nread);
        if (a <= 0) {
            std::cout << "Error, could not read: " << strerror(errno) << std::endl;
        }
        assert(a>0);
        buf += a;
        nread += a;
    }
    assert(nread <= nbytes);
}

template <typename T>
void preada_trunc(int f, T * tbuf, size_t nbytes, size_t off) {
    size_t nread = 0;
    char * buf = (char*)tbuf;
    while(nread<nbytes) {
        size_t a = pread(f, buf, nbytes-nread, off+nread);
        if (a == 0) {
            // set rest to 0
     //       std::cout << "WARNING: file was not long enough - filled with zeros. " << std::endl;
            memset(buf, 0, nbytes-nread);
            return;
        }
        buf += a;
        nread += a;
    }

} 

template <typename T>
size_t readfull(int f, T ** buf) {
     off_t sz = lseek(f, 0, SEEK_END);
     lseek(f, 0, SEEK_SET);
     *buf = (char*)malloc(sz);
    preada(f, *buf, sz, 0);
    return sz;
}
 template <typename T>
void pwritea(int f, T * tbuf, size_t nbytes, size_t off) {
    size_t nwritten = 0;
    assert(f>0);
    char * buf = (char*)tbuf;
    while(nwritten<nbytes) {
        size_t a = pwrite(f, buf, nbytes-nwritten, off+nwritten);
        if (a == size_t(-1)) {
            logstream(LOG_ERROR) << "f:" << f << " nbytes: " << nbytes << " written: " << nwritten << " off:" << 
                off << " f: " << f << " error:" <<  strerror(errno) << std::endl;
            assert(false);
        }
        assert(a>0);
        buf += a;
        nwritten += a;
    }
} 
template <typename T>
void writea(int f, T * tbuf, size_t nbytes) {
    size_t nwritten = 0;
    char * buf = (char*)tbuf;
    while(nwritten<nbytes) {
        size_t a = write(f, buf, nbytes-nwritten);
        assert(a>0);
        if (a == size_t(-1)) {
            logstream(LOG_ERROR) << "Could not write " << (nbytes-nwritten) << " bytes!" << " error:" <<  strerror(errno) << std::endl; 
            assert(false);
        }
        buf += a;
        nwritten += a;
    }

} 

template <typename T>
void checkarray_filesize(std::string fname, size_t nelements) {
    // Check the vertex file is correct size
    int f = open(fname.c_str(),  O_RDWR | O_CREAT, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
    if (f < 1) {
        logstream(LOG_ERROR) << "Error initializing the data-file: " << fname << " error:" <<  strerror(errno) << std::endl;    }
    assert(f>0);
    int err = ftruncate(f, nelements * sizeof(T));
    if (err != 0) {
        logstream(LOG_ERROR) << "Error in adjusting file size: " << fname << " to size: " << nelements * sizeof(T)    
                 << " error:" <<  strerror(errno) << std::endl;
    }
    assert(err == 0);
    close(f);
}

#endif



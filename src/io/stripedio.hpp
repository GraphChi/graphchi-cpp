
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
 * I/O manager.
 */



#ifndef DEF_STRIPEDIO_HPP
#define DEF_STRIPEDIO_HPP

#include <iostream> 

#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <stdint.h>
#include <pthread.h>
#include <errno.h>
#include <sys/mman.h>


#include <vector>

#include "logger/logger.hpp"
#include "metrics/metrics.hpp"
#include "util/synchronized_queue.hpp"
#include "util/ioutil.hpp"
#include "util/cmdopts.hpp"

#define CACHED_SESSION_ID (-1)


namespace graphchi {
    
    static size_t get_filesize(std::string filename);
    
    /**
     * Defines a striped file access.
     */
    struct io_descriptor {
        std::string filename;    
        std::vector<int> readdescs;
        std::vector<int> writedescs;
        
        int start_mplex;
        bool open;
        bool compressed;
    };
    
    struct mmap_info {
        void * ptr;
        size_t length;
        int filedesc;
    };
    
    
    enum BLOCK_ACTION { READ, WRITE };
    
    // Very simple ref count system
    struct refcountptr {
        char * ptr;
        volatile int count;
        refcountptr(char * ptr, int count) : ptr(ptr), count(count) {}
    };
    
    // Forward declaration
    class stripedio;
    
    struct iotask {
        BLOCK_ACTION action;
        int fd;
        int session;
        refcountptr * ptr;
        size_t length;
        size_t offset;
        size_t ptroffset;
        bool free_after;
        stripedio * iomgr;
        bool compressed;
        bool closefd;
        volatile int * doneptr;
        
        iotask() : action(READ), fd(0), session(0), ptr(NULL), length(0), offset(0), ptroffset(0), free_after(false), iomgr(NULL), compressed(false), closefd(false), doneptr(NULL) {}
        iotask(stripedio * iomgr, BLOCK_ACTION act, int fd, int session,  refcountptr * ptr, size_t length, size_t offset, size_t ptroffset, bool free_after, bool compressed, bool closefd=false) :
        action(act), fd(fd), session(session), ptr(ptr),length(length), offset(offset), ptroffset(ptroffset), free_after(free_after), iomgr(iomgr),compressed(compressed), closefd(closefd) {
            if (closefd) assert(free_after);
            doneptr = NULL;
        }
    };
    
    struct thrinfo {
        synchronized_queue<iotask> * readqueue;
        synchronized_queue<iotask> * commitqueue;
        synchronized_queue<iotask> * prioqueue;
        
        bool running;
        metrics * m;
        volatile int pending_writes;
        volatile int pending_reads;
        int mplex;
    };
    
    // Forward declaration
    static void * io_thread_loop(void * _info);
    
    struct stripe_chunk {
        int mplex_thread;
        size_t offset;
        size_t len;
        stripe_chunk(int mplex_thread, size_t offset, size_t len) : mplex_thread(mplex_thread), offset(offset), len(len) {}
    };
    
    struct cached_block {
        size_t len;
        void * data;
        bool was_compressed;
        
        cached_block(size_t len, void * data, bool was_compressed) : len(len), data(data), was_compressed(was_compressed) {}
        
        ~cached_block() {
            free(data);
            data = NULL;
        }
    };
    
    
    /**
      * Simple cache attached to the io manager. 
      */
    class block_cache {
        size_t cache_budget_bytes;
        size_t cache_size;
        mutex lock;  // TODO: read-write-lock
        bool full;
        std::map<std::string, cached_block *> cachemap;
        
        size_t hits, misses;
        
    public:
    
        block_cache(size_t cache_budget_bytes) : cache_budget_bytes(cache_budget_bytes), cache_size(0), full(false) {
            hits = misses = 0;
        }
        
        ~block_cache() {
            if (hits + misses > 0) {
                logstream(LOG_INFO) << "Cache stats: hits=" << hits << " misses=" << misses << std::endl;
                logstream(LOG_INFO) << " -- in total had " << (cache_size / 1024 / 1024) << " MB in cache." << std::endl;
            }
            std::map<std::string, cached_block *>::iterator it = cachemap.begin();
            for(; it != cachemap.end(); ++it) {
                delete it->second;
            }
        }
        
         
        
        bool consider_caching(std::string filename, void * data, size_t len, bool was_compresssed) {
            bool did_cache = false;
            if (!full && len + cache_size <= cache_budget_bytes) {
                lock.lock();
                if (len + cache_size <= cache_budget_bytes) {
                    cache_size += len;
                    did_cache = true;
                    if (cachemap.size() % 40 == 0) {
                        logstream(LOG_DEBUG) << "Cache size: " << cache_size << " / " << cache_budget_bytes << std::endl;
                    }
                    cachemap.insert(std::pair<std::string, cached_block*>(filename, new cached_block(len, data, was_compresssed)));
                }
                if (cache_size > cache_budget_bytes) {
                    full = true; // If full, we can avoid locking
                }
                lock.unlock();
            }
            return did_cache;
        }
        
        void * get_cached(std::string filename) {
            bool acquired_mutex = false;
            if (!full) {
                acquired_mutex = true;
                lock.lock();
            }
            
            void * ret = NULL;
            std::map<std::string, cached_block *>::iterator lookup = cachemap.find(filename);
            if (lookup != cachemap.end()) {
                ret =  lookup->second->data;
                hits++;
            } else {
                misses++;
            }
            
            if (acquired_mutex) {
                lock.unlock();
            }
            return ret;
        }
        friend class stripedio;
    };
    
    
    class stripedio {
        
        std::vector<io_descriptor *> sessions;
        mutex mlock;
        int stripesize;
        int multiplex;
        std::string multiplex_root;
        
        synchronized_queue<iotask> * mplex_readtasks;
        synchronized_queue<iotask> * mplex_writetasks;
        synchronized_queue<iotask> * mplex_priotasks;

        std::vector< pthread_t > threads;
        std::vector< thrinfo * > thread_infos;
        metrics &m;        
        
        int niothreads; // threads per mplex
        
        block_cache cache;
        
        
    private:
        // MMAP 
        mutex mmaplock;
        std::map<std::string, mmap_info> mmaped;
        
    public:
        stripedio( metrics &_m) : m(_m), cache(0) {
            stripesize = get_option_int("io.stripesize", 1024 * 1024 / 2);

            multiplex = get_option_int("multiplex", 1);
            if (multiplex>1) {
                multiplex_root = get_option_string("multiplex_root", "<not-set>");
                logstream(LOG_FATAL) << "Multiplexing files is currently not supported! Let akyrola@cs.cmu.edu know if you need this support :)." << std::endl;
                assert(multiplex == 1);
            } else {
                multiplex_root = "";
                stripesize = 1024*1024*1024;
            }
            m.set("stripesize", (size_t)stripesize);
            
            // Start threads (niothreads is now threads per multiplex)
            niothreads = get_option_int("niothreads", 1);
            m.set("niothreads", (size_t)niothreads);
       
            logstream(LOG_DEBUG) << "Start io-manager with " << niothreads << " threads." << std::endl;

            // Each multiplex partition has its own queues
            mplex_readtasks = new synchronized_queue<iotask>[multiplex * niothreads];
            mplex_writetasks = new synchronized_queue<iotask>[multiplex * niothreads];
            mplex_priotasks = new synchronized_queue<iotask>[multiplex * niothreads];
            
            int k = 0;
            for(int i=0; i < multiplex; i++) {
                for(int j=0; j < niothreads; j++) {
                    thrinfo * cthreadinfo = new thrinfo();
                    cthreadinfo->commitqueue = &mplex_writetasks[k];
                    cthreadinfo->readqueue = &mplex_readtasks[k];
                    cthreadinfo->prioqueue = &mplex_priotasks[k];
                    cthreadinfo->running = true;
                    cthreadinfo->pending_writes = 0;
                    cthreadinfo->pending_reads = 0;
                    cthreadinfo->mplex = i;
                    cthreadinfo->m = &m;
                    thread_infos.push_back(cthreadinfo);
                    
                    pthread_t iothread;
                    int ret = pthread_create(&iothread, NULL, io_thread_loop, cthreadinfo);
                    threads.push_back(iothread);
                    assert(ret>=0);
                    k++;
                }
            }
        }
        
        ~stripedio() {
            int mplex = (int) thread_infos.size();
            // Quit all threads
            for(int i=0; i<mplex; i++) {
                thread_infos[i]->running=false;
            }
            size_t nthreads = threads.size();
            for(unsigned int i=0; i<nthreads; i++) {
                pthread_join(threads[i], NULL);
            }
            for(int i=0; i<mplex; i++) {
                delete thread_infos[i];
            }
            
            for(int j=0; j<(int)sessions.size(); j++) {
                if (sessions[j] != NULL) {
                    close_session(j);
                    delete sessions[j];
                    sessions[j] = NULL;
                }
            }
            
            std::map<std::string, mmap_info>::iterator mmit = mmaped.begin();
            for(; mmit != mmaped.end(); ++mmit) {
                mmap_info minfo = mmit->second;
                munmap((void*)minfo.ptr, minfo.length);
                close(minfo.filedesc);
            }
            mmaped.clear();
        }
        
        void set_cache_budget(size_t c) {
            cache.cache_budget_bytes = c;
            cache.full = false;
        }
        
        block_cache & get_block_cache() {
            return cache;
        }
        
        /**
          * Write to disk cached blocks.
          */
        void commit_cached_blocks() {
            std::map<std::string, cached_block *>::iterator it = cache.cachemap.begin();
            for(; it != cache.cachemap.end(); ++it) {
                std::string fname = it->first;
                cached_block * block = it->second;
                
                int session = open_session(fname, false, block->was_compressed);
                pwritea_now(session, block->data, block->len, 0);
                close_session(session);
            }

        }
        
        bool multiplexed() {
            return multiplex>1;
        }
        
        void print_session(int session) {
            for(int i=0; i<multiplex; i++) {
                std::cout << "multiplex: " << multiplex << std::endl;
                std::cout << "Read desc: " << sessions[session]->readdescs[i] << std::endl;
            }
            
            for(int i=0; i<(int)sessions[session]->writedescs.size(); i++) {
                std::cout << "multiplex: " << multiplex << std::endl;
                std::cout << "Read desc: " << sessions[session]->writedescs[i] << std::endl;
            }
        }
        
        // Compute a hash for filename which is used for
        // permuting the stripes. It is important the permutation
        // is same regardless of when the file is opened.
        int hash(std::string filename) {
            const char * cstr = filename.c_str();
            int hash = 1;
            int l = (int) strlen(cstr);
            for(int i=0; i<l; i++) {
                hash = 31*hash + cstr[i];
            }
            return std::abs(hash);
        }
        
        int open_session(std::string filename, bool readonly=false, bool compressed=false) {
            mlock.lock();
            // FIXME: known memory leak: sessions table is never shrunk
            int session_id = (int) sessions.size();
            io_descriptor * iodesc = new io_descriptor();
            iodesc->open = true;
            iodesc->compressed = compressed;
            iodesc->filename = filename;
            iodesc->start_mplex = hash(filename) % multiplex;
            sessions.push_back(iodesc);
            mlock.unlock();
            
            
            
            for(int i=0; i<multiplex; i++) {
                std::string fname = multiplexprefix(i) + filename;
                for(int j=0; j<niothreads+(multiplex == 1 ? 1 : 0); j++) { // Hack to have one fd for synchronous
                    int rddesc = open(fname.c_str(), (readonly ? O_RDONLY : O_RDWR));
                    if (rddesc < 0) logstream(LOG_ERROR)  << "Could not open: " << fname << " session: " << session_id
                        << " error: " << strerror(errno) << std::endl;
                    assert(rddesc>=0);
                    iodesc->readdescs.push_back(rddesc);
#ifdef F_NOCACHE
                    if (!readonly)
                        fcntl(rddesc, F_NOCACHE, 1);
#endif
                    if (!readonly) {
                        int wrdesc = rddesc; // Change by Aapo: Aug 11, 2012. I don't think we need separate wrdesc?

                        if (wrdesc < 0) logstream(LOG_ERROR)  << "Could not open for writing: " << fname << " session: " << session_id
                            << " error: " << strerror(errno) << std::endl;
                        assert(wrdesc>=0);
#ifdef F_NOCACHE
                        fcntl(wrdesc, F_NOCACHE, 1);
                        
#endif
                        iodesc->writedescs.push_back(wrdesc);
                    }
                }
            }
            iodesc->filename = filename;
            return session_id;
        }
        
        void close_session(int session) {
            mlock.lock();
            // Note: currently io-descriptors are left into the vertex array
            // in purpose to make managed memory work. Should be fixed as this is 
            // a (relatively minor) memory leak.
            bool wasopen;
            io_descriptor * iodesc = sessions[session];
            wasopen = iodesc->open;
            iodesc->open = false;
            mlock.unlock();
            if (wasopen) {
                for(std::vector<int>::iterator it=iodesc->readdescs.begin(); it!=iodesc->readdescs.end(); ++it) {
                    close(*it);
                }
            }
        }
        
        void first_pass_finished() {
            // Optimization
            cache.full = true;
        }
        
        std::string & get_session_filename(int session) {
            return sessions[session]->filename;
        }
        
        int mplex_for_offset(int session, size_t off) {
            return ((int) (off / stripesize) + sessions[session]->start_mplex) % multiplex;
        }
        
        // Returns vector of <mplex, offset> 
        std::vector< stripe_chunk > stripe_offsets(int session, size_t nbytes, size_t off) {
            size_t end = off+nbytes;
            size_t idx = off;
            size_t bufoff = 0;
            std::vector<stripe_chunk> stripelist;
            while(idx<end) {
                size_t blockoff = idx % stripesize;
                size_t blocklen = std::min(stripesize-blockoff, end-idx);
                
                int mplex_thread = (int) mplex_for_offset(session, idx) * niothreads + (int) (random() % niothreads);
                stripelist.push_back(stripe_chunk(mplex_thread, bufoff, blocklen));
                
                bufoff += blocklen;
                idx += blocklen;
            }
            return stripelist;
        }
        
        template <typename T>
        void preada_async(int session,  T * tbuf, size_t nbytes, size_t off, volatile int * doneptr = NULL) {
            std::vector<stripe_chunk> stripelist = stripe_offsets(session, nbytes, off);
            if (compressed_session(session)) {
                assert(stripelist.size() == 1);
                assert(off == 0);
            }
            refcountptr * refptr = new refcountptr((char*)tbuf, (int)stripelist.size());
            for(int i=0; i<(int)stripelist.size(); i++) {
                stripe_chunk chunk = stripelist[i];
                __sync_add_and_fetch(&thread_infos[chunk.mplex_thread]->pending_reads, 1);
                iotask task = iotask(this, READ, sessions[session]->readdescs[chunk.mplex_thread],
                                     session,
                                     refptr, chunk.len, chunk.offset+off, chunk.offset, false,
                                     compressed_session(session));
                task.doneptr = doneptr;
                mplex_readtasks[chunk.mplex_thread].push(task);
            }
        }
       
        
        
        
        
        bool compressed_session(int session) {
            return sessions[session]->compressed;
        }
        
       
        
        
        // Note: data is freed after write!
        template <typename T>
        void pwritea_async(int session, T * tbuf, size_t nbytes, size_t off, bool free_after, bool close_fd=false) {
            std::vector<stripe_chunk> stripelist = stripe_offsets(session, nbytes, off);
            refcountptr * refptr = new refcountptr((char*)tbuf, (int) stripelist.size());
            if (compressed_session(session)) {
                assert(stripelist.size() == 1);
                assert(off == 0);
            }
            for(int i=0; i<(int)stripelist.size(); i++) {
                stripe_chunk chunk = stripelist[i];
                __sync_add_and_fetch(&thread_infos[chunk.mplex_thread]->pending_writes, 1);
                mplex_writetasks[chunk.mplex_thread].push(iotask(this, WRITE, sessions[session]->writedescs[chunk.mplex_thread], session,
                                                                 refptr, chunk.len, chunk.offset+off, chunk.offset, free_after, compressed_session(session),
                                                                        close_fd));
            }
        }
        
        template <typename T>
        void preada_now(int session,  T * tbuf, size_t nbytes, size_t off, bool dupfd=false) {
            metrics_entry me = m.start_time();
            if (compressed_session(session)) {
                // Compressed sessions do not support multiplexing for now
                assert(off == 0);
                read_compressed(sessions[session]->readdescs[0], tbuf, nbytes);
                m.stop_time(me, "preada_now", false);
                return;
            }

            if (multiplex > 1) {
                std::vector<stripe_chunk> stripelist = stripe_offsets(session, nbytes, off);
                size_t checklen=0;
                refcountptr * refptr = new refcountptr((char*)tbuf, (int) stripelist.size());
                refptr->count++; // Take a reference so we can spin on it
                for(int i=0; i < (int)stripelist.size(); i++) {
                    stripe_chunk chunk = stripelist[i];
                    __sync_add_and_fetch(&thread_infos[chunk.mplex_thread]->pending_reads, 1);
                    
                    // Use prioritized task queue
                    mplex_priotasks[chunk.mplex_thread].push(iotask(this, READ, sessions[session]->readdescs[chunk.mplex_thread], session,
                                                                    refptr, chunk.len, chunk.offset+off, chunk.offset, false,
                                                                        false));
                    checklen += chunk.len;
                }
                assert(checklen == nbytes);
                
                // Spin
                while(refptr->count>1) {
                    usleep(5000);
                }
                delete refptr;
            } else {
                if (!dupfd) {
                    preada(sessions[session]->readdescs[threads.size()], tbuf, nbytes, off);
                } else {
                    int filedesc = dup(sessions[session]->readdescs[threads.size()]);
                    preada(filedesc, tbuf, nbytes, off);
                    close(filedesc);

                }
            }
            m.stop_time(me, "preada_now", false);
        }
        
        template <typename T>
        void pwritea_now(int session, T * tbuf, size_t nbytes, size_t off) {
            metrics_entry me = m.start_time();

            if (compressed_session(session)) {
                // Compressed sessions do not support multiplexing for now
                assert(off == 0);
                write_compressed(sessions[session]->writedescs[0], tbuf, nbytes);
                m.stop_time(me, "pwritea_now", false);

                return;
            }
            std::vector<stripe_chunk> stripelist = stripe_offsets(session, nbytes, off);
            size_t checklen=0;
            
            for(int i=0; i<(int)stripelist.size(); i++) {
                stripe_chunk chunk = stripelist[i];
                pwritea(sessions[session]->writedescs[chunk.mplex_thread], (char*)tbuf+chunk.offset, chunk.len, chunk.offset+off);
                checklen += chunk.len;
            }
            assert(checklen == nbytes);
            m.stop_time(me, "pwritea_now", false);
            
        }
        
        
        
        /** 
         * Memory managed version of the I/O functions. Note, currently this management is not
         * used. Prior there was a file preloading option that allowed it, but it has been now disabled.
         */
        
        template <typename T>
        void managed_pwritea_async(int session, T ** tbuf, size_t nbytes, size_t off, bool free_after, bool close_fd=false) {
            pwritea_async(session, *tbuf, nbytes, off, free_after, close_fd);
        }
        
        template <typename T>
        void managed_preada_now(int session,  T ** tbuf, size_t nbytes, size_t off) {
            preada_now(session, *tbuf, nbytes,  off);
        }
        
        template <typename T>
        void managed_pwritea_now(int session, T ** tbuf, size_t nbytes, size_t off) {
            pwritea_now(session, *tbuf, nbytes, off);
        }
        
        template<typename T>
        void managed_malloc(int session, T ** tbuf, size_t nbytes, size_t noff) {
            *tbuf = (T*) malloc(nbytes);
        }
        
        /**
          * @param doneptr is decremented to zero when task is ready
          */
        template <typename T>
        void managed_preada_async(int session, T ** tbuf, size_t nbytes, size_t off, volatile int * doneptr = NULL) {
            preada_async(session, *tbuf, nbytes,  off, doneptr);
        }
        
        template <typename T>
        void managed_release(int session, T ** ptr) {
            assert(*ptr != NULL);
            free(*ptr);
            *ptr = NULL;
        }
        
        
        void truncate(int session, size_t nbytes) {
            assert(multiplex <= 1);  // We do not support truncating on multiplex yet
            int stat = ftruncate(sessions[session]->writedescs[0], nbytes); 
            if (stat != 0) {
                logstream(LOG_ERROR) << "Could not truncate " << sessions[session]->filename <<
                " error: " << strerror(errno) << std::endl;
                assert(false);
            }
        }
        
        void wait_for_reads() {
            metrics_entry me = m.start_time();
            int loops = 0;
            int mplex = (int) thread_infos.size();
            for(int i=0; i<mplex; i++) {
                while(thread_infos[i]->pending_reads > 0) {
                    usleep(10000);
                    loops++;
                }
            }
            m.stop_time(me, "stripedio_wait_for_reads", false);
        }
        
        void wait_for_writes() {
            metrics_entry me = m.start_time();
            int mplex = (int) thread_infos.size();
            for(int i=0; i<mplex; i++) {
                while(thread_infos[i]->pending_writes>0) {
                    usleep(10000);
                }
            }
            m.stop_time(me, "stripedio_wait_for_writes", false);
        }
        
        
        std::string multiplexprefix(int stripe) {
            if (multiplex > 1) {
                char mstr[255];
                sprintf(mstr, "%d/", 1+stripe%multiplex);
                return multiplex_root + std::string(mstr);
            } else return "";
        }
        
        std::string multiplexprefix_random() {
            return multiplexprefix((int)random() % multiplex);
        }
        
        /**
          * MMAP support
          */
        
    public:
        void * get_mmaped_file(std::string &filename, bool write) {
            std::string cachekey = (write ? filename + "?w" : filename);
            void * ptr = NULL;
            mmaplock.lock();
            
            if (mmaped.find(cachekey) == mmaped.end()) {
                logstream(LOG_DEBUG) << "Mmap: " << filename << std::endl;
                /* Not mmaped yet, so open */
                size_t mmap_length = get_filesize(filename);
                int filedesc = open(filename.c_str(), (write ? O_RDWR : O_RDONLY));
                ptr =  mmap(NULL, mmap_length, (write ? PROT_READ | PROT_WRITE : PROT_READ), MAP_SHARED, filedesc, 0);
                if (!ptr) {
                    logstream(LOG_FATAL) << "Could not mmap " << filename << std::endl;
                }
                assert(ptr);
                mmap_info minfo;
                minfo.ptr = ptr;
                minfo.length = mmap_length;
                minfo.filedesc = filedesc;
                mmaped[cachekey] = minfo;
            } else {
                ptr = mmaped[cachekey].ptr;
            }
            mmaplock.unlock();
            return ptr;
        }
        
    };
    
    
    static void * io_thread_loop(void * _info) {
        iotask task;
        thrinfo * info = (thrinfo*)_info;
        int ntasks = 0;
        // logstream(LOG_INFO) << "Thread for multiplex :" << info->mplex << " starting." << std::endl;
        while(info->running) {
            bool success;
            if (info->pending_reads>0) {  // Prioritize read queue
                success = info->prioqueue->safepop(&task);
                if (!success) {
                    success = info->readqueue->safepop(&task);
                }
            } else {
                success = info->commitqueue->safepop(&task);
            }
            if (success) {
                ++ntasks;
                if (task.action == WRITE) {  // Write
                    metrics_entry me = info->m->start_time();
                    
                    if (task.compressed) {
                        assert(task.offset == 0);
                        write_compressed(task.fd, task.ptr->ptr, task.length);
                    } else {
                        pwritea(task.fd, task.ptr->ptr + task.ptroffset, task.length, task.offset);
                    }
                    if (task.free_after) {
                        // Threead-safe method of memory managment - ugly!
                        if (__sync_sub_and_fetch(&task.ptr->count, 1) == 0) {
                            free(task.ptr->ptr);
                            delete task.ptr;
                            if (task.closefd) {
                                task.iomgr->close_session(task.session);
                            }
                        }
                    }
                   
                    __sync_sub_and_fetch(&info->pending_writes, 1);
                    info->m->stop_time(me, "commit_thr");
                } else {
                    if (task.compressed) {
                        assert(task.offset == 0);
                        read_compressed(task.fd, task.ptr->ptr, task.length);

                    } else {
                        preada(task.fd, task.ptr->ptr+task.ptroffset, task.length, task.offset);
                    }
                    __sync_sub_and_fetch(&info->pending_reads, 1);
                    if (__sync_sub_and_fetch(&task.ptr->count, 1) == 0) {
                        free(task.ptr);
                        if (task.closefd) {
                            task.iomgr->close_session(task.session);
                        }
                    }
                }
                if (task.doneptr != NULL) {
                    __sync_sub_and_fetch(task.doneptr, 1);
                }
            } else {
                usleep(50000); // 50 ms
            }
        }
        // logstream(LOG_INFO) << "I/O thread exists. Handled " << ntasks << " i/o tasks." << std::endl;
        return NULL;
    }
    
    
   
    static size_t get_filesize(std::string filename) {
        std::string fname = filename;
        int f = open(fname.c_str(), O_RDONLY);
        
        if (f < 0) {
            logstream(LOG_ERROR) << "Could not open file " << filename << " error: " << strerror(errno) << std::endl;
            assert(false);
        }
        
        off_t sz = lseek(f, 0, SEEK_END);
        close(f);
        return sz;
    }
    
    
    
}


#endif



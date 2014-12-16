#ifndef DEF_PTHREAD_TOOLS_HPP
#define DEF_PTHREAD_TOOLS_HPP

// Stolen from GraphLab

#include <cstdlib>
#include <memory.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/time.h>
#include <vector>
#include <cassert>
#include <list>
#include <iostream>

#undef _POSIX_SPIN_LOCKS
#define _POSIX_SPIN_LOCKS -1




/**
 * \file pthread_tools.hpp A collection of utilities for threading
 */
namespace graphchi {



    /**
     * \class mutex
     *
     * Wrapper around pthread's mutex On single core systems mutex
     * should be used.  On multicore systems, spinlock should be used.
     */
    class mutex {
    private:
        // mutable not actually needed
        mutable pthread_mutex_t m_mut;
    public:
        mutex() {
            int error = pthread_mutex_init(&m_mut, NULL);
            assert(!error);
        }
        inline void lock() const {
            int error = pthread_mutex_lock( &m_mut  );
            assert(!error);
        }
        inline void unlock() const {
            int error = pthread_mutex_unlock( &m_mut );
            assert(!error);
        }
        inline bool try_lock() const {
            return pthread_mutex_trylock( &m_mut ) == 0;
        }
        ~mutex(){
            int error = pthread_mutex_destroy( &m_mut );
            if (error)
              perror("Error: failed to destroy mutex");
            assert(!error);
        }
        friend class conditional;
    }; // End of Mutex

#if _POSIX_SPIN_LOCKS >= 0
    // We should change this to use a test for posix_spin_locks eventually

    // #ifdef __linux__
    /**
     * \class spinlock
     *
     * Wrapper around pthread's spinlock On single core systems mutex
     * should be used.  On multicore systems, spinlock should be used.
     * If pthread_spinlock is not available, the spinlock will be
     * typedefed to a mutex
     */
    class spinlock {
    private:
        // mutable not actually needed
        mutable pthread_spinlock_t m_spin;
    public:
        spinlock () {
            int error = pthread_spin_init(&m_spin, PTHREAD_PROCESS_PRIVATE);
            assert(!error);
        }

        inline void lock() const {
            int error = pthread_spin_lock( &m_spin  );
            assert(!error);
        }
        inline void unlock() const {
            int error = pthread_spin_unlock( &m_spin );
            assert(!error);
        }
        inline bool try_lock() const {
            return pthread_spin_trylock( &m_spin ) == 0;
        }
        ~spinlock(){
            int error = pthread_spin_destroy( &m_spin );
            assert(!error);
        }
        friend class conditional;
    }; // End of spinlock
#define SPINLOCK_SUPPORTED 1
#else
    //! if spinlock not supported, it is typedef it to a mutex.
    typedef mutex spinlock;
#define SPINLOCK_SUPPORTED 0
#endif


    /**
     * \class conditional
     * Wrapper around pthread's condition variable
     */
    class conditional {
    private:
        mutable pthread_cond_t  m_cond;
    public:
        conditional() {
            int error = pthread_cond_init(&m_cond, NULL);
            assert(!error);
        }
        inline void wait(const mutex& mut) const {
            int error = pthread_cond_wait(&m_cond, &mut.m_mut);
            assert(!error);
        }
        inline int timedwait(const mutex& mut, int sec) const {
            struct timespec timeout;
            struct timeval tv;
            struct timezone tz;
            gettimeofday(&tv, &tz);
            timeout.tv_nsec = 0;
            timeout.tv_sec = tv.tv_sec + sec;
            return pthread_cond_timedwait(&m_cond, &mut.m_mut, &timeout);
        }
        inline void signal() const {
            int error = pthread_cond_signal(&m_cond);
            assert(!error);
        }
        inline void broadcast() const {
            int error = pthread_cond_broadcast(&m_cond);
            assert(!error);
        }
        ~conditional() {
            int error = pthread_cond_destroy(&m_cond);
            assert(!error);
        }
    }; // End conditional





#define atomic_xadd(P, V) __sync_fetch_and_add((P), (V))
#define cmpxchg(P, O, N) __sync_val_compare_and_swap((P), (O), (N))
#define atomic_inc(P) __sync_add_and_fetch((P), 1)

    /**
     * \class spinrwlock
     * rwlock built around "spinning"
     * source adapted from http://locklessinc.com/articles/locks/
     * "Scalable Reader-Writer Synchronization for Shared-Memory Multiprocessors"
     * John Mellor-Crummey and Michael Scott
     */
    class spinrwlock {

        union rwticket {
            unsigned u;
            unsigned short us;
            __extension__ struct {
                unsigned char write;
                unsigned char read;
                unsigned char users;
            } s;
        };
        mutable bool writing;
        mutable volatile rwticket l;
    public:
        spinrwlock() {
            memset(const_cast<rwticket*>(&l), 0, sizeof(rwticket));
        }
        inline void writelock() const {
            unsigned me = atomic_xadd(&l.u, (1<<16));
            unsigned char val = me >> 16;

            while (val != l.s.write) sched_yield();
            writing = true;
        }

        inline void wrunlock() const{
            rwticket t = *const_cast<rwticket*>(&l);

            t.s.write++;
            t.s.read++;

            *(volatile unsigned short *) (&l) = t.us;
            writing = false;
            __asm("mfence");
        }

        inline void readlock() const {
            unsigned me = atomic_xadd(&l.u, (1<<16));
            unsigned char val = me >> 16;

            while (val != l.s.read) sched_yield();
            l.s.read++;
        }

        inline void rdunlock() const {
            atomic_inc(&l.s.write);
        }

        inline void unlock() const {
            if (!writing) rdunlock();
            else wrunlock();
        }
    };

#undef atomic_xadd
#undef cmpxchg
#undef atomic_inc


    /**
     * \class rwlock
     * Wrapper around pthread's rwlock
     */
    class rwlock {
    private:
        mutable pthread_rwlock_t m_rwlock;
    public:
        rwlock() {
            int error = pthread_rwlock_init(&m_rwlock, NULL);
            assert(!error);
        }
        ~rwlock() {
            int error = pthread_rwlock_destroy(&m_rwlock);
            assert(!error);
        }
        inline void readlock() const {
            pthread_rwlock_rdlock(&m_rwlock);
            //assert(!error);
        }
        inline void writelock() const {
            pthread_rwlock_wrlock(&m_rwlock);
            //assert(!error);
        }
        inline void unlock() const {
            pthread_rwlock_unlock(&m_rwlock);
            //assert(!error);
        }
        inline void rdunlock() const {
            unlock();
        }
        inline void wrunlock() const {
            unlock();
        }
    }; // End rwlock

    /**
     * \class barrier
     * Wrapper around pthread's barrier
     */
#ifdef __linux__
    /**
     * \class barrier
     * Wrapper around pthread's barrier
     */
    class barrier {
    private:
        mutable pthread_barrier_t m_barrier;
    public:
        barrier(size_t numthreads) { pthread_barrier_init(&m_barrier, NULL, numthreads); }
        ~barrier() { pthread_barrier_destroy(&m_barrier); }
        inline void wait() const { pthread_barrier_wait(&m_barrier); }
    };

#else
    /**
     * \class barrier
     * Wrapper around pthread's barrier
     */
    class barrier {
    private:
        mutex m;
        int needed;
        int called;
        conditional c;

        // we need the following to protect against spurious wakeups
        std::vector<unsigned char> waiting;
    public:

        barrier(size_t numthreads) {
            needed = (int)numthreads;
            called = 0;
            waiting.resize(numthreads);
            std::fill(waiting.begin(), waiting.end(), 0);
        }

        ~barrier() {}


        inline void wait() {
            m.lock();
            // set waiting;
            size_t myid = called;
            waiting[myid] = 1;
            called++;

            if (called == needed) {
                // if I have reached the required limit, wait up. Set waiting
                // to 0 to make sure everyone wakes up

                called = 0;
                // clear all waiting
                std::fill(waiting.begin(), waiting.end(), 0);
                c.broadcast();
            }
            else {
                // while no one has broadcasted, sleep
                while(waiting[myid]) c.wait(m);
            }
            m.unlock();
        }
    };
#endif



    inline void prefetch_range(void *addr, size_t len) {
        char *cp;
        char *end = (char*)(addr) + len;

        for (cp = (char*)(addr); cp < end; cp += 64) __builtin_prefetch(cp, 0);
    }
    inline void prefetch_range_write(void *addr, size_t len) {
        char *cp;
        char *end = (char*)(addr) + len;

        for (cp = (char*)(addr); cp < end; cp += 64) __builtin_prefetch(cp, 1);
    }


};
#endif


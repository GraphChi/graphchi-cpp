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
 * For more about this software visit:
 *
 *      http://www.graphlab.ml.cmu.edu
 *
 */


#include <pthread.h>

#include <set>
#include <iostream>
#include <fstream>

#include <boost/random.hpp>
#include <boost/integer_traits.hpp>

#include "util/pthread_tools.hpp"
#include "api/graphlab2_1_GAS_api/graphlab.hpp"



namespace graphlab {
    namespace random {
        
        /**
         * A truely nondeterministic generator
         */
        class nondet_generator {
        public:
            static nondet_generator& global() {
                static nondet_generator global_gen;
                return global_gen;
            }
            
            typedef size_t result_type;
            BOOST_STATIC_CONSTANT(result_type, min_value =
                                  boost::integer_traits<result_type>::const_min);
            BOOST_STATIC_CONSTANT(result_type, max_value =
                                  boost::integer_traits<result_type>::const_max);
            result_type min BOOST_PREVENT_MACRO_SUBSTITUTION () const { return min_value; }
            result_type max BOOST_PREVENT_MACRO_SUBSTITUTION () const { return max_value; }
            
            nondet_generator() {
                rnd_dev.open("/dev/urandom", std::ios::binary | std::ios::in);
                ASSERT_TRUE(rnd_dev.good());
            }
            // Close the random number generator
            ~nondet_generator() { rnd_dev.close(); }
            // read a size_t from the source
            result_type operator()() {
                // read a machine word into result
                result_type result(0);
                mut.lock();
                ASSERT_TRUE(rnd_dev.good());
                rnd_dev.read(reinterpret_cast<char*>(&result), sizeof(result_type));
                ASSERT_TRUE(rnd_dev.good());
                mut.unlock();
                //        std::cout << result << std::endl;
                return result;
            }
        private:
            std::ifstream rnd_dev;
            mutex mut;
        };
        //nondet_generator global_nondet_rng;
        
        
        
        
        
        
        /**
         * This class represents a master registery of all active random
         * number generators
         */
        struct source_registry {
            std::set<generator*> generators;
            generator master;
            mutex mut;
            
            static source_registry& global() {
                static source_registry registry;
                return registry;
            }
            /**
             * Seed all threads using the default seed
             */
            void seed() {
                mut.lock();
                master.seed();
                foreach(generator* generator, generators) {
                    ASSERT_TRUE(generator != NULL);
                    generator->seed(master);
                }
                mut.unlock();
            }
            
            /**
             * Seed all threads using the default seed
             */
            void nondet_seed() {
                mut.lock();
                master.nondet_seed();
                foreach(generator* generator, generators) {
                    ASSERT_TRUE(generator != NULL);
                    generator->seed(master);
                }
                mut.unlock();
            }
            
            
            /**
             * Seed all threads using the default seed
             */
            void time_seed() {
                mut.lock();
                master.time_seed();
                foreach(generator* generator, generators) {
                    ASSERT_TRUE(generator != NULL);
                    generator->seed(master);
                }
                mut.unlock();
            }
            
            
            /**
             *  Seed all threads with a fixed number
             */
            void seed(const size_t number) {
                mut.lock();
                master.seed(number);
                foreach(generator* generator, generators) {
                    ASSERT_TRUE(generator != NULL);
                    generator->seed(master);
                }
                mut.unlock();
            }
            
            /**
             * Register a source with the registry and seed it based on the
             * master.
             */
            void register_generator(generator* tls_ptr) {
                ASSERT_TRUE(tls_ptr != NULL);
                mut.lock();
                generators.insert(tls_ptr);
                tls_ptr->seed(master);
                // std::cout << "Generator created" << std::endl;
                // __print_back_trace();
                mut.unlock();
            }
            
            /**
             * Unregister a source from the registry
             */
            void unregister_source(generator* tls_ptr) {
                mut.lock();
                generators.erase(tls_ptr);
                mut.unlock();
            }
        };
        // source_registry registry;
        
        
        
        
        
        
        
        
        //////////////////////////////////////////////////////////////
        /// Pthread TLS code
        
        /**
         * this function is responsible for destroying the random number
         * generators
         */
        void destroy_tls_data(void* ptr) {
            generator* tls_rnd_ptr =
            reinterpret_cast<generator*>(ptr);
            if(tls_rnd_ptr != NULL) {
                source_registry::global().unregister_source(tls_rnd_ptr);
                delete tls_rnd_ptr;
            }
        }
        
        
        /**
         * Simple struct used to construct the thread local storage at
         * startup.
         */
        struct tls_key_creator {
            pthread_key_t TLS_RANDOM_SOURCE_KEY;
            tls_key_creator() : TLS_RANDOM_SOURCE_KEY(0) {
                pthread_key_create(&TLS_RANDOM_SOURCE_KEY,
                                   destroy_tls_data);
            }
        };
        // This function is to be called prior to any access to the random
        // source
        static pthread_key_t get_random_source_key() {
            static const tls_key_creator key;
            return key.TLS_RANDOM_SOURCE_KEY;
        }
        // This forces __init_keys__ to be called prior to main.
        static pthread_key_t __unused_init_keys__(get_random_source_key());
        
        // the combination of the two mechanisms above will force the
        // thread local store to be initialized
        // 1: before main
        // 2: before any use of random by global variables.
        // KNOWN_ISSUE: if a global variable (initialized before main)
        //               spawns threads which then call random. Things explode.
        
        
        /////////////////////////////////////////////////////////////
        //// Implementation of header functions
        
        
        
        generator& get_source() {
            // get the thread local storage
            generator* tls_rnd_ptr =
            reinterpret_cast<generator*>
            (pthread_getspecific(get_random_source_key()));
            // Create a tls_random_source if none was provided
            if(tls_rnd_ptr == NULL) {
                tls_rnd_ptr = new generator();
                assert(tls_rnd_ptr != NULL);
                // This will seed it with the master rng
                source_registry::global().register_generator(tls_rnd_ptr);
                pthread_setspecific(get_random_source_key(), 
                                    tls_rnd_ptr);      
            }
            // assert(tls_rnd_ptr != NULL);
            return *tls_rnd_ptr;
        } // end of get local random source
        
        
        
        void seed() { source_registry::global().seed();  } 
        
        void nondet_seed() { source_registry::global().nondet_seed(); } 
        
        void time_seed() { source_registry::global().time_seed(); } 
        
        void seed(const size_t seed_value) { 
            source_registry::global().seed(seed_value);  
        } 
        
        
        void generator::nondet_seed() {
            // Get the global nondeterministic random number generator.
            nondet_generator& nondet_rnd(nondet_generator::global());
            mut.lock();
            // std::cout << "initializing real rng" << std::endl;
            real_rng.seed(nondet_rnd());
            // std::cout << "initializing discrete rng" << std::endl;
            discrete_rng.seed(nondet_rnd());
            // std::cout << "initializing fast discrete rng" << std::endl;
            fast_discrete_rng.seed(nondet_rnd());
            mut.unlock();
        }
        
        
        void pdf2cdf(std::vector<double>& pdf) {
            double Z = 0;
            for(size_t i = 0; i < pdf.size(); ++i) Z += pdf[i];
            for(size_t i = 0; i < pdf.size(); ++i)
                pdf[i] = pdf[i]/Z + ((i>0)? pdf[i-1] : 0);
        } // end of pdf2cdf
        
        
        
        
    }; // end of namespace random
    
};// end of namespace graphlab


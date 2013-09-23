

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
 * Output interface for GraphChi programs. Use GraphChi context to access and initialize.
 */

#ifndef DEF_OUTPUT_HPP
#define DEF_OUTPUT_HPP

#include "graphchi_types.hpp"
#include "util/pthread_tools.hpp"
#include <fstream>

namespace graphchi {
    
    template <typename VT, typename ET>
    class ioutput{
        
    public:
        virtual void output_edge(vid_t from, vid_t to) = 0;
        
        
         
        virtual void output_edge(vid_t from, vid_t to, float value) = 0;

        virtual void output_edge(vid_t from, vid_t to, double value) = 0;
        
        virtual void output_edge(vid_t from, vid_t to, int value) = 0;

        virtual void output_edge(vid_t from, vid_t to, size_t value) = 0;

       // virtual void output_edgeval(vid_t from, vid_t to, ET value) = 0;

        virtual void output_value(vid_t vid, VT value) = 0;
        
        
        // Called automatically at the end
        virtual void close() = 0;
        
    };
    
    
    template <typename VT, typename ET>
    class basic_text_output : public ioutput<VT, ET> {
        
        std::ofstream strm;
        std::string delimiter;
        mutex lock;
        
    public:
        
        basic_text_output(std::string filename, std::string delimiter="\t") : strm(filename.c_str(),std::ofstream::out), 
                delimiter(delimiter) {
        }
        
        ~basic_text_output() {
            strm.close();
        }
   
        
    protected:
        template <typename T>
        void _output_edge(vid_t from, vid_t to, T val) {
            lock.lock();
            strm << from << delimiter << to << delimiter << val << "\n";
            lock.unlock();
        }
        
    public:
        void output_edge(vid_t from, vid_t to) {
            lock.lock();
            strm << from << delimiter << to << "\n";
            lock.unlock();
        }
        
        
         
        
        virtual void output_edge(vid_t from, vid_t to, float value) {
            _output_edge(from, to, value);
        }
        
        virtual void output_edge(vid_t from, vid_t to, double value) {
            _output_edge(from, to, value);
        }
        
        
        virtual void output_edge(vid_t from, vid_t to, int value)  {
            _output_edge(from, to, value);
        }
        
        virtual void output_edge(vid_t from, vid_t to, size_t value)  {
            _output_edge(from, to, value);
        }
        
        void output_edgeval(vid_t from, vid_t to, ET value) {
            assert(false);
        }
        
        
        void output_value(vid_t vid, VT value) {
            lock.lock();
            strm << vid << delimiter << value << "\n";    
            lock.unlock();
        }
        
        
        
        void close() {
            strm.close();
        }
        
    };
    
    

    
}





#endif


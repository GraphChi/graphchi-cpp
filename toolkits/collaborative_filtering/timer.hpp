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


#ifndef GRAPHLAB_TIMER_HPP
#define GRAPHLAB_TIMER_HPP

#include <sys/time.h>
#include <stdio.h>

#include <iostream>

  /**
   * \ingroup util 
   *
   * \brief A simple class that can be used for benchmarking/timing up
   * to microsecond resolution.
   *
   * Standard Usage
   * =================
   *
   * The timer is used by calling \ref graphlab::timer::start and then
   * by getting the current time since start by calling 
   * \ref graphlab::timer::current_time.
   * 
   * For example:
   * 
   * \code
   *
   * graphlab::timer timer;
   * timer.start();
   * // do something
   * std::cout << "Elapsed time: " << timer.current_time() << std::endl; 
   * \endcode
   *
   * Fast approximate time
   * ====================
   *
   * Calling current item in a tight loop can be costly and so we
   * provide a faster less accurate timing primitive which reads a
   * local time variable that is updated roughly every 100 millisecond.
   * These are the \ref graphlab::timer::approx_time_seconds and
   * \ref graphlab::timer::approx_time_millis.
   */
  class timer {
  private:
    /**
     * \brief The internal start time for this timer object
     */
    timeval start_time_;   
  public:
    /**
     * \brief The timer starts on construction but can be restarted by
     * calling \ref graphlab::timer::start.
     */
    inline timer() { start(); }
    
    /**
     * \brief Reset the timer.
     */
    inline void start() { gettimeofday(&start_time_, NULL); }
    
    /** 
     * \brief Returns the elapsed time in seconds since 
     * \ref graphlab::timer::start was last called.
     *
     * @return time in seconds since \ref graphlab::timer::start was called.
     */
    inline double current_time() const {
      timeval current_time;
      gettimeofday(&current_time, NULL);
      double answer = 
       // (current_time.tv_sec + ((double)current_time.tv_usec)/1.0E6) -
       // (start_time_.tv_sec + ((double)start_time_.tv_usec)/1.0E6);
        (double)(current_time.tv_sec - start_time_.tv_sec) + 
        ((double)(current_time.tv_usec - start_time_.tv_usec))/1.0E6;
       return answer;
    } // end of current_time

    /** 
     * \brief Returns the elapsed time in milliseconds since 
     * \ref graphlab::timer::start was last called.
     *
     * @return time in milliseconds since \ref graphlab::timer::start was called.
     */
    inline double current_time_millis() const { return current_time() * 1000; }

  }; // end of Timer
  

#endif


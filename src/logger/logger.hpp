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


/**
 * @file logger.hpp
 * Usage:
 * First include logger.hpp. To logger, use the logger() function
 * There are 2 output levels. A "soft" output level which is
 * set by calling global_logger.set_log_level(), as well as a "hard" output
 * level OUTPUTLEVEL which is set in the source code (logger.h).
 *
 * when you call "logger()" with a loglevel and if the loglevel is greater than
 * both of the output levels, the string will be written.
 * written to a logger file. Otherwise, logger() has no effect.
 *
 * The difference between the hard level and the soft level is that the
 * soft level can be changed at runtime, while the hard level optimizes away
 * logging calls at compile time.
 *
 * @author Yucheng Low (ylow)
 */

/** 
  * NOTICE: This file taken from GraphLab (as stated in the license above).
  * I have merged the CPP and HPP files.
  * @author Aapo Kyrola
  */

#ifndef GRAPHCHI_LOG_LOG_HPP
#define GRAPHCHI_LOG_LOG_HPP
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <cstring>
#include <cstdarg>
#include <pthread.h>
/**
 * \def LOG_FATAL
 *   Used for fatal and probably irrecoverable conditions
 * \def LOG_ERROR
 *   Used for errors which are recoverable within the scope of the function
 * \def LOG_WARNING
 *   Logs interesting conditions which are probably not fatal
 * \def LOG_INFO
 *   Used for providing general useful information
 * \def LOG_DEBUG
 *   Debugging purposes only
 */
#define LOG_NONE 5
#define LOG_FATAL 4
#define LOG_ERROR 3
#define LOG_WARNING 2
#define LOG_INFO 1
#define LOG_DEBUG 0

/**
 * \def OUTPUTLEVEL
 *  The minimum level to logger at
 * \def LOG_NONE
 *  OUTPUTLEVEL to LOG_NONE to disable logging
 */

#ifndef OUTPUTLEVEL
#define OUTPUTLEVEL LOG_DEBUG
#endif
/// If set, logs to screen will be printed in color
#define COLOROUTPUT


/**
 * \def logger(lvl,fmt,...)
 *    extracts the filename, line number
 *     and function name and calls _log. It will be optimized
 *     away if LOG_NONE is set
 *     This relies on a few compiler macros. As far as I know, these
 *     macros are pretty standard among most other C++ compilers. 
 */
#if OUTPUTLEVEL == LOG_NONE
// totally disable logging
#define logger(lvl,fmt,...)
#define logbuf(lvl,fmt,...)
#define logstream
#else

#define logger(lvl,fmt,...)                 \
    (log_dispatch<(lvl >= OUTPUTLEVEL)>::exec(lvl,__FILE__, __func__ ,__LINE__,fmt,##__VA_ARGS__))

    
#define logbuf(lvl,buf,len)                 \
    (log_dispatch<(lvl >= OUTPUTLEVEL)>::exec(lvl,__FILE__,     \
                        __func__ ,__LINE__,buf,len))

#define logstream(lvl)                      \
    (log_stream_dispatch<(lvl >= OUTPUTLEVEL)>::exec(lvl,__FILE__, __func__ ,__LINE__) )
#endif

static const char* messages[] = {  "DEBUG:    ",
    "INFO:     ",
    "WARNING:  ",
    "ERROR:    ",
    "FATAL:    "};

namespace logger_impl {
struct streambuff_tls_entry {
  std::stringstream streambuffer;
  bool streamactive;
};
}

 
/**
  logging class.
  This writes to a file, and/or the system console.
*/
class file_logger{
 public:

 

  /** Closes the current logger file if one exists.
      if 'file' is not an empty string, it will be opened and 
      all subsequent logger output will be written into 'file'.
      Any existing content of 'file' will be cleared.
      Return true on success and false on failure.
  */
 
  /// If consolelog is true, subsequent logger output will be written to stderr
  void set_log_to_console(bool consolelog) {
    log_to_console = consolelog;
  }

  /// Returns the current logger file.
  std::string get_log_file(void) {
    return log_file;
  }

  /// Returns true if output is being written to stderr
  bool get_log_to_console() {
    return log_to_console;
  }

  /// Returns the current logger level
  int get_log_level() {
    return log_level;
  }

   
  template <typename T>
  file_logger& operator<<(T a) {
    // get the stream buffer
    logger_impl::streambuff_tls_entry* streambufentry = reinterpret_cast<logger_impl::streambuff_tls_entry*>(
                                          pthread_getspecific(streambuffkey));
    if (streambufentry != NULL) {
      std::stringstream& streambuffer = streambufentry->streambuffer;
      bool& streamactive = streambufentry->streamactive;

      if (streamactive) streambuffer << a;
    }
    return *this;
  }

  file_logger& operator<<(const char* a) {
    // get the stream buffer
    logger_impl::streambuff_tls_entry* streambufentry = reinterpret_cast<logger_impl::streambuff_tls_entry*>(
                                          pthread_getspecific(streambuffkey));
    if (streambufentry != NULL) {
      std::stringstream& streambuffer = streambufentry->streambuffer;
      bool& streamactive = streambufentry->streamactive;

      if (streamactive) {
        streambuffer << a;
        if (a[strlen(a)-1] == '\n') {
          stream_flush();
        }
      }
    }
    return *this;
  }

  file_logger& operator<<(std::ostream& (*f)(std::ostream&)){
    // get the stream buffer
    logger_impl::streambuff_tls_entry* streambufentry = reinterpret_cast<logger_impl::streambuff_tls_entry*>(
                                          pthread_getspecific(streambuffkey));
    if (streambufentry != NULL) {
      std::stringstream& streambuffer = streambufentry->streambuffer;
      bool& streamactive = streambufentry->streamactive;

      typedef std::ostream& (*endltype)(std::ostream&);
      if (streamactive) {
        if (endltype(f) == endltype(std::endl)) {
          streambuffer << "\n";
          stream_flush();
          if(streamloglevel == LOG_FATAL) {
              throw "log fatal";
            // exit(EXIT_FAILURE);
          }
        }
      }
    }
    return *this;
  }



  /** Sets the current logger level. All logging commands below the current
      logger level will not be written. */
  void set_log_level(int new_log_level) {
    log_level = new_log_level;
  }

 
    

    
    
    
    static void streambuffdestructor(void* v){
        logger_impl::streambuff_tls_entry* t = 
        reinterpret_cast<logger_impl::streambuff_tls_entry*>(v);
        delete t;
    }
    
   
    
    /** Default constructor. By default, log_to_console is off,
     there is no logger file, and logger level is set to LOG_WARNING
     */
    file_logger() {
        log_file = "";
        log_to_console = true;
        log_level = LOG_DEBUG; 
        pthread_mutex_init(&mut, NULL);
        pthread_key_create(&streambuffkey, streambuffdestructor);
    }
    
    ~file_logger() {
        if (fout.good()) {
            fout.flush();
            fout.close();
        }
        
        pthread_mutex_destroy(&mut);
    }
    
    bool set_log_file(std::string file) {
        // close the file if it is open
        if (fout.good()) {
            fout.flush();
            fout.close();
            log_file = "";
        }
        // if file is not an empty string, open the new file
        if (file.length() > 0) {
            fout.open(file.c_str());
            if (fout.fail()) return false;
            log_file = file;
        }
        return true;
    }
    
    
    
#define RESET   0
#define BRIGHT    1
#define DIM   2
#define UNDERLINE   3
#define BLINK   4
#define REVERSE   7
#define HIDDEN    8
    
#define BLACK     0
#define RED   1
#define GREEN   2
#define YELLOW    3
#define BLUE    4
#define MAGENTA   5
#define CYAN    6
#define WHITE   7
    
    void textcolor(FILE* handle, int attr, int fg)
    {
        char command[13];
        /* Command is the control command to the terminal */
        sprintf(command, "%c[%d;%dm", 0x1B, attr, fg + 30);
        fprintf(handle, "%s", command);
    }
    
    void reset_color(FILE* handle)
    {
        char command[20];
        /* Command is the control command to the terminal */
        sprintf(command, "%c[0m", 0x1B);
        fprintf(handle, "%s", command);
    }
    
    
    
    void _log(int lineloglevel,const char* file,const char* function,
                           int line,const char* fmt, va_list ap ){
        // if the logger level fits
        if (lineloglevel >= 0 && lineloglevel <= 3 && lineloglevel >= log_level){
            // get just the filename. this line found on a forum on line.
            // claims to be from google.
            file = ((strrchr(file, '/') ? : file- 1) + 1);
            
            char str[1024];
            
            // write the actual header
            int byteswritten = snprintf(str,1024, "%s%s(%s:%d): ",
                                        messages[lineloglevel],file,function,line);
            // write the actual logger
            
            byteswritten += vsnprintf(str + byteswritten,1024 - byteswritten,fmt,ap);
            
            str[byteswritten] = '\n';
            str[byteswritten+1] = 0;
            // write the output
            if (fout.good()) {
                pthread_mutex_lock(&mut);
                fout << str;;
                pthread_mutex_unlock(&mut);
            }
            if (log_to_console) {
#ifdef COLOROUTPUT
                if (lineloglevel == LOG_FATAL) {
                    textcolor(stderr, BRIGHT, RED);
                }
                else if (lineloglevel == LOG_ERROR) {
                    textcolor(stderr, BRIGHT, RED);
                }
                else if (lineloglevel == LOG_WARNING) {
                    textcolor(stderr, BRIGHT, GREEN);
                }
#endif
                std::cerr << str;;
#ifdef COLOROUTPUT
                reset_color(stderr);
#endif
            }
        }
    }
    
    
    
    void _logbuf(int lineloglevel,const char* file,const char* function,
                              int line,const char* buf, int len) {
        // if the logger level fits
        if (lineloglevel >= 0 && lineloglevel <= 3 && lineloglevel >= log_level){
            // get just the filename. this line found on a forum on line.
            // claims to be from google.
            file = ((strrchr(file, '/') ? : file- 1) + 1);
            
            // length of the 'head' of the string
            size_t headerlen = snprintf(NULL,0,"%s%s(%s:%d): ",
                                        messages[lineloglevel],file,function,line);
            
            if (headerlen> 2047) {
                std::cerr << "Header length exceed buffer length!";
            }
            else {
                char str[2048];
                const char *newline="\n";
                // write the actual header
                int byteswritten = snprintf(str,2047,"%s%s(%s:%d): ",
                                            messages[lineloglevel],file,function,line);
                _lograw(lineloglevel,str, byteswritten);
                _lograw(lineloglevel,buf, len);
                _lograw(lineloglevel,newline, (int)strlen(newline));
            }
        }
    }
    
    void _lograw(int lineloglevel, const char* buf, int len) {
        if (fout.good()) {
            pthread_mutex_lock(&mut);
            fout.write(buf,len);
            pthread_mutex_unlock(&mut);
        }
        if (log_to_console) {
#ifdef COLOROUTPUT
            if (lineloglevel == LOG_FATAL) {
                textcolor(stderr, BRIGHT, RED);
            }
            else if (lineloglevel == LOG_ERROR) {
                textcolor(stderr, BRIGHT, RED);
            }
            else if (lineloglevel == LOG_WARNING) {
                textcolor(stderr, BRIGHT, GREEN);
            }
            else if (lineloglevel == LOG_DEBUG) {
                textcolor(stderr, BRIGHT, YELLOW);
            }
            
#endif
            std::cerr.write(buf,len);
#ifdef COLOROUTPUT
            reset_color(stderr);
#endif
        }
    }
    
    file_logger& start_stream(int lineloglevel,const char* file,const char* function, int line) {
        // get the stream buffer
        logger_impl::streambuff_tls_entry* streambufentry = reinterpret_cast<logger_impl::streambuff_tls_entry*>(
                                                                                                                 pthread_getspecific(streambuffkey));
        // create the key if it doesn't exist
        if (streambufentry == NULL) {
            streambufentry = new logger_impl::streambuff_tls_entry;
            pthread_setspecific(streambuffkey, streambufentry);
        }
        std::stringstream& streambuffer = streambufentry->streambuffer;
        bool& streamactive = streambufentry->streamactive;
        
        file = ((strrchr(file, '/') ? : file- 1) + 1);
        
        if (lineloglevel >= log_level){
            if (streambuffer.str().length() == 0) {
                streambuffer << messages[lineloglevel] << file
                << "(" << function << ":" <<line<<"): ";
            }
            streamactive = true;
            streamloglevel = lineloglevel;
        }
        else {
            streamactive = false;
        }
        return *this;
    }
    
    

  void stream_flush() {
    // get the stream buffer
    logger_impl::streambuff_tls_entry* streambufentry = reinterpret_cast<logger_impl::streambuff_tls_entry*>(
                                          pthread_getspecific(streambuffkey));
    if (streambufentry != NULL) {
      std::stringstream& streambuffer = streambufentry->streambuffer;

      streambuffer.flush();
      _lograw(streamloglevel,
              streambuffer.str().c_str(),
              (int)(streambuffer.str().length()));
      streambuffer.str("");
    }
  }
 private:
  std::ofstream fout;
  std::string log_file;
  
  pthread_key_t streambuffkey;
  
  int streamloglevel;
  pthread_mutex_t mut;
  
  bool log_to_console;
  int log_level;

};


static file_logger& global_logger();

/**
Wrapper to generate 0 code if the output level is lower than the log level
*/
template <bool dostuff>
struct log_dispatch {};

template <>
struct log_dispatch<true> {
  inline static void exec(int loglevel,const char* file,const char* function,
                int line,const char* fmt, ... ) {
  va_list argp;
	va_start(argp, fmt);
	global_logger()._log(loglevel, file, function, line, fmt, argp);
	va_end(argp);
  }
};

template <>
struct log_dispatch<false> {
  inline static void exec(int loglevel,const char* file,const char* function,
                int line,const char* fmt, ... ) {}
};


struct null_stream {
  template<typename T>
  inline null_stream operator<<(T t) { return null_stream(); }
  inline null_stream operator<<(const char* a) { return null_stream(); }
  inline null_stream operator<<(std::ostream& (*f)(std::ostream&)) { return null_stream(); }
};


template <bool dostuff>
struct log_stream_dispatch {};

template <>
struct log_stream_dispatch<true> {
  inline static file_logger& exec(int lineloglevel,const char* file,const char* function, int line) {
    return global_logger().start_stream(lineloglevel, file, function, line);
  }
};

template <>
struct log_stream_dispatch<false> {
  inline static null_stream exec(int lineloglevel,const char* file,const char* function, int line) {
    return null_stream();
  }
};

void textcolor(FILE* handle, int attr, int fg);
void reset_color(FILE* handle);

static file_logger& global_logger() {
    static file_logger l;
    return l;
}


#endif


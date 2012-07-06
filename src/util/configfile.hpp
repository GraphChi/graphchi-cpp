

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
 * Parses a simple configuration file.
 * Why did I write my own?
 */
#ifndef GRAPHCHI_CONFIGFILE_DEF
#define GRAPHCHI_CONFIGFILE_DEF

#include <iostream>
#include <cstdio>
#include <string>
#include <map>
#include <assert.h>

namespace graphchi {
    
    // Code for trimming strings copied from + modified
    // http://stackoverflow.com/questions/479080/trim-is-not-part-of-the-standard-c-c-library
    const std::string whiteSpaces( " \f\n\r\t\v" );
    
    
    static void trimRight( std::string &str,
                          const std::string& trimChars )
    {
        std::string::size_type pos = str.find_last_not_of( trimChars );
        str.erase( pos + 1 );    
    }
    
    
    static void trimLeft( std::string &str,
                         const std::string& trimChars )
    {
        std::string::size_type pos = str.find_first_not_of( trimChars );
        str.erase( 0, pos );
    }
    
    
    static std::string trim( std::string str)
    {
        std::string trimChars = " \f\n\r\t\v";
        trimRight( str, trimChars );
        trimLeft( str, trimChars );
        return str;
    }
    
    
    // Removes \n from the end of line
    static void _FIXLINE(char * s) {
        int len = (int)strlen(s)-1; 	  
        if(s[len] == '\n') s[len] = 0;
    }
    
    /**
     * Returns a key-value map of a configuration file key-values.
     * If file is not found, fails with an assertion.
     * @param filename filename of the configuration file
     * @param secondary_filename secondary filename if the first version is not found.
     */
    static std::map<std::string, std::string> loadconfig(std::string filename, std::string secondary_filename) {
        FILE * f = fopen(filename.c_str(), "r");
        if (f == NULL) {
            f = fopen(secondary_filename.c_str(), "r");
            if (f == NULL) {
                std::cout << "ERROR: Could not read configuration file: " << filename << std::endl;
                std::cout << "Please define environment variable GRAPHCHI_ROOT or run the program from that directory." << std::endl;
            }
            assert(f != NULL);
        }
        
        char s[4096];
        std::map<std::string, std::string> conf;
        
        // I like C parsing more than C++, that is why this is such a mess
        while(fgets(s, 4096, f) != NULL) {
            _FIXLINE(s);
            if (s[0] == '#') continue; // Comment
            if (s[0] == '%') continue; // Comment
            
            char delims[] = "=";
            char * t;
            t = strtok(s, delims);
            const char * ckey = t;
            t = strtok(NULL, delims);
            const char * cval = t;
            
            if (ckey != NULL && cval != NULL) {
                std::string key = trim(std::string(ckey));
                std::string val = trim(std::string(cval));
                conf[key] = val;
            }
        }
        
        
        return conf;
    }
    
};


#endif


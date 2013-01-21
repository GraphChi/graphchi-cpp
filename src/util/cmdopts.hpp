
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
 * Command line options.
 */

#ifndef GRAPHCHI_CMDOPTS_DEF
#define GRAPHCHI_CMDOPTS_DEF


#include <string>
#include <iostream>
#include <stdint.h>

#include "api/chifilenames.hpp"
#include "util/configfile.hpp"

namespace graphchi { 
    
    /** GNU COMPILER HACK TO PREVENT IT FOR COMPILING METHODS WHICH ARE NOT USED IN
        THE PARTICULAR APP BEING BUILT */
#ifdef __GNUC__
#define VARIABLE_IS_NOT_USED __attribute__ ((unused))
#else
#define VARIABLE_IS_NOT_USED
#endif
    
    static bool _cmd_configured = false;
    
    static int _argc;
    static char **_argv;
    static std::map<std::string, std::string> conf;
    
    
    static void VARIABLE_IS_NOT_USED set_conf(std::string key, std::string value) {
        conf[key] = value;
    }
    
    // Config file
    static std::string VARIABLE_IS_NOT_USED get_config_option_string(const char *option_name) {
        if (conf.find(option_name) != conf.end()) {
            return conf[option_name];
        } else {
            std::cout << "ERROR: could not find option " << option_name << " from config.";
            assert(false);
        }
    }
    
    static  std::string VARIABLE_IS_NOT_USED get_config_option_string(const char *option_name,
                                                 std::string default_value) {
        if (conf.find(option_name) != conf.end()) {
            return conf[option_name];
        } else {
            return default_value;
        }
        
    }
    static int VARIABLE_IS_NOT_USED get_config_option_int(const char *option_name, int default_value) {
        if (conf.find(option_name) != conf.end()) {
            return atoi(conf[option_name].c_str());
        } else {
            return default_value;
        }
    }
    
    static int VARIABLE_IS_NOT_USED get_config_option_int(const char *option_name) {
        if (conf.find(option_name) != conf.end()) {
            return atoi(conf[option_name].c_str());
        } else {
            std::cout << "ERROR: could not find option " << option_name << " from config.";
            assert(false);
        }
    }
    
    static uint64_t VARIABLE_IS_NOT_USED get_config_option_long(const char *option_name, uint64_t default_value) {
        if (conf.find(option_name) != conf.end()) {
            return atol(conf[option_name].c_str());
        } else {
            return default_value;
        }
    }
    static double VARIABLE_IS_NOT_USED get_config_option_double(const char *option_name, double default_value) {
        if (conf.find(option_name) != conf.end()) {
            return atof(conf[option_name].c_str());
        } else {
            return default_value;
        }
    }
    
    static void set_argc(int argc, const char ** argv);
    static void set_argc(int argc, const char ** argv) {
        _argc = argc;
        _argv = (char**)argv;
        _cmd_configured = true;
        conf = loadconfig(filename_config_local(), filename_config());
        
        /* Load --key=value type arguments into the conf map */
        std::string prefix = "--";
        for (int i = 1; i < argc; i++) {
            std::string arg = std::string(_argv[i]);
            
            if (arg.substr(0, prefix.size()) == prefix) {
                arg = arg.substr(prefix.size());
                size_t a = arg.find_first_of("=", 0);
                if (a != arg.npos) {
                    std::string key = arg.substr(0, a);
                    std::string val = arg.substr(a + 1);
                    
                    std::cout << "[" << key << "]" << " => " << "[" << val << "]" << std::endl;
                    conf[key] = val;
                }
            }
        }

    }
    
    static void graphchi_init(int argc, const char ** argv);
    static void graphchi_init(int argc, const char ** argv) {
        set_argc(argc, argv);
            }
    
    static void check_cmd_init() {
        if (!_cmd_configured) {
            std::cout << "ERROR: command line options not initialized." << std::endl;
            std::cout << "       You need to call set_argc() in the beginning of the program." << std::endl;
        }
    }
    
    

    
    static std::string VARIABLE_IS_NOT_USED get_option_string(const char *option_name,
                                         std::string default_value)
    {
        check_cmd_init();
        int i;
        
        for (i = _argc - 2; i >= 0; i -= 1)
            if (strcmp(_argv[i], option_name) == 0)
                return std::string(_argv[i + 1]);
        return get_config_option_string(option_name, default_value);
    }
    
    static std::string VARIABLE_IS_NOT_USED get_option_string(const char *option_name)
    {
        int i;
        check_cmd_init();
        
        for (i = _argc - 2; i >= 0; i -= 1)
            if (strcmp(_argv[i], option_name) == 0)
                return std::string(_argv[i + 1]);
        return get_config_option_string(option_name);
    }
    
    static std::string VARIABLE_IS_NOT_USED get_option_string_interactive(const char *option_name, std::string options)
    {
        int i;
        check_cmd_init();
        
        for (i = _argc - 2; i >= 0; i -= 1)
            if (strcmp(_argv[i], option_name) == 0)
                return std::string(_argv[i + 1]);
        if (conf.find(option_name) != conf.end()) {
            return conf[option_name];
        } 

        std::cout << "Please enter value for command-line argument [" << std::string(option_name) << "]"<< std::endl;
        std::cout << "  (Options are: " << options << ")" << std::endl;
        
        std::string val;
        std::cin >> val;
        
        return val;
    }
    
    
    
    
    
    static int VARIABLE_IS_NOT_USED get_option_int(const char *option_name, int default_value)
    {
        int i;
        check_cmd_init();
        
        for (i = _argc - 2; i >= 0; i -= 1)
            if (strcmp(_argv[i], option_name) == 0)
                return atoi(_argv[i + 1]);
        
        return get_config_option_int(option_name, default_value);
    }
    
    static int VARIABLE_IS_NOT_USED get_option_int(const char *option_name)
    {
        int i;
        check_cmd_init();
        
        for (i = _argc - 2; i >= 0; i -= 1)
            if (strcmp(_argv[i], option_name) == 0)
                return atoi(_argv[i + 1]);
        
        return get_config_option_int(option_name);

    }

    
    
    static uint64_t VARIABLE_IS_NOT_USED get_option_long(const char *option_name, uint64_t default_value)
    {
        int i;
        check_cmd_init();
        
        for (i = _argc - 2; i >= 0; i -= 1)
            if (strcmp(_argv[i], option_name) == 0)
                return atol(_argv[i + 1]);
        return get_config_option_long(option_name, default_value);
    }
    
    static float VARIABLE_IS_NOT_USED get_option_float(const char *option_name, float default_value)
    {
        int i;
        check_cmd_init();
        
        for (i = _argc - 2; i >= 0; i -= 1)
            if (strcmp(_argv[i], option_name) == 0)
                return (float)atof(_argv[i + 1]);
        return (float) get_config_option_double(option_name, default_value);
    }
    
} // End namespace


#endif




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
 * Includes for the GraphChi - GraphLab v2.1 Gather-Apply-Scatter API.
 */

#ifndef DEF_GRAPHLAB_GAS_API_V2_1
#define DEF_GRAPHLAB_GAS_API_V2_1


#include "api/graphlab2_1_GAS_api/assertions.hpp"
#include "api/graphlab2_1_GAS_api/graphchi_graphlabv2_1.hpp"


#include <boost/foreach.hpp>
#include <stdint.h>

// if GNUC is available, this checks if the file which included
// macros_def.hpp is the same file which included macros_undef.hpp
#ifdef __GNUC__
#define GRAPHLAB_MACROS_INC_LEVEL __INCLUDE_LEVEL__
#endif


// prevent this file from being included before other graphlab headers
#ifdef GRAPHLAB_MACROS
#error "Repeated include of <macros_def.hpp>. This probably means that macros_def.hpp was not the last include, or some header file failed to include <macros_undef.hpp>"
#endif

#define GRAPHLAB_MACROS

/** A macro to disallow the copy constructor and operator= functions
 This should be used in the private: declarations for a class */
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
TypeName(const TypeName&);               \
void operator=(const TypeName&);



// Shortcut macro definitions
//! see http://www.boost.org/doc/html/foreach.html
#define foreach BOOST_FOREACH

#define rev_foreach BOOST_REVERSE_FOREACH


#include "api/graphlab2_1_GAS_api/random.hpp"

#endif




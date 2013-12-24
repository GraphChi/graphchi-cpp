#**
# * @file
# * @author  Danny Bickson
# * @version 1.0
# *
# * @section LICENSE
# *
# * Copyright [2012] [Carngie Mellon University]
# * 
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# * 
# * http://www.apache.org/licenses/LICENSE-2.0
# * 
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

#  Makefile for compiling graphchi collaborative filtering library
#  Written by Danny Bickson
#  Thanks to Yucheng Low for fixing the Makefile
INCFLAGS = -I/usr/local/include/ -I../../src/ -I.

# NOTE: Uncomment the flag GRAPHCHI_USE_GSL if you want to compile pmf
CPPFLAGS = -O3  $(INCFLAGS) -DEIGEN_NDEBUG -fopenmp -Wno-strict-aliasing  -Wall
# NOTE: uncomment the flag -lgsl if you want to compile pmf
LINKFLAGS = -lz 
# Note : on Ubuntu on some compilers -lz is not detected properly so it is 
# deliberatively set to be the last flag.
CPP = g++
CXX = g++
headers=$(wildcard *.h**)
all: $(patsubst %.cpp, %, $(wildcard *.cpp))

%: %.cpp $(headers)
	$(CPP) $(CPPFLAGS) $<  -o $@ $(LINKFLAGS)

clean:
	rm -f $(patsubst %.cpp, %, $(wildcard *.cpp))

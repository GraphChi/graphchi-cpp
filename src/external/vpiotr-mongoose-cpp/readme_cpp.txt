C++ wrapper for Mongoose http server.

This code repository contains C++ wrapper for mongoose http server library.

C++ wrapper author:
  Piotr Likus

wrapper project home:
  http://code.google.com/r/vpiotr-mongoose-cpp/
    
mongoose project home:
  http://code.google.com/p/mongoose/


Files and directories:
------------------------
\bin         - test executable for wrapper (mongotest.exe)
\build       - project files for library & test program
\lib         - output directory for library (mongoose)
\test        - source code for wrapper test server (mongotest.cpp)

Changes:
------------------------
mongoose.h, mongoose.c - contains extra function mg_read_user_data

New code:
------------------------
mongcpp.h, mongcpp.cpp - C++ wrapper for mongoose web server library

Example for C++ web server:
---------------------------
\test\mongotest.cpp

To run it on Windows:
- compile library project "mongoose" - in \build directory
- compile program project "mongotest" - in \build directory
- copy "html" directory from \examples to \bin
- execute "\bin\mongoose.exe" program
- open in your browser address "http://127.0.0.1:8080/"

Tested compilers:
-----------------
- VS2010 Express SP1
- Code::Blocks 10.05 + MinGW + gcc 4.5

Final notes
---------------------------
Sources of mongoose project updated basing on code
repository on 2010/12/15:

http://code.google.com/p/mongoose/


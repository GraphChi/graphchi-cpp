/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2010 Piotr Likus
// Name:        mongotest.cpp
// Project:     mongoose
// Purpose:     Test program (main) for C++ wrapper.
// Author:      Piotr Likus
// Modified by:
// Created:     16/12/2010
// Licence:     MIT
/////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <time.h>
#include <sstream>

#include "mongcpp.h"


using namespace mongoose;
using namespace std;

template <class T>
inline std::string toString (const T& t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}

string ipToString(long ip)
{
    string res;
    long workIp, a, b, c, d;
    workIp = ip;
    d = workIp % 0x100;
    workIp = workIp >> 8;
    c = workIp % 0x100;
    workIp = workIp >> 8;
    b = workIp % 0x100;
    workIp = workIp >> 8;
    a = workIp;
    res = toString(a)+"."+toString(b)+"."+toString(c)+"."+toString(d);
    return res;
}

class TestMongoServer: public MongooseServer {
public:
    TestMongoServer(): MongooseServer() {}
    virtual ~TestMongoServer() {}
protected:
    virtual bool handleEvent(ServerHandlingEvent eventCode, MongooseConnection &connection, const MongooseRequest &request, MongooseResponse &response) {
        bool res = false;

        if (eventCode == MG_NEW_REQUEST) {
            if (request.getUri() == string("/info")) {
                handleInfo(request, response);
                res = true;
            }
        }

        return res;
    }

    void handleInfo(const MongooseRequest &request, MongooseResponse &response) {
        response.setStatus(200);
        response.setConnectionAlive(false);
        response.setCacheDisabled();
        response.setContentType("text/html");
        response.addContent(generateInfoContent(request));
        response.write();
    }

    const string generateInfoContent(const MongooseRequest &request) {
        string result;
        result = "<h1>Sample Info Page</h1>";
        result += "<br />Request URI: " + request.getUri();
        result += "<br />Your IP: " + ipToString(request.getRemoteIp());

	time_t tim;
	time(&tim);

        result += "<br />Current date & time: " + toString(ctime(&tim));
        result += "<br /><br /><a href=\"/\">Index page</a>";

        return result;
    }
};

int main()
{
    TestMongoServer server;

    server.setOption("document_root", "html");
    server.setOption("listening_ports", "8080");
    server.setOption("num_threads", "5");

    server.start();

    cout << "Test server started, press enter to quit..." << endl;
    cin.ignore();

    server.stop();
    cout << "Test server stopped" << endl;
}

/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2010 Piotr Likus
// Name:        mongcpp.h
// Project:     mongoose
// Purpose:     C++ wrapper for mongoose.
// Author:      Piotr Likus
// Modified by:
// Created:     15/12/2010
// Licence:     MIT
/////////////////////////////////////////////////////////////////////////////

#ifndef _MONGCPP_H__
#define _MONGCPP_H__

// ----------------------------------------------------------------------------
// Description
// ----------------------------------------------------------------------------
/// \file mongcpp.h
///
/// Mongoose wrapper for C++

// ----------------------------------------------------------------------------
// Headers
// ----------------------------------------------------------------------------
#include <cstddef>
#include <cstdlib>

#include "mongoose.h"

#include <string>
#include <map>
#include <vector>
#include <memory>

namespace mongoose
{

// ----------------------------------------------------------------------------
// Simple type definitions
// ----------------------------------------------------------------------------
enum MongooseRequestMethodCode {
    rmcUndef,
    rmcGet,
    rmcPost,
    rmcHead,
    rmcPut,
    rmcDelete,
    rmcTrace,
    rmcOptions
};

typedef std::map<std::string, std::string> ServerOptionSet;
typedef std::vector<std::string> ServerOptionList;
typedef std::vector<const char *> ServerOptionStorage;
typedef std::map<std::string, std::string> RequestValueSet;
typedef std::map<std::string, int> ResponseValueIndex;
typedef std::vector< std::pair<std::string, std::string> > ResponseValueList;
typedef enum mg_event ServerHandlingEvent;
typedef std::map<std::string, MongooseRequestMethodCode> MethodMap;
typedef std::auto_ptr<MethodMap> MethodMapGuard;

// ----------------------------------------------------------------------------
// Forward class definitions
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Class definitions
// ----------------------------------------------------------------------------

class MongooseConnection {
public:
    MongooseConnection(struct mg_connection *conn);
    virtual ~MongooseConnection();
    int write(const void *buf, size_t len);
    int write(const std::string &text);
    int read(void *buf, size_t len);
    void sendAuthorizationRequest(const std::string &nonce = "");
    bool getHeader(const std::string &name, std::string &output) const;
    bool getCookie(const std::string &name, std::string &output) const;
protected:
    struct mg_connection *getInfo();
protected:
    struct mg_connection *m_conn;
};

class MongooseRequest {
public:
    MongooseRequest(struct mg_connection *conn, mg_request_info* info);
    virtual ~MongooseRequest();
    const std::string getRequestMethod() const;
    MongooseRequestMethodCode getRequestMethodCode() const;
    const std::string getUri() const;
    const std::string getHttpVersion() const;
    const std::string getQueryString() const;
    const std::string readQueryString() const;
    const std::string getRemoteUser() const;
    const std::string getLogMessage() const;
    long getRemoteIp() const;
    int getRemotePort() const;
    int getStatusCode() const;
    bool isSsl() const;
    bool getVar(const std::string &name, std::string &output) const;
    static MongooseRequestMethodCode methodTextToCode(const std::string &text);
protected:
    mg_request_info* getInfo() const;
protected:
    mg_request_info* m_info;
    struct mg_connection *m_conn;
};

class MongooseResponse {
public:
    MongooseResponse(struct mg_connection *conn);
    virtual ~MongooseResponse();
    virtual void write();
    void setStatus(int code, const std::string &statusDesc = "", const std::string &httpVer = "");
    void setSetCookie(const std::string &name, const std::string &value);
    void setLocation(const std::string &value = "");
    void setContentType(const std::string &value = "");
    bool getHeaderValue(const std::string &name, std::string &output);
    void addHeaderValue(const std::string &name, const std::string &value);
    void setHeaderValue(const std::string &name, const std::string &value);
    void setConnectionAlive(bool keepAlive = false);
    void setCacheDisabled();
    void addHeader();
    void addContent(const std::string &text, bool addLen = true);
    void addText(const std::string &text);
    void addTextLine(const std::string &text);
    static const char *getHttpStatusDesc(int statusCode);
protected:
    ResponseValueList *prepareHeaderValues();
    void addHeaderValueToText(const std::string &name, const std::string &value);
protected:
    struct mg_connection *m_conn;
    std::string m_text;
    std::auto_ptr<ResponseValueList> m_headerValues;
    std::auto_ptr<ResponseValueIndex> m_headerValuesIndex;
    std::auto_ptr<std::string> m_statusText;
};

class MongooseServer {
public:
    // construct
    MongooseServer();
    virtual ~MongooseServer();
    // attributes
    void setOptions(const ServerOptionSet &options);
    void setOption(const std::string &name, const std::string &value);
    void getOptions(ServerOptionSet &options) const;
    bool getOptionValue(const std::string &name, std::string &value) const;
    void getOptionValue(const std::string &name, std::string &value, const std::string &defValue) const;
    static void getValidOptions(ServerOptionList &output);
    static std::string getVersion();
    static void calcMD5(const std::string &text, std::string &output);
    static MongooseRequestMethodCode methodTextToCode(const std::string &text);
    // run
    virtual void init();
    void start();
    void stop();
    bool isRunning();
    virtual void *handleEvent(ServerHandlingEvent eventCode,
                              struct mg_connection *conn,
                              const struct mg_request_info *request_info);
protected:
    virtual bool handleEvent(ServerHandlingEvent eventCode, MongooseConnection &connection, const MongooseRequest &request, MongooseResponse &response);
    virtual MongooseConnection *newConnection(struct mg_connection *conn);
    virtual MongooseRequest *newRequest(struct mg_connection *conn, const struct mg_request_info *request);
    virtual MongooseResponse *newResponse(struct mg_connection *conn);
    void checkStopped();
    const char **prepareOptions();
    void unprepareOptions();
    void checkMethodMap();
protected:
    ServerOptionSet m_options;
    ServerOptionStorage m_optionStorage;
    bool m_statusRunning;
    bool m_prepared;
    struct mg_context *m_ctx;
    static MethodMapGuard m_methodMap;
};

}

#endif // _MONGCPP_H__
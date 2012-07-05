/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2010 Piotr Likus
// Name:        mongcpp.cpp
// Project:     mongoose
// Purpose:     C++ wrapper for mongoose.
// Author:      Piotr Likus
// Modified by:
// Created:     15/12/2010
// Licence:     MIT
/////////////////////////////////////////////////////////////////////////////

#include "mongcpp.h"

#include <sstream>
#include <cstring>

using namespace mongoose;

MethodMapGuard MongooseServer::m_methodMap;

template <class T>
inline std::string toString (const T& t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}

inline unsigned int stringToUIntDef(const std::string &str, unsigned int defVal)
{
    using namespace std;
    unsigned int res;
    istringstream cStream(str);
    if (!(cStream >> res)) {
        res = defVal;
    }
    return res;
}

MongooseConnection::MongooseConnection(struct mg_connection *conn): m_conn(conn)
{
}

MongooseConnection::~MongooseConnection()
{
}

int MongooseConnection::write(const void *buf, size_t len)
{
  return mg_write(m_conn, buf, len);
}

int MongooseConnection::write(const std::string &text)
{
    return write(text.c_str(), text.length());
}

int MongooseConnection::read(void *buf, size_t len)
{
  return mg_read(m_conn, buf, len);
}

void MongooseConnection::sendAuthorizationRequest(const std::string &nonce)
{
    if (nonce.empty())
      mg_send_authorization_request(m_conn, NULL);
    else
      mg_send_authorization_request(m_conn, nonce.c_str());
}

bool MongooseConnection::getHeader(const std::string &name, std::string &output) const
{
    const char *value = mg_get_header(m_conn, name.c_str());
    if (value != NULL)
    {
        output = std::string(value);
        return true;
    } else {
        output = "";
        return false;
    }
}

bool MongooseConnection::getCookie(const std::string &name, std::string &output) const
{
    const int BUF_LEN = 4096;
    char buffer[BUF_LEN];

    int readCnt = mg_get_cookie(m_conn, name.c_str(), buffer, BUF_LEN - 1);

    if (readCnt >= 0) {
        buffer[BUF_LEN - 1] = '\0';
        output = buffer;
        return true;
    } else {
        output = "";
        return false;
    }
}

struct mg_connection *MongooseConnection::getInfo()
{
    return m_conn;
}

//-----------------------------------------------------------------------
MongooseRequest::MongooseRequest(struct mg_connection *conn, mg_request_info* info):
  m_conn(conn), m_info(info)
{
}

MongooseRequest::~MongooseRequest()
{
}

const std::string MongooseRequest::getRequestMethod() const
{
    return std::string(m_info->request_method);
}

MongooseRequestMethodCode MongooseRequest::getRequestMethodCode() const
{
    //return methodTextToCode(getRequestMethod());
    return MongooseServer::methodTextToCode(getRequestMethod());
}

const std::string MongooseRequest::getUri() const
{
    return std::string(m_info->uri);
}

const std::string MongooseRequest::getHttpVersion() const
{
    return std::string(m_info->http_version);
}

/// use only for GET
const std::string MongooseRequest::getQueryString() const
{
    if (m_info->query_string != NULL)
      return std::string(m_info->query_string);
    else
      return "";
}

/// use for POST, PUT
const std::string MongooseRequest::readQueryString() const
{
  const char *cl = mg_get_header(m_conn, "Content-Length");
  size_t buf_len;
  if (cl != NULL)
    buf_len = stringToUIntDef(std::string(cl), 0);
  else
    buf_len = 0;

  std::string res;

  if (buf_len > 0)
  {
    char *buf = new char[buf_len+1];
    //TODO: verify if we need exception handling here

    /* Read in two pieces, to test continuation */
    if (buf_len > 2) {
      mg_read(m_conn, buf, 2);
      mg_read(m_conn, buf + 2, buf_len - 2);
    } else {
      mg_read(m_conn, buf, buf_len);
    }

    buf[buf_len] = '\0';
    res = std::string(buf);
    delete[] buf;
  }

  return res;
}

const std::string MongooseRequest::getRemoteUser() const
{
    return std::string(m_info->remote_user);
}

const std::string MongooseRequest::getLogMessage() const
{
    return std::string(m_info->log_message);
}

long MongooseRequest::getRemoteIp() const
{
    return m_info->remote_ip;
}

int MongooseRequest::getRemotePort() const
{
    return m_info->remote_port;
}

int MongooseRequest::getStatusCode() const
{
    return m_info->status_code;
}

bool MongooseRequest::isSsl() const
{
    return (m_info->is_ssl > 0);
}

mg_request_info* MongooseRequest::getInfo() const
{
    return m_info;
}

bool MongooseRequest::getVar(const std::string &name, std::string &output) const
{
    const int MAX_VAR_LEN = 4096;
    char buffer[MAX_VAR_LEN];
    const char *qs = m_info->query_string;
    int readCnt = mg_get_var(qs, strlen(qs == NULL ? "" : qs), name.c_str(), buffer, MAX_VAR_LEN - 1);
    if (readCnt >= 0) {
      buffer[MAX_VAR_LEN - 1] = '\0';
      output = buffer;
      return true;
    } else {
      output.clear();
      return false;
    }
}

//-----------------------------------------------------------------------
MongooseResponse::MongooseResponse(struct mg_connection *conn): m_conn(conn)
{
}

MongooseResponse::~MongooseResponse()
{
}

void MongooseResponse::write()
{
   mg_write(m_conn, m_text.c_str(), m_text.length());
   m_text.clear();
}

const char *MongooseResponse::getHttpStatusDesc(int statusCode)
{
      const char *res;

      switch(statusCode) {
      case 100: res = "Continue";
          break;
      case 101: res = "Switching Protocols";
          break;
      //--------------------------------------
      case 200: res = "OK";
          break;
      case 201: res = "Created";
          break;
      case 202: res = "Accepted";
          break;
      case 203: res = "Non-Authoritative Information";
          break;
      case 204: res = "No Content";
          break;
      case 205: res = "Reset Content";
          break;
      case 206: res = "Partial Content";
          break;
      //--------------------------------------
      case 300: res = "Multiple Choices";
          break;
      case 301: res = "Moved Permanently";
          break;
      case 302: res = "Found";
          break;
      case 303: res = "See Other";
          break;
      case 304: res = "Not Modified";
          break;
      case 305: res = "Use Proxy";
          break;
      case 306: res = "Switch Proxy";
          break;
      case 307: res = "Temporary Redirect";
          break;
      //--------------------------------------
      case 400: res = "Bad Request";
          break;
      case 401: res = "Unauthorized";
          break;
      case 402: res = "Payment Required";
          break;
      case 403: res = "Forbidden";
          break;
      case 404: res = "Not Found";
          break;
      case 405: res = "Method Not Allowed";
          break;
      case 406: res = "Not Acceptable";
          break;
      case 407: res = "Proxy Authentication Required";
          break;
      case 408: res = "Request Timeout";
          break;
      case 409: res = "Conflict";
          break;
      case 410: res = "Gone";
          break;
      case 411: res = "Length Required";
          break;
      case 412: res = "Precondition Failed";
          break;
      case 413: res = "Request Entity Too Large";
          break;
      case 414: res = "Request-URI Too Long";
          break;
      case 415: res = "Unsupported Media Type";
          break;
      case 416: res = "Requested Range Not Satisfiable";
          break;
      case 417: res = "Expectation Failed";
          break;
      //--------------------------------------
      case 500: res = "Internal Server Error";
          break;
      case 501: res = "Not Implemented";
          break;
      case 502: res = "Bad Gateway";
          break;
      case 503: res = "Service Unavailable";
          break;
      case 504: res = "Gateway Timeout";
          break;
      case 505: res = "HTTP Version Not Supported";
          break;
      default:
          res = "";
      } // switch

      return res;
}

void MongooseResponse::setStatus(int code, const std::string &statusDesc, const std::string &httpVer)
{
    const char *realStatusDesc = NULL;
    const char *realHttpVer = NULL;

    if (!httpVer.empty()) {
        realHttpVer = httpVer.c_str();
    }
    else {
        realHttpVer = "HTTP/1.1";
    }

    if (!statusDesc.empty()) {
        realStatusDesc = statusDesc.c_str();
    } else {
        realStatusDesc = MongooseResponse::getHttpStatusDesc(code);
    } // if / else

    std::string output = std::string(realHttpVer) + " " + toString(code) + " " + std::string(realStatusDesc);
    m_statusText.reset(new std::string(output));
}

void MongooseResponse::setSetCookie(const std::string &name, const std::string &value)
{
    setHeaderValue("Set-Cookie", name+"="+value);
}

void MongooseResponse::setLocation(const std::string &value)
{
    setHeaderValue("Location", value);
}

void MongooseResponse::setContentType(const std::string &value)
{
    if (value.empty())
      setHeaderValue("Content-type", "text/html");
    else
      setHeaderValue("Content-type", value);
}

bool MongooseResponse::getHeaderValue(const std::string &name, std::string &output)
{
    prepareHeaderValues();

    ResponseValueIndex::const_iterator cit = m_headerValuesIndex->find(name);
    bool res = false;

    if (cit != m_headerValuesIndex->end()) {
        int idx = cit->second;
        output = (*m_headerValues)[idx].second;
        res = true;
    } else {
        output.clear();
    }

    return res;
}

// note: this method allows duplicates (like "Set-Cookie")
void MongooseResponse::addHeaderValue(const std::string &name, const std::string &value)
{
    prepareHeaderValues();

    int idx = m_headerValues->size();

    ResponseValueIndex::iterator cit = m_headerValuesIndex->find(name);
    if (cit != m_headerValuesIndex->end()) {
        cit->second = idx;
    } else {
        m_headerValuesIndex->insert(std::make_pair<std::string, int>(name, idx));
    }

    m_headerValues->push_back(std::make_pair<std::string, std::string>(name, value));
}

void MongooseResponse::setHeaderValue(const std::string &name, const std::string &value)
{
    prepareHeaderValues();
    ResponseValueIndex::const_iterator cit = m_headerValuesIndex->find(name);
    if (cit != m_headerValuesIndex->end()) {
        int idx = cit->second;
        (*m_headerValues)[idx].second = value;
    } else {
        int idx = m_headerValues->size();
        m_headerValuesIndex->insert(std::make_pair<std::string, int>(name, idx));
        m_headerValues->push_back(std::make_pair<std::string, std::string>(name, value));
    }
}

void MongooseResponse::addContent(const std::string &text, bool addLen)
{
    if (addLen) {
        setHeaderValue("Content-length", toString(text.length()));
        addHeader();
        addTextLine("");
        addText(text);
    } else {
        addHeader();
        addText(text);
    }
}

void MongooseResponse::setConnectionAlive(bool keepAlive)
{
    if (keepAlive)
      setHeaderValue("Connection", "Keep-Alive");
    else
      setHeaderValue("Connection", "close");
}

void MongooseResponse::setCacheDisabled()
{
    setHeaderValue("Cache-Control", "no-cache, must-revalidate");
    setHeaderValue("Expires", "Sat, 26 Jul 1997 05:00:00 GMT"); // Date in the past
}

void MongooseResponse::addHeader()
{
    if (m_statusText.get() != NULL) {
        addTextLine(*m_statusText);
        m_statusText.reset();
    }

    if (m_headerValues.get() != NULL) {
        for(ResponseValueList::const_iterator cit = m_headerValues->begin(), epos = m_headerValues->end(); cit != epos; ++cit)
        {
            addHeaderValueToText(cit->first, cit->second);
        }
    }

    m_headerValues.reset();
    m_headerValuesIndex.reset();
}

void MongooseResponse::addHeaderValueToText(const std::string &name, const std::string &value)
{
    addTextLine(name + std::string(": ") + value);
}

void MongooseResponse::addText(const std::string &text)
{
    m_text += text;
}

void MongooseResponse::addTextLine(const std::string &text)
{
    addText(text+std::string("\r\n"));
}

ResponseValueList *MongooseResponse::prepareHeaderValues()
{
    if (m_headerValues.get() == NULL) {
        m_headerValues.reset(new ResponseValueList);
        m_headerValuesIndex.reset(new ResponseValueIndex);
    }

    return m_headerValues.get();
}

//-----------------------------------------------------------------------
static void *LocalMongooseEventHandler(enum mg_event eventCode,
                           struct mg_connection *conn,
                           const struct mg_request_info *request_info)
{
  MongooseServer* server = (MongooseServer* )mg_read_user_data(conn);
  return server->handleEvent(eventCode, conn, request_info);
}

// construct
MongooseServer::MongooseServer(): m_statusRunning(false), m_prepared(false), m_ctx(NULL)
{
}

MongooseServer::~MongooseServer()
{
    checkStopped();
}

// attributes
void MongooseServer::setOptions(const ServerOptionSet &options)
{
    m_options = options;
}

void MongooseServer::setOption(const std::string &name, const std::string &value)
{
    m_options[name] = value;
}

void MongooseServer::getOptions(ServerOptionSet &options) const
{
    options = m_options;
}

bool MongooseServer::getOptionValue(const std::string &name, std::string &value) const
{
    ServerOptionSet::const_iterator citer = m_options.find(name);
    if (citer != m_options.end()) {
        value = citer->second;
        return true;
    } else {
        value = "";
        return false;
    }
}

void MongooseServer::getOptionValue(const std::string &name, std::string &value, const std::string &defValue) const
{
    bool found = getOptionValue(name, value);
    if (!found) {
        value = defValue;
    }
}

void MongooseServer::getValidOptions(ServerOptionList &output)
{
   const char **names;
   const char SEP_CHAR = ';';

   names = mg_get_valid_option_names();

   for (int i = 0; names[i] != NULL; i += 3) {
       output.push_back(
          std::string(names[i]) + SEP_CHAR +
          std::string(names[i] + 1) + SEP_CHAR +
          std::string(names[i + 2] == NULL ? "" : names[i + 2])
       );
   }
}

std::string MongooseServer::getVersion()
{
    return std::string(mg_version());
}

void MongooseServer::calcMD5(const std::string &text, std::string &output)
{
    char buf[33];
    mg_md5(buf, text.c_str());
    output = buf;
}

// run
void MongooseServer::init()
{
    checkMethodMap();
}

void MongooseServer::start()
{
    if (!m_prepared) {
        init();
        m_prepared = true;
    }

    checkStopped();

    m_ctx = mg_start(LocalMongooseEventHandler, (void *) this, prepareOptions());
    m_statusRunning = true;
}

void MongooseServer::stop()
{
    if (!isRunning())
        return;
    mg_stop(m_ctx);
    unprepareOptions();
    m_ctx = NULL;
    m_statusRunning = false;
}

bool MongooseServer::isRunning()
{
    return m_statusRunning;
}

// disable "unused args" warning
#pragma warning( disable : 4716 )
#pragma warning( disable : 4100 )
bool MongooseServer::handleEvent(ServerHandlingEvent eventCode, MongooseConnection &connection, const MongooseRequest &request, MongooseResponse &response)
{
    return NULL;
}

void *MongooseServer::handleEvent(ServerHandlingEvent eventCode,
                            struct mg_connection *conn,
                            const struct mg_request_info *request_info)
{
    void *processed = reinterpret_cast<void *> (const_cast<char *>("yes"));
    std::auto_ptr<MongooseConnection> connection(newConnection(conn));
    std::auto_ptr<MongooseRequest> request(newRequest(conn, request_info));
    std::auto_ptr<MongooseResponse> response(newResponse(conn));

    if (handleEvent(eventCode, *connection, *request, *response))
        return processed;
    else
        return NULL;
}

MongooseConnection *MongooseServer::newConnection(struct mg_connection *conn)
{
    return new MongooseConnection(conn);
}

MongooseRequest *MongooseServer::newRequest(struct mg_connection *conn, const struct mg_request_info *request)
{
    return new MongooseRequest(conn, const_cast<struct mg_request_info *>(request));
}

MongooseResponse *MongooseServer::newResponse(struct mg_connection *conn)
{
    return new MongooseResponse(conn);
}

void MongooseServer::checkStopped()
{
    if (m_statusRunning)
        stop();
}

const char **MongooseServer::prepareOptions()
{
    unprepareOptions();
    for(ServerOptionSet::const_iterator cit = m_options.begin(), epos = m_options.end(); cit != epos; ++cit)
    {
        m_optionStorage.push_back(cit->first.c_str());
        m_optionStorage.push_back(cit->second.c_str());
    }
    m_optionStorage.push_back(NULL);
    return &(*m_optionStorage.begin());
}

void MongooseServer::unprepareOptions()
{
    m_optionStorage.clear();
}

void MongooseServer::checkMethodMap()
{
    if (m_methodMap.get() == NULL)
    {
        m_methodMap.reset(new MethodMap);

        m_methodMap->insert(std::make_pair("GET", rmcGet));
        m_methodMap->insert(std::make_pair("POST", rmcPost));
        m_methodMap->insert(std::make_pair("HEAD", rmcHead));
        m_methodMap->insert(std::make_pair("PUT", rmcPut));
        m_methodMap->insert(std::make_pair("DELETE", rmcDelete));
        m_methodMap->insert(std::make_pair("TRACE", rmcTrace));
        m_methodMap->insert(std::make_pair("OPTIONS", rmcOptions));
    }
}

MongooseRequestMethodCode MongooseServer::methodTextToCode(const std::string &text)
{
    MethodMap::const_iterator cit = m_methodMap->find(text);
    if (cit == m_methodMap->end())
        return rmcUndef;
    else
        return cit->second;
}

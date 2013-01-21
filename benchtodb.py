#!/usr/bin/python

import os
import sys
#import io
import MySQLdb

def db_init():
    global _cursor
    import MySQLdb
    # TODO: move to a config file
    dbconn = MySQLdb.connect (host = "multi6.aladdin.cs.cmu.edu", port = 3306,
                                                            user = "graphchi",
                                                            passwd = "chihuahua9231",
                                                            db = "graphchi")

    _cursor = dbconn.cursor ()  # Underscore makes the _cursor non-public outside module    		
    dbconn.autocommit(1)

def cur():
	return _cursor
	
db_init()

trtable = {"execute_updates": "exec_updates", "execthreads": "nthreads", "memoryshard_create_edges": "load_memshard"}

def translate(k):
    if k in trtable:
        return trtable[k]
    else: return k
    
columns = ["file"
          , "determ"
          , "niothreads"
          , "scheduler"
          , "work"
          , "safeupdates"
          , "stripesize"
          ,"nthreads"
          ,"epochsize"
          ,"niters"
          ,"runtime"
          ,"cacheratio"
          ,"loadtime"
          ,"multiplex"
          ,"load_adj"
          ,"load_inv"
          ,"load_outv"
          ,"sort_edgereqs"
          ,"sort_edges"
          ,"edata_in_bytes"
          ,"edata_out_bytes"
          ,"adjbytes"
          ,"committime"
          ,"max_indegree"
          ,"max_outdegree"
          ,"app"
          ,"nshards"
          ,"shard_preada"
          ,"shard_create_edges"
          ,"commit"
          ,"commit_thr"
          ,"blockload"
          ,"updates"
          ,"stream_ahead"
          ,"load_memshard"
          ,"blocksize"
          ,"exec_updates"
          , "memshard_commit"
          , "membudget_mb"
          ,"nedges"
          ,"nvertices"
          , "subwindow"
          , "compression"
	  , "loadthreads"]
 
# This is terrible code!
def insert(cols, percs, vals):
  sql = "insert into runs (" + cols + ") values(" + percs + ")"
  print sql
  cur().execute(sql, vals)
 
(sysname, nodename, release, version, machine) = os.uname()


fname = sys.argv[-1]
print "loading", fname
cols = ""
percs = ""
vals=None
lines = open(fname,"r").readlines()
lines = [l[:-1] for l in lines] # strip \n
for l in lines:
  
  if l[0] == "[":
    if vals != None:
      insert(cols,percs,vals)
    cols = "time,host"
    percs = "now(),'%s'" %nodename
    vals = []

  tok = l[1:].split("=")  # Strip "." from beginning
  key = tok[0]
  key = translate(key.replace("-", "_"))
  print key, tok[0]
  value = tok[-1]
  

  if columns.count(key)==1 and cols.find(","+key)<0:
    cols += "," + key
    percs += ",%s"
    if key == "app":
      value = value.split("/")[-1]
    vals.append(value)
    
insert(cols,percs,vals)

cur().execute("select last_insert_id()")
lastid = cur().fetchone()[0]

os.system("mv %s %s.%d" %(fname, fname, lastid))

from optparse import OptionParser
import random
import heapq
from operator import itemgetter
from collections import defaultdict

class Split:

    def __init__(self):
        self.train = {}
        self.test = {}
        self.counts = {}

    def add(self,user,trustees):
        if len(trustees) >= opts.min_trustees:
            self.counts[user] = len(trustees)
            random.shuffle(trustees)
            self.train[user] = trustees[:opts.given]
            self.test[user] = trustees[opts.given:]

    def map_ids(self):
        utrans = IndexTranslator()
        ttrans = IndexTranslator()
        train_idx = defaultdict(list)
        for user,trustees in self.train.iteritems():
            uidx = utrans.idx(user)
            for t in trustees:
                train_idx[uidx].append(ttrans.idx(t))
        test_idx = defaultdict(list)
        for user,trustees in self.test.iteritems():
            uidx = utrans.idx(user,allow_update=False)
            assert(uidx is not None)  # shouldn't have any unique users
            for t in trustees:
                tidx = ttrans.idx(t,allow_update=False)
                if tidx is not None:
                    test_idx[uidx].append(tidx)
        self.train = train_idx
        self.test = test_idx

class IndexTranslator:

    def __init__(self):
        self.index = {}

    def idx(self,key,allow_update=True):
        if allow_update and key not in self.index:
            self.index[key] = len(self.index)+1
        return self.index.get(key,None)

class MMWriter:

    def __init__(self,filepath):
        self.filepath = filepath

    def write(self,mat):
        f = open(self.filepath,'w')
        self.write_header(f,mat)
        self.write_data(f,mat)

    def write_header(self,f,mat):
        tot = 0
        maxid = 0
        for user,trustees in mat.iteritems():
            tot += len(trustees)
            maxid = max(maxid,max(trustees))
        print >>f,'%%MatrixMarket matrix coordinate integer general'
        print >>f,'{0} {1} {2}'.format(max(mat.keys()),maxid,tot)

    def write_data(self,f,mat):
        for user,trustees in mat.iteritems():
            for t in trustees:
                print >>f,user,t,1

parser = OptionParser()
parser.add_option('-i','--infile',dest='infile',help='input dataset')
parser.add_option('-o','--outpath',dest='outpath',help='root path for output datasets [default=infile]')
parser.add_option('-m','--min_trustees',dest='min_trustees',type='int',help='omit users with fewer trustees')
parser.add_option('-g','--given',dest='given',type='int',help='retain this many trustees in training set')
parser.add_option('-d','--discard_top',dest='discard_top',type='int',default=3,help='discard this many overall top popular users [default=%default]')

(opts,args) = parser.parse_args()
if not opts.min_trustees or not opts.given or not opts.infile:
    parser.print_help()
    raise SystemExit

if not opts.outpath:
    opts.outpath = opts.infile

overall = defaultdict(list)
counts = defaultdict(int)
f = open(opts.infile)
for line in f:
    if not line.startswith('%'):
        break
for line in f:
    user,trustee,score = map(int,line.strip().split())
    if score > 0:
        counts[trustee] += 1
top = heapq.nlargest(opts.discard_top,counts.iteritems(),key=itemgetter(1))
for user,_ in top:
    counts[user] = 0  # so we don't include them

f = open(opts.infile)
for line in f:
    if not line.startswith('%'):
        break
for line in f:
    user,trustee,score = map(int,line.strip().split())
    if score > 0 and counts[trustee] >= opts.min_trustees:
        overall[user].append(trustee)

split = Split()
for user,trustees in overall.iteritems():
    split.add(user,trustees)
split.map_ids()

w = MMWriter(opts.outpath+'_train')
w.write(split.train)
w = MMWriter(opts.outpath+'_test')
w.write(split.test)

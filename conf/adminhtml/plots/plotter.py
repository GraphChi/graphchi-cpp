#!/usr/bin/python

import sys
import os
import matplotlib

import numpy
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter


def getArg(param, default=""):
    if (sys.argv.count(param) == 0): return default
    i = sys.argv.index(param)
    return sys.argv[i + 1]

lastsecs = int(getArg("lastsecs", 240))

fname = sys.argv[1]
try:
	tdata = numpy.loadtxt(fname, delimiter=" ")
except:
	exit(0)


if len(tdata.shape) < 2 or tdata.shape[0] < 2 or tdata.shape[1] < 2:
    print "Too small data - do not try to plot yet."
    exit(0)

times = tdata[:, 0]
values = tdata[:, 1]

lastt = max(times)


#majorFormatter = FormatStrFormatter('%.2f')

fig = plt.figure(figsize=(3.5, 2.0))
plt.plot(times[times > lastt - lastsecs], values[times > lastt - lastsecs])
plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = 7, prune = 'lower') )
plt.xlim([max(0, lastt - lastsecs), lastt])
#plt.ylim([lastt - lastsecs, lastt])

plt.gca().yaxis.set_major_locator( MaxNLocator(nbins = 7, prune = 'lower') )

#plt.gca().yaxis.set_major_formatter(majorFormatter)
plt.savefig(fname.replace(".dat", ".png"), format="png", bbox_inches='tight')



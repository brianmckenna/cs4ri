import datetime
import numpy as np
import mydict

#gfs = np.load('gfs.npz')
#for g in gfs:
#    print g
#    #print gfs[g]
#    #print type(gfs[g])
#
#d = gfs['arr_0']
##for k in d:
##    print k
#
keys = [
    'k245',
    'k246',
    'k247u',
    'k247v',
    'k485',
    'k486',
    'k487u',
    'k487v',
    'k725',
    'k726',
    'k727u',
    'k727v',
]


for k in mydict.d:
    print len(mydict.d[k].keys())
    print len(mydict.d[k].values())

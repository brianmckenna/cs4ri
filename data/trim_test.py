import datetime
import numpy as np
import time

def _interpolate(i, e, dt, d):

    # seconds for interpolation
    _dt = np.array([time.mktime(t.timetuple()) for t in dt])

    # fix -9999.9 in original data
    _i = np.where(d == -9999.9)
    _e = np.where(d != -9999.9)
    d[_i] = np.interp(_dt[e][_i], _dt[e][_e], d[_e])

    out = np.empty(dt.shape)
    out[e] = d
    out[i] = np.interp(_dt[i], _dt[e], d)

    return out

data_tst = np.load('test.npz')

# SANDY FULL INDEX 23/00Z to 31/00Z
_dt = data_tst['dt']
#print(_dt.shape)

# check all 6 minute intervals are there, add if not
add = []
t = _dt.min()
while True:
    if t > _dt.max():
        break
    if t not in _dt:
        add.append(t)
    t += datetime.timedelta(minutes=6)

# update _dt, SANDY FULL INDEX with every 6 minutes
dt = sorted(add + _dt.tolist())

# index of existing values
idx_tst = [dt.index(t) for t in _dt.tolist()] # TODO interpolate where None values
idx_add = [dt.index(t) for t in add]

# numpy this
dt = np.array(dt)
#print(dt.shape)

'''
# SANDY DISPLAY INDEX
def sandy(dt):
    mask = np.zeros(dt.shape, dtype=bool)
    _ = np.logical_and((dt >= datetime.datetime(2012,10,23,0,0,0)), (dt <= datetime.datetime(2012,10,31,0,0,0)))
    return np.logical_or(mask, _)

display_dt = dt[sandy(dt)]
'''

# every output should be interpolated
kw = {}
for k in data_tst:
    if k == 'dt' or k == 'mask':
        continue
    d = data_tst[k]
    o = _interpolate(idx_add, idx_tst, dt, d)
    print(o.shape)
    print(o.min(), o.mean(), o.max())
    kw[k] = o
print(dt.shape)
kw['dt'] = dt
print(kw.keys())
np.savez('test_filled.npz', **kw)

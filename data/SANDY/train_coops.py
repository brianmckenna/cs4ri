import datetime
import dateutil.parser
import itertools
import numpy as np

import mydict

def offset_days(dt, d):
    days = datetime.timedelta(days=d)
    return [dt.index(t) if t in dt else None for t in [t-days for t in dt]]

data = mydict.d

# --------------------
# LOAD DATA
# --------------------
with np.load('press.npz') as data:
    pdt = data['dt']
    p = data['press']
with np.load('temp.npz') as data:
    tdt = data['dt']
    t = data['temp']
with np.load('u.npz') as data:
    udt = data['dt']
    u = data['u']
with np.load('v.npz') as data:
    vdt = data['dt']
    v = data['v']
with np.load('wl.npz') as data:
    zdt = data['dt']
    z = data['wl']
with np.load('wlp.npz') as data:
    ydt = data['dt']
    y = data['wlp']

# ---------------------
# RANGE OF DATES
# ---------------------
test_dates = [
    #list(map(dateutil.parser.parse, ['2007-04-12T00:00:00', '2007-04-18T00:00:00'])),  # peak at 2007-04-16T10:54:00 (ish)
    #list(map(dateutil.parser.parse, ['2012-12-23T00:00:00', '2012-12-29T00:00:00'])),  # peak at 2012-12-27T12:00:00 (ish)
    list(map(dateutil.parser.parse, ['2016-02-05T00:00:00', '2016-02-11T00:00:00'])),  # peak at 2016-02-09T13:00:00 (ish)
]
print( test_dates)


# mask out all values NOT in our testing date range, so we can see if we have any datasets with zero
def date_window_mask(t, test_dates):
    mask = np.zeros(t.shape, dtype=bool)
    for td in test_dates:
        _ = np.logical_and((t >= td[0]), (t <= td[1]))
        mask = np.logical_or(mask, _)
    return mask
pdtm = date_window_mask(pdt, test_dates)
tdtm = date_window_mask(tdt, test_dates)
udtm = date_window_mask(udt, test_dates)
vdtm = date_window_mask(vdt, test_dates)
zdtm = date_window_mask(zdt, test_dates)
ydtm = date_window_mask(ydt, test_dates)
print( np.sum(pdtm))
print( np.sum(tdtm))
print( np.sum(udtm))
print( np.sum(vdtm))
print( np.sum(zdtm))
print( np.sum(ydtm))

# get the non-masked values
masked_pdt = pdt[pdtm]
masked_tdt = tdt[tdtm]
masked_udt = udt[udtm]
masked_vdt = vdt[vdtm]
masked_zdt = zdt[zdtm]
masked_ydt = ydt[ydtm]

# find what values are missing from one of the arrays
missing_y_z = np.setxor1d(masked_ydt, masked_zdt)
missing_y_u = np.setxor1d(masked_ydt, masked_udt)
missing_y_t = np.setxor1d(masked_ydt, masked_tdt)
missing_y_p = np.setxor1d(masked_ydt, masked_pdt)
missing_z_u = np.setxor1d(masked_zdt, masked_udt)
missing_z_t = np.setxor1d(masked_zdt, masked_tdt)
missing_z_p = np.setxor1d(masked_zdt, masked_pdt)
missing_u_t = np.setxor1d(masked_udt, masked_tdt)
missing_u_p = np.setxor1d(masked_udt, masked_pdt)
missing_t_p = np.setxor1d(masked_tdt, masked_pdt)

missing_dates = np.array(list(set(
    missing_y_z.tolist()
  + missing_y_u.tolist()
  + missing_y_t.tolist()
  + missing_y_p.tolist()
  + missing_z_u.tolist()
  + missing_z_t.tolist()
  + missing_z_p.tolist()
  + missing_u_t.tolist()
  + missing_u_p.tolist()
  + missing_t_p.tolist()
)))

def date_missing_mask(t, missing_dates):
    return np.logical_not(np.in1d(t, missing_dates))

# mask these missing dates
pdtm2 = date_missing_mask(pdt, missing_dates)
tdtm2 = date_missing_mask(tdt, missing_dates)
udtm2 = date_missing_mask(udt, missing_dates)
vdtm2 = date_missing_mask(vdt, missing_dates)
zdtm2 = date_missing_mask(zdt, missing_dates)
ydtm2 = date_missing_mask(ydt, missing_dates)

# FINAL MASK (in testing window and all present)
pm = np.logical_and(pdtm, pdtm2)
tm = np.logical_and(tdtm, tdtm2)
um = np.logical_and(udtm, udtm2)
vm = np.logical_and(vdtm, vdtm2)
zm = np.logical_and(zdtm, zdtm2)
ym = np.logical_and(ydtm, ydtm2)

print( pm.shape)
print( tm.shape)
print( um.shape)
print( vm.shape)
print( zm.shape)
print( ym.shape)

print( np.sum(pm))
print( p[pm])
print( p[pm].shape)

# all empty sets indicate same
print( np.setdiff1d(pdt[pm], tdt[tm]))
print( np.setdiff1d(pdt[pm], udt[um]))
print( np.setdiff1d(pdt[pm], zdt[zm]))
print( np.setdiff1d(pdt[pm], ydt[ym]))

# index of offsets
dt = pdt[pm]
i24 = offset_days(dt.tolist(), 1)
i48 = offset_days(dt.tolist(), 2)
i72 = offset_days(dt.tolist(), 3)

print((dt))
print((i24))
print((i48))
print((i72))

z00 = z[zm]
print( z00.min(), z00.max(), z00.mean())
y00 = y[ym]
print( y00.min(), y00.max(), y00.mean())

# 24 hour persistence
#p24 = np.ma.masked_values([p[ti] if ti else -9999.9 for ti in i24], -9999.9)
#t24 = np.ma.masked_values([t[ti] if ti else -9999.9 for ti in i24], -9999.9)
#u24 = np.ma.masked_values([u[ti] if ti else -9999.9 for ti in i24], -9999.9)
#v24 = np.ma.masked_values([v[ti] if ti else -9999.9 for ti in i24], -9999.9)
#z24 = np.ma.masked_values([z[ti] if ti else -9999.9 for ti in i24], -9999.9)
p24 = np.array([p[ti] if ti else -9999.9 for ti in i24])
t24 = np.array([t[ti] if ti else -9999.9 for ti in i24])
u24 = np.array([u[ti] if ti else -9999.9 for ti in i24])
v24 = np.array([v[ti] if ti else -9999.9 for ti in i24])
z24 = np.array([z[ti] if ti else -9999.9 for ti in i24])
print( p24.min(), p24.max(), p24.mean())
print( t24.min(), t24.max(), t24.mean())
print( u24.min(), u24.max(), u24.mean())
print( v24.min(), v24.max(), v24.mean())
print( z24.min(), z24.max(), z24.mean())

# 48 hour persistence
#p48 = np.ma.masked_values([p[ti] if ti else -9999.9 for ti in i48], -9999.9)
#t48 = np.ma.masked_values([t[ti] if ti else -9999.9 for ti in i48], -9999.9)
#u48 = np.ma.masked_values([u[ti] if ti else -9999.9 for ti in i48], -9999.9)
#v48 = np.ma.masked_values([v[ti] if ti else -9999.9 for ti in i48], -9999.9)
#z48 = np.ma.masked_values([z[ti] if ti else -9999.9 for ti in i48], -9999.9)
p48 = np.array([p[ti] if ti else -9999.9 for ti in i48])
t48 = np.array([t[ti] if ti else -9999.9 for ti in i48])
u48 = np.array([u[ti] if ti else -9999.9 for ti in i48])
v48 = np.array([v[ti] if ti else -9999.9 for ti in i48])
z48 = np.array([z[ti] if ti else -9999.9 for ti in i48])
print( p48.min(), p48.max(), p48.mean())
print( t48.min(), t48.max(), t48.mean())
print( u48.min(), u48.max(), u48.mean())
print( v48.min(), v48.max(), v48.mean())
print( z48.min(), z48.max(), z48.mean())

# 72 hour persistence
#p72 = np.ma.masked_values([p[ti] if ti else -9999.9 for ti in i72], -9999.9)
#t72 = np.ma.masked_values([t[ti] if ti else -9999.9 for ti in i72], -9999.9)
#u72 = np.ma.masked_values([u[ti] if ti else -9999.9 for ti in i72], -9999.9)
#v72 = np.ma.masked_values([v[ti] if ti else -9999.9 for ti in i72], -9999.9)
#z72 = np.ma.masked_values([z[ti] if ti else -9999.9 for ti in i72], -9999.9)
p72 = np.array([p[ti] if ti else -9999.9 for ti in i72])
t72 = np.array([t[ti] if ti else -9999.9 for ti in i72])
u72 = np.array([u[ti] if ti else -9999.9 for ti in i72])
v72 = np.array([v[ti] if ti else -9999.9 for ti in i72])
z72 = np.array([z[ti] if ti else -9999.9 for ti in i72])
print( p72.min(), p72.max(), p72.mean())
print( t72.min(), t72.max(), t72.mean())
print( u72.min(), u72.max(), u72.mean())
print( v72.min(), v72.max(), v72.mean())
print( z72.min(), z72.max(), z72.mean())

print( dt)

print( dt.shape)
print( z00.shape)
print( p24.shape)#, p24.count(), t24.count(), u24.count(), v24.count(), z24.count())
print( p48.shape)#, p48.count(), t48.count(), u48.count(), v48.count(), z48.count())
print( p72.shape)#, p72.count(), t72.count(), u72.count(), v72.count(), z72.count())

print( type(dt))
print( type(z00))
print( type(p24))
print( type(p48))
print( type(p72))

# mask on the 3 days we're running
test_dates = [
    #list(map(dateutil.parser.parse, ['2007-04-15T00:00:00', '2007-04-18T00:00:00'])),  # peak at 2007-04-16T10:54:00 (ish)
    #list(map(dateutil.parser.parse, ['2012-12-26T00:00:00', '2012-12-29T00:00:00'])),  # peak at 2012-12-27T12:00:00 (ish)
    list(map(dateutil.parser.parse, ['2016-02-08T00:00:00', '2016-02-11T00:00:00'])),  # peak at 2016-02-09T13:00:00 (ish)
]
print( test_dates)

mask = []
for td in test_dates:
    mask.append(np.logical_or(np.zeros(dt.shape, dtype=bool), np.logical_and((dt >= td[0]), (dt <= td[1]))).tolist())
mask = np.array(mask, dtype=bool)
print( mask, np.sum(mask), np.sum(mask, axis=1), mask.shape)


np.savez('TRAIN.npz',
    dt    = dt,

    mask  = mask,

    k241  = p24,
    k242  = t24,
    k243u = u24,
    k243v = v24,
    k244  = z24,

    k481  = p48,
    k482  = t48,
    k483u = u48,
    k483v = v48,
    k484  = z48,

    k721  = p72,
    k722  = t72,
    k723u = u72,
    k723v = v72,
    k724  = z72,

    k009  = y00,
    k004  = z00

)

import numpy as np

train1 = np.load('train1_filled.npz')
train2 = np.load('train2_filled.npz')
train3 = np.load('train3_filled.npz')

k1 = sorted([k for k in train1])
k2 = sorted([k for k in train2])
k3 = sorted([k for k in train3])

assert((k1 == k2) & (k2 == k3))


kw = {}
for k in k1:
    a = train1[k]
    b = train2[k]
    c = train3[k]
    d = np.concatenate((a, b, c), axis=0)
    kw[k] = d
mask = np.concatenate((np.ones(a.shape)*1.0, np.ones(a.shape)*2.0, np.ones(a.shape)*3.0), axis=0)
kw['mask'] = np.array([(mask==1), (mask==2), (mask==3)])
print(kw['mask'].shape)


np.savez('train_filled.npz', **kw)

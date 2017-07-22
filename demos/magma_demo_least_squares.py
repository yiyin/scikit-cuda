#!/usr/bin/env python

"""
Demo of how to call low-level MAGMA wrappers to perform QR decomposition.
"""

from ypcutil.importcuda import *
import numpy as np
import pycuda.driver as cuda
import skcuda.magma as magma
from ypcutil.timing import func_timer
import ypcutil.parray as parray
import pycuda.gpuarray as garray
import ypcutil.random as random
import pdb

magma.magma_init()
ngpu = 4
K = 4
N = 2048*8*K
M = 2048*8*K
dbThres = 40

print N,M
x = cuda.pagelocked_empty((N, M), np.float32)

for i in range(K):
    d_x = random.randn((2048*8, M), dtype=x.dtype)
    x[2048*8*i:2048*8*(i+1),:] = d_x.get()
del d_x
c = random.rand((M,1), dtype = x.dtype).get()

c_orig = c.copy()
x_orig = x.T.copy()

n, m = x.shape

tau = np.empty(min(m,n), x.dtype)
nb = magma.magma_get_sgeqrf_nb(m, n)
Lwork = nb*n
workspace = cuda.pagelocked_empty(Lwork, x.dtype)

status = func_timer(magma.magma_sgeqrf_m)(ngpu, m, n, x.ctypes.data, m,
                                          tau.ctypes.data,
                                          workspace.ctypes.data, Lwork)


status = func_timer(magma.magma_sormqr_m)(ngpu, 'L', 'T', m, 1, n,
                                          x.ctypes.data, m,
                                          tau.ctypes.data,
                                          c.ctypes.data, m,
                                          workspace.ctypes.data, Lwork)

status = func_timer(magma.magma_strsm_m)(ngpu, 'L', 'U', 'N', 'N', n, 1, 1.0,
                                         x.ctypes.data, m,
                                         c.ctypes.data, m)


del x, workspace

print -20 * np.log10( np.linalg.norm(np.dot(x_orig,c)-c_orig)/np.linalg.norm(c_orig) )
assert( (-20 * np.log10( np.linalg.norm(np.dot(x_orig,c)-c_orig)/np.linalg.norm(c_orig) )) > dbThres)

magma.magma_finalize()

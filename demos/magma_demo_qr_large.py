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
x = cuda.pagelocked_empty((N, M), np.float32)

for i in range(K):
    d_x = random.randn((2048*8, M), dtype=x.dtype)
    x[2048*8*i:2048*8*(i+1),:] = d_x.get()
del d_x
c = random.rand((M,1), dtype = x.dtype).get()

#x_orig = x.T.copy()

n, m = x.shape

tau = np.empty(min(m,n), x.dtype)
nb = magma.magma_get_sgeqrf_nb(m, n)
Lwork = nb*n
workspace = cuda.pagelocked_empty(Lwork, x.dtype)

status = func_timer(magma.magma_sgeqrf_m)(ngpu, m, n, x.ctypes.data, m,
                                          tau.ctypes.data,
                                          workspace.ctypes.data, Lwork)

#R = func_timer(np.triu)(x.T)

status = func_timer(magma.magma_sormqr_m)(2, 'L', 'N', m, n, m,
                                          x.ctypes.data, m,
                                          tau.ctypes.data,
                                          c.ctypes.data, m,
                                          workspace.ctypes.data, Lwork)

del x, workspace
#status = func_timer(magma.magma_sorgqr2)(m, n, min(m,n), x.ctypes.data, m,
#                                         tau.ctypes.data)
#Q = x.T

#q,r = func_timer(np.linalg.qr)(x_orig)

#print np.abs(R-r).max()#, np.abs(Q-q).max()

magma.magma_finalize()

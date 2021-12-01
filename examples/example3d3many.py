"""
Demonstrate the type 1 NUFFT using cuFINUFFT
"""

import numpy as np

import pycuda.autoinit
from pycuda.gpuarray import GPUArray, to_gpu

from cufinufft import cufinufft

# Set up parameters for problem.
M = 100                         # Number of nonuniform input points
N = 50                          # Number of nonuniform output points
n_transf = 2                    # Number of input arrays
eps = 1e-6                      # Requested tolerance
dtype = np.float32              # Datatype (real)
complex_dtype = np.complex64    # Datatype (complex)

# Generate coordinates of non-uniform points.
kx = np.random.uniform(-np.pi, np.pi, size=M)
ky = np.random.uniform(-np.pi, np.pi, size=M)
kz = np.random.uniform(-np.pi, np.pi, size=M)
s = np.random.uniform(-np.pi, np.pi, size=N)
t = np.random.uniform(-np.pi, np.pi, size=N)
u = np.random.uniform(-np.pi, np.pi, size=N)

# Generate source strengths.
c = (np.random.standard_normal((n_transf, M))
     + 1j * np.random.standard_normal((n_transf, M)))

# Cast to desired datatype.
kx = kx.astype(dtype)
ky = ky.astype(dtype)
kz = kz.astype(dtype)
s = s.astype(dtype)
t = t.astype(dtype)
u = u.astype(dtype)
c = c.astype(complex_dtype)

# Allocate memory for the uniform grid on the GPU.
fk_gpu = GPUArray((n_transf, N), dtype=complex_dtype)

# Initialize the plan and set the points.
plan = cufinufft(3, 3, n_transf, eps=eps, dtype=dtype)
plan.set_pts(to_gpu(kx), to_gpu(ky), to_gpu(kz), to_gpu(s), to_gpu(t), to_gpu(u))

# Execute the plan, reading from the strengths array c and storing the
# result in fk_gpu.
plan.execute(to_gpu(c), fk_gpu)

# Retreive the result from the GPU.
fk = fk_gpu.get()

# Check accuracy of the transform at position.
nt = int(0.37 * N)

for i in range(n_transf):
    # Calculate the true value of the type 1 transform at the uniform grid
    # point (nt1, nt2), which corresponds to the coordinate nt1 - N1 // 2 and
    # nt2 - N2 // 2.
    fk_true = np.sum(c[i] * np.exp(1j * (s[nt] * kx + t[nt] * ky + u[nt] * kz)))

    # Calculate the absolute and relative error.
    err = np.abs(fk[i, nt] - fk_true)
    rel_err = err / np.max(np.abs(fk[i]))

    print(f"[{i}] Absolute error on mode [{nt}] is {err:.3g}")
    print(f"[{i}] Relative error on mode [{nt}] is {rel_err:.3g}")

    assert(rel_err < 10 * eps)

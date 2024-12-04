import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp

# PART A. FOURIER WITH PERIODIC BC


# Define parameters
tp = [0, 4]
tspan = np.arange(0, 4.5, 0.5)
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny
beta = 1
D_in = 0.1

# Define spatial domain and initial conditions
x2 = np.linspace(-Lx / 2, Lx / 2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly / 2, Ly / 2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)

m = 1;  # number of spirals

# define initial conditions
u = np.tanh(np.sqrt(X ** 2 + Y ** 2)) * np.cos(m * np.angle(X + 1j * Y) - (np.sqrt(X ** 2 + Y ** 2))) + 1j * np.zeros(
    (nx, ny));
v = np.tanh(np.sqrt(X ** 2 + Y ** 2)) * np.sin(m * np.angle(X + 1j * Y) - (np.sqrt(X ** 2 + Y ** 2))) + 1j * np.zeros(
    (nx, ny));

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx / 2), np.arange(-nx / 2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny / 2), np.arange(-ny / 2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX ** 2 + KY ** 2


# Define the ODE system
def spc_rhs(t, zt0, nx, ny, N, KX, KY, K, beta, D):
    ut = zt0[:N].reshape((nx, ny)) + 1j * zt0[N:2 * N].reshape((nx, ny))
    vt = zt0[2 * N:3 * N].reshape((nx, ny)) + 1j * zt0[3 * N:].reshape((nx, ny))

    u = ifft2(ut)  # why do I take reals here?
    v = ifft2(vt)

    A = u ** 2 + v ** 2
    lamda = 1 - A
    w = - beta * (A)

    rhs_U = (-D * K * ut + fft2(lamda * u - w * v)).reshape(N)
    rhs_V = (-D * K * vt + fft2(w * u + lamda * v)).reshape(N)

    return np.hstack([np.real(rhs_U), np.imag(rhs_U),
                      np.real(rhs_V), np.imag(rhs_V)])


# Solve the ODE and plot the results

zt0 = np.hstack([np.real(fft2(u)).reshape(N), np.imag(fft2(u)).reshape(N),
                 np.real(fft2(v)).reshape(N), np.imag(fft2(v)).reshape(N)])

ztsol = solve_ivp(spc_rhs, tp, zt0, method='RK45', t_eval=tspan, args=(nx, ny, N, KX, KY, K, beta, D_in))
ztsol = ztsol.y.T

# in fourier for submission:
zsol_u_f = np.zeros((len(tspan), N), dtype=np.complex128)
zsol_v_f = np.zeros((len(tspan), N), dtype=np.complex128)

# need to inverse fourier transform each row of output ztsol to get out of fourier space
for i, zt in enumerate(ztsol):
    z_real_u = zt[:N]
    z_imag_u = zt[N:2 * N]

    z_real_v = zt[2 * N:3 * N]
    z_imag_v = zt[3 * N:]

    zsol_u_f[i] = z_real_u + 1j * z_imag_u
    zsol_v_f[i] = z_real_v + 1j * z_imag_v

A1 = np.hstack([zsol_u_f, zsol_v_f])
A1 = A1.T
A1 = np.real(A1)

# ifft for plotting:

# outside fourier:
zsol_u = np.zeros((len(tspan), N))
zsol_v = np.zeros((len(tspan), N))

# need to inverse fourier transform each row of output ztsol to get out of fourier space
for i, zt in enumerate(ztsol):
    z_real_u = zt[:N].reshape((nx, ny))
    z_imag_u = zt[N:2 * N].reshape((nx, ny))

    z_real_v = zt[2 * N:3 * N].reshape((nx, ny))
    z_imag_v = zt[3 * N:].reshape((nx, ny))

    z_complex_u = z_real_u + 1j * z_imag_u
    z_complex_v = z_real_v + 1j * z_imag_v

    z_spatial_u = (ifft2(z_complex_u)).reshape(N)
    z_spatial_v = (ifft2(z_complex_v)).reshape(N)

    zsol_u[i] = np.real((z_spatial_u))
    zsol_v[i] = np.real((z_spatial_v))

A1_inv = np.hstack([zsol_u, zsol_v])
A1_inv = A1_inv.T



import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy import kron

# PART B. Chebyshev


# Parameters
tp = [0, 4]
tspan = np.arange(0, 4.5, 0.5)
beta = 1
D_in = 0.1

def sech(x):
    return 1 / np.cosh(x)

def tanh(x):
    return np.sinh(x) / np.cosh(x)

def cheb(N):
    if N == 0:
        D = 0.; x = 1.
    else:
        n = np.arange(0, N+1)
        x = np.cos(np.pi * n / N).reshape(N+1, 1)
        c = (np.hstack(([2.], np.ones(N-1), [2.])) * (-1)**n).reshape(N+1, 1)
        X = np.tile(x, (1, N+1))
        dX = X - X.T
        D = np.dot(c, 1. / c.T) / (dX + np.eye(N+1))
        D -= np.diag(np.sum(D.T, axis=0))
    return D, x.reshape(N+1)

# Chebyshev grid and Laplacian
N = 30
D, x = cheb(N)

D[N, :] = 0
D[0, :] = 0

D2 = np.dot(D, D) # Second derivative matrix

I = np.eye(len(D2))
L = kron(I, D2) + kron(D2, I)  # 2D Laplacian
L = L / 100

# Rescale the grid to [-10, 10]
x = x*10
y = x
X, Y = np.meshgrid(x, y)

# Initial conditions for U and V
m = 1
U = np.tanh(np.sqrt(X**2 + Y**2)) * np.cos(m * np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))
V = np.tanh(np.sqrt(X**2 + Y**2)) * np.sin(m * np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))

# Flatten and stack the initial conditions
u = U.reshape((N + 1) ** 2)
v = V.reshape((N + 1) ** 2)
z0 = np.hstack([u, v])

# Reaction-diffusion system
def spc_rhs(t, z0, L, beta, D):
    u0 = z0[:(N+1)**2]
    v0 = z0[(N+1)**2:]
    A = u0 ** 2 + v0 ** 2
    lamda = 1 - A
    w = -beta * (A)

    rhs_U = lamda * u0 - w * v0 + D * np.dot(L, u0)
    rhs_V = w * u0 + lamda * v0 + D * np.dot(L, v0)
    return np.hstack([rhs_U, rhs_V])

# Solve the system of equations
zsol = solve_ivp(spc_rhs, tp, z0, method='RK45', t_eval=tspan, args=(L, beta, D_in))
zsol = zsol.y.T
A2 = zsol.T


print(A2)
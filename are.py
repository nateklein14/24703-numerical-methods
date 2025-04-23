import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.linalg import solve_continuous_lyapunov
from scipy.io import loadmat
from scipy import signal
from time import time
import random
import matplotlib.pyplot as plt

def ARE(A, G, Q, X):
    return A.T@X+X@A+X@G@X+Q

def exact_line_search(A, G, Q, X0=None):
    # If starting solution is specified, use it, otherwise use 0
    if np.any(X0 != None):
        X = X0
    else:    
        X = np.zeros(A.shape)
    t = 1
    R = ARE(A, G, Q, X)
    while np.linalg.norm(R, 'fro') > 1e-3:
        N = -solve_continuous_lyapunov((A+G@X).T, R)
        t = find_t(G, N, R)
        X_new = X + t*N
        X = X_new
        R = ARE(A, G, Q, X)
    return X

def find_t(G, N, R):
    dt = 0.0001
    roots = []
    for i in range(0, int(2/dt)+1):
        t = i*dt
        val = f2(R, G, N, t)
        if val / f2(R, G, N, t+dt) < 0:
            while abs(val) > 1e-6:
                dfdt = (f2(R, G, N, t+dt) - val) / dt
                t = t - val / dfdt
                val = f2(R, G, N, t)
            roots.append(t)
    return roots[0]

def f2(R, G, N, t):
    V = N@G@N
    return -2*np.trace((R-2*t*V)@((1-t)*R+t**2*V))

def newton_kleinman(A, G, Q, X0=None):
    # If starting solution is specified, use it, otherwise use 0
    if np.any(X0 != None):
        X = X0
    else:    
        X = np.zeros(A.shape)
    while np.linalg.norm(ARE(A, G, Q, X), 'fro') > 1e-3:
        X_new = solve_continuous_lyapunov((A+G@X).T, X@G@X-Q)
        X = X_new
    return X

def create_system(order):

    # Generate random stable poles for the system
    reals = [random.uniform(-100, -1e-10) for i in range(int(order/2))]
    imags = [random.uniform(0, 100) for i in range(int(order/2))]
    zeros = [random.uniform(-100, 100) for i in range(random.randint(0, order-1))]
    poles = []
    for i in range(len(reals)):
        poles.append(reals[i]+1j*imags[i])
        poles.append(reals[i]-1j*imags[i])
    if len(poles)<order:
        poles.append(random.uniform(-100, -1e-10))

    # Create state space system with generated poles and zeros
    A, B, C, _= signal.zpk2ss(zeros, poles, random.uniform(1,10))
    Q = C.T@C
    H = np.block([[A, B@B.T], [-Q, -A.T]])
    H_evals = np.linalg.eigvals(H)
    flag = np.any(abs(np.real(H_evals))<=1e-15) or (not is_pos_sdef(B@B.T)) or (not is_pos_sdef(Q))

    return A, B, Q, flag

def is_pos_sdef(M):
    return np.all(np.linalg.eigvals(M) >= 0)

nk_times = []
nk_norms = []
els_times = []
els_norms = []
sp_times = []
sp_norms = []

n_max = 8

for order in [2**i for i in range(1,n_max+1)]:
    data = loadmat(f'sys{order}.mat')
    A = np.array(data['A'])
    B = np.array(data['B'])
    C = np.array(data['C'])
    G = -B@B.T
    Q = C.T@C

    t = time()
    X = solve_continuous_are(A, B, Q, 1)
    sp_times.append(1000*(time() - t))
    R = ARE(A, G, Q, X)
    sp_norms.append(np.linalg.norm(R, 'fro'))
    # print(X)
    
    t = time()
    X = newton_kleinman(A, G, Q)
    nk_times.append(1000*(time() - t))
    R = ARE(A, G, Q, X)
    nk_norms.append(np.linalg.norm(R, 'fro'))
    # print(X)
    
    t = time()
    X = exact_line_search(A, G, Q)
    els_times.append(1000*(time() - t))
    R = ARE(A, G, Q, X)
    els_norms.append(np.linalg.norm(R, 'fro'))
    # print(X)
    print(order)

plt.plot(range(1,n_max+1), sp_times)
plt.plot(range(1,n_max+1), nk_times)
plt.plot(range(1,n_max+1), els_times)
plt.title("Runtime")
plt.xlabel("log2(Model Order)")
plt.ylabel("Runtime (ms)")
# plt.legend(["SciPy", "Newton-Kleinman", "Exact Line Search"])
plt.show()

plt.plot(range(1,n_max+1), sp_norms)
plt.plot(range(1,n_max+1), nk_norms)
plt.plot(range(1,n_max+1), els_norms)
plt.title("Frobenius Norm")
plt.xlabel("log2(Model Order)")
plt.ylabel("Norm")
# plt.legend(["SciPy", "Newton-Kleinman", "Exact Line Search"])
plt.show()
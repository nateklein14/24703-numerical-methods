import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.linalg import solve_continuous_lyapunov
from scipy.io import loadmat
from time import time
import matplotlib.pyplot as plt

# The Algebraic Riccati equation
def ARE(A, G, Q, X):
    return A.T@X+X@A+X@G@X+Q

def exact_line_search(A, G, Q, tol, X0=None):
    # If starting solution is specified, use it, otherwise use 0
    if np.any(X0 != None):
        X = X0
    else:    
        X = np.zeros(A.shape)

    # The algorithm described in Benner
    i = 0
    t = 1
    R = ARE(A, G, Q, X)
    while np.linalg.norm(R, 'fro') > tol:
        N = -solve_continuous_lyapunov((A+G@X).T, R)
        t = find_t(G, N, R)
        X_new = X + t*N
        X = X_new
        R = ARE(A, G, Q, X)
        i += 1
    return X, i

# Newton-Raphson root finding for minimizing the error estimate
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
    return min(roots)

# The derivative of the error estimate as derived by Benner
def f2(R, G, N, t):
    V = N@G@N
    return -2*np.trace((R-2*t*V)@((1-t)*R+t**2*V))

# Standard iterative Newton-Kleinman method
def newton_kleinman(A, G, Q, tol, X0=None):
    # If starting solution is specified, use it, otherwise use 0
    if np.any(X0 != None):
        X = X0
    else:    
        X = np.zeros(A.shape)

    i = 0
    while np.linalg.norm(ARE(A, G, Q, X), 'fro') > tol:
        X_new = solve_continuous_lyapunov((A+G@X).T, X@G@X-Q)
        X = X_new
        i += 1
    return X, i

# Run each solver on systems of order 2^n for n = 1, 2, ..., n_max
# Systems must be pre-saved as "sys{n}.mat"
# Stores the runtime and final solution error of each method on each order system and saves the data to .npy files
def run_tests(n_max):
    nk_times = []
    nk_norms = []
    nk_iters = []
    els_times = []
    els_norms = []
    els_iters = []
    sp_times = []
    sp_norms = []
    stacked_times = []
    stacked_norms = []
    stacked_iters = []

    tol = 1e-4

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
        
        t = time()
        X, i = newton_kleinman(A, G, Q, tol)
        nk_times.append(1000*(time() - t))
        R = ARE(A, G, Q, X)
        nk_norms.append(np.linalg.norm(R, 'fro'))
        nk_iters.append(i)
        
        t = time()
        X, i = exact_line_search(A, G, Q, tol)
        els_times.append(1000*(time() - t))
        R = ARE(A, G, Q, X)
        els_norms.append(np.linalg.norm(R, 'fro'))
        els_iters.append(i)

        t = time()
        X, i_nk = newton_kleinman(A, G, Q, 10*tol)
        X, i_els = exact_line_search(A, G, Q, tol, X0=X)
        stacked_times.append(1000*(time() - t))
        R = ARE(A, G, Q, X)
        stacked_norms.append(np.linalg.norm(R, 'fro'))
        stacked_iters.append(i_nk + i_els)

        print(order)

    np.save("sp_times.npy", sp_times)
    np.save("nk_times.npy", nk_times)
    np.save("els_times.npy", els_times)
    np.save("stacked_times.npy", stacked_times)
    np.save("sp_norms.npy", sp_norms)
    np.save("nk_norms.npy", nk_norms)
    np.save("els_norms.npy", els_norms)
    np.save("stacked_norms.npy", stacked_norms)
    np.save("nk_iters.npy", nk_iters)
    np.save("els_iters.npy", els_iters)
    np.save("stacked_iters.npy", stacked_iters)

# Load results from the .npy files
def load_results():
    sp_times = np.load("sp_times.npy")
    nk_times = np.load("nk_times.npy")
    els_times = np.load("els_times.npy")
    stacked_times = np.load("stacked_times.npy")
    sp_norms = np.load("sp_norms.npy")
    nk_norms = np.load("nk_norms.npy")
    els_norms = np.load("els_norms.npy")
    stacked_norms = np.load("stacked_norms.npy")
    nk_iters = np.load("nk_iters.npy")
    els_iters = np.load("els_iters.npy")
    stacked_iters = np.load("stacked_iters.npy")
    return sp_times, nk_times, els_times, stacked_times, sp_norms, nk_norms, els_norms, stacked_norms, nk_iters, els_iters, stacked_iters

# Plot the results of the tests
def plot_results(sp_times, nk_times, els_times, stacked_times, sp_norms, nk_norms, els_norms, stacked_norms, nk_iters, els_iters, stacked_iters):

    plt.xlabel("$log_2(Model Order)$")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtimes")
    plt.plot(range(1,len(sp_times)+1), sp_times, linewidth=3, color='m')
    # plt.show()

    plt.xlabel("$log_2(Model Order)$")
    plt.ylabel("Runtime (ms)")
    # plt.title("Newton-Kleinman Runtime")
    plt.plot(range(1,len(nk_times)+1), nk_times, linewidth=3, color='r')
    # plt.show()

    plt.xlabel("$log_2(Model Order)$")
    plt.ylabel("Runtime (ms)")
    # plt.title("Stacked Methods Runtime")
    plt.plot(range(1,len(stacked_times)+1), stacked_times, linewidth=3, color='g')
    plt.legend(["SciPy", "Newton-Kleinman", "Stacked Methods"])
    plt.show()

    plt.xlabel("$log_2(Model Order)$")
    plt.ylabel("Runtime (ms)")
    plt.title("Exact Line Search Runtime")
    plt.plot(range(1,len(els_times)+1), els_times, linewidth=3, color='b')
    plt.show()

    plt.xlabel("$log_2(Model Order)$")
    plt.ylabel("Iterations")
    plt.title("Iterations by Method")
    plt.plot(range(1,len(nk_iters)+1), nk_iters, linewidth=3, color='r')
    plt.plot(range(1,len(els_iters)+1), els_iters, linewidth=3, color='b')
    plt.plot(range(1,len(stacked_iters)+1), stacked_iters, linewidth=3, linestyle='--', color='g')
    plt.legend(["Newton-Kleinman", "Exact Line Search", "Stacked Methods"])
    plt.show()

    plt.plot(range(1,len(sp_norms)+1), sp_norms, linewidth=3, color='m')
    plt.plot(range(1,len(nk_norms)+1), nk_norms, linewidth=3, color='r')
    plt.plot(range(1,len(els_norms)+1), els_norms, linewidth=3, color='b')
    plt.plot(range(1,len(stacked_norms)+1), stacked_norms, linewidth=3, linestyle='--', color='g')
    plt.xlabel("$log_2(Model Order)$")
    plt.ylabel("$||R(X_j)||_F$")
    plt.title("Solution Error")
    plt.legend(["SciPy", "Newton-Kleinman", "Exact Line Search", "Stacked Methods"])
    plt.show()

# run_tests(n_max=7) # Comment this out if just plotting previously run tests
sp_times, nk_times, els_times, stacked_times, sp_norms, nk_norms, els_norms, stacked_norms, nk_iters, els_iters, stacked_iters = load_results()
plot_results(sp_times, nk_times, els_times, stacked_times, sp_norms, nk_norms, els_norms, stacked_norms, nk_iters, els_iters, stacked_iters)

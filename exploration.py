import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def upper_powered_matrix(N, rho):
    """
    Create an N x N upper-triangular matrix with:
      - 1s on the diagonal
      - rho on the first upper diagonal
      - rho^2 on the second upper diagonal
      - ...
      - rho^k on the k-th upper diagonal

    Parameters:
    ----------
    N : int
        Size of the matrix (NxN).
    rho : float
        Constant base used for the exponential decay along upper diagonals.

    Returns:
    -------
    A : np.ndarray
        An N x N matrix where A[i, j] = rho^{j - i} for j >= i.
    """
    A = np.eye(N)  # start with identity matrix
    for k in range(1, N):
        A += np.diag([rho**k] * (N - k), k=k)
    return A

# number of individuals, time periods, and covariates
N, T, J = 100, 10, 3

# autoregressive parameter
rho = 0.7

# intercept
beta0 = 1.0

# coefficients
beta = np.array([0.5, -0.3, 0.2])

# error term standard deviation
sigma_eps = 1.0

# Simulate covariates x_{jit}
x = np.random.normal(size=(N, T, J))

# Individual fixed effects
f = np.random.normal(scale=1.0, size=N)

# Error terms
eps = np.random.normal(scale=sigma_eps, size=(N, T))

# --------------------
# Model (1): y depends on lagged y
# --------------------
A = upper_powered_matrix(T, rho)

x_coll = x @ beta + f[:, np.newaxis] + eps + beta0
y1 = x_coll @ A

# --------------------
# Model (2): y depends on AR(1) error u_{it}
# --------------------
u = eps @ A

y2 = x @ beta + f[:, np.newaxis] + u + beta0


# --------------------
# Visualize the simulated data
# --------------------

# plot mean, median, and 10-90% quantile of each data
def summary_plot(y, title, ax):
    t = np.arange(y.shape[1])
    mean = np.mean(y, axis=0)
    median = np.median(y, axis=0)
    q10 = np.quantile(y, 0.10, axis=0)
    q90 = np.quantile(y, 0.90, axis=0)

    ax.plot(t, mean, label='Mean', color='blue')
    ax.plot(t, median, label='Median', color='green')
    ax.fill_between(t, q10, q90, alpha=0.2, color='gray', label='10–90% Quantile')
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.grid(True)
    ax.legend()

# Assume y1 and y2 are (N, T)
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

summary_plot(y1, 'Panel A: Summary of y1', axs[0])
summary_plot(y2, 'Panel B: Summary of y2', axs[1])

plt.tight_layout()
plt.show()



# plot 5 random individuals for each model
idx1 = np.random.choice(y1.shape[0], size=5, replace=False)
idx2 = np.random.choice(y2.shape[0], size=5, replace=False)

t = np.arange(y1.shape[1])  # time periods

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Panel A: 5 individuals from y1
for i in idx1:
    axs[0].plot(t, y1[i], label=f'Ind {i}')
axs[0].set_title('Panel A: 5 Random Individuals from y1')
axs[0].set_ylabel('Value')
axs[0].legend()
axs[0].grid(True)

# Panel B: 5 individuals from y2
for i in idx2:
    axs[1].plot(t, y2[i], label=f'Ind {i}')
axs[1].set_title('Panel B: 5 Random Individuals from y2')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Value')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()


# --------------------
# Estimate the model to find the bias
# --------------------

# Paper notes that model (2) can e written in the form of model (1)
# For simplicity, we will only continue with model 1

def dgp(N, T, J, rho, beta0, beta, sigma_eps):
    """
    Generate panel data according to a dynamic panel data model with fixed effects.
    
    This function creates simulated panel data following the structure:
    y_{it} = rho * y_{i,t-1} + x_{it}' * beta + f_i + u_{it}
    where u_{it} = eps_{it} + rho * eps_{i,t-1} + ... + rho^(t-1) * eps_{i1}
    
    Parameters
    ----------
    N : int
        Number of individuals (cross-sectional units)
    T : int
        Number of time periods
    J : int
        Number of explanatory variables (excluding lagged dependent variable)
    rho : float
        Autoregressive parameter for the lagged dependent variable
    beta0 : float
        Intercept term
    beta : array-like
        Coefficient vector for explanatory variables, shape (J,)
    sigma_eps : float
        Standard deviation of the innovation term epsilon
    
    Returns
    -------
    tuple
        A tuple containing:
        - y1 : ndarray, shape (N, T)
            Dependent variable for model 1 (with moving average errors)
        - x : ndarray, shape (N, T, J) or None if J=0
            Explanatory variables
        - f : ndarray, shape (N,)
            Individual fixed effects
        - eps : ndarray, shape (N, T)
            Innovation terms
        - u : ndarray, shape (N, T)
            Composite error terms for model 2
    
    Notes
    -----
    Model 1: y1_{it} = rho * y1_{i,t-1} + x_{it}' * beta + f_i + u_{it}
    where u_{it} follows a moving average process.
    """

    if J > 0:
        x = np.random.normal(size=(N, T, J))
        x_beta = x @ beta
    else:
        x = None
        x_beta = 0
    
    f = np.random.normal(scale=1.0, size=N)
    eps = np.random.normal(scale=sigma_eps, size=(N, T))

    A = upper_powered_matrix(T, rho)

    x_coll = x_beta + f[:, np.newaxis] + eps + beta0
    y1 = x_coll @ A

    u = eps @ A

    return y1, x, f, eps, u

N, T, J, rho, beta0, beta, sigma_eps = 100, 10, 3, 0.7, 1.0, np.array([0.5, -0.3, 0.2]), 1.0
N, T, J, rho, beta0, beta, sigma_eps = 100, 10, 0, 0.7, 1.0, None, 1.0
y1, x, f, eps, u = dgp(N, T, J, rho, beta0, beta, sigma_eps)



# Demeaned variables
y_dm = y1 - y1.mean(axis=1, keepdims=True)
if J > 0:
    assert x is not None  # type assertion for linter
    x_dm = x - x.mean(axis=1, keepdims=True)
else:
    x_dm = None
eps_dm = eps - eps.mean(axis=1, keepdims=True)

# Lagged demeaned y
y_lag = np.zeros_like(y1)
y_lag[:, 1:] = y1[:, :-1]
y_lag_dm = y_lag - y_lag.mean(axis=1, keepdims=True)


y_dep = y_dm[:, 1:].ravel()                         
y_lag_flat = y_lag_dm[:, 1:].reshape(-1, 1)         
if J > 0:
    assert x is not None and x_dm is not None  # type assertion for linter
    x_flat = x_dm[:, 1:, :].reshape(-1, x.shape[2])     
    # Combine regressors
    X = np.hstack([y_lag_flat, x_flat]) 
else:
    X = y_lag_flat
X = sm.add_constant(X)

# Run regression that pools all time periods
model = sm.OLS(y_dep, X).fit()
print(model.summary())
print('DGP parameters: beta0 (const):', beta0, 'rho (x1): ', rho, 'beta1, beta2, beta3 (x1, x2, x3): ', beta)
# Here beta1 to beta3 are not biased, because they are not correlated with the fixed effects. 

# Run regression for only one time period t <= T
t = 4
if J > 0:
    assert x_dm is not None  # type assertion for linter
    X = np.hstack([y_lag_dm[:, t].reshape(-1, 1), x_dm[:, t, :]])
else:
    X = y_lag_dm[:, t].reshape(-1, 1)
X = sm.add_constant(X)
model = sm.OLS(y_dm[:, t], X).fit()
print(model.summary())
if J > 0:
    print('DGP parameters: beta0 (const):', beta0, 'rho (x1): ', rho, 'beta1, beta2, beta3 (x1, x2, x3): ', beta)
else:
    print('DGP parameters: beta0 (const):', beta0, 'rho (x1): ', rho)



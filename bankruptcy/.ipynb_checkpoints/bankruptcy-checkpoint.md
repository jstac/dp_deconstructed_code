---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## Comparative Timings for the Bankruptcy Model


If $a=b$

```python
import numpy as np
import quantecon as qe
from numba import njit
```

```python
z_size = 3
kappa_size = 3
eta_size = 3
d_size = 3
```

```python
d_max = 10
d_grid = np.linspace(0, d_max, d_size)
```

```python
mc = qe.tauchen(0.9, 0.1, n=z_size)
```

```python
P = mc.P
```

```python
z_grid = np.exp(mc.state_values)
```

```python
z_grid.min()
```

```python
beta = 0.99
gamma = 0.5
r_bar = 0.1
```

```python
@njit
def q(z):
    return 1 + 0.1 * z
```

```python
@njit
def u(c):
    return c
```

```python
kappa_min, kappa_max = 0.5, 2.0
kappa_grid = np.linspace(kappa_min, kappa_max, kappa_size)
kappa_p = np.ones(kappa_size) / kappa_size  # Uniform dist
```

```python
eta_min, eta_max = 0.5, 2.0
eta_grid = np.linspace(eta_min, eta_max, eta_size)
eta_p = np.ones(eta_size) / eta_size  # Uniform dist
```

```python
@njit
def bellman(vR, vB, vE):
    
    vR_new = np.empty_like(vR)
    vB_new = np.empty_like(vB)
    vE_new = np.empty_like(vE)
    
    # First update vR
    # Here's all the states
    for i_d, d in enumerate(d_grid):
        for i_z, z in enumerate(z_grid):
            for i_eta, η in enumerate(eta_grid):
                for i_kappa, κ in enumerate(kappa_grid):
                    # For each state, eval RHS of Bellman at all dp and record largest
                    current_max = -1e10
                    for i_dp, dp in enumerate(d_grid):                        
                        # First compute the expectation
                        e = 0.0
                        for i_zp in range(z_size):
                            for i_etap in range(eta_size):
                                for i_kappap in range(kappa_size):
                                    e += max(vR[i_dp, i_zp, i_etap, i_kappap], vB[i_zp, i_etap]) * P[i_z, i_zp]
                        e = e * (1 / eta_size) * (1 / kappa_size)
                        candidate = u(dp * q(z) - κ - d + η * z) + beta * e
                        if candidate > current_max:
                            current_max = candidate
                    # Largest recorded is new value
                    vR_new[i_d, i_z, i_eta, i_kappa] = current_max
                            
    # Next update vB
    # Here's all the states
    for i_z, z in enumerate(z_grid):
        for i_eta, η in enumerate(eta_grid):
            # Compute the expectation
            e = 0.0
            for i_zp in range(z_size):
                for i_etap in range(eta_size):
                    for i_kappap in range(kappa_size):
                        e += max(vR[0, i_zp, i_etap, i_kappap], vE[i_zp, i_etap, i_kappap]) * P[i_z, i_zp]
            e = e * (1 / eta_size) * (1 / kappa_size)
            vB_new[i_z, i_eta] = u((1 - gamma) * z * η) + beta * e
            
    # Finally, update vE
    # Here's all the states
    for i_z, z in enumerate(z_grid):
        for i_eta, η in enumerate(eta_grid):
            for i_kappa, κ in enumerate(kappa_grid):
                i_d_hat = np.searchsorted(d_grid, (κ - gamma * z * η) * (1 + r_bar))
                # Compute the expectation
                e = 0.0
                for i_zp in range(z_size):
                    for i_etap in range(eta_size):
                        for i_kappap in range(kappa_size):
                            e += max(vR[i_d_hat, i_zp, i_etap, i_kappap], vB[i_zp, i_etap]) * P[i_z, i_zp]
                e = e * (1 / eta_size) * (1 / kappa_size)
                vE_new[i_z, i_eta, i_kappa] = u((1 - gamma) * z * η) + beta * e

    return vR_new, vB_new, vE_new
```

```python
vR = np.ones((d_size, z_size, eta_size, kappa_size))
vB = np.ones((z_size, eta_size))
vE = np.ones((z_size, eta_size, kappa_size))
```

```python
tol = 0.001
iter_max = 10_000
eps = tol + 1
i = 0

while eps > tol and i < iter_max:
    vR_new, vB_new, vE_new = bellman(vR, vB, vE)
    eps_R = np.max(np.abs(vR_new - vR))
    eps_B = np.max(np.abs(vB_new - vB))
    eps_E = np.max(np.abs(vB_new - vB))
    eps = max(eps_R, eps_B, eps_E)
    vR, vB, vE = vR_new, vB_new, vE_new
    i += 1

print(f"Terminated at iteration {i} with error {eps}.")
```

```python

```

```python

```

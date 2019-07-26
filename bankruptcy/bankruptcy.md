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

The value of repaying one's debts, $v^R$, satisfies the 
Bellman equation

\begin{equation*}
	v^R (d,z, \eta, \kappa) = \max_{c, \, d'}
	\left[
	u(c) + \beta E_{z', \eta', \kappa' \mid z} \max 
	\left\{
	v^R (d', z', \eta', \kappa'), \, v^B (z', \eta')
	\right\}
	\right]
\end{equation*}

subject to $c + d + \kappa \leq \bar{e} z \eta + q (z) d'$.

Here $v^B$ is the value of declaring bankruptcy, which satisfies

\begin{equation*}
	v^B (z, \eta) = u(c) + 
	\beta E_{z', \eta', \kappa' \mid z} \max 
	\left\{ 
	v^R (0, z', \eta', \kappa'), \, v^E (z', \eta', \kappa')
	\right\}
\end{equation*}

subject to 

$$ c = \hat c := (1 - \gamma) \bar{e} z \eta . $$ 

Finally, $v^E$ is the value of defaulting on expense debt, which satisfies 

\begin{equation*}
	v^E (z, \eta, \kappa) = u (c ) + 
	\beta E_{z', \eta', \kappa' \mid z} \max 
	\left\{
	v^R (d', z', \eta', \kappa'), \, v^B (z', \eta')
	\right\}
\end{equation*}

subject to 

$$ 
    c = \hat c
   \quad \text{and} \quad
   d' = \hat d := (\kappa - \gamma \bar{e} z \eta) (1 + \bar{r}) 
   .
$$


### Set Up and Parameters

```python
import numpy as np
import quantecon as qe
from numba import njit
```

```python
z_size = 10
kappa_size = 10
eta_size = 10
d_size = 10
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
kappa_prob = np.ones(kappa_size) / kappa_size  # Uniform dist
```

```python
eta_min, eta_max = 0.5, 2.0
eta_grid = np.linspace(eta_min, eta_max, eta_size)
eta_prob = np.ones(eta_size) / eta_size  # Uniform dist
```

### The Regular Bellman Equation

```python
@njit
def T(vR, vB, vE):
    
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

Let's just check that iteration with the Bellman operator converges.

```python
vR = np.ones((d_size, z_size, eta_size, kappa_size))
vB = np.ones((z_size, eta_size))
vE = np.ones((z_size, eta_size, kappa_size))
tol = 0.001
iter_max = 10_000
eps = tol + 1
i = 0

while eps > tol and i < iter_max:
    vR_new, vB_new, vE_new = T(vR, vB, vE)
    eps_R = np.max(np.abs(vR_new - vR))
    eps_B = np.max(np.abs(vB_new - vB))
    eps_E = np.max(np.abs(vB_new - vB))
    eps = max(eps_R, eps_B, eps_E)
    vR, vB, vE = vR_new, vB_new, vE_new
    i += 1

print(f"Terminated at iteration {i} with error {eps}.")
```

### The Refactored Bellman Operator


After the refactoring, the modified Bellman equations become

\begin{equation*}
	g^{D} (z, d') = 
	E_{z', \eta', \kappa' \mid z} \max 
	\left\{
	\max_{c', \, d''} 
	\left[
	u (c') + \beta g^{D} (z', d'')
	\right], \,
	u (\tilde{c}) + \beta g^{E} (z')
	\right\}
\end{equation*}

subject to

$$
    c'  =  z' \eta' + q(z') d'' - d' - \kappa'
    \quad \text{and} \quad
    \tilde{c} = (1 - \gamma)  z' \eta'
$$

and

\begin{equation*}
	g^{E} (z) = 
	E_{z', \eta', \kappa' \mid z} \max 
	\left\{ 
	\max_{c', \, d''} 
	\left[
	u (c') + \beta g^{D} (z', d'')
	\right], \,
	u (\tilde{c}) + \beta g^{D} (z', \tilde{d})
	\right\}
\end{equation*}

subject to 

$$
    c' = z' \eta' + q(z') d'' - \kappa' 
    \quad \text{and} \quad
    \tilde{d} = (\kappa' - \gamma z' \eta') (1 + \bar{r}).
$$


```python
@njit
def S(gD, gE):
    
    gD_new = np.empty_like(gD)
    gE_new = np.empty_like(gE)
    
    # First update gD
    # Step through all states:
    for i_z, z in enumerate(z_grid):
        for i_dp, dp in enumerate(d_grid):
            e = 0.0  # Will hold the expectation
            for i_zp, zp in enumerate(z_grid):
                for i_etap, etap in enumerate(eta_grid):
                    for i_kappap, kappap in enumerate(kappa_grid):
                        # Compute the max of two terms, L and R (left and right)
                        # Start with R
                        c_tilde = (1 - gamma) * zp * etap
                        R = u(c_tilde) + beta * gE[i_zp]
                        # Next, L
                        current_max = -1e10
                        for i_dpp, dpp in enumerate(d_grid):
                            util = u(zp * etap + q(zp) * dpp - dp - kappap)
                            m = util + beta * gD[i_zp, i_dpp]
                            if m > current_max:
                                current_max = m
                        L = current_max
                        e += max(L, R) * P[i_z, i_zp]
            e = e * (1 / eta_size) * (1 / kappa_size)
            gD_new[i_z, i_dp] = e
                
    # Next update gE
    # Step through all states:
    for i_z, z in enumerate(z_grid):
        e = 0.0  # Will hold the expectation
        for i_zp, zp in enumerate(z_grid):
            for i_etap, etap in enumerate(eta_grid):
                for i_kappap, kappap in enumerate(kappa_grid):
                    # Compute the max of two terms, L and R (left and right)
                    # Start with R
                    c_tilde = (1 - gamma) * zp * etap
                    i_d_tilde = np.searchsorted(d_grid, (kappap - gamma * zp * etap) * (1 + r_bar))
                    R = u(c_tilde) + beta * gD[i_zp, i_d_tilde]
                    # Next, L
                    current_max = -1e10
                    for i_dpp, dpp in enumerate(d_grid):
                        util = u(zp * etap + q(zp) * dpp - kappap)
                        m = util + beta * gD[i_zp, i_dpp]
                        if m > current_max:
                            current_max = m
                    L = current_max
                    e += max(L, R) * P[i_z, i_zp]
        e = e * (1 / eta_size) * (1 / kappa_size)
        gE_new[i_z] = e
                
    return gD_new, gE_new
```


Let's check that iteration with $S$ converges.

```python
gD = np.ones((z_size, d_size))
gE = np.ones(z_size)
tol = 0.001
iter_max = 10_000
eps = tol + 1
i = 0

while eps > tol and i < iter_max:
    gD_new, gE_new = S(gD, gE)
    eps_D = np.max(np.abs(gD_new - gD))
    eps_E = np.max(np.abs(gE_new - gE))
    eps = max(eps_D, eps_E)
    gD, gE = gD_new, gE_new
    i += 1

print(f"Terminated at iteration {i} with error {eps}.")
```


### Timing


```python

gD = np.ones((z_size, d_size))
gE = np.ones(z_size)

vR = np.ones((d_size, z_size, eta_size, kappa_size))
vB = np.ones((z_size, eta_size))
vE = np.ones((z_size, eta_size, kappa_size))

```


```python
%%time
vR_new, vB_new, vE_new = T(vR, vB, vE)

```

```python
%%time
gD_new, gE_new = S(gD, gE)

```

```python

```

```python

```

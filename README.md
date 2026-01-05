# Risk-Aware Linear Quadratic Control

Implementation of the risk-constrained LQR and LQG controllers from:

> A. Tsiamis, D. S. Kalogerias, A. Ribeiro, and G. J. Pappas, "Linear Quadratic Control with Risk Constraints," *IEEE Transactions on Automatic Control*, 2023.

## Problem Formulation

Consider a discrete-time linear system:

$$x_{t+1} = A x_t + B u_t + w_{t+1}$$

$$y_t = C x_t + v_t$$

where \(x_t \in \mathbb{R}^n\) is the state, \(u_t \in \mathbb{R}^p\) is the control input, \(y_t \in \mathbb{R}^m\) is the measurement, and \(w_t, v_t\) are process and measurement noise.

The standard LQ cost minimizes expected cumulative cost:

$$\min_u \mathbb{E} \left[ x_N' Q x_N + \sum_{t=0}^{N-1} x_t' Q x_t + u_t' R u_t \right]$$

This risk-neutral formulation optimizes average performance but may be ineffective under infrequent yet significant extreme events (e.g., large wind gusts, sudden disturbances).

### Risk-Constrained Formulation

To explicitly limit tail risk, the paper introduces a predictive variance constraint:

$$\min_u \mathbb{E} \left[ x_N' Q x_N + \sum_{t=0}^{N-1} x_t' Q x_t + u_t' R u_t \right]$$

$$\text{s.t.} \quad \mathbb{E} \left[ \sum_{t=1}^{N} \left( x_t' Q x_t - \mathbb{E}[x_t' Q x_t | \mathcal{F}_{t-1}] \right)^2 \right] \leq \epsilon$$

where \(\mathcal{F}_{t-1}\) is the information available at time \(t-1\). The constraint bounds the cumulative expected predictive variance of the state penalty, forcing the controller to account for higher-order noise statistics.

## Mathematical Details

### Noise Statistics

For process noise with innovation \(\delta_t = w_t - \bar{w}\), the relevant statistics are:

- **Covariance**: \(W = \mathbb{E}[\delta_t \delta_t']\)
- **Third moment** (skewness): \(m_3 = 2Q \mathbb{E}[\delta_t \delta_t' Q \delta_t]\)
- **Fourth moment**: \(m_4 = \mathbb{E}\left[({\delta_t' Q \delta_t - \text{tr}(QW)})^2\right]\)

The third moment \(m_3\) captures asymmetry in the noise distribution. For Gaussian noise, \(m_3 = 0\).

### Risk-Aware LQR (Fully Observed Systems)

Using Lagrangian duality with multiplier \(\mu \geq 0\), the optimal control law is (Theorem 3):

$$u_t^* = K_t x_t + l_t$$

where the gain \(K_t\) satisfies a Riccati equation with **inflated** state penalty:

$$Q_\mu = Q + 4\mu Q W Q$$

The Riccati recursion (backward in time):

$$V_{t-1} = A' V_t A + Q_\mu - A' V_t B (B' V_t B + R)^{-1} B' V_t A$$

$$K_{t-1} = -(B' V_t B + R)^{-1} B' V_t A$$

with terminal condition \(V_N = Q_\mu\).

The affine term \(l_t\) accounts for noise mean and skewness:

$$\xi_{t-1} = (A + BK_{t-1})' (\xi_t + V_t \bar{w}) + \mu m_3$$

$$l_{t-1} = -(B' V_t B + R)^{-1} B' (\xi_t + V_t \bar{w})$$

with terminal condition \(\xi_N = \mu m_3\).

**Interpretation:**

1. The term \(4\mu Q W Q\) in the inflated cost penalizes state directions where both the cost \(Q\) and noise covariance \(W\) are large simultaneously.

2. The affine term \(l_t\) pushes the state away from directions where the noise has heavy tails (captured by \(m_3\)).

3. As \(\mu \to 0\), we recover risk-neutral LQR. As \(\mu \to \infty\), the controller becomes maximally risk-aware, treating noise as adversarial.

### Risk-Aware LQG (Partially Observed Systems)

For Gaussian noise with partial observations (Theorem 7), the optimal control law is:

$$u_t^* = K_t \hat{x}_{t|t} + l_t$$

where \(\hat{x}_{t|t}\) is the Kalman filter estimate. The time-varying inflated cost is:

$$Q_{\mu,t} = Q + 4\mu Q W_t Q$$

where \(W_t\) is the Kalman filter prediction error covariance:

$$W_{t+1} = A W_t A' + W - A W_t C' (C W_t C' + S)^{-1} C W_t A'$$

The Riccati equation becomes (backward):

$$V_{t-1} = (A + BK_t)' V_t (A + BK_t) + K_t' R K_t + Q_{\mu,t-1}$$

Unlike risk-neutral LQG, the control design depends on the estimation process (no certainty equivalence). However, separation still holds: the optimal estimator remains the minimum mean-square error Kalman filter.

For Gaussian noise, \(m_3 = 0\), so there is no affine term. Risk awareness is achieved purely through the inflated gain.

### Comparison with LEQG

The Linear Exponential Quadratic Gaussian (LEQG) controller minimizes:

$$\mathbb{E}\left[ \exp\left(\theta \sum_{t} x_t' Q x_t + u_t' R u_t \right) \right]$$

LEQG has two limitations addressed by this approach:

1. **Well-posedness**: LEQG requires the moment generating function to exist, excluding heavy-tailed distributions. The predictive variance approach only requires finite fourth moments.

2. **Stability**: LEQG can become unstable near the "neurotic breakdown" point where \(\theta\) is too large. The risk-aware LQR/LQG controllers remain stable for all \(\mu \geq 0\).

## Example: Flying Robot with Wind Disturbance

The simulations model a robot moving on a 2D plane as a double integrator:

$$x_{k+1} = \begin{bmatrix} 1 & T_s & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & T_s \\ 0 & 0 & 0 & 1 \end{bmatrix} x_k + \begin{bmatrix} T_s^2/2 & 0 \\ T_s & 0 \\ 0 & T_s^2/2 \\ 0 & T_s \end{bmatrix} (u_k + d_k)$$

where \(d_k\) is wind disturbance. The wind model has:
- Strong direction: bimodal mixture of \(\mathcal{N}(30, 30)\) (80%) and \(\mathcal{N}(80, 60)\) (20%)
- Weak direction: \(\mathcal{N}(0, 5)\)

This models infrequent but large wind gusts.

## Results

### LQR with Bimodal Wind

The risk-aware controller (\(\mu = 1\)):
- Achieves lower average and maximum state penalties than risk-neutral LQR
- Pushes the state away from the direction of large gusts
- Uses more control effort to regulate risky directions more strictly

![Motivating Example](figures/figure1_motivation.png)

*Figure 1: Response to a rare shock at time 6. Risk-aware controllers pre-compensate, limiting peak deviation.*

![State Penalties](figures/figure2_lqr_penalties.png)

*Figure 2: Evolution of state penalties. Risk-aware LQR limits variability during wind gusts.*

![CDF of Penalties](figures/figure3_lqr_cdf.png)

*Figure 3: Time-empirical CDF. Risk-aware LQR significantly reduces probability of large penalties.*

### LQG with Gaussian Noise

For partially observed systems with Gaussian noise:
- Risk-aware LQG regulates the state more strictly than LQG
- Increasing \(\mu\) reduces state penalties at the cost of increased control effort
- LEQG shows similar behavior for small \(\theta\) but becomes poorly behaved near the neurotic breakdown point

![LQG State Penalties](figures/figure5_lqg_penalties.png)

*Figure 5: LQG state penalty evolution. Risk-aware controller achieves tighter regulation.*

![LQG CDFs](figures/figure6_lqg_cdfs.png)

*Figure 6: Trade-off curves. Risk-aware LQG offers a wider, more intuitive range of risk-performance trade-offs than LEQG.*

## Usage

```python
import numpy as np
from risk_aware_lqr import (
    SystemParams, 
    RiskAwareLQR, 
    RiskNeutralLQR,
    compute_noise_statistics
)

# Define system
A = np.array([[1, 0.5], [0, 1]])
B = np.array([[0.125], [0.5]])
Q = np.eye(2)
R = np.array([[1]])

sys = SystemParams(A=A, B=B, Q=Q, R=R)

# Generate noise samples and compute statistics
noise_samples = np.random.randn(10000, 2) @ np.array([[1, 0], [0.5, 0.5]])
noise_stats = compute_noise_statistics(noise_samples, Q)

# Create controllers
N = 100  # horizon
mu = 1.0  # risk sensitivity

lqr = RiskNeutralLQR(sys, noise_stats, N)
risk_lqr = RiskAwareLQR(sys, noise_stats, mu, N)

# Simulate
x = np.array([1.0, 0.0])
for t in range(N):
    u_neutral = lqr.control(x, t)
    u_risk = risk_lqr.control(x, t)
    # Apply control...
```

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Simulations

```bash
python risk_aware_lqr.py
```

This generates all figures in the `figures/` directory.

## Dependencies

- numpy
- scipy
- matplotlib

## References

1. A. Tsiamis, D. S. Kalogerias, A. Ribeiro, and G. J. Pappas, "Linear Quadratic Control with Risk Constraints," *IEEE Transactions on Automatic Control*, 2023.

2. P. Whittle, "Risk-Sensitive Linear/Quadratic/Gaussian Control," *Advances in Applied Probability*, vol. 13, pp. 764-777, 1981.

3. J. L. Speyer and W. H. Chung, *Stochastic Processes, Estimation, and Control*, SIAM, 2008.

## License

MIT License


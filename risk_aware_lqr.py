"""
Risk-Aware Linear Quadratic Control

Implementation of "Linear Quadratic Control with Risk Constraints"
by Tsiamis, Kalogerias, Ribeiro, and Pappas (IEEE TAC 2023).

This module provides risk-aware variants of the classical LQR and LQG
controllers that explicitly trade off average performance against
tail-risk protection.
"""

import numpy as np
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class SystemParams:
    """
    Linear time-invariant system parameters.
    
    System dynamics: x_{k+1} = A x_k + B u_k + w_{k+1}
    Output equation: y_k = C x_k + v_k (for partially observed systems)
    
    Attributes:
        A: State transition matrix (n x n)
        B: Input matrix (n x p)
        Q: State penalty matrix (n x n), positive semi-definite
        R: Input penalty matrix (p x p), positive definite
        C: Output matrix (m x n), optional for LQG
        S: Measurement noise covariance (m x m), optional for LQG
    """
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    C: np.ndarray = None
    S: np.ndarray = None


def compute_noise_statistics(noise_samples: np.ndarray, Q: np.ndarray) -> Dict:
    """
    Compute noise statistics required for risk-aware control.
    
    For a zero-mean noise with samples delta_k, computes:
    - W: Covariance E[delta delta']
    - m3: Third moment 2Q E[delta delta' Q delta] (captures skewness)
    - m4: Fourth moment E[(delta'Q delta - tr(QW))^2]
    
    Args:
        noise_samples: Array of shape (num_samples, n) with noise realizations
        Q: State penalty matrix for computing weighted moments
        
    Returns:
        Dictionary with keys 'w_bar', 'W', 'm3', 'm4'
    """
    n = noise_samples.shape[1]
    
    # Mean (should be approximately zero for centered noise)
    w_bar = np.mean(noise_samples, axis=0)
    
    # Centered samples
    delta = noise_samples - w_bar
    
    # Covariance matrix W = E[delta delta']
    W = np.cov(delta, rowvar=False)
    if W.ndim == 0:
        W = np.array([[W]])
    
    # Third moment m3 = 2Q E[delta delta' Q delta]
    # This term captures the skewness of the noise distribution
    m3 = np.zeros(n)
    for sample in delta:
        # Compute delta delta' Q delta = (delta' Q delta) * delta
        quad_form = sample @ Q @ sample
        m3 += 2 * Q @ (np.outer(sample, sample) @ Q @ sample)
    m3 /= len(delta)
    
    # Fourth moment m4 = E[(delta'Q delta - tr(QW))^2]
    trace_QW = np.trace(Q @ W)
    m4 = 0.0
    for sample in delta:
        quad_form = sample @ Q @ sample
        m4 += (quad_form - trace_QW) ** 2
    m4 /= len(delta)
    
    return {
        'w_bar': w_bar,
        'W': W,
        'm3': m3,
        'm4': m4
    }


class RiskAwareLQR:
    """
    Risk-Aware Linear Quadratic Regulator.
    
    Implements Theorem 3 from the paper. The optimal control law is:
        u_t = K_t x_t + l_t
    
    where K_t comes from a Riccati equation with inflated cost matrix
    Q_mu = Q + 4*mu*Q*W*Q, and l_t accounts for noise mean and skewness.
    
    The parameter mu controls the risk-performance tradeoff:
    - mu = 0: Risk-neutral LQR
    - mu > 0: Increasing risk-awareness
    - mu -> inf: Maximally risk-aware (treats noise as adversarial)
    
    Args:
        sys: System parameters
        noise_stats: Dictionary with noise statistics from compute_noise_statistics
        mu: Risk-sensitivity parameter (>= 0)
        N: Planning horizon
    """
    
    def __init__(self, sys: SystemParams, noise_stats: Dict, mu: float, N: int):
        self.sys = sys
        self.noise_stats = noise_stats
        self.mu = mu
        self.N = N
        
        # Inflated state penalty: Q_mu = Q + 4*mu*Q*W*Q (Lemma 1)
        self.Q_mu = sys.Q + 4 * mu * sys.Q @ noise_stats['W'] @ sys.Q
        self._solve_riccati()
    
    def _solve_riccati(self):
        """Solve the risk-aware Riccati difference equation (Theorem 3)."""
        A, B, R = self.sys.A, self.sys.B, self.sys.R
        Q_mu = self.Q_mu
        w_bar = self.noise_stats['w_bar']
        m3 = self.noise_stats['m3']
        n = A.shape[0]
        
        # Initialize storage for time-varying gains
        self.V = [None] * (self.N + 1)   # Value function matrices
        self.K = [None] * self.N          # State feedback gains
        self.xi = [None] * (self.N + 1)   # Affine term accumulators
        self.l = [None] * self.N          # Affine control terms
        
        # Terminal conditions: V_N = Q_mu, xi_N = mu * m3
        self.V[self.N] = Q_mu.copy()
        self.xi[self.N] = self.mu * m3.copy()
        
        # Backward recursion (equations 13-17)
        for t in range(self.N - 1, -1, -1):
            V_next = self.V[t + 1]
            xi_next = self.xi[t + 1]
            
            # Gain computation (equation 14)
            inv_term = np.linalg.inv(B.T @ V_next @ B + R)
            K_t = -inv_term @ B.T @ V_next @ A
            self.K[t] = K_t
            
            # Affine term (equation 16)
            l_t = -inv_term @ B.T @ (xi_next + V_next @ w_bar)
            self.l[t] = l_t
            
            # Value function update (equation 13)
            A_cl = A + B @ K_t
            self.V[t] = A.T @ V_next @ A + Q_mu - A.T @ V_next @ B @ inv_term @ B.T @ V_next @ A
            
            # Affine accumulator update (equation 15)
            self.xi[t] = A_cl.T @ (xi_next + V_next @ w_bar) + self.mu * m3
    
    def control(self, x: np.ndarray, t: int) -> np.ndarray:
        """
        Compute the risk-aware control action.
        
        Args:
            x: Current state
            t: Current time step
            
        Returns:
            Control input u_t = K_t x_t + l_t
        """
        if t >= self.N:
            return np.zeros(self.sys.B.shape[1])
        return self.K[t] @ x + self.l[t]


class RiskNeutralLQR:
    """
    Standard risk-neutral LQR controller.
    
    This is the classical LQR solution that minimizes expected quadratic cost
    without explicit consideration of tail risk.
    
    Args:
        sys: System parameters
        noise_stats: Dictionary with noise statistics
        N: Planning horizon
    """
    
    def __init__(self, sys: SystemParams, noise_stats: Dict, N: int):
        self.sys = sys
        self.noise_stats = noise_stats
        self.N = N
        
        A, B, Q, R = sys.A, sys.B, sys.Q, sys.R
        w_bar = noise_stats['w_bar']
        
        # Solve infinite-horizon DARE for reference
        self.P = solve_discrete_are(A, B, Q, R)
        self.K_inf = -np.linalg.inv(B.T @ self.P @ B + R) @ B.T @ self.P @ A
        
        # Steady-state affine term
        A_cl = A + B @ self.K_inf
        inv_term = np.linalg.inv(B.T @ self.P @ B + R)
        self.l_inf = -inv_term @ B.T @ np.linalg.inv(np.eye(A.shape[0]) - A_cl.T) @ self.P @ w_bar
        
        self._solve_riccati()
    
    def _solve_riccati(self):
        """Solve the standard Riccati difference equation."""
        A, B, Q, R = self.sys.A, self.sys.B, self.sys.Q, self.sys.R
        w_bar = self.noise_stats['w_bar']
        
        self.V = [None] * (self.N + 1)
        self.K = [None] * self.N
        self.xi = [None] * (self.N + 1)
        self.l = [None] * self.N
        
        self.V[self.N] = Q.copy()
        self.xi[self.N] = np.zeros(A.shape[0])
        
        for t in range(self.N - 1, -1, -1):
            V_next = self.V[t + 1]
            xi_next = self.xi[t + 1]
            
            inv_term = np.linalg.inv(B.T @ V_next @ B + R)
            K_t = -inv_term @ B.T @ V_next @ A
            self.K[t] = K_t
            
            l_t = -inv_term @ B.T @ (xi_next + V_next @ w_bar)
            self.l[t] = l_t
            
            A_cl = A + B @ K_t
            self.V[t] = A.T @ V_next @ A + Q - A.T @ V_next @ B @ inv_term @ B.T @ V_next @ A
            self.xi[t] = A_cl.T @ (xi_next + V_next @ w_bar)
    
    def control(self, x: np.ndarray, t: int) -> np.ndarray:
        """Compute the risk-neutral control action."""
        if t >= self.N:
            return np.zeros(self.sys.B.shape[1])
        return self.K[t] @ x + self.l[t]


class LEQGController:
    """
    Linear Exponential Quadratic Gaussian (LEQG) controller.
    
    Minimizes E[exp(theta * cost)] which penalizes high-cost events
    exponentially. Valid only when the moment generating function exists,
    which requires theta < theta_critical (neurotic breakdown point).
    
    Note: LEQG assumes Gaussian noise; using it with non-Gaussian noise
    is only a heuristic comparison.
    
    Args:
        sys: System parameters
        W: Process noise covariance
        theta: Exponential risk-sensitivity parameter
        N: Planning horizon
    """
    
    def __init__(self, sys: SystemParams, W: np.ndarray, theta: float, N: int):
        self.sys = sys
        self.W = W
        self.theta = theta
        self.N = N
        self._solve_riccati()
    
    def _solve_riccati(self):
        """Solve the LEQG Riccati equation."""
        A, B, Q, R = self.sys.A, self.sys.B, self.sys.Q, self.sys.R
        W = self.W
        theta = self.theta
        n = A.shape[0]
        
        self.V = [None] * (self.N + 1)
        self.K = [None] * self.N
        
        self.V[self.N] = Q.copy()
        
        for t in range(self.N - 1, -1, -1):
            V_next = self.V[t + 1]
            
            # LEQG modification: Phi = I - theta * W * V
            Phi = np.eye(n) - theta * W @ V_next
            
            # Check for neurotic breakdown
            if np.min(np.linalg.eigvals(Phi)) <= 0:
                Phi = np.eye(n) - 0.99 * theta * W @ V_next
            
            Phi_inv = np.linalg.inv(Phi)
            V_tilde = V_next @ Phi_inv
            
            inv_term = np.linalg.inv(B.T @ V_tilde @ B + R)
            K_t = -inv_term @ B.T @ V_tilde @ A
            self.K[t] = K_t
            
            A_cl = A + B @ K_t
            self.V[t] = Q + A_cl.T @ V_tilde @ A_cl + K_t.T @ R @ K_t
    
    def control(self, x: np.ndarray, t: int) -> np.ndarray:
        """Compute the LEQG control action (no affine term for zero-mean noise)."""
        if t >= self.N:
            return np.zeros(self.sys.B.shape[1])
        return self.K[t] @ x


class RiskAwareLQG:
    """
    Risk-Aware Linear Quadratic Gaussian controller.
    
    Implements Theorem 7 from the paper for partially observed systems
    with Gaussian noise. The time-varying inflated cost is:
        Q_{mu,t} = Q + 4*mu*Q*W_t*Q
    
    where W_t is the Kalman filter prediction error covariance.
    
    Unlike the fully-observed case, the control design depends on the
    estimation process (no certainty equivalence), though separation
    still holds for the estimator design.
    
    Args:
        sys: System parameters (must include C and S for LQG)
        W_process: Process noise covariance
        mu: Risk-sensitivity parameter
        N: Planning horizon
    """
    
    def __init__(self, sys: SystemParams, W_process: np.ndarray, mu: float, N: int):
        self.sys = sys
        self.mu = mu
        self.N = N
        self.W_process = W_process
        self._run_kalman_filter()
        self._solve_riccati()
    
    def _run_kalman_filter(self):
        """Compute the Kalman filter prediction error covariance sequence."""
        A, C = self.sys.A, self.sys.C
        W = self.W_process
        S = self.sys.S if self.sys.S is not None else np.eye(C.shape[0])
        
        n = A.shape[0]
        self.W_kf = [np.zeros((n, n))]
        
        W_t = np.zeros((n, n))
        for t in range(self.N):
            # Kalman gain
            if np.linalg.matrix_rank(C @ W_t @ C.T + S) < C.shape[0]:
                K_kf = np.zeros((n, C.shape[0]))
            else:
                K_kf = W_t @ C.T @ np.linalg.inv(C @ W_t @ C.T + S)
            
            # Updated covariance after measurement
            W_t_updated = (np.eye(n) - K_kf @ C) @ W_t
            
            # Prediction covariance for next step
            W_t = A @ W_t_updated @ A.T + W
            self.W_kf.append(W_t.copy())
    
    def _solve_riccati(self):
        """Solve the risk-aware LQG Riccati equation (Theorem 7, equations 30-33)."""
        A, B, Q, R = self.sys.A, self.sys.B, self.sys.Q, self.sys.R
        
        self.V = [None] * (self.N + 1)
        self.K = [None] * self.N
        
        # Terminal condition with time-varying Q_mu
        Q_mu_N = Q + 4 * self.mu * Q @ self.W_kf[self.N - 1] @ Q
        self.V[self.N - 1] = Q_mu_N.copy()
        
        for t in range(self.N - 2, -1, -1):
            V_next = self.V[t + 1]
            Q_mu_t = Q + 4 * self.mu * Q @ self.W_kf[t] @ Q
            
            inv_term = np.linalg.inv(B.T @ V_next @ B + R)
            K_t = -inv_term @ B.T @ V_next @ A
            self.K[t] = K_t
            
            A_cl = A + B @ K_t
            self.V[t] = Q_mu_t + A_cl.T @ V_next @ A_cl + K_t.T @ R @ K_t
    
    def control(self, x_hat: np.ndarray, t: int) -> np.ndarray:
        """
        Compute the risk-aware LQG control action.
        
        Args:
            x_hat: Current state estimate (from Kalman filter)
            t: Current time step
            
        Returns:
            Control input u_t = K_t * x_hat (no affine term for Gaussian noise)
        """
        if t >= self.N - 1 or self.K[t] is None:
            return np.zeros(self.sys.B.shape[1])
        return self.K[t] @ x_hat


def create_flying_robot_system(Ts: float = 0.5) -> SystemParams:
    """
    Create the double integrator system for the flying robot example.
    
    Models a robot moving on a 2D plane with position and velocity states.
    The state is [pos_x, vel_x, pos_y, vel_y] and control is acceleration.
    
    Args:
        Ts: Sampling time
        
    Returns:
        SystemParams for the double integrator
    """
    A = np.array([
        [1, Ts, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, Ts],
        [0, 0, 0, 1]
    ])
    
    B = np.array([
        [Ts**2 / 2, 0],
        [Ts, 0],
        [0, Ts**2 / 2],
        [0, Ts]
    ])
    
    Q = np.diag([1, 0.1, 2, 0.1])
    R = np.eye(2)
    
    return SystemParams(A=A, B=B, Q=Q, R=R)


def generate_bimodal_wind(N: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate bimodal wind disturbance sequence.
    
    The wind model has:
    - Strong direction: mixture of N(30, 30) (80%) and N(80, 60) (20%)
    - Weak direction: N(0, 5)
    
    This represents infrequent but strong wind gusts.
    
    Args:
        N: Number of time steps
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (wind samples d, expected wind E[d])
    """
    if seed is not None:
        np.random.seed(seed)
    
    mixture_weights = np.random.rand(N)
    d1 = np.where(
        mixture_weights < 0.8,
        np.random.normal(30, np.sqrt(30), N),
        np.random.normal(80, np.sqrt(60), N)
    )
    d2 = np.random.normal(0, np.sqrt(5), N)
    d = np.column_stack([d1, d2])
    
    E_d = np.array([0.8 * 30 + 0.2 * 80, 0])
    
    return d, E_d


def compute_bimodal_noise_stats(Q: np.ndarray, B: np.ndarray, 
                                 num_samples: int = 100000) -> Dict:
    """
    Compute noise statistics for the bimodal wind distribution.
    
    Args:
        Q: State penalty matrix
        B: Input matrix
        num_samples: Number of Monte Carlo samples
        
    Returns:
        Noise statistics dictionary
    """
    d, E_d = generate_bimodal_wind(num_samples)
    w_samples = (d - E_d) @ B.T
    return compute_noise_statistics(w_samples, Q)


def simulate_system(sys: SystemParams, controller, noise_sequence: np.ndarray,
                    x0: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the closed-loop system.
    
    Args:
        sys: System parameters
        controller: Controller object with control(x, t) method
        noise_sequence: Pre-generated noise sequence
        x0: Initial state
        N: Simulation horizon
        
    Returns:
        Tuple of (states, inputs) arrays
    """
    n = sys.A.shape[0]
    p = sys.B.shape[1]
    
    states = np.zeros((N + 1, n))
    inputs = np.zeros((N, p))
    
    states[0] = x0
    
    for t in range(N):
        u_t = controller.control(states[t], t)
        inputs[t] = u_t
        states[t + 1] = sys.A @ states[t] + sys.B @ u_t + noise_sequence[t]
    
    return states, inputs


def run_lqr_simulation():
    """
    Run the LQR simulation from Section VIII of the paper.
    
    Compares risk-neutral LQR, risk-aware LQR, and LEQG controllers
    on the flying robot example with bimodal wind disturbance.
    
    Returns:
        Dictionary with simulation results
    """
    print("=" * 60)
    print("Risk-Aware LQR Simulation (Section VIII)")
    print("=" * 60)
    
    Ts = 0.5
    sys = create_flying_robot_system(Ts)
    N = 5000
    
    print("\nComputing noise statistics...")
    noise_stats = compute_bimodal_noise_stats(sys.Q, sys.B, num_samples=200000)
    
    print("\nCreating controllers...")
    lqr = RiskNeutralLQR(sys, noise_stats, N)
    risk_lqr = RiskAwareLQR(sys, noise_stats, mu=1.0, N=N)
    leqg = LEQGController(sys, noise_stats['W'], theta=0.001, N=N)
    
    print("\nSteady-state gains:")
    print(f"  LQR K:\n{lqr.K[0]}")
    print(f"  Risk-Aware LQR K (mu=1):\n{risk_lqr.K[0]}")
    
    print("\nGenerating noise sequence...")
    d, E_d = generate_bimodal_wind(N, seed=42)
    w_sequence = (d - E_d) @ sys.B.T
    
    x0 = np.zeros(4)
    
    print("\nSimulating systems...")
    states_lqr, inputs_lqr = simulate_system(sys, lqr, w_sequence, x0, N)
    states_risk, inputs_risk = simulate_system(sys, risk_lqr, w_sequence, x0, N)
    states_leqg, inputs_leqg = simulate_system(sys, leqg, w_sequence, x0, N)
    
    penalties_lqr = np.array([x @ sys.Q @ x for x in states_lqr])
    penalties_risk = np.array([x @ sys.Q @ x for x in states_risk])
    penalties_leqg = np.array([x @ sys.Q @ x for x in states_leqg])
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"\nAverage state penalty E[x'Qx]:")
    print(f"  LQR:            {np.mean(penalties_lqr):.2f}")
    print(f"  Risk-Aware LQR: {np.mean(penalties_risk):.2f}")
    print(f"  LEQG:           {np.mean(penalties_leqg):.2f}")
    
    print(f"\nMax state penalty:")
    print(f"  LQR:            {np.max(penalties_lqr):.2f}")
    print(f"  Risk-Aware LQR: {np.max(penalties_risk):.2f}")
    print(f"  LEQG:           {np.max(penalties_leqg):.2f}")
    
    return {
        'states_lqr': states_lqr,
        'states_risk': states_risk,
        'states_leqg': states_leqg,
        'inputs_lqr': inputs_lqr,
        'inputs_risk': inputs_risk,
        'inputs_leqg': inputs_leqg,
        'penalties_lqr': penalties_lqr,
        'penalties_risk': penalties_risk,
        'penalties_leqg': penalties_leqg,
        'sys': sys,
        'N': N
    }


def run_lqg_simulation():
    """
    Run the LQG simulation from Section VIII.A of the paper.
    
    Compares risk-neutral LQG, risk-aware LQG with various mu values,
    and LEQG controllers on a partially-observed system.
    
    Returns:
        Dictionary with simulation results
    """
    print("\n" + "=" * 60)
    print("Risk-Aware LQG Simulation (Section VIII.A)")
    print("=" * 60)
    
    Ts = 0.5
    A = np.array([
        [1, Ts, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, Ts],
        [0, 0, 0, 1]
    ])
    B = np.array([
        [Ts**2 / 2, 0],
        [Ts, 0],
        [0, Ts**2 / 2],
        [0, Ts]
    ])
    Q = np.diag([1, 0.5, 2, 0.5])
    R = np.eye(2)
    C = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])
    S = np.array([
        [5, 2],
        [2, 2]
    ])
    
    sys = SystemParams(A=A, B=B, Q=Q, R=R, C=C, S=S)
    N = 3000
    
    # Process noise covariance
    W_process = B @ np.diag([30, 5]) @ B.T
    
    noise_stats = {
        'w_bar': np.zeros(4),
        'W': W_process,
        'm3': np.zeros(4),
        'm4': 0
    }
    
    print("\nCreating LQG controllers...")
    lqg = RiskNeutralLQR(sys, noise_stats, N)
    risk_lqg_05 = RiskAwareLQG(sys, W_process, mu=0.5, N=N)
    risk_lqg_1 = RiskAwareLQG(sys, W_process, mu=1.0, N=N)
    risk_lqg_100 = RiskAwareLQG(sys, W_process, mu=100.0, N=N)
    leqg_003 = LEQGController(sys, noise_stats['W'], theta=0.003, N=N)
    leqg_006 = LEQGController(sys, W_process, theta=0.006, N=N)
    leqg_01 = LEQGController(sys, W_process, theta=0.01, N=N)
    
    print("\nSimulating LQG systems...")
    np.random.seed(42)
    
    d1 = np.random.normal(0, np.sqrt(30), N)
    d2 = np.random.normal(0, np.sqrt(5), N)
    d = np.column_stack([d1, d2])
    w_sequence = d @ B.T
    
    x0 = np.zeros(4)
    
    states_lqg, inputs_lqg = simulate_system(sys, lqg, w_sequence, x0, N)
    states_risk_05, inputs_risk_05 = simulate_system(sys, risk_lqg_05, w_sequence, x0, N)
    states_risk_1, inputs_risk_1 = simulate_system(sys, risk_lqg_1, w_sequence, x0, N)
    states_risk_100, inputs_risk_100 = simulate_system(sys, risk_lqg_100, w_sequence, x0, N)
    states_leqg_003, inputs_leqg_003 = simulate_system(sys, leqg_003, w_sequence, x0, N)
    states_leqg_006, inputs_leqg_006 = simulate_system(sys, leqg_006, w_sequence, x0, N)
    states_leqg_01, inputs_leqg_01 = simulate_system(sys, leqg_01, w_sequence, x0, N)
    
    results = {
        'Q': Q, 'R': R,
        'penalties_lqg': np.array([x @ Q @ x for x in states_lqg]),
        'penalties_risk_05': np.array([x @ Q @ x for x in states_risk_05]),
        'penalties_risk_1': np.array([x @ Q @ x for x in states_risk_1]),
        'penalties_risk_100': np.array([x @ Q @ x for x in states_risk_100]),
        'penalties_leqg_003': np.array([x @ Q @ x for x in states_leqg_003]),
        'penalties_leqg_006': np.array([x @ Q @ x for x in states_leqg_006]),
        'penalties_leqg_01': np.array([x @ Q @ x for x in states_leqg_01]),
        'input_penalties_lqg': np.array([u @ R @ u for u in inputs_lqg]),
        'input_penalties_risk_05': np.array([u @ R @ u for u in inputs_risk_05]),
        'input_penalties_risk_1': np.array([u @ R @ u for u in inputs_risk_1]),
        'input_penalties_risk_100': np.array([u @ R @ u for u in inputs_risk_100]),
        'input_penalties_leqg_003': np.array([u @ R @ u for u in inputs_leqg_003]),
        'input_penalties_leqg_006': np.array([u @ R @ u for u in inputs_leqg_006]),
        'input_penalties_leqg_01': np.array([u @ R @ u for u in inputs_leqg_01]),
    }
    
    print("\n" + "=" * 60)
    print("LQG Results Summary")
    print("=" * 60)
    print(f"\nAverage state penalty E[x'Qx]:")
    print(f"  LQG:                   {np.mean(results['penalties_lqg']):.2f}")
    print(f"  Risk-Aware LQG mu=0.5: {np.mean(results['penalties_risk_05']):.2f}")
    print(f"  Risk-Aware LQG mu=1:   {np.mean(results['penalties_risk_1']):.2f}")
    print(f"  Risk-Aware LQG mu=100: {np.mean(results['penalties_risk_100']):.2f}")
    print(f"  LEQG theta=0.006:      {np.mean(results['penalties_leqg_006']):.2f}")
    
    return results


# Plotting functions

def setup_paper_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 1.2,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,
        'figure.dpi': 150,
    })


def plot_figure1(save_path: str):
    """
    Reproduce Figure 1: Motivating 1D example.
    
    Shows the response of risk-neutral, risk-aware, and maximally
    risk-aware controllers to a rare but large shock at time 6.
    """
    setup_paper_style()
    
    beta = 8.0
    N = 15
    
    w_sequence = np.zeros(N)
    w_sequence[5] = beta
    
    E_w = 1.0
    
    # Risk-neutral: u = -x - E[w]
    x_lqr = np.zeros(N + 1)
    for t in range(N):
        u_t = -x_lqr[t] - E_w
        x_lqr[t + 1] = x_lqr[t] + u_t + w_sequence[t]
    
    # Risk-aware (mu = 1): u = -x - E[w] - mu/(1+2*mu) * (beta - 2)
    mu = 1.0
    x_risk = np.zeros(N + 1)
    for t in range(N):
        u_t = -x_risk[t] - E_w - mu / (1 + 2*mu) * (beta - 2)
        x_risk[t + 1] = x_risk[t] + u_t + w_sequence[t]
    
    # Maximally risk-aware: u = -x - beta/2
    x_max_risk = np.zeros(N + 1)
    for t in range(N):
        u_t = -x_max_risk[t] - beta / 2
        x_max_risk[t + 1] = x_max_risk[t] + u_t + w_sequence[t]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    t_range = np.arange(N + 1)
    
    ax.plot(t_range, x_lqr, color='#1f77b4', linestyle='--', 
            label='Risk-Neutral', linewidth=1.5)
    ax.plot(t_range, x_risk, color='#d62728', linestyle='-', 
            label='Risk-Aware', linewidth=1.5)
    ax.plot(t_range, x_max_risk, color='#e6a000', linestyle=':', 
            label='Maximally Risk-Aware', linewidth=1.5)
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2, linewidth=0.5)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x_t$')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim([0, 14])
    ax.set_ylim([-4, 8])
    ax.set_yticks([-4, 0, 4, 8])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/figure1_motivation.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_figure2(results: Dict, save_path: str):
    """
    Reproduce Figure 2: Evolution of state penalties x'Qx.
    """
    setup_paper_style()
    
    fig, ax = plt.subplots(figsize=(7, 4))
    t_range = np.arange(51)
    
    ax.plot(t_range, results['penalties_lqr'][:51], color='#1f77b4', 
            linestyle='--', label='LQR', linewidth=1.2)
    ax.plot(t_range, results['penalties_risk'][:51], color='#d62728', 
            linestyle='-', label=r'Risk-Aware LQR, $\mu = 1$', linewidth=1.2)
    ax.plot(t_range, results['penalties_leqg'][:51], color='#e6a000', 
            linestyle=':', marker='o', markersize=3, 
            label=r'LEQG, $\theta$=0.001', linewidth=1.2)
    
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r"$x_t'Qx_t$")
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 800])
    ax.set_yticks([0, 200, 400, 600, 800])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/figure2_lqr_penalties.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_figure3(results: Dict, save_path: str):
    """
    Reproduce Figure 3: Time-empirical CDF of state penalties.
    """
    setup_paper_style()
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    sorted_lqr = np.sort(results['penalties_lqr'])
    sorted_risk = np.sort(results['penalties_risk'])
    sorted_leqg = np.sort(results['penalties_leqg'])
    
    n_points = len(sorted_lqr)
    cdf_y = np.arange(1, n_points + 1) / n_points
    
    ax.semilogx(sorted_lqr, cdf_y, color='#1f77b4', linestyle='--', 
                label='LQR', linewidth=1.2)
    ax.semilogx(sorted_risk, cdf_y, color='#d62728', linestyle='-', 
                label=r'Risk-Aware LQR, $\mu = 1$', linewidth=1.2)
    ax.semilogx(sorted_leqg, cdf_y, color='#e6a000', linestyle=':', 
                label=r'LEQG, $\theta$=0.001', linewidth=1.2)
    
    ax.set_xlabel(r"State penalty values $x'Qx$")
    ax.set_ylabel('Time-empirical CDF')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim([1e0, 1e3])
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/figure3_lqr_cdf.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_figure4(results: Dict, save_path: str):
    """
    Reproduce Figure 4: State x_{k,1} and input u_{k,1} evolution.
    """
    setup_paper_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
    t_range = np.arange(51)
    
    # State evolution
    ax1.plot(t_range, results['states_lqr'][:51, 0], color='#1f77b4', 
             linestyle='--', label='LQR', linewidth=1.2)
    ax1.plot(t_range, results['states_risk'][:51, 0], color='#d62728', 
             linestyle='-', label=r'Risk-Aware LQR, $\mu = 1$', linewidth=1.2)
    ax1.plot(t_range, results['states_leqg'][:51, 0], color='#e6a000', 
             linestyle=':', marker='o', markersize=3, 
             label=r'LEQG, $\theta$=0.001', linewidth=1.2)
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$x_{t,1}$')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_xlim([0, 50])
    ax1.set_ylim([-20, 30])
    ax1.set_yticks([-20, -10, 0, 10, 20, 30])
    ax1.grid(True, alpha=0.3)
    
    # Input evolution
    ax2.plot(t_range[:-1], results['inputs_lqr'][:50, 0], color='#1f77b4', 
             linestyle='--', label='LQR', linewidth=1.2)
    ax2.plot(t_range[:-1], results['inputs_risk'][:50, 0], color='#d62728', 
             linestyle='-', label=r'Risk-Aware LQR, $\mu = 1$', linewidth=1.2)
    ax2.plot(t_range[:-1], results['inputs_leqg'][:50, 0], color='#e6a000', 
             linestyle=':', marker='o', markersize=3, 
             label=r'LEQG, $\theta$=0.001', linewidth=1.2)
    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(r'$u_{t,1}$')
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.set_xlim([0, 50])
    ax2.set_ylim([-80, 40])
    ax2.set_yticks([-80, -60, -40, -20, 0, 20, 40])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/figure4_lqr_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_figure5(results: Dict, save_path: str):
    """
    Reproduce Figure 5: LQG state penalties evolution.
    """
    setup_paper_style()
    
    fig, ax = plt.subplots(figsize=(7, 4))
    t_range = np.arange(51)
    
    ax.plot(t_range, results['penalties_lqg'][:51], color='#8b0000', 
            linestyle='-', marker='^', markersize=4, markevery=5, 
            label='LQG', linewidth=1.2)
    ax.plot(t_range, results['penalties_risk_1'][:51], color='#d65f00', 
            linestyle='-', label=r'Risk-Aware LQG, $\mu = 1$', linewidth=1.2)
    ax.plot(t_range, results['penalties_leqg_003'][:51], color='#228b22', 
            linestyle='--', marker='o', markersize=3, markevery=5, 
            label=r'LEQG, $\theta$=0.003', linewidth=1.2)
    
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r"$x_t'Qx_t$")
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 200])
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/figure5_lqg_penalties.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_figure6(results: Dict, save_path: str):
    """
    Reproduce Figure 6: Time-empirical CDFs for state and input penalties.
    """
    setup_paper_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))
    
    n = len(results['penalties_lqg'])
    cdf_y = np.arange(1, n + 1) / n
    
    # State penalty CDF
    ax1.semilogx(np.sort(results['penalties_risk_05']), cdf_y, color='#d65f00', 
                 linestyle='-', label=r'Risk-Aware LQG $\mu$=0.5', linewidth=1.2)
    ax1.semilogx(np.sort(results['penalties_risk_100']), cdf_y, color='#e6a000', 
                 linestyle='-', label=r'Risk-Aware LQG $\mu$=100', linewidth=1.2)
    ax1.semilogx(np.sort(results['penalties_leqg_006']), cdf_y, color='#228b22', 
                 linestyle='--', label=r'LEQG, $\theta$=0.006', linewidth=1.2)
    ax1.semilogx(np.sort(results['penalties_leqg_01']), cdf_y, color='#00bfff', 
                 linestyle='--', label=r'LEQG, $\theta$=0.01', linewidth=1.2)
    ax1.semilogx(np.sort(results['penalties_lqg']), cdf_y, color='#8b0000', 
                 linestyle='-', marker='^', markersize=3, markevery=150, 
                 label='LQG', linewidth=1.2)
    
    ax1.set_xlabel(r"State penalty values $x'Qx$")
    ax1.set_ylabel('Time-empirical CDF')
    ax1.legend(loc='lower right', framealpha=0.9, fontsize=8)
    ax1.set_xlim([1e1, 1e3])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    
    # Input penalty CDF
    n_inputs = len(results['input_penalties_lqg'])
    cdf_y_inputs = np.arange(1, n_inputs + 1) / n_inputs
    
    ax2.semilogx(np.sort(results['input_penalties_risk_05']), cdf_y_inputs, 
                 color='#d65f00', linestyle='-', 
                 label=r'Risk-Aware LQG $\mu$=0.5', linewidth=1.2)
    ax2.semilogx(np.sort(results['input_penalties_risk_100']), cdf_y_inputs, 
                 color='#e6a000', linestyle='-', 
                 label=r'Risk-Aware LQG $\mu$=100', linewidth=1.2)
    ax2.semilogx(np.sort(results['input_penalties_leqg_006']), cdf_y_inputs, 
                 color='#228b22', linestyle='--', 
                 label=r'LEQG, $\theta$=0.006', linewidth=1.2)
    ax2.semilogx(np.sort(results['input_penalties_leqg_01']), cdf_y_inputs, 
                 color='#00bfff', linestyle='--', 
                 label=r'LEQG, $\theta$=0.01', linewidth=1.2)
    ax2.semilogx(np.sort(results['input_penalties_lqg']), cdf_y_inputs, 
                 color='#8b0000', linestyle='-', marker='^', markersize=3, 
                 markevery=150, label='LQG', linewidth=1.2)
    
    ax2.set_xlabel(r"Input penalty values $u'Ru$")
    ax2.set_ylabel('Time-empirical CDF')
    ax2.legend(loc='lower right', framealpha=0.9, fontsize=8)
    ax2.set_xlim([1e1, 1e4])
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/figure6_lqg_cdfs.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import os
    
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Figure 1 (motivation)...")
    plot_figure1(output_dir)
    
    lqr_results = run_lqr_simulation()
    
    print("\nGenerating Figure 2 (LQR penalties)...")
    plot_figure2(lqr_results, output_dir)
    
    print("Generating Figure 3 (LQR CDF)...")
    plot_figure3(lqr_results, output_dir)
    
    print("Generating Figure 4 (LQR trajectories)...")
    plot_figure4(lqr_results, output_dir)
    
    lqg_results = run_lqg_simulation()
    
    print("\nGenerating Figure 5 (LQG penalties)...")
    plot_figure5(lqg_results, output_dir)
    
    print("Generating Figure 6 (LQG CDFs)...")
    plot_figure6(lqg_results, output_dir)
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}/")
    print("=" * 60)


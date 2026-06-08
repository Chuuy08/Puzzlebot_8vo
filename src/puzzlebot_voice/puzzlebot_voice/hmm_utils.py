import numpy as np

class HMMUtils:
    def __init__(self, n_states, n_symbols):
        self.N = n_states
        self.M = n_symbols
        
        # A: State transition probabilities (left-right model)
        self.A = np.zeros((n_states, n_states))
        for i in range(n_states):
            if i < n_states - 1:
                self.A[i, i] = 0.5
                self.A[i, i+1] = 0.5
            else:
                self.A[i, i] = 1.0
        
        # B: Observation symbol probabilities (Emission matrix)
        self.B = np.full((n_states, n_symbols), 1.0 / n_symbols)
        
        # Pi: Initial state distribution (always start in state 0)
        self.pi = np.zeros(n_states)
        self.pi[0] = 1.0

    def _forward_scaled(self, obs_seq):
        """Forward algorithm with scaling to prevent underflow."""
        T = len(obs_seq)
        alpha = np.zeros((T, self.N))
        scale = np.zeros(T)

        alpha[0, :] = self.pi * self.B[:, obs_seq[0]]
        scale[0] = alpha[0, :].sum() + 1e-300
        alpha[0, :] /= scale[0]

        for t in range(1, T):
            alpha[t, :] = (alpha[t-1, :] @ self.A) * self.B[:, obs_seq[t]]
            scale[t] = alpha[t, :].sum() + 1e-300
            alpha[t, :] /= scale[t]

        return alpha, scale

    def forward(self, obs_seq):
        """Problem 1: Calculate Log P(O|lambda) using scaled Forward Algorithm."""
        if len(obs_seq) == 0:
            return -np.inf
        _, scale = self._forward_scaled(obs_seq)
        return np.sum(np.log(scale + 1e-300))

    def train(self, sequences, max_iter=20):
        """Problem 3: Viterbi training to estimate emission matrix B."""
        for iteration in range(max_iter):
            new_B = np.ones((self.N, self.M)) * 1e-6  # Laplace smoothing
            for seq in sequences:
                states = self.viterbi(seq)
                for t, s in enumerate(states):
                    new_B[s, seq[t]] += 1
            new_B /= new_B.sum(axis=1, keepdims=True)

            # Stop early if B converged
            if np.max(np.abs(new_B - self.B)) < 1e-4:
                print(f"Converged at iteration {iteration}")
                break
            self.B = new_B

    def viterbi(self, obs_seq):
        """Problem 2: Find the most likely state sequence (log-domain, vectorized)."""
        T = len(obs_seq)
        if T == 0:
            return np.array([], dtype=int)

        log_A  = np.log(self.A + 1e-300)
        log_B  = np.log(self.B + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        delta = np.zeros((T, self.N))
        phi   = np.zeros((T, self.N), dtype=int)

        # Initialization
        delta[0, :] = log_pi + log_B[:, obs_seq[0]]

        # Recursion — vectorized: no inner loop over j
        for t in range(1, T):
            trans = delta[t-1, :, None] + log_A   # (N, N): trans[i,j] = delta[t-1,i] + logA[i,j]
            phi[t, :]   = np.argmax(trans, axis=0) # best predecessor for each state j
            delta[t, :] = np.max(trans, axis=0) + log_B[:, obs_seq[t]]

        # Backtracking
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1, :])
        for t in range(T-2, -1, -1):
            states[t] = phi[t+1, states[t+1]]

        return states
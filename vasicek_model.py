import numpy as np

class VasicekModel:
    def __init__(self, alpha, beta, sigma, r0):
        self.alpha = alpha  # Mean reversion speed
        self.beta = beta    # Long-term mean rate
        self.sigma = sigma  # Volatility
        self.r0 = r0        # Initial interest rate

    def simulate_path(self, time_horizon, steps):
        dt = time_horizon / steps
        rates = np.zeros(steps)
        rates[0] = self.r0

        for i in range(1, steps):
            dr = self.alpha * (self.beta - rates[i - 1]) * dt + self.sigma * np.sqrt(dt) * np.random.normal()
            rates[i] = rates[i - 1] + dr

        return rates
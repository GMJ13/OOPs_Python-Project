import numpy as np
from scipy.optimize import fsolve

class BDTModel:
    def __init__(self, volatility_data, discount_factors, r0):
        self.volatility_data = volatility_data
        self.discount_factors = discount_factors
        self.r0 = r0  # Initial interest rate

    def solve_initial_rate(self, D):
        """Solve for initial 6-month rate using D(0.5)"""
        def equation(r):
            return (0.5 + 0.5) / (1 + r / 2) - D[0]
        return fsolve(equation, 0.05)[0]

    def simulate_path(self, time_horizon, steps):
        """Simulate short rate paths using the BDT model"""
        dt = time_horizon / steps
        rate_tree = np.zeros((steps, steps))
        rate_tree[0, 0] = self.r0

        for period in range(1, steps):
            # Use the last available volatility if the period exceeds the length of volatility_data
            sigma = self.volatility_data[min(period - 1, len(self.volatility_data) - 1)]
            rates_at_period = self._solve_rates_for_period(period, rate_tree, sigma, dt)
            for j, rate in enumerate(rates_at_period):
                rate_tree[j, period] = rate

        # Extract the first row (short rate path)
        return rate_tree[0, :]

    def _solve_rates_for_period(self, period, rate_tree, sigma, dt):
        """Solve rates for a specific period"""
        rates = []
        r_prev = rate_tree[0:period, period - 1]

        for i in range(period + 1):
            if i == 0:  # Lowest rate
                r = r_prev[0] * np.exp(-sigma**2 * dt / 2 - sigma * np.sqrt(dt))
            elif i == period:  # Highest rate
                r = r_prev[-1] * np.exp(sigma**2 * dt / 2 + sigma * np.sqrt(dt))
            else:  # Middle rates
                r = r_prev[i - 1] * np.exp(sigma**2 * dt)
            rates.append(r)

        return rates
import numpy as np
from scipy.optimize import fsolve
from interest_rate_model import InterestRateModel


class BDTModel(InterestRateModel):
    def __init__(self, volatility_data, discount_factors, r0):
        self.volatility_data = volatility_data
        self.discount_factors = discount_factors
        self.r0 = r0  # Initial interest rate
        self.rate_tree = None  # Will store the rate tree after simulation
        self.time_points = None  # Will store the time points after simulation

    def solve_initial_rate(self, D):
        """Solve for initial 6-month rate using D(0.5)"""

        def equation(r):
            return (0.5 + 0.5) / (1 + r / 2) - D[0]

        return fsolve(equation, 0.05)[0]

    def simulate_path(self, time_horizon, steps):
        """
        Simulate short rate paths using the BDT model.

        Parameters:
        -----------
        time_horizon : float
            The time horizon in years for the simulation.
        steps : int
            The number of time steps in the simulation.

        Returns:
        --------
        ndarray
            An array containing the simulated short rates.
        """
        dt = time_horizon / steps
        self.time_points = np.linspace(0, time_horizon, steps)
        rate_tree = np.zeros((steps, steps))
        rate_tree[0, 0] = self.r0

        for period in range(1, steps):
            # Use the last available volatility if the period exceeds the length of volatility_data
            sigma = self.volatility_data[min(period - 1, len(self.volatility_data) - 1)]
            rates_at_period = self._solve_rates_for_period(period, rate_tree, sigma, dt)
            for j, rate in enumerate(rates_at_period):
                rate_tree[j, period] = rate

        # Store the rate tree for later use in calculating discount factors
        self.rate_tree = rate_tree

        # Extract the first row (short rate path)
        return rate_tree[0, :]

    def _solve_rates_for_period(self, period, rate_tree, sigma, dt):
        """Solve rates for a specific period"""
        rates = []
        r_prev = rate_tree[0:period, period - 1]

        for i in range(period + 1):
            if i == 0:  # Lowest rate
                r = r_prev[0] * np.exp(-sigma ** 2 * dt / 2 - sigma * np.sqrt(dt))
            elif i == period:  # Highest rate
                r = r_prev[-1] * np.exp(sigma ** 2 * dt / 2 + sigma * np.sqrt(dt))
            else:  # Middle rates
                r = r_prev[i - 1] * np.exp(sigma ** 2 * dt)
            rates.append(r)

        return rates

    def get_discount_factor(self, t):
        """
        Calculate the discount factor for time t under the BDT model.

        For the BDT model, this requires traversing the interest rate tree.

        Parameters:
        -----------
        t : float
            The time in years for which to calculate the discount factor.

        Returns:
        --------
        float
            The discount factor for time t.
        """
        # If the rate tree hasn't been simulated yet, return approximation
        if self.rate_tree is None or self.time_points is None:
            # Simple approximation based on the initial rate
            return np.exp(-self.r0 * t)

        # Find the closest time index
        idx = np.abs(self.time_points - t).argmin()

        # If we're at time 0, just return 1
        if idx == 0:
            return 1.0

        # For BDT model, we need to traverse all possible paths in the tree
        # and average the discount factors

        # This is a simplified approach - for a given time point, we average
        # the discount factors across all nodes at that level
        rates_at_time = self.rate_tree[:idx + 1, idx]
        discount_factors = np.exp(-rates_at_time * t)

        # Weight each path equally (can be refined with proper probabilities)
        avg_discount_factor = np.mean(discount_factors)

        return avg_discount_factor

    def get_zero_coupon_bond_price(self, t, T):
        """
        Calculate the price of a zero-coupon bond maturing at time T.

        Parameters:
        -----------
        t : float
            Current time.
        T : float
            Maturity time of the bond.

        Returns:
        --------
        float
            The price of the zero-coupon bond.
        """
        if t >= T:
            return 1.0

        # Time to maturity
        tau = T - t

        # Get the discount factor for the remaining time
        return self.get_discount_factor(tau)
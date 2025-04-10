import numpy as np
from interest_rate_model import InterestRateModel


class VasicekModel(InterestRateModel):
    def __init__(self, alpha, beta, sigma, r0):
        self.alpha = alpha  # Mean reversion speed
        self.beta = beta  # Long-term mean rate
        self.sigma = sigma  # Volatility
        self.r0 = r0  # Initial interest rate

    def simulate_path(self, time_horizon, steps):
        """
        Simulate short rate paths using the Vasicek model.

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
        rates = np.zeros(steps)
        rates[0] = self.r0

        for i in range(1, steps):
            dr = self.alpha * (self.beta - rates[i - 1]) * dt + self.sigma * np.sqrt(dt) * np.random.normal()
            rates[i] = rates[i - 1] + dr

        return rates

    def get_discount_factor(self, t):
        """
        Calculate the discount factor for time t under the Vasicek model.

        For the Vasicek model, this is derived analytically from the bond pricing formula.

        Parameters:
        -----------
        t : float
            The time in years for which to calculate the discount factor.

        Returns:
        --------
        float
            The discount factor for time t.
        """
        # Calculate term B(t) from Vasicek bond pricing formula
        if self.alpha > 0:
            B = (1 - np.exp(-self.alpha * t)) / self.alpha
        else:
            # Handle the case when alpha is very close to zero
            B = t

        # Calculate term A(t) from Vasicek bond pricing formula
        A = (self.beta - (self.sigma ** 2) / (2 * self.alpha ** 2)) * (B - t) - \
            (self.sigma ** 2 * B ** 2) / (4 * self.alpha)

        # Return the discount factor
        return np.exp(A - B * self.r0)

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
from abc import ABC, abstractmethod
import numpy as np


class InterestRateModel(ABC):
    """
    Abstract base class for interest rate models.

    This class defines the common interface that all interest rate models
    should implement, following the object-oriented principle of polymorphism.
    """

    @abstractmethod
    def simulate_path(self, time_horizon, steps):
        """
        Simulate short rate paths for the specified time horizon and number of steps.

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
        pass

    @abstractmethod
    def get_discount_factor(self, t):
        """
        Calculate the discount factor for time t.

        Parameters:
        -----------
        t : float
            The time in years for which to calculate the discount factor.

        Returns:
        --------
        float
            The discount factor for time t.
        """
        pass

    def get_yield_curve(self, times):
        """
        Generate a yield curve for the specified time points.

        Parameters:
        -----------
        times : array-like
            An array of time points (in years) for which to calculate yields.

        Returns:
        --------
        ndarray
            An array containing the yields for each time point.
        """
        discount_factors = np.array([self.get_discount_factor(t) for t in times])
        # Convert discount factors to continuously compounded yields
        yields = -np.log(discount_factors) / times
        return yields

    def get_forward_rates(self, times):
        """
        Calculate forward rates from the yield curve.

        Parameters:
        -----------
        times : array-like
            An array of time points (in years) for which to calculate forward rates.

        Returns:
        --------
        ndarray
            An array containing the forward rates between consecutive time points.
        """
        if len(times) <= 1:
            return np.array([])

        discount_factors = np.array([self.get_discount_factor(t) for t in times])
        forward_rates = np.zeros(len(times) - 1)

        for i in range(len(times) - 1):
            dt = times[i + 1] - times[i]
            forward_rates[i] = (np.log(discount_factors[i]) - np.log(discount_factors[i + 1])) / dt

        return forward_rates
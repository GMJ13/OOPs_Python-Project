import numpy as np
import matplotlib.pyplot as plt
from vasicek_model import VasicekModel
from cir_model import CIRModel
from bdt_model import BDTModel


def plot_short_rates(models, time_axis, model_names):
    """Plot short rate paths for the given models."""
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        rates = model.simulate_path(time_axis[-1], len(time_axis))
        plt.plot(time_axis, rates, label=model_names[i])

    plt.title("Comparison of Short Rate Evolution")
    plt.xlabel("Time (Years)")
    plt.ylabel("Short Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_yield_curves(models, times, model_names):
    """Plot yield curves for the given models."""
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        yields = model.get_yield_curve(times)
        plt.plot(times, yields, 'o-', label=model_names[i])

    plt.title("Yield Curves")
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Yield")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_discount_factors(models, times, model_names):
    """Plot discount factors for the given models."""
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        discount_factors = np.array([model.get_discount_factor(t) for t in times])
        plt.plot(times, discount_factors, 'o-', label=model_names[i])

    plt.title("Discount Factors")
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Discount Factor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_forward_rates(models, times, model_names):
    """Plot forward rates for the given models."""
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        forward_rates = model.get_forward_rates(times)
        # Plot forward rates at the midpoint of each interval
        midpoints = [(times[j] + times[j + 1]) / 2 for j in range(len(times) - 1)]
        plt.plot(midpoints, forward_rates, 'o-', label=model_names[i])

    plt.title("Forward Rates")
    plt.xlabel("Time (Years)")
    plt.ylabel("Forward Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def main():
    # Standardized inputs
    r0 = 0.03  # Initial short rate
    time_horizon = 10  # Time horizon in years
    steps = 100  # Number of time steps
    alpha = 0.1  # Mean reversion speed
    beta = 0.05  # Long-term mean rate
    sigma = 0.02  # Volatility

    # Discount factors and volatility data for BDT model
    # Extend the arrays to match the number of steps
    discount_factors = np.array([0.9724765, 0.9445693, 0.9163238, 0.8879337, 0.8597406,
                                 0.8320438, 0.805077, 0.7789041, 0.7534803, 0.7287341])
    volatility_data = np.array([0.1, 0.12, 0.135, 0.15, 0.16, 0.162, 0.164, 0.162, 0.16, 0.157])

    # Extend discount_factors and volatility_data to match the number of steps
    time_axis = np.linspace(0, time_horizon, steps)
    discount_factors_extended = np.interp(time_axis, np.linspace(0, time_horizon, len(discount_factors)),
                                          discount_factors)
    volatility_data_extended = np.interp(time_axis, np.linspace(0, time_horizon, len(volatility_data)), volatility_data)

    # Initialize models
    vasicek = VasicekModel(alpha=alpha, beta=beta, sigma=sigma, r0=r0)
    cir = CIRModel(alpha=alpha, beta=beta, sigma=sigma, r0=r0)
    bdt = BDTModel(volatility_data=volatility_data_extended, discount_factors=discount_factors_extended, r0=r0)

    # Create a list of models and their names
    models = [vasicek, cir, bdt]
    model_names = ["Vasicek Model", "CIR Model", "BDT Model"]

    # Plot short rate evolution
    plot_short_rates(models, time_axis, model_names)

    # Define time points for yield curve and discount factor calculations
    yield_curve_times = np.linspace(0.5, 10, 20)  # From 6 months to 10 years

    # Initialize model simulations to build the rate trees
    for model in models:
        model.simulate_path(time_horizon, steps)

    # Plot yield curves
    plot_yield_curves(models, yield_curve_times, model_names)

    # Plot discount factors
    plot_discount_factors(models, yield_curve_times, model_names)

    # Plot forward rates
    plot_forward_rates(models, yield_curve_times, model_names)

    plt.show()


if __name__ == "__main__":
    main()
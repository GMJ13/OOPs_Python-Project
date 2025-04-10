import numpy as np
import matplotlib.pyplot as plt
from vasicek_model import VasicekModel
from cir_model import CIRModel
from bdt_model import BDTModel

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
    discount_factors = np.interp(np.linspace(0, len(discount_factors) - 1, steps), np.arange(len(discount_factors)), discount_factors)
    volatility_data = np.interp(np.linspace(0, len(volatility_data) - 1, steps), np.arange(len(volatility_data)), volatility_data)

    # Initialize models
    vasicek = VasicekModel(alpha=alpha, beta=beta, sigma=sigma, r0=r0)
    cir = CIRModel(alpha=alpha, beta=beta, sigma=sigma, r0=r0)
    bdt = BDTModel(volatility_data=volatility_data, discount_factors=discount_factors, r0=r0)

    # Simulate short rate paths
    vasicek_rates = vasicek.simulate_path(time_horizon, steps)
    cir_rates = cir.simulate_path(time_horizon, steps)
    bdt_rates = bdt.simulate_path(time_horizon, steps)

    # Plot results
    time_axis = np.linspace(0, time_horizon, steps)
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, vasicek_rates, label="Vasicek Model")
    plt.plot(time_axis, cir_rates, label="CIR Model")
    plt.plot(time_axis, bdt_rates, label="BDT Model")
    plt.title("Comparison of Short Rate Evolution")
    plt.xlabel("Time (Years)")
    plt.ylabel("Short Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
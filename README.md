# 🌐 Forex Options Portfolio Simulation

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen)
![Jupyter](https://img.shields.io/badge/jupyter-compatible-orange)

A comprehensive financial modeling and simulation tool for pricing and analyzing European FX options on the EUR/TND currency pair. This project implements multiple sophisticated stochastic pricing models with a focus on market realism and performance evaluation.

## 📊 Key Features

- **Multiple Pricing Models**: 
  - 📈 Black-Scholes (Garman-Kohlhagen) with historical volatility
  - 📉 E-GARCH Monte Carlo simulation capturing asymmetric volatility responses
  - 🔄 Merton Jump-Diffusion to handle sudden market movements
  - 📊 SABR model for stochastic volatility with correlation

- **Realistic Market Data Generation**:
  - Customizable spot rate paths with mean reversion and jumps
  - Time-varying volatility surfaces
  - Dynamic interest rate term structures

- **Portfolio Management**:
  - Generate realistic option portfolios with configurable parameters
  - Track active notional exposure over time
  - Calculate Greeks and risk metrics (Delta, Gamma, Vega, Theta, Rho)

- **Performance Evaluation**:
  - Model comparison based on pricing accuracy and PnL
  - Statistical error metrics (RMSE, MAE, MAPE)
  - Visualization of volatility smiles and option pricing dynamics

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Simulation

```bash
python main.py
```

Optional arguments:
- `--config`: Path to configuration file (default: `config.yaml`)
- `--generate`: Generate new data instead of loading existing data
- `--output-dir`: Output directory for results
- `--skip-pricing`: Skip portfolio pricing phase
- `--skip-evaluation`: Skip performance evaluation phase
- `--skip-visualization`: Skip visualization phase

### Interactive Exploration

Run the Jupyter notebook for interactive exploration:

```bash
jupyter notebook notebooks/portfolio_evaluation.ipynb
```

## 🧮 Model Implementation Details

This project implements several sophisticated option pricing models:

### 1. Black-Scholes (Garman-Kohlhagen)
The classic model adapted for FX options with domestic and foreign interest rates.

### 2. E-GARCH Monte Carlo
Implements the Exponential GARCH model that captures asymmetric responses to positive and negative shocks in returns, a common feature in financial markets.

### 3. Merton Jump-Diffusion
Extends the standard pricing framework to include discrete jumps in the price process, accounting for unexpected market events.

### 4. SABR Model
Sophisticated stochastic volatility model that accounts for the relationship between spot price movements and volatility changes.

## 📁 Project Structure

```
├── config.yaml                  # Configuration parameters
├── main.py                      # Main script for full simulation
├── requirements.txt             # Python dependencies
├── data/                        # Generated and market data
│   ├── generated/               # Synthetic option contracts
│   └── market/                  # Simulated market data
├── notebooks/                   # Jupyter notebooks
│   └── portfolio_evaluation.ipynb  # Interactive analysis
├── output/                      # Simulation results and charts
└── src/                         # Source code
    ├── data_generation/         # Option contract generation
    ├── market_data/             # Market data simulation
    ├── models/                  # Pricing models implementation
    ├── portfolio/               # Portfolio management
    ├── evaluation/              # Performance metrics
    └── visualization/           # Plotting and dashboard
```

## 📈 Example Visualizations

The project generates various visualizations to analyze option pricing and model performance:

- Spot rate and volatility evolution
- Option strike distribution and maturity profile
- Active notional exposure over time
- Volatility smile visualization
- Model price comparison
- Error metrics evaluation

## ⚙️ Configuration

The `config.yaml` file allows customization of various simulation parameters:

- Portfolio constraints (notional limits, number of options)
- Time parameters (date range, maturity limits)
- Market parameters (initial rates, volatilities)
- Model-specific parameters for each pricing model
- Simulation and output settings

## 🔍 Performance Insights

This simulation allows for comprehensive comparison of different option pricing models:

- Traditional Black-Scholes provides a baseline but may underestimate tail risks
- E-GARCH captures time-varying and asymmetric volatility effects
- Jump-Diffusion excels during periods of market turbulence
- SABR model handles volatility smile/skew effects in the options market

## 🛠️ Development

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- Hull, J. C. (2018). Options, Futures, and Other Derivatives
- Garman, M. B., & Kohlhagen, S. W. (1983). Foreign currency option values
- Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach
- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous
- Hagan, P. S., et al. (2002). Managing smile risk (The SABR model)

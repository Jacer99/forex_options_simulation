"""
E-GARCH-based Monte Carlo Simulation Model

This module implements a Monte Carlo simulation for pricing FX options based on
the Exponential GARCH (E-GARCH) volatility model introduced by Nelson (1991).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import logging
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EGARCHMCModel:
    """
    Implements an E-GARCH-based Monte Carlo simulation model for pricing FX options.
    
    The E-GARCH model allows for asymmetric responses to positive and negative shocks,
    which is a common feature in financial time series.
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize the E-GARCH Monte Carlo model.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.egarch_params = self.config['models']['egarch']
        self.num_simulations = self.egarch_params['num_simulations']
        
        # E-GARCH parameters
        self.omega = self.egarch_params['omega']
        self.alpha = self.egarch_params['alpha']
        self.gamma = self.egarch_params['gamma']
        self.beta = self.egarch_params['beta']
        
        logger.info("E-GARCH Monte Carlo model initialized")
    
    def _load_config(self, config_path):
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to the configuration file.
            
        Returns:
            dict: Configuration parameters.
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def simulate_egarch_process(self, returns, n_days, initial_vol=None):
        """
        Simulate asset returns using an E-GARCH process.
        
        Args:
            returns (pandas.Series): Historical returns series.
            n_days (int): Number of days to simulate.
            initial_vol (float, optional): Initial volatility. If None, it's estimated from returns.
            
        Returns:
            tuple: (simulated_returns, simulated_log_variance)
        """
        # Estimate initial log-variance if not provided
        if initial_vol is None:
            initial_vol = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Initial log-variance
        log_var = np.log(initial_vol**2 / 252)  # Convert to daily and take log
        
        # Arrays to store simulated values
        sim_returns = np.zeros(n_days)
        sim_log_var = np.zeros(n_days)
        
        # Generate random innovations
        z = np.random.normal(0, 1, n_days)
        
        # Simulate the E-GARCH process
        for t in range(n_days):
            # Store the current log-variance
            sim_log_var[t] = log_var
            
            # Calculate the current conditional variance and volatility
            variance = np.exp(log_var)
            volatility = np.sqrt(variance)
            
            # Generate the return for the current period
            sim_returns[t] = volatility * z[t]
            
            # Update the log-variance using the E-GARCH formula
            log_var = self.omega + self.beta * log_var + \
                      self.alpha * (np.abs(z[t]) - np.sqrt(2/np.pi)) + \
                      self.gamma * z[t]
        
        return sim_returns, sim_log_var
    
    def run_monte_carlo(self, spot, strike, days_to_maturity, domestic_rate, foreign_rate, 
                       volatility, returns=None, option_type='call'):
        """
        Run a Monte Carlo simulation to price a European FX option using E-GARCH.
        
        Args:
            spot (float): Current spot exchange rate.
            strike (float): Strike price.
            days_to_maturity (int): Number of days to maturity.
            domestic_rate (float): Domestic risk-free interest rate (annualized).
            foreign_rate (float): Foreign risk-free interest rate (annualized).
            volatility (float): Annualized volatility of the exchange rate.
            returns (pandas.Series, optional): Historical returns for calibration.
            option_type (str): 'call' or 'put'.
            
        Returns:
            float: Option price in domestic currency.
        """
        # Convert days to years
        T = days_to_maturity / 365.0
        
        # Handle edge cases
        if days_to_maturity <= 0:
            if option_type.lower() == 'call':
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)
        
        # Initialize arrays for the simulated prices
        num_sims = self.num_simulations
        final_prices = np.zeros(num_sims)
        
        # Run simulations
        for sim in range(num_sims):
            # Simulate returns using E-GARCH
            sim_returns, _ = self.simulate_egarch_process(
                returns=returns,
                n_days=days_to_maturity,
                initial_vol=volatility
            )
            
            # Calculate the path of the exchange rate
            # The drift term is the interest rate differential (domestic - foreign)
            drift = (domestic_rate - foreign_rate) / 365.0  # Daily drift
            path = np.zeros(days_to_maturity + 1)
            path[0] = spot
            
            for t in range(days_to_maturity):
                path[t+1] = path[t] * np.exp(drift + sim_returns[t])
            
            # Store the final price
            final_prices[sim] = path[-1]
        
        # Calculate option payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(0, final_prices - strike)
        else:
            payoffs = np.maximum(0, strike - final_prices)
        
        # Discount the payoffs
        option_price = np.mean(payoffs) * np.exp(-domestic_rate * T)
        
        return option_price
    
    def price_options_portfolio(self, options_data, market_data):
        """
        Price a portfolio of European FX options using the E-GARCH Monte Carlo model.
        
        Args:
            options_data (list or pandas.DataFrame): Portfolio of options.
            market_data (tuple): Tuple of (spot_rates, volatility, interest_rates) DataFrames.
            
        Returns:
            pandas.DataFrame: Options data with added pricing information.
        """
        if isinstance(options_data, list):
            df = pd.DataFrame(options_data)
        else:
            df = options_data.copy()
        
        # Unpack market data
        spot_rates, volatilities, interest_rates = market_data
        
        # Calculate returns from spot rates
        returns = spot_rates['EUR/TND'].pct_change().dropna()
        
        # Add E-GARCH Monte Carlo price column if it doesn't exist
        if 'egarch_price' not in df.columns:
            df['egarch_price'] = np.nan
        
        # Price each option
        for i, option in df.iterrows():
            try:
                # Get issue date
                issue_date = pd.to_datetime(option['issue_date'])
                
                # Get spot rate at issue date
                spot = spot_rates[spot_rates['date'] <= issue_date].iloc[-1]['EUR/TND']
                
                # Get volatility at issue date
                vol = volatilities[volatilities['date'] <= issue_date].iloc[-1]['historical_vol']
                
                # Get interest rates at issue date
                rates = interest_rates[interest_rates['date'] <= issue_date].iloc[-1]
                domestic_rate = rates['EUR_rate']
                foreign_rate = rates['TND_rate']
                
                # Get returns up to issue date for calibration
                historical_returns = returns[spot_rates['date'] < issue_date].tail(252)  # Use up to 1 year of data
                
                # Run Monte Carlo simulation
                price = self.run_monte_carlo(
                    spot=spot,
                    strike=option['strike_price'],
                    days_to_maturity=option['days_to_maturity'],
                    domestic_rate=domestic_rate,
                    foreign_rate=foreign_rate,
                    volatility=vol,
                    returns=historical_returns,
                    option_type=option['type']
                )
                
                # Update option price
                df.at[i, 'egarch_price'] = price
                
            except Exception as e:
                logger.error(f"Error pricing option {option['option_id']} with E-GARCH MC: {e}")
        
        return df
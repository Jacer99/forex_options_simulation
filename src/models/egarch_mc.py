"""
Optimized E-GARCH-based Monte Carlo Simulation Model

This module implements an optimized Monte Carlo simulation for pricing FX options based on
the Exponential GARCH (E-GARCH) volatility model with improved performance.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import logging
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EGARCHMCModel:
    """
    Implements an optimized E-GARCH-based Monte Carlo simulation model for pricing FX options.
    
    Optimizations:
    - Vectorized calculations
    - Batch processing
    - Optional parallel execution
    - Improved memory management
    - Numerical stability enhancements
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize the E-GARCH Monte Carlo model.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.egarch_params = self.config['models']['egarch']
        
        # Limit number of simulations for better performance
        self.num_simulations = min(5000, self.egarch_params.get('num_simulations', 10000))
        
        # E-GARCH parameters
        self.omega = self.egarch_params['omega']
        self.alpha = self.egarch_params['alpha']
        self.gamma = self.egarch_params['gamma']
        self.beta = self.egarch_params['beta']
        
        # Determine optimal number of processes
        self.use_parallel = True  # Can be set to False to disable parallelism
        self.max_workers = min(4, multiprocessing.cpu_count())
        
        logger.info(f"Optimized E-GARCH Monte Carlo model initialized with {self.num_simulations} simulations")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            raise
    
    def _simulate_egarch_batch(self, n_days, n_paths, initial_vol, seed=None):
        """
        Simulate multiple EGARCH paths in a single batch.
        
        Args:
            n_days (int): Number of days to simulate.
            n_paths (int): Number of paths to simulate.
            initial_vol (float): Initial volatility.
            seed (int, optional): Random seed for reproducibility.
            
        Returns:
            tuple: (simulated_returns, final_prices)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize arrays
        log_var = np.log(initial_vol**2 / 252) * np.ones(n_paths)  # Daily variance
        sim_returns = np.zeros((n_paths, n_days))
        
        # Pre-generate all random innovations
        z = np.random.normal(0, 1, (n_paths, n_days))
        
        # Simulate paths
        for t in range(n_days):
            # Calculate volatility from log-variance
            volatility = np.sqrt(np.exp(log_var))
            
            # Generate returns
            sim_returns[:, t] = volatility * z[:, t]
            
            # Update log-variance vectorized across all paths
            log_var = self.omega + self.beta * log_var + \
                      self.alpha * (np.abs(z[:, t]) - np.sqrt(2/np.pi)) + \
                      self.gamma * z[:, t]
        
        return sim_returns
    
    def _worker_task(self, params):
        """Task function for parallel simulation."""
        spot, strike, days_to_maturity, daily_drift, volatility, option_type, batch_size, seed = params
        
        # Simulate returns
        sim_returns = self._simulate_egarch_batch(days_to_maturity, batch_size, volatility, seed)
        
        # Calculate price paths and final prices
        paths = np.zeros((batch_size, days_to_maturity + 1))
        paths[:, 0] = spot  # Initial price
        
        # Efficiently update all paths
        for t in range(days_to_maturity):
            paths[:, t+1] = paths[:, t] * np.exp(daily_drift + sim_returns[:, t])
        
        # Get final prices
        final_prices = paths[:, -1]
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(0, final_prices - strike)
        else:
            payoffs = np.maximum(0, strike - final_prices)
        
        return np.mean(payoffs)  # Return average payoff for this batch
    
    def run_monte_carlo(self, spot, strike, days_to_maturity, domestic_rate, foreign_rate, 
                       volatility, returns=None, option_type='call'):
        """
        Run an optimized Monte Carlo simulation to price a European FX option using E-GARCH.
        
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
        # Handle edge cases
        if days_to_maturity <= 0:
            if option_type.lower() == 'call':
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)
        
        # Convert days to years
        T = days_to_maturity / 365.0
        
        # Calculate daily drift once
        daily_drift = (domestic_rate - foreign_rate) / 365.0
        
        # Determine batch configuration
        num_sims = self.num_simulations
        num_batches = min(self.max_workers * 2, 16) if self.use_parallel else 1
        batch_size = num_sims // num_batches
        
        # Ensure positive volatility
        volatility = max(0.001, volatility)
        
        try:
            # Run in parallel if enabled
            if self.use_parallel and num_batches > 1:
                # Prepare parameters for each batch
                batch_params = [
                    (spot, strike, days_to_maturity, daily_drift, volatility, option_type, batch_size, i) 
                    for i in range(num_batches)
                ]
                
                # Run batches in parallel
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(self._worker_task, params) for params in batch_params]
                    batch_results = [future.result() for future in as_completed(futures)]
                
                # Average results from all batches
                mean_payoff = np.mean(batch_results)
                
            else:
                # Run single-threaded for smaller problems or when parallel is disabled
                sim_returns = self._simulate_egarch_batch(days_to_maturity, num_sims, volatility)
                
                # Calculate price paths efficiently
                paths = np.zeros((num_sims, days_to_maturity + 1))
                paths[:, 0] = spot
                
                for t in range(days_to_maturity):
                    paths[:, t+1] = paths[:, t] * np.exp(daily_drift + sim_returns[:, t])
                
                # Calculate final prices and payoffs
                final_prices = paths[:, -1]
                
                if option_type.lower() == 'call':
                    payoffs = np.maximum(0, final_prices - strike)
                else:
                    payoffs = np.maximum(0, strike - final_prices)
                
                mean_payoff = np.mean(payoffs)
            
            # Calculate option price (discounted expected payoff)
            option_price = mean_payoff * np.exp(-domestic_rate * T)
            
            return option_price
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            # Fall back to Black-Scholes as backup
            d1 = (np.log(spot / strike) + (domestic_rate - foreign_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
            d2 = d1 - volatility * np.sqrt(T)
            
            if option_type.lower() == 'call':
                return spot * np.exp(-foreign_rate * T) * norm.cdf(d1) - strike * np.exp(-domestic_rate * T) * norm.cdf(d2)
            else:
                return strike * np.exp(-domestic_rate * T) * norm.cdf(-d2) - spot * np.exp(-foreign_rate * T) * norm.cdf(-d1)
    
    def price_options_portfolio(self, options_data, market_data):
        """
        Price a portfolio of European FX options using the optimized E-GARCH Monte Carlo model.
        
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
        
        # Process options in batches for better memory management
        batch_size = min(20, len(df))  # Process 20 options at a time
        
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch = df.iloc[batch_start:batch_end]
            
            for i, option in batch.iterrows():
                try:
                    # Get issue date
                    issue_date = pd.to_datetime(option['issue_date'])
                    
                    # Get spot rate at issue date
                    spot_data = spot_rates[spot_rates['date'] <= issue_date]
                    if spot_data.empty:
                        logger.warning(f"No spot rate data for {issue_date} for option {option['option_id']}")
                        continue
                    spot = spot_data.iloc[-1]['EUR/TND']
                    
                    # Get volatility at issue date
                    vol_data = volatilities[volatilities['date'] <= issue_date]
                    if vol_data.empty:
                        logger.warning(f"No volatility data for {issue_date} for option {option['option_id']}")
                        vol = 0.15  # Default value
                    else:
                        vol = vol_data.iloc[-1]['historical_vol']
                    
                    # Get interest rates at issue date
                    rates_data = interest_rates[interest_rates['date'] <= issue_date]
                    if rates_data.empty:
                        logger.warning(f"No interest rate data for {issue_date} for option {option['option_id']}")
                        continue
                    rates = rates_data.iloc[-1]
                    domestic_rate = rates['EUR_rate']
                    foreign_rate = rates['TND_rate']
                    
                    # Get days to maturity and ensure it's positive
                    days_to_maturity = max(1, option['days_to_maturity'])
                    
                    # Run Monte Carlo simulation with optimized algorithm
                    price = self.run_monte_carlo(
                        spot=spot,
                        strike=option['strike_price'],
                        days_to_maturity=days_to_maturity,
                        domestic_rate=domestic_rate,
                        foreign_rate=foreign_rate,
                        volatility=vol,
                        returns=returns,
                        option_type=option['type']
                    )
                    
                    # Update option price
                    df.at[i, 'egarch_price'] = price
                    
                except Exception as e:
                    logger.error(f"Error pricing option {option.get('option_id', i)}: {e}")
        
        return df
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
        # Reduce number of simulations for faster runtime
        self.num_simulations = min(1000, self.egarch_params['num_simulations'])  # Limit to 1000 max
        
        # E-GARCH parameters
        self.omega = self.egarch_params['omega']
        self.alpha = self.egarch_params['alpha']
        self.gamma = self.egarch_params['gamma']
        self.beta = self.egarch_params['beta']
        
        logger.info(f"E-GARCH Monte Carlo model initialized with {self.num_simulations} simulations")
    
    def _load_config(self, config_path):
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to the configuration file.
            
        Returns:
            dict: Configuration parameters.
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
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
            if returns is not None and len(returns) > 0:
                initial_vol = returns.std() * np.sqrt(252)  # Annualized
            else:
                initial_vol = 0.2  # Default value
        
        # Initial log-variance
        log_var = np.log(initial_vol**2 / 252)  # Convert to daily
        
        # Pre-allocate arrays for efficiency
        sim_returns = np.zeros(n_days)
        sim_log_var = np.zeros(n_days)
        
        # Generate all random innovations at once (more efficient)
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
        try:
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
            
            # Pre-allocate arrays for efficiency
            num_sims = self.num_simulations
            final_prices = np.zeros(num_sims)
            
            # Generate all simulations in batches for better memory management
            batch_size = min(1000, num_sims)  # Process in batches of 1000 max
            for batch_start in range(0, num_sims, batch_size):
                batch_end = min(batch_start + batch_size, num_sims)
                batch_size_actual = batch_end - batch_start
                
                # Initialize paths for this batch
                paths = np.zeros((batch_size_actual, days_to_maturity + 1))
                paths[:, 0] = spot  # Set initial price
                
                # Generate returns for all simulations in this batch
                for i in range(batch_start, batch_end):
                    batch_idx = i - batch_start
                    sim_returns, _ = self.simulate_egarch_process(
                        returns=returns,
                        n_days=days_to_maturity,
                        initial_vol=volatility
                    )
                    
                    # Calculate price path (vectorized over time)
                    for t in range(days_to_maturity):
                        paths[batch_idx, t+1] = paths[batch_idx, t] * np.exp(daily_drift + sim_returns[t])
                    
                    # Store final price
                    final_prices[i] = paths[batch_idx, -1]
            
            # Calculate option payoffs (vectorized)
            if option_type.lower() == 'call':
                payoffs = np.maximum(0, final_prices - strike)
            else:
                payoffs = np.maximum(0, strike - final_prices)
            
            # Calculate option price (discounted expected payoff)
            option_price = np.mean(payoffs) * np.exp(-domestic_rate * T)
            
            return option_price
            
        except (ValueError, ZeroDivisionError) as e:
            logger.error(f"Numerical error in Monte Carlo simulation: {e}")
            # Return a sensible default based on Black-Scholes
            from scipy.stats import norm
            # Calculate d1 and d2
            T = days_to_maturity / 365.0
            d1 = (np.log(spot / strike) + (domestic_rate - foreign_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
            d2 = d1 - volatility * np.sqrt(T)
            
            if option_type.lower() == 'call':
                return spot * np.exp(-foreign_rate * T) * norm.cdf(d1) - strike * np.exp(-domestic_rate * T) * norm.cdf(d2)
            else:
                return strike * np.exp(-domestic_rate * T) * norm.cdf(-d2) - spot * np.exp(-foreign_rate * T) * norm.cdf(-d1)
        except Exception as e:
            logger.error(f"Unexpected error in Monte Carlo simulation: {e}")
            return 0.0
    
    def price_options_portfolio(self, options_data, market_data):
        """
        Price a portfolio of European FX options using the E-GARCH Monte Carlo model.
        
        Args:
            options_data (list or pandas.DataFrame): Portfolio of options.
            market_data (tuple): Tuple of (spot_rates, volatility, interest_rates) DataFrames.
            
        Returns:
            pandas.DataFrame: Options data with added pricing information.
        """
        try:
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
            
            # Process options in batches to avoid memory issues
            batch_size = min(50, len(df))  # Process 50 options at a time
            
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
                        
                        # Get historical returns up to issue date
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
                        
                    except KeyError as e:
                        logger.error(f"Missing data for option {option.get('option_id', i)}: {e}")
                    except ValueError as e:
                        logger.error(f"Invalid value for option {option.get('option_id', i)}: {e}")
                    except Exception as e:
                        logger.error(f"Error pricing option {option.get('option_id', i)} with E-GARCH MC: {e}")
            
            return df
            
        except Exception as e:
            logger.error(f"Unexpected error in pricing portfolio: {e}")
            return options_data  # Return original data in case of error
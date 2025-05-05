"""
Merton Jump-Diffusion Model

This module implements the Merton Jump-Diffusion model for pricing FX options.
The model extends the Black-Scholes framework by adding jumps in the price process.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
import logging
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JumpDiffusionModel:
    """
    Implements the Merton Jump-Diffusion model for pricing FX options.
    
    The model assumes that the asset price follows a diffusion process with
    occasional jumps, where the jump sizes follow a log-normal distribution.
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize the Jump-Diffusion model.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.jd_params = self.config['models']['jump_diffusion']
        
        # Jump-Diffusion parameters
        self.lambda_jump = self.jd_params['lambda']  # Jump intensity
        self.jump_mean = self.jd_params['jump_mean']  # Mean jump size
        self.jump_std = self.jd_params['jump_std']    # Standard deviation of jump size
        
        logger.info("Merton Jump-Diffusion model initialized")
    
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
    
    def price_option(self, spot, strike, days_to_maturity, domestic_rate, foreign_rate, 
                    volatility, option_type='call', n_terms=10):
        """
        Calculate the price of a European FX option using the Merton Jump-Diffusion model.
        
        Args:
            spot (float): Spot exchange rate.
            strike (float): Strike price.
            days_to_maturity (int): Number of days to maturity.
            domestic_rate (float): Domestic risk-free interest rate (annualized).
            foreign_rate (float): Foreign risk-free interest rate (annualized).
            volatility (float): Annualized volatility of the exchange rate (for diffusion part).
            option_type (str): 'call' or 'put'.
            n_terms (int): Number of terms in the series expansion.
            
        Returns:
            float: Option price in domestic currency.
        """
        # Convert days to years
        T = days_to_maturity / 365.0
        
        # Handle edge cases
        if T <= 0 or volatility <= 0:
            if option_type.lower() == 'call':
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)
        
        # Adjust the drift term to account for jumps
        # Expected jump contribution to return: lambda * (exp(jump_mean + 0.5*jump_std^2) - 1)
        jump_expected_return = self.lambda_jump * (np.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1)
        adjusted_drift = domestic_rate - foreign_rate - jump_expected_return
        
        # Calculate the option price as a weighted sum of Black-Scholes prices
        # Each term corresponds to a specific number of jumps
        option_price = 0.0
        
        for n in range(n_terms):
            # Probability of exactly n jumps during time T
            p_n_jumps = poisson.pmf(n, self.lambda_jump * T)
            
            # Adjusted volatility for n jumps
            vol_n = np.sqrt(volatility**2 + n * self.jump_std**2 / T)
            
            # Adjusted drift for n jumps
            drift_n = adjusted_drift + n * self.jump_mean / T
            
            # Black-Scholes price for n jumps
            d1 = (np.log(spot / strike) + (drift_n + 0.5 * vol_n**2) * T) / (vol_n * np.sqrt(T))
            d2 = d1 - vol_n * np.sqrt(T)
            
            if option_type.lower() == 'call':
                bs_price = spot * np.exp(-foreign_rate * T) * norm.cdf(d1) - strike * np.exp(-domestic_rate * T) * norm.cdf(d2)
            else:
                bs_price = strike * np.exp(-domestic_rate * T) * norm.cdf(-d2) - spot * np.exp(-foreign_rate * T) * norm.cdf(-d1)
            
            # Add the weighted Black-Scholes price to the total
            option_price += p_n_jumps * bs_price
        
        return option_price
    
    def simulate_price_path(self, spot, days, domestic_rate, foreign_rate, volatility):
        """
        Simulate a price path using the Jump-Diffusion process.
        
        Args:
            spot (float): Initial spot price.
            days (int): Number of days to simulate.
            domestic_rate (float): Domestic risk-free interest rate (annualized).
            foreign_rate (float): Foreign risk-free interest rate (annualized).
            volatility (float): Annualized volatility of the exchange rate (for diffusion part).
            
        Returns:
            numpy.ndarray: Simulated price path.
        """
        # Convert parameters to daily values
        daily_drift = (domestic_rate - foreign_rate) / 365.0
        daily_vol = volatility / np.sqrt(252)
        daily_jump_intensity = self.lambda_jump / 252
        
        # Simulate price path
        prices = np.zeros(days + 1)
        prices[0] = spot
        
        for t in range(days):
            # Diffusion component
            diff_return = daily_drift + daily_vol * np.random.normal(0, 1)
            
            # Jump component
            jump_return = 0
            # Simulate number of jumps (usually 0 or 1)
            n_jumps = np.random.poisson(daily_jump_intensity)
            
            if n_jumps > 0:
                # Simulate jump sizes
                jump_sizes = np.random.normal(self.jump_mean, self.jump_std, n_jumps)
                jump_return = np.sum(jump_sizes)
            
            # Update price
            prices[t+1] = prices[t] * np.exp(diff_return + jump_return)
        
        return prices
    
    def monte_carlo_price(self, spot, strike, days_to_maturity, domestic_rate, foreign_rate, 
                         volatility, option_type='call', n_simulations=10000):
        """
        Price an option using Monte Carlo simulation of the Jump-Diffusion process.
        
        Args:
            spot (float): Spot exchange rate.
            strike (float): Strike price.
            days_to_maturity (int): Number of days to maturity.
            domestic_rate (float): Domestic risk-free interest rate (annualized).
            foreign_rate (float): Foreign risk-free interest rate (annualized).
            volatility (float): Annualized volatility of the exchange rate (for diffusion part).
            option_type (str): 'call' or 'put'.
            n_simulations (int): Number of Monte Carlo simulations.
            
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
        
        # Initialize array for final prices
        final_prices = np.zeros(n_simulations)
        
        # Run simulations
        for i in range(n_simulations):
            # Simulate price path
            prices = self.simulate_price_path(
                spot=spot,
                days=days_to_maturity,
                domestic_rate=domestic_rate,
                foreign_rate=foreign_rate,
                volatility=volatility
            )
            
            # Store final price
            final_prices[i] = prices[-1]
        
        # Calculate option payoff
        if option_type.lower() == 'call':
            payoffs = np.maximum(0, final_prices - strike)
        else:
            payoffs = np.maximum(0, strike - final_prices)
        
        # Calculate option price
        option_price = np.mean(payoffs) * np.exp(-domestic_rate * T)
        
        return option_price
    
    def price_options_portfolio(self, options_data, market_data, use_monte_carlo=False):
        """
        Price a portfolio of European FX options using the Merton Jump-Diffusion model.
        
        Args:
            options_data (list or pandas.DataFrame): Portfolio of options.
            market_data (tuple): Tuple of (spot_rates, volatility, interest_rates) DataFrames.
            use_monte_carlo (bool): Whether to use Monte Carlo simulation instead of analytical formula.
            
        Returns:
            pandas.DataFrame: Options data with added pricing information.
        """
        if isinstance(options_data, list):
            df = pd.DataFrame(options_data)
        else:
            df = options_data.copy()
        
        # Unpack market data
        spot_rates, volatilities, interest_rates = market_data
        
        # Add Jump-Diffusion price column if it doesn't exist
        if 'jd_price' not in df.columns:
            df['jd_price'] = np.nan
        
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
                
                # Price the option
                if use_monte_carlo:
                    price = self.monte_carlo_price(
                        spot=spot,
                        strike=option['strike_price'],
                        days_to_maturity=option['days_to_maturity'],
                        domestic_rate=domestic_rate,
                        foreign_rate=foreign_rate,
                        volatility=vol,
                        option_type=option['type']
                    )
                else:
                    price = self.price_option(
                        spot=spot,
                        strike=option['strike_price'],
                        days_to_maturity=option['days_to_maturity'],
                        domestic_rate=domestic_rate,
                        foreign_rate=foreign_rate,
                        volatility=vol,
                        option_type=option['type']
                    )
                
                # Update option price
                df.at[i, 'jd_price'] = price
                
            except Exception as e:
                logger.error(f"Error pricing option {option['option_id']} with Jump-Diffusion: {e}")
        
        return df
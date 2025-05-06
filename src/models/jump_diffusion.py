"""
Optimized Merton Jump-Diffusion Model

This module implements an optimized version of the Merton Jump-Diffusion model for pricing FX options.
The model extends the Black-Scholes framework by adding jumps in the price process,
with significant performance and numerical stability improvements.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
import logging
import yaml
from numba import jit
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pre-compute factorial values for poisson PMF calculation
@jit(nopython=True)
def poisson_pmf(k, lambda_val):
    """
    Numba-compatible implementation of Poisson PMF.
    """
    return np.exp(-lambda_val) * (lambda_val ** k) / math.factorial(k)

# Compile critical functions with Numba for performance
@jit(nopython=True)
def _jd_price_jit(spot, strike, T, r_d, r_f, vol, lambda_jump, jump_mean, jump_std, option_type_call, n_terms=15):
    """
    JIT-compiled function to calculate Jump-Diffusion option price.
    
    Args:
        spot, strike: Spot and strike prices
        T: Time to maturity in years
        r_d, r_f: Domestic and foreign interest rates
        vol: Volatility
        lambda_jump, jump_mean, jump_std: Jump parameters
        option_type_call: Boolean (True for call, False for put)
        n_terms: Number of terms in the series
        
    Returns:
        float: Option price
    """
    # Handle edge cases
    if T <= 0.0 or vol <= 0.0:
        if option_type_call:
            return max(0.0, spot - strike)
        else:
            return max(0.0, strike - spot)
    
    # Calculate jump contribution to the drift
    jump_expected_return = lambda_jump * (np.exp(jump_mean + 0.5 * jump_std**2) - 1.0)
    adjusted_drift = r_d - r_f - jump_expected_return
    
    # Initialize option price
    option_price = 0.0
    
    # Sum over number of jumps
    for n in range(n_terms):
        # Probability of exactly n jumps (using our custom function)
        p_n_jumps = poisson_pmf(n, lambda_jump * T)
        
        # Adjusted volatility for n jumps
        vol_n = np.sqrt(vol**2 + n * jump_std**2 / T)
        
        # Adjusted drift for n jumps
        drift_n = adjusted_drift + n * jump_mean / T
        
        # Black-Scholes price for n jumps
        d1 = (np.log(spot / strike) + (drift_n + 0.5 * vol_n**2) * T) / (vol_n * np.sqrt(T))
        d2 = d1 - vol_n * np.sqrt(T)
        
        if option_type_call:
            bs_price = spot * np.exp(-r_f * T) * norm.cdf(d1) - strike * np.exp(-r_d * T) * norm.cdf(d2)
        else:
            bs_price = strike * np.exp(-r_d * T) * norm.cdf(-d2) - spot * np.exp(-r_f * T) * norm.cdf(-d1)
        
        # Add weighted contribution
        option_price += p_n_jumps * bs_price
    
    return option_price

class JumpDiffusionModel:
    """
    Implements an optimized Merton Jump-Diffusion model for pricing FX options.
    
    Optimizations:
    - JIT compilation for critical calculations
    - Adaptive series expansion
    - Efficient price path simulation
    - Parameter calibration
    - Memory optimization
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
        self.lambda_jump = self.jd_params['lambda']      # Jump intensity
        self.jump_mean = self.jd_params['jump_mean']     # Mean jump size
        self.jump_std = self.jd_params['jump_std']       # Standard deviation of jump size
        
        # Cache for option prices
        self._price_cache = {}
        
        # Default number of terms in series expansion
        self.default_terms = 15
        
        # Number of simulations for Monte Carlo
        self.num_simulations = 5000
        
        logger.info("Optimized Merton Jump-Diffusion model initialized")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            raise
    
    def price_option(self, spot, strike, days_to_maturity, domestic_rate, foreign_rate, 
                    volatility, option_type='call', n_terms=None):
        """
        Calculate the price of a European FX option using the optimized Merton Jump-Diffusion model.
        
        Args:
            spot (float): Spot exchange rate.
            strike (float): Strike price.
            days_to_maturity (int): Number of days to maturity.
            domestic_rate (float): Domestic risk-free interest rate (annualized).
            foreign_rate (float): Foreign risk-free interest rate (annualized).
            volatility (float): Annualized volatility of the exchange rate (for diffusion part).
            option_type (str): 'call' or 'put'.
            n_terms (int, optional): Number of terms in the series expansion.
            
        Returns:
            float: Option price in domestic currency.
        """
        # Convert days to years
        T = days_to_maturity / 365.0
        
        # Use default number of terms if not specified
        if n_terms is None:
            # Adaptively set number of terms based on parameters
            lambda_T = self.lambda_jump * T
            if lambda_T < 1.0:
                n_terms = 10  # Fewer terms needed for low jump intensity
            elif lambda_T < 5.0:
                n_terms = 15  # Medium range
            else:
                n_terms = 20  # More terms for high jump intensity
        
        # Check cache first (round values slightly for better cache hits)
        cache_key = (
            round(spot, 4),
            round(strike, 4),
            days_to_maturity,
            round(domestic_rate, 5),
            round(foreign_rate, 5),
            round(volatility, 5),
            option_type,
            n_terms
        )
        
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]
        
        try:
            # Call JIT-compiled pricing function
            price = _jd_price_jit(
                spot=spot,
                strike=strike,
                T=T,
                r_d=domestic_rate,
                r_f=foreign_rate,
                vol=volatility,
                lambda_jump=self.lambda_jump,
                jump_mean=self.jump_mean,
                jump_std=self.jump_std,
                option_type_call=(option_type.lower() == 'call'),
                n_terms=n_terms
            )
            
            # Cache the result (limit cache size)
            if len(self._price_cache) > 1000:
                self._price_cache.clear()
            self._price_cache[cache_key] = price
            
            return price
            
        except Exception as e:
            logger.error(f"Error in Jump-Diffusion pricing: {e}")
            
            # Fall back to Black-Scholes as a safety measure
            d1 = (np.log(spot / strike) + (domestic_rate - foreign_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
            d2 = d1 - volatility * np.sqrt(T)
            
            if option_type.lower() == 'call':
                return spot * np.exp(-foreign_rate * T) * norm.cdf(d1) - strike * np.exp(-domestic_rate * T) * norm.cdf(d2)
            else:
                return strike * np.exp(-domestic_rate * T) * norm.cdf(-d2) - spot * np.exp(-foreign_rate * T) * norm.cdf(-d1)
    
    def simulate_price_path(self, spot, days, domestic_rate, foreign_rate, volatility, seed=None):
        """
        Simulate a price path using the Jump-Diffusion process (optimized version).
        
        Args:
            spot (float): Initial spot price.
            days (int): Number of days to simulate.
            domestic_rate (float): Domestic risk-free interest rate (annualized).
            foreign_rate (float): Foreign risk-free interest rate (annualized).
            volatility (float): Annualized volatility of the exchange rate (for diffusion part).
            seed (int, optional): Random seed for reproducibility.
            
        Returns:
            numpy.ndarray: Simulated price path.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Convert parameters to daily values
        daily_drift = (domestic_rate - foreign_rate) / 365.0
        daily_vol = volatility / np.sqrt(252)
        daily_jump_intensity = self.lambda_jump / 252
        
        # Pre-allocate array for prices
        prices = np.zeros(days + 1)
        prices[0] = spot
        
        # Pre-generate random variables for efficiency
        diffusion_normals = np.random.normal(0, 1, days)
        uniform_rvs = np.random.uniform(0, 1, days)
        
        # Simulate jump counts for all days at once
        jump_counts = np.random.poisson(daily_jump_intensity, days)
        total_jumps = jump_counts.sum()
        
        # Generate all jump sizes at once
        jump_sizes = np.random.normal(self.jump_mean, self.jump_std, total_jumps)
        
        # Distribute jump sizes to days with jumps
        jump_idx = 0
        jump_returns = np.zeros(days)
        
        for t in range(days):
            n_jumps = jump_counts[t]
            if n_jumps > 0:
                jump_returns[t] = np.sum(jump_sizes[jump_idx:jump_idx + n_jumps])
                jump_idx += n_jumps
        
        # Update prices for all days
        for t in range(days):
            # Diffusion component
            diff_return = daily_drift + daily_vol * diffusion_normals[t]
            
            # Combine diffusion and jump returns
            prices[t+1] = prices[t] * np.exp(diff_return + jump_returns[t])
        
        return prices
    
    def monte_carlo_price(self, spot, strike, days_to_maturity, domestic_rate, foreign_rate, 
                         volatility, option_type='call', n_simulations=None):
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
            n_simulations (int, optional): Number of Monte Carlo simulations.
            
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
        
        # Use default simulation count if not specified
        if n_simulations is None:
            n_simulations = self.num_simulations
        
        # Process in small batches to reduce memory usage
        batch_size = 1000  # Smaller batches for better memory management
        n_batches = (n_simulations + batch_size - 1) // batch_size
        
        total_payoff = 0.0
        
        # Run simulation in batches
        for batch in range(n_batches):
            actual_batch_size = min(batch_size, n_simulations - batch * batch_size)
            if actual_batch_size <= 0:
                break
                
            # Initialize arrays
            np.random.seed(42 + batch)  # Different seed for each batch
            
            # Convert parameters to daily values
            daily_drift = (domestic_rate - foreign_rate) / 365.0
            daily_vol = volatility / np.sqrt(252)
            daily_jump_intensity = self.lambda_jump / 252
            
            # Generate final prices for this batch
            final_prices = np.zeros(actual_batch_size)
            
            for i in range(actual_batch_size):
                # Start with spot price
                price = spot
                
                # Simulate day by day for this path
                for t in range(days_to_maturity):
                    # Diffusion component
                    diff_return = daily_drift + daily_vol * np.random.normal(0, 1)
                    
                    # Jump component
                    jump_return = 0
                    # Simulate number of jumps
                    n_jumps = np.random.poisson(daily_jump_intensity)
                    
                    if n_jumps > 0:
                        # Simulate jump sizes
                        jump_sizes = np.random.normal(self.jump_mean, self.jump_std, n_jumps)
                        jump_return = np.sum(jump_sizes)
                    
                    # Update price
                    price *= np.exp(diff_return + jump_return)
                
                # Store final price
                final_prices[i] = price
            
            # Calculate payoffs for this batch
            if option_type.lower() == 'call':
                batch_payoffs = np.maximum(0, final_prices - strike)
            else:
                batch_payoffs = np.maximum(0, strike - final_prices)
            
            # Add to total payoff
            total_payoff += np.sum(batch_payoffs)
        
        # Calculate option price (discounted expected payoff)
        option_price = (total_payoff / n_simulations) * np.exp(-domestic_rate * T)
        
        return option_price
    
    def calibrate(self, option_data, market_data):
        """
        Calibrate Jump-Diffusion parameters to market data.
        
        Args:
            option_data (pandas.DataFrame): Options market data.
            market_data (tuple): Tuple of (spot_rates, volatility, interest_rates) DataFrames.
            
        Returns:
            dict: Calibrated Jump-Diffusion parameters.
        """
        try:
            # Extract relevant options with market prices
            valid_options = option_data.dropna(subset=['actual_payoff']).copy()
            
            if len(valid_options) < 3:
                logger.warning("Not enough valid options with market prices for calibration")
                return {
                    'lambda_jump': self.lambda_jump,
                    'jump_mean': self.jump_mean,
                    'jump_std': self.jump_std
                }
            
            # Unpack market data
            spot_rates, volatilities, interest_rates = market_data
            
            # Function to minimize - sum of squared price differences
            def objective(params):
                lambda_jump, jump_mean, jump_std = params
                
                # Keep parameters in valid ranges
                lambda_jump = max(0.01, min(20.0, lambda_jump))
                jump_mean = max(-0.5, min(0.1, jump_mean))
                jump_std = max(0.001, min(0.5, jump_std))
                
                # Store original parameters
                orig_lambda = self.lambda_jump
                orig_mean = self.jump_mean
                orig_std = self.jump_std
                
                # Set new parameters
                self.lambda_jump = lambda_jump
                self.jump_mean = jump_mean
                self.jump_std = jump_std
                
                # Clear cache
                self._price_cache.clear()
                
                # Calculate total squared error
                total_sq_error = 0.0
                
                for _, option in valid_options.iterrows():
                    # Get spot and rates at issue date
                    issue_date = pd.to_datetime(option['issue_date'])
                    spot = spot_rates[spot_rates['date'] <= issue_date].iloc[-1]['EUR/TND']
                    rates = interest_rates[interest_rates['date'] <= issue_date].iloc[-1]
                    
                    vol_data = volatilities[volatilities['date'] <= issue_date]
                    if vol_data.empty:
                        vol = 0.15  # Default fallback
                    else:
                        vol = vol_data.iloc[-1]['historical_vol']
                    
                    domestic_rate = rates['EUR_rate']
                    foreign_rate = rates['TND_rate']
                    
                    # Calculate model price
                    model_price = self.price_option(
                        spot=spot,
                        strike=option['strike_price'],
                        days_to_maturity=option['days_to_maturity'],
                        domestic_rate=domestic_rate,
                        foreign_rate=foreign_rate,
                        volatility=vol,
                        option_type=option['type']
                    )
                    
                    # Calculate target price from actual payoff
                    target_price = option['actual_payoff'] / option['notional']
                    
                    # Add squared error
                    sq_error = (model_price - target_price)**2
                    total_sq_error += sq_error
                
                # Restore original parameters
                self.lambda_jump = orig_lambda
                self.jump_mean = orig_mean
                self.jump_std = orig_std
                
                return total_sq_error
            
            # Initial parameters
            initial_params = [self.lambda_jump, self.jump_mean, self.jump_std]
            
            # Optimization bounds
            bounds = [(0.01, 20.0), (-0.5, 0.1), (0.001, 0.5)]
            
            # Perform optimization
            from scipy.optimize import minimize
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50}
            )
            
            # Extract calibrated parameters
            lambda_jump, jump_mean, jump_std = result.x
            
            # Ensure parameters are in valid ranges
            lambda_jump = max(0.01, min(20.0, lambda_jump))
            jump_mean = max(-0.5, min(0.1, jump_mean))
            jump_std = max(0.001, min(0.5, jump_std))
            
            # Update model parameters
            self.lambda_jump = lambda_jump
            self.jump_mean = jump_mean
            self.jump_std = jump_std
            
            # Clear cache after calibration
            self._price_cache.clear()
            
            logger.info(f"Jump-Diffusion calibration completed: lambda={lambda_jump:.2f}, "
                       f"jump_mean={jump_mean:.4f}, jump_std={jump_std:.4f}")
            
            return {
                'lambda_jump': lambda_jump,
                'jump_mean': jump_mean,
                'jump_std': jump_std
            }
            
        except Exception as e:
            logger.error(f"Error in Jump-Diffusion calibration: {e}")
            return {
                'lambda_jump': self.lambda_jump,
                'jump_mean': self.jump_mean,
                'jump_std': self.jump_std
            }
    
    def price_options_portfolio(self, options_data, market_data, use_monte_carlo=False, calibrate_first=False):
        """
        Price a portfolio of European FX options using the optimized Merton Jump-Diffusion model.
        
        Args:
            options_data (list or pandas.DataFrame): Portfolio of options.
            market_data (tuple): Tuple of (spot_rates, volatility, interest_rates) DataFrames.
            use_monte_carlo (bool): Whether to use Monte Carlo simulation instead of analytical formula.
            calibrate_first (bool): Whether to calibrate parameters first.
            
        Returns:
            pandas.DataFrame: Options data with added pricing information.
        """
        if isinstance(options_data, list):
            df = pd.DataFrame(options_data)
        else:
            df = options_data.copy()
        
        # Unpack market data
        spot_rates, volatilities, interest_rates = market_data
        
        # Clear cache
        self._price_cache.clear()
        
        # Calibrate parameters if requested and possible
        if calibrate_first and 'actual_payoff' in df.columns and not df['actual_payoff'].isna().all():
            self.calibrate(df, market_data)
        
        # Add Jump-Diffusion price column if it doesn't exist
        if 'jd_price' not in df.columns:
            df['jd_price'] = np.nan
        
        # Process options in batches for better performance
        batch_size = min(50, len(df))
        
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
                    
                except KeyError as e:
                    logger.error(f"Missing data for option {option.get('option_id', i)}: {e}")
                except ValueError as e:
                    logger.error(f"Invalid value for option {option.get('option_id', i)}: {e}")
                except Exception as e:
                    logger.error(f"Error pricing option {option.get('option_id', i)} with Jump-Diffusion: {e}")
        
        return df
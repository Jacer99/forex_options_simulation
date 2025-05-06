"""
Optimized SABR Model for Stochastic Volatility

This module implements an optimized version of the SABR (Stochastic Alpha, Beta, Rho) model 
for pricing FX options with enhanced numerical stability, calibration, and performance.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import logging
import yaml
from scipy.optimize import minimize
from numba import jit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use JIT compilation to speed up critical functions
@jit(nopython=True)
def _sabr_vol_jit(F, K, T, alpha, beta, rho, nu):
    """
    JIT-compiled function to calculate SABR implied volatility.
    This provides significant speedup for the critical calculation.
    """
    # Handle edge cases
    if T <= 0.0 or alpha <= 0.0:
        return 0.15  # Default fallback volatility
    
    # For ATM case (F ≈ K)
    if abs(F - K) < 1e-10:
        # ATM SABR formula
        powFmB = F**(1.0 - beta)
        logFK = 0.0  # For ATM, log(F/K) = 0
        zz = (nu/alpha) * powFmB * logFK
        
        if abs(beta - 1.0) < 1e-10:
            # Log-normal case (β = 1)
            I1 = 1.0
        else:
            # Non-log-normal case
            I1 = 1.0 + (((1.0 - beta)**2)/24.0) * ((alpha/powFmB)**2) * T

        I2 = 1.0 + 0.25 * rho * beta * nu * alpha * T / powFmB
        I3 = 1.0 + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2 * T
        
        return (alpha / powFmB) * I1 * I2 * I3
    
    # For non-ATM case
    logFK = np.log(F / K)
    powFmB = F**(1.0 - beta)
    powKmB = K**(1.0 - beta)
    powFmBpowKmB = (powFmB + powKmB) / 2.0
    
    # Calculate z
    z = (nu/alpha) * powFmBpowKmB * logFK
    
    # Calculate x(z)
    if abs(z) < 1e-6:
        # For small z, use Taylor expansion
        xz = logFK * (1.0 + (1.0 - rho) * z / 2.0 + (2.0 - 3.0 * rho**2) * z**2 / 12.0)
    else:
        # Standard formula
        temp = np.sqrt(1.0 - 2.0 * rho * z + z**2)
        xz = np.log((temp + z - rho) / (1.0 - rho))
    
    # Calculate the volatility multiplier terms
    if abs(beta - 1.0) < 1e-10:
        # Log-normal case (β = 1)
        I1 = 1.0
    else:
        # Non-log-normal case
        I1 = 1.0 + (((1.0 - beta)**2)/24.0) * ((alpha/powFmBpowKmB)**2) * T
    
    I2 = 1.0 + 0.25 * rho * beta * nu * alpha * T / powFmBpowKmB
    I3 = 1.0 + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2 * T
    
    # Calculate implied volatility
    vol = (alpha / powFmBpowKmB) * (logFK / xz) * I1 * I2 * I3
    
    return max(0.001, vol)  # Ensure positive volatility

class SABRModel:
    """
    Implements an optimized SABR model for pricing FX options.
    
    Optimizations:
    - JIT compilation for critical calculations
    - Improved numerical stability
    - Enhanced parameter calibration
    - Memory-efficient Monte Carlo
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize the SABR model.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.sabr_params = self.config['models']['sabr']
        
        # SABR parameters
        self.alpha = self.sabr_params['alpha']  # Initial volatility
        self.beta = self.sabr_params['beta']    # CEV parameter (0 = normal, 1 = lognormal)
        self.rho = self.sabr_params['rho']      # Correlation between spot and vol
        self.nu = self.sabr_params['nu']        # Volatility of volatility
        
        # Reduced number of simulations for faster runtime
        self.num_simulations = 1000
        
        # Cache for storing calculated implied volatilities
        self._vol_cache = {}
        
        logger.info("Optimized SABR model initialized")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            raise
    
    def implied_volatility(self, spot, strike, days_to_maturity, alpha=None, beta=None, rho=None, nu=None):
        """
        Calculate the implied volatility using the SABR model with caching and JIT acceleration.
        
        Args:
            spot (float): Spot exchange rate.
            strike (float): Strike price.
            days_to_maturity (int): Number of days to maturity.
            alpha, beta, rho, nu: SABR parameters.
            
        Returns:
            float: Implied volatility according to the SABR model.
        """
        # Use default SABR parameters if not provided
        alpha = alpha if alpha is not None else self.alpha
        beta = beta if beta is not None else self.beta
        rho = rho if rho is not None else self.rho
        nu = nu if nu is not None else self.nu
        
        # Convert days to years
        T = max(days_to_maturity / 365.0, 1e-10)
        
        # Create cache key - round inputs slightly for better cache hit rate
        key = (round(spot, 4), round(strike, 4), days_to_maturity, 
               round(alpha, 4), round(beta, 2), round(rho, 2), round(nu, 2))
        
        # Check cache first
        if key in self._vol_cache:
            return self._vol_cache[key]
        
        try:
            # Call JIT-compiled function for speed
            vol = _sabr_vol_jit(spot, strike, T, alpha, beta, rho, nu)
            
            # Store in cache
            if len(self._vol_cache) > 1000:  # Prevent unlimited growth
                self._vol_cache.clear()
            self._vol_cache[key] = vol
            
            return vol
            
        except Exception as e:
            logger.warning(f"Error in SABR implied volatility calculation: {e}")
            # Fallback to a reasonable value
            return max(0.1, alpha)
    
    def price_option(self, spot, strike, days_to_maturity, domestic_rate, foreign_rate,
                   alpha=None, beta=None, rho=None, nu=None, option_type='call'):
        """
        Calculate the price of a European FX option using the optimized SABR model.
        
        Args:
            spot (float): Spot exchange rate.
            strike (float): Strike price.
            days_to_maturity (int): Number of days to maturity.
            domestic_rate (float): Domestic risk-free interest rate (annualized).
            foreign_rate (float): Foreign risk-free interest rate (annualized).
            alpha, beta, rho, nu: SABR parameters.
            option_type (str): 'call' or 'put'.
            
        Returns:
            float: Option price in domestic currency.
        """
        # Convert days to years
        T = max(days_to_maturity / 365.0, 1e-10)
        
        # Handle edge cases
        if T <= 1e-10:
            if option_type.lower() == 'call':
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)
        
        try:
            # Calculate the implied volatility using the SABR model
            vol = self.implied_volatility(spot, strike, days_to_maturity, alpha, beta, rho, nu)
            
            # Use Black-Scholes formula with the SABR implied volatility
            d1 = (np.log(spot / strike) + (domestic_rate - foreign_rate + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)
            
            if option_type.lower() == 'call':
                # Call option price
                price = spot * np.exp(-foreign_rate * T) * norm.cdf(d1) - strike * np.exp(-domestic_rate * T) * norm.cdf(d2)
            else:
                # Put option price
                price = strike * np.exp(-domestic_rate * T) * norm.cdf(-d2) - spot * np.exp(-foreign_rate * T) * norm.cdf(-d1)
            
            return max(0.0, price)  # Ensure non-negative price
            
        except Exception as e:
            logger.warning(f"Error in SABR pricing: {e}")
            # Fall back to intrinsic value
            if option_type.lower() == 'call':
                return max(0, spot - strike * np.exp(-domestic_rate * T))
            else:
                return max(0, strike * np.exp(-domestic_rate * T) - spot)
    
    def calibrate(self, option_data, market_data):
        """
        Calibrate SABR parameters to market data.
        
        Args:
            option_data (pandas.DataFrame): Options market data.
            market_data (tuple): Tuple of (spot_rates, volatility, interest_rates) DataFrames.
            
        Returns:
            dict: Calibrated SABR parameters.
        """
        try:
            # Extract relevant options with market prices
            valid_options = option_data.dropna(subset=['actual_payoff']).copy()
            
            if len(valid_options) < 3:
                logger.warning("Not enough valid options with market prices for calibration")
                return {'alpha': self.alpha, 'beta': self.beta, 'rho': self.rho, 'nu': self.nu}
            
            # Function to minimize - sum of squared price differences
            def objective(params):
                alpha, beta, rho, nu = params
                
                # Keep parameters in valid ranges
                alpha = max(0.001, min(1.0, alpha))
                beta = max(0.0, min(1.0, beta))
                rho = max(-0.999, min(0.999, rho))
                nu = max(0.001, min(2.0, nu))
                
                total_sq_error = 0.0
                
                for _, option in valid_options.iterrows():
                    # Get spot and rates at issue date
                    issue_date = pd.to_datetime(option['issue_date'])
                    spot = spot_rates[spot_rates['date'] <= issue_date].iloc[-1]['EUR/TND']
                    rates = interest_rates[interest_rates['date'] <= issue_date].iloc[-1]
                    domestic_rate = rates['EUR_rate']
                    foreign_rate = rates['TND_rate']
                    
                    # Calculate model price
                    model_price = self.price_option(
                        spot=spot,
                        strike=option['strike_price'],
                        days_to_maturity=option['days_to_maturity'],
                        domestic_rate=domestic_rate,
                        foreign_rate=foreign_rate,
                        alpha=alpha,
                        beta=beta,
                        rho=rho,
                        nu=nu,
                        option_type=option['type']
                    )
                    
                    # Calculate target price from actual payoff
                    target_price = option['actual_payoff'] / option['notional']
                    
                    # Add squared error
                    sq_error = (model_price - target_price)**2
                    total_sq_error += sq_error
                
                return total_sq_error
            
            # Initial parameters
            initial_params = [self.alpha, self.beta, self.rho, self.nu]
            
            # Optimization bounds
            bounds = [(0.001, 1.0), (0.0, 1.0), (-0.999, 0.999), (0.001, 2.0)]
            
            # Unpack market data
            spot_rates, volatilities, interest_rates = market_data
            
            # Perform optimization
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50}
            )
            
            # Extract calibrated parameters
            alpha, beta, rho, nu = result.x
            
            # Ensure parameters are in valid ranges
            alpha = max(0.001, min(1.0, alpha))
            beta = max(0.0, min(1.0, beta))
            rho = max(-0.999, min(0.999, rho))
            nu = max(0.001, min(2.0, nu))
            
            # Update model parameters
            self.alpha = alpha
            self.beta = beta
            self.rho = rho
            self.nu = nu
            
            # Clear cache after calibration
            self._vol_cache.clear()
            
            logger.info(f"SABR calibration completed: alpha={alpha:.4f}, beta={beta:.4f}, rho={rho:.4f}, nu={nu:.4f}")
            
            return {'alpha': alpha, 'beta': beta, 'rho': rho, 'nu': nu}
            
        except Exception as e:
            logger.error(f"Error in SABR calibration: {e}")
            return {'alpha': self.alpha, 'beta': self.beta, 'rho': self.rho, 'nu': self.nu}
    
    def price_options_portfolio(self, options_data, market_data, use_monte_carlo=False, calibrate_first=False):
        """
        Price a portfolio of European FX options using the optimized SABR model.
        
        Args:
            options_data (list or pandas.DataFrame): Portfolio of options.
            market_data (tuple): Tuple of (spot_rates, volatility, interest_rates) DataFrames.
            use_monte_carlo (bool): Whether to use Monte Carlo simulation.
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
        
        # Clear volatility cache
        self._vol_cache.clear()
        
        # Calibrate parameters if requested and possible
        if calibrate_first and 'actual_payoff' in df.columns and not df['actual_payoff'].isna().all():
            self.calibrate(df, market_data)
        
        # Add SABR price column if it doesn't exist
        if 'sabr_price' not in df.columns:
            df['sabr_price'] = np.nan
        
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
                    
                    # Get volatility at issue date for initial alpha
                    vol_data = volatilities[volatilities['date'] <= issue_date]
                    if vol_data.empty:
                        logger.warning(f"No volatility data for {issue_date} for option {option['option_id']}")
                        hist_vol = 0.15  # Default value
                    else:
                        hist_vol = vol_data.iloc[-1]['historical_vol']
                    
                    # Use implied volatility if available, otherwise historical vol for alpha
                    local_alpha = option.get('implied_volatility', hist_vol)
                    
                    # Get interest rates at issue date
                    rates_data = interest_rates[interest_rates['date'] <= issue_date]
                    if rates_data.empty:
                        logger.warning(f"No interest rate data for {issue_date} for option {option['option_id']}")
                        continue
                    rates = rates_data.iloc[-1]
                    domestic_rate = rates['EUR_rate']
                    foreign_rate = rates['TND_rate']
                    
                    # Price the option using analytical formula
                    price = self.price_option(
                        spot=spot,
                        strike=option['strike_price'],
                        days_to_maturity=option['days_to_maturity'],
                        domestic_rate=domestic_rate,
                        foreign_rate=foreign_rate,
                        alpha=local_alpha,
                        beta=self.beta,
                        rho=self.rho,
                        nu=self.nu,
                        option_type=option['type']
                    )
                    
                    # Update option price
                    df.at[i, 'sabr_price'] = price
                    
                except KeyError as e:
                    logger.error(f"Missing data for option {option.get('option_id', i)}: {e}")
                except ValueError as e:
                    logger.error(f"Invalid value for option {option.get('option_id', i)}: {e}")
                except Exception as e:
                    logger.error(f"Error pricing option {option.get('option_id', i)} with SABR: {e}")
        
        return df
    
    def monte_carlo_price(self, spot, strike, days_to_maturity, domestic_rate, foreign_rate,
                         alpha=None, beta=None, rho=None, nu=None, option_type='call', n_steps=50):
        """
        Price an option using a memory-efficient Monte Carlo simulation of the SABR model.
        
        Args:
            spot (float): Spot exchange rate.
            strike (float): Strike price.
            days_to_maturity (int): Number of days to maturity.
            domestic_rate (float): Domestic risk-free interest rate (annualized).
            foreign_rate (float): Foreign risk-free interest rate (annualized).
            alpha, beta, rho, nu: SABR parameters.
            option_type (str): 'call' or 'put'.
            n_steps (int): Number of time steps in the simulation.
            
        Returns:
            float: Option price in domestic currency.
        """
        # Use default SABR parameters if not provided
        alpha = alpha if alpha is not None else self.alpha
        beta = beta if beta is not None else self.beta
        rho = rho if rho is not None else self.rho
        nu = nu if nu is not None else self.nu
        
        # Convert days to years
        T = days_to_maturity / 365.0
        
        # Handle edge cases
        if T <= 0:
            if option_type.lower() == 'call':
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)
        
        # Time step
        dt = T / n_steps
        
        # Number of simulations
        n_sims = self.num_simulations
        
        # Drift adjustment for risk-neutral measure
        drift = domestic_rate - foreign_rate
        
        # Initialize arrays for efficiency
        np.random.seed(42)  # For reproducibility
        
        # Process in small batches to reduce memory usage
        batch_size = 1000  # Smaller batches for better memory management
        n_batches = (n_sims + batch_size - 1) // batch_size
        
        total_payoff = 0.0
        
        # Run simulation in batches
        for batch in range(n_batches):
            actual_batch_size = min(batch_size, n_sims - batch * batch_size)
            if actual_batch_size <= 0:
                break
                
            # Initialize paths for this batch
            spot_paths = np.full(actual_batch_size, spot)
            vol_paths = np.full(actual_batch_size, alpha)
            
            # Simulate paths for this batch
            for t in range(n_steps):
                # Generate correlated random variables
                z1 = np.random.normal(0, 1, actual_batch_size)
                z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, actual_batch_size)
                
                # Update volatility (prevent negative values)
                vol_paths = np.maximum(0.0001, vol_paths * np.exp(-0.5 * nu**2 * dt + nu * np.sqrt(dt) * z2))
                
                # Update spot price (handle numerical issues based on beta)
                if abs(beta - 0.0) < 1e-10:  # Normal case (β = 0)
                    spot_paths = np.maximum(0.0001, spot_paths + 
                                  drift * spot_paths * dt + 
                                  vol_paths * np.sqrt(dt) * z1)
                elif abs(beta - 1.0) < 1e-10:  # Lognormal case (β = 1)
                    spot_paths = np.maximum(0.0001, spot_paths * 
                                  np.exp((drift - 0.5 * vol_paths**2) * dt + 
                                        vol_paths * np.sqrt(dt) * z1))
                else:  # General CEV case
                    spot_paths = np.maximum(0.0001, spot_paths * 
                                  np.exp((drift - 0.5 * (vol_paths * spot_paths**(beta-1))**2) * dt + 
                                        vol_paths * spot_paths**(beta-1) * np.sqrt(dt) * z1))
            
            # Calculate payoffs for this batch
            if option_type.lower() == 'call':
                batch_payoffs = np.maximum(0, spot_paths - strike)
            else:
                batch_payoffs = np.maximum(0, strike - spot_paths)
            
            # Add to total payoff sum
            total_payoff += np.sum(batch_payoffs)
        
        # Calculate option price (discounted expected payoff)
        option_price = (total_payoff / n_sims) * np.exp(-domestic_rate * T)
        
        return option_price
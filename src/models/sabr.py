"""
SABR Model for Stochastic Volatility

This module implements the SABR (Stochastic Alpha, Beta, Rho) model for pricing FX options.
The model accounts for the stochastic nature of volatility and its correlation with the
underlying asset price.
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

class SABRModel:
    """
    Implements the SABR model for pricing FX options.
    
    The SABR model assumes that both the asset price and its volatility follow
    stochastic processes, with a correlation between the two.
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize the SABR model.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        try:
            self.config = self._load_config(config_path)
            self.sabr_params = self.config['models']['sabr']
            
            # SABR parameters
            self.alpha = self.sabr_params['alpha']  # Initial volatility
            self.beta = self.sabr_params['beta']    # CEV parameter (0 = normal, 1 = lognormal)
            self.rho = self.sabr_params['rho']      # Correlation between spot and vol
            self.nu = self.sabr_params['nu']        # Volatility of volatility
            
            # Reduce number of simulations for Monte Carlo
            self.num_simulations = 1000  # Reduced for faster runtime
            
            logger.info("SABR model initialized")
        except Exception as e:
            logger.error(f"Error initializing SABR model: {e}")
            raise
    
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
    
    def implied_volatility(self, spot, strike, days_to_maturity, alpha=None, beta=None, rho=None, nu=None):
        """
        Calculate the implied volatility using the SABR model.
        
        Args:
            spot (float): Spot exchange rate.
            strike (float): Strike price.
            days_to_maturity (int): Number of days to maturity.
            alpha (float, optional): Initial volatility parameter. Defaults to self.alpha.
            beta (float, optional): CEV parameter. Defaults to self.beta.
            rho (float, optional): Correlation parameter. Defaults to self.rho.
            nu (float, optional): Volatility of volatility parameter. Defaults to self.nu.
            
        Returns:
            float: Implied volatility according to the SABR model.
        """
        try:
            # Use default SABR parameters if not provided
            alpha = alpha if alpha is not None else self.alpha
            beta = beta if beta is not None else self.beta
            rho = rho if rho is not None else self.rho
            nu = nu if nu is not None else self.nu
            
            # Convert days to years
            T = max(days_to_maturity / 365.0, 1e-10)  # Avoid division by zero
            
            # Handle edge cases for numerical stability
            if strike < 1e-10:
                strike = 1e-10
            
            # Handle ATM case separately to avoid numerical issues
            if abs(spot - strike) < 1e-10:
                # ATM formula with improved numerical stability
                spotPow1 = spot ** (1 - beta)
                zxz = (alpha * spotPow1) ** 2 * T
                
                atm_vol = alpha / spotPow1 * (1 + 
                          (((1 - beta)**2 / 24) * zxz + 
                          0.25 * rho * beta * nu * alpha * T / spotPow1 + 
                          ((2 - 3 * rho**2) / 24) * nu**2 * T))
                return max(0.001, atm_vol)  # Ensure positive volatility
            
            # Calculate the log of the forward/strike
            log_fk = np.log(spot / strike)
            
            # For beta = 1 (lognormal case)
            if abs(beta - 1.0) < 1e-10:
                # Improved numerical stability for small z values
                z = nu / alpha * log_fk
                
                if abs(z) < 1e-6:
                    # Use Taylor series for small z
                    x_z = log_fk * (1 + (1 - rho) * z / 2 + (2 - 3 * rho**2) * z**2 / 12)
                else:
                    x_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
                
                # Calculate implied volatility
                vol_multiplier = 1 + (((1 - beta)**2 / 24) * alpha**2 
                                  + 0.25 * rho * beta * nu * alpha 
                                  + ((2 - 3 * rho**2) / 24) * nu**2) * T
                
                imp_vol = alpha * log_fk / x_z * vol_multiplier
                return max(0.001, imp_vol)  # Ensure positive volatility
            
            # For beta < 1 (including beta = 0 for normal case)
            # Compute the intermediate values with improved numerical stability
            f_mid = spot**(1 - beta) * strike**beta  # Effective forward price
            z = nu / alpha * f_mid**(beta - 1) * log_fk
            
            # Handle small z values to avoid numerical issues
            if abs(z) < 1e-6:
                # Use Taylor series approximation for small z
                x_z = log_fk * (1 + (1 - rho) * z / 2 + (2 - 3 * rho**2) * z**2 / 12)
            else:
                # Use standard formula with safeguards
                discriminant = max(1e-10, 1 - 2*rho*z + z**2)  # Ensure positive
                x_z = np.log((np.sqrt(discriminant) + z - rho) / max(1e-10, 1 - rho))
            
            # Calculate volatility multiplier with safeguards for numerical stability
            f_avg = 0.5 * (spot + strike)
            f_avg_pow = max(1e-10, f_avg**(2 - 2*beta))  # Prevent division by zero
            
            vol_multiplier = (1 + (((1 - beta)**2 / 24) * alpha**2 / f_avg_pow 
                             + 0.25 * rho * beta * nu * alpha / (f_avg**(1 - beta)) 
                             + ((2 - 3 * rho**2) / 24) * nu**2) * T)
            
            # Calculate the final implied volatility
            vol_term = 1 + (1 - beta)**2 * log_fk**2 / 24 + (1 - beta)**4 * log_fk**4 / 1920
            imp_vol = alpha * vol_term * z / x_z * vol_multiplier
            
            # Ensure positive volatility with a reasonable lower bound
            return max(0.001, imp_vol)
            
        except (ZeroDivisionError, ValueError) as e:
            logger.warning(f"Numerical error in SABR implied volatility calculation: {e}")
            # Fall back to a reasonable volatility value
            return max(0.15, alpha)  # Return input alpha or 15% as fallback
        except Exception as e:
            logger.error(f"Unexpected error in implied volatility calculation: {e}")
            return 0.15  # Default fallback
    
    def price_option(self, spot, strike, days_to_maturity, domestic_rate, foreign_rate,
                   alpha=None, beta=None, rho=None, nu=None, option_type='call'):
        """
        Calculate the price of a European FX option using the SABR model.
        
        Args:
            spot (float): Spot exchange rate.
            strike (float): Strike price.
            days_to_maturity (int): Number of days to maturity.
            domestic_rate (float): Domestic risk-free interest rate (annualized).
            foreign_rate (float): Foreign risk-free interest rate (annualized).
            alpha (float, optional): Initial volatility parameter. Defaults to self.alpha.
            beta (float, optional): CEV parameter. Defaults to self.beta.
            rho (float, optional): Correlation parameter. Defaults to self.rho.
            nu (float, optional): Volatility of volatility parameter. Defaults to self.nu.
            option_type (str): 'call' or 'put'.
            
        Returns:
            float: Option price in domestic currency.
        """
        try:
            # Convert days to years
            T = max(days_to_maturity / 365.0, 1e-10)  # Avoid division by zero
            
            # Handle edge cases
            if T <= 1e-10:
                if option_type.lower() == 'call':
                    return max(0, spot - strike)
                else:
                    return max(0, strike - spot)
            
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
            
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Numerical error in SABR pricing: {e}")
            # Fall back to intrinsic value
            if option_type.lower() == 'call':
                return max(0, spot - strike * np.exp(-domestic_rate * T))
            else:
                return max(0, strike * np.exp(-domestic_rate * T) - spot)
        except Exception as e:
            logger.error(f"Unexpected error in option pricing: {e}")
            return 0.0  # Default fallback
    
    def monte_carlo_price(self, spot, strike, days_to_maturity, domestic_rate, foreign_rate,
                         alpha=None, beta=None, rho=None, nu=None, option_type='call', dt=1/252):
        """
        Price an option using Monte Carlo simulation of the SABR model.
        
        Args:
            spot (float): Spot exchange rate.
            strike (float): Strike price.
            days_to_maturity (int): Number of days to maturity.
            domestic_rate (float): Domestic risk-free interest rate (annualized).
            foreign_rate (float): Foreign risk-free interest rate (annualized).
            alpha (float, optional): Initial volatility parameter. Defaults to self.alpha.
            beta (float, optional): CEV parameter. Defaults to self.beta.
            rho (float, optional): Correlation parameter. Defaults to self.rho.
            nu (float, optional): Volatility of volatility parameter. Defaults to self.nu.
            option_type (str): 'call' or 'put'.
            dt (float): Time step for simulation (in years).
            
        Returns:
            float: Option price in domestic currency.
        """
        try:
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
            
            # Number of time steps
            n_steps = max(1, int(T / dt))
            dt = T / n_steps  # Adjust dt to ensure exact maturity
            
            # Number of simulations (limited for performance)
            n_simulations = self.num_simulations
            
            # Pre-allocate arrays for efficiency
            spot_paths = np.full(n_simulations, spot)
            vol_paths = np.full(n_simulations, alpha)
            
            # Generate correlated random variables for all steps at once
            np.random.seed(42)  # For reproducibility
            z1 = np.random.normal(0, 1, (n_simulations, n_steps))
            z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, (n_simulations, n_steps))
            
            # Drift adjustment for risk-neutral measure
            drift = domestic_rate - foreign_rate
            
            # Simulate paths (vectorized over simulations)
            for t in range(n_steps):
                # Update volatility first (CEV process)
                vol_paths = np.maximum(0.0001, vol_paths * np.exp(-0.5 * nu**2 * dt + nu * np.sqrt(dt) * z2[:, t]))
                
                # Update spot price
                if beta == 0:  # Normal case (avoid numerical issues)
                    spot_paths = np.maximum(0.0001, spot_paths + drift * spot_paths * dt + 
                                        vol_paths * spot_paths**beta * np.sqrt(dt) * z1[:, t])
                else:
                    spot_paths = np.maximum(0.0001, spot_paths * np.exp((drift - 0.5 * (vol_paths * spot_paths**(beta-1))**2) * dt + 
                                                     vol_paths * spot_paths**(beta-1) * np.sqrt(dt) * z1[:, t]))
            
            # Calculate payoffs (vectorized)
            if option_type.lower() == 'call':
                payoffs = np.maximum(0, spot_paths - strike)
            else:
                payoffs = np.maximum(0, strike - spot_paths)
            
            # Calculate option price (discounted expected payoff)
            option_price = np.mean(payoffs) * np.exp(-domestic_rate * T)
            
            return option_price
            
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as e:
            logger.warning(f"Numerical error in Monte Carlo simulation: {e}")
            # Fall back to analytical formula
            return self.price_option(spot, strike, days_to_maturity, domestic_rate, 
                                  foreign_rate, alpha, beta, rho, nu, option_type)
        except Exception as e:
            logger.error(f"Unexpected error in Monte Carlo simulation: {e}")
            return 0.0  # Default fallback
    
    def price_options_portfolio(self, options_data, market_data, use_monte_carlo=False):
        """
        Price a portfolio of European FX options using the SABR model.
        
        Args:
            options_data (list or pandas.DataFrame): Portfolio of options.
            market_data (tuple): Tuple of (spot_rates, volatility, interest_rates) DataFrames.
            use_monte_carlo (bool): Whether to use Monte Carlo simulation instead of analytical formula.
            
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
            
            # Add SABR price column if it doesn't exist
            if 'sabr_price' not in df.columns:
                df['sabr_price'] = np.nan
            
            # Process options in batches for better performance
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
                        
                        # Get historical volatility at issue date (for initial alpha)
                        vol_data = volatilities[volatilities['date'] <= issue_date]
                        if vol_data.empty:
                            logger.warning(f"No volatility data for {issue_date} for option {option['option_id']}")
                            hist_vol = 0.15  # Default value
                        else:
                            hist_vol = vol_data.iloc[-1]['historical_vol']
                        
                        # Use implied volatility if available, otherwise historical vol
                        alpha = option.get('implied_volatility', hist_vol)
                        
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
                                alpha=alpha,
                                option_type=option['type']
                            )
                        else:
                            price = self.price_option(
                                spot=spot,
                                strike=option['strike_price'],
                                days_to_maturity=option['days_to_maturity'],
                                domestic_rate=domestic_rate,
                                foreign_rate=foreign_rate,
                                alpha=alpha,
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
            
        except Exception as e:
            logger.error(f"Unexpected error in pricing portfolio: {e}")
            return options_data  # Return original data in case of error
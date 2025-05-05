"""
SABR Model for Stochastic Volatility

This module implements the SABR (Stochastic Alpha, Beta, Rho) model for pricing FX options.
The model accounts for the stochastic nature of volatility and the correlation between
the asset price and volatility.
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
        self.config = self._load_config(config_path)
        self.sabr_params = self.config['models']['sabr']
        
        # SABR parameters
        self.alpha = self.sabr_params['alpha']  # Initial volatility
        self.beta = self.sabr_params['beta']    # CEV parameter (0 = normal, 1 = lognormal)
        self.rho = self.sabr_params['rho']      # Correlation between spot and vol
        self.nu = self.sabr_params['nu']        # Volatility of volatility
        
        logger.info("SABR model initialized")
    
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
        # Use default SABR parameters if not provided
        alpha = alpha if alpha is not None else self.alpha
        beta = beta if beta is not None else self.beta
        rho = rho if rho is not None else self.rho
        nu = nu if nu is not None else self.nu
        
        # Convert days to years
        T = days_to_maturity / 365.0
        
        # Handle edge cases
        if T <= 0:
            return alpha
        
        # Handle ATM case separately to avoid numerical issues
        if abs(spot - strike) < 1e-10:
            # ATM formula
            atm_vol = alpha * (1 + (((1 - beta)**2 / 24) * alpha**2 / (spot**(2 - 2*beta)) 
                               + 0.25 * rho * beta * nu * alpha / (spot**(1 - beta)) 
                               + ((2 - 3 * rho**2) / 24) * nu**2) * T)
            return atm_vol
        
        # Calculate the log of the forward/strike
        log_fk = np.log(spot / strike)
        
        # Handle strike near zero for beta < 1
        if strike < 1e-10 and beta < 1:
            return alpha / (strike**(1 - beta))
        
        # For beta = 1 (lognormal case)
        if abs(beta - 1.0) < 1e-10:
            # Compute the intermediate values
            z = nu / alpha * log_fk
            x_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
            
            # Calculate implied volatility
            imp_vol = alpha * log_fk / x_z * (1 + (((1 - beta)**2 / 24) * alpha**2 
                                               + 0.25 * rho * beta * nu * alpha 
                                               + ((2 - 3 * rho**2) / 24) * nu**2) * T)
            return imp_vol
        
        # For beta < 1 (including beta = 0 for normal case)
        # Compute the intermediate values
        f_avg = 0.5 * (spot + strike)
        f_mid = spot**(1 - beta) * strike**beta
        z = nu / alpha * f_mid**(beta - 1) * log_fk
        
        # Handle small z values to avoid numerical issues
        if abs(z) < 1e-6:
            # Use Taylor expansion for small z
            x_z = log_fk
        else:
            x_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
        
        # Calculate the volatility multiplier
        vol_multiplier = (1 + (((1 - beta)**2 / 24) * alpha**2 / f_avg**(2 - 2*beta) 
                             + 0.25 * rho * beta * nu * alpha / f_avg**(1 - beta) 
                             + ((2 - 3 * rho**2) / 24) * nu**2) * T)
        
        # Calculate the final implied volatility
        imp_vol = alpha * (1 + (1 - beta)**2 * log_fk**2 / 24 + (1 - beta)**4 * log_fk**4 / 1920) * z / x_z * vol_multiplier
        
        return imp_vol
    
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
        # Convert days to years
        T = days_to_maturity / 365.0
        
        # Handle edge cases
        if T <= 0:
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
        
        return price
    
    def monte_carlo_price(self, spot, strike, days_to_maturity, domestic_rate, foreign_rate,
                         alpha=None, beta=None, rho=None, nu=None, option_type='call', n_simulations=10000, dt=1/252):
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
            n_simulations (int): Number of Monte Carlo simulations.
            dt (float): Time step for simulation (in years).
            
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
        
        # Number of time steps
        n_steps = int(T / dt)
        if n_steps < 1:
            n_steps = 1
            dt = T
        
        # Initialize arrays for spot price and volatility
        spot_paths = np.zeros((n_simulations, n_steps + 1))
        vol_paths = np.zeros((n_simulations, n_steps + 1))
        
        # Set initial values
        spot_paths[:, 0] = spot
        vol_paths[:, 0] = alpha
        
        # Generate standard normal random variables
        z1 = np.random.normal(0, 1, (n_simulations, n_steps))
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, (n_simulations, n_steps))
        
        # Drift adjustment for risk-neutral measure
        drift = domestic_rate - foreign_rate
        
        # Simulate paths
        for t in range(n_steps):
            # Handle the spot price
            dW1 = np.sqrt(dt) * z1[:, t]
            
            # Handle beta = 0 and beta = 1 cases separately to avoid numerical issues
            if beta == 0:
                # Normal case
                spot_paths[:, t+1] = spot_paths[:, t] + drift * spot_paths[:, t] * dt + vol_paths[:, t] * dW1
            elif beta == 1:
                # Log-normal case
                spot_paths[:, t+1] = spot_paths[:, t] * np.exp((drift - 0.5 * vol_paths[:, t]**2) * dt + vol_paths[:, t] * dW1)
            else:
                # General CEV case
                spot_paths[:, t+1] = spot_paths[:, t] * np.exp((drift - 0.5 * (vol_paths[:, t] * spot_paths[:, t]**(beta-1))**2) * dt 
                                                             + vol_paths[:, t] * spot_paths[:, t]**(beta-1) * dW1)
            
            # Ensure spot prices don't go negative
            spot_paths[:, t+1] = np.maximum(spot_paths[:, t+1], 1e-10)
            
            # Handle the volatility
            dW2 = np.sqrt(dt) * z2[:, t]
            vol_paths[:, t+1] = vol_paths[:, t] * np.exp((-0.5 * nu**2) * dt + nu * dW2)
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(0, spot_paths[:, -1] - strike)
        else:
            payoffs = np.maximum(0, strike - spot_paths[:, -1])
        
        # Calculate option price
        option_price = np.mean(payoffs) * np.exp(-domestic_rate * T)
        
        return option_price
    
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
        if isinstance(options_data, list):
            df = pd.DataFrame(options_data)
        else:
            df = options_data.copy()
        
        # Unpack market data
        spot_rates, volatilities, interest_rates = market_data
        
        # Add SABR price column if it doesn't exist
        if 'sabr_price' not in df.columns:
            df['sabr_price'] = np.nan
        
        # Price each option
        for i, option in df.iterrows():
            try:
                # Get issue date
                issue_date = pd.to_datetime(option['issue_date'])
                
                # Get spot rate at issue date
                spot = spot_rates[spot_rates['date'] <= issue_date].iloc[-1]['EUR/TND']
                
                # Get historical volatility at issue date (for initial alpha)
                hist_vol = volatilities[volatilities['date'] <= issue_date].iloc[-1]['historical_vol']
                alpha = option.get('implied_volatility', hist_vol)  # Use implied vol if available
                
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
                
            except Exception as e:
                logger.error(f"Error pricing option {option['option_id']} with SABR: {e}")
        
        return df
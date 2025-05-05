"""
Black-Scholes (Garman-Kohlhagen) Model for FX Options

This module implements the Black-Scholes model for European FX options,
also known as the Garman-Kohlhagen model.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BlackScholesModel:
    """Implements the Black-Scholes (Garman-Kohlhagen) model for FX options."""
    
    def __init__(self):
        """Initialize the Black-Scholes model."""
        logger.info("Black-Scholes (Garman-Kohlhagen) model initialized")
    
    def price_option(self, spot, strike, days_to_maturity, domestic_rate, foreign_rate, 
                     volatility, option_type='call'):
        """
        Calculate the price of a European FX option using the Black-Scholes (Garman-Kohlhagen) model.
        
        Args:
            spot (float): Spot exchange rate (price of 1 unit of foreign currency in domestic currency).
            strike (float): Strike price.
            days_to_maturity (int): Number of days to maturity.
            domestic_rate (float): Domestic risk-free interest rate (annualized).
            foreign_rate (float): Foreign risk-free interest rate (annualized).
            volatility (float): Annualized volatility of the exchange rate.
            option_type (str): 'call' or 'put'.
            
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
        
        # Black-Scholes (Garman-Kohlhagen) formula for FX options
        d1 = (np.log(spot / strike) + (domestic_rate - foreign_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        
        if option_type.lower() == 'call':
            # Call option price (in domestic currency)
            price = spot * np.exp(-foreign_rate * T) * norm.cdf(d1) - strike * np.exp(-domestic_rate * T) * norm.cdf(d2)
        else:
            # Put option price (in domestic currency)
            price = strike * np.exp(-domestic_rate * T) * norm.cdf(-d2) - spot * np.exp(-foreign_rate * T) * norm.cdf(-d1)
        
        return price
    
    def calculate_greeks(self, spot, strike, days_to_maturity, domestic_rate, foreign_rate, 
                          volatility, option_type='call'):
        """
        Calculate the option Greeks using the Black-Scholes (Garman-Kohlhagen) model.
        
        Args:
            spot (float): Spot exchange rate.
            strike (float): Strike price.
            days_to_maturity (int): Number of days to maturity.
            domestic_rate (float): Domestic risk-free interest rate (annualized).
            foreign_rate (float): Foreign risk-free interest rate (annualized).
            volatility (float): Annualized volatility of the exchange rate.
            option_type (str): 'call' or 'put'.
            
        Returns:
            dict: Dictionary containing the option Greeks (delta, gamma, theta, vega, rho).
        """
        # Convert days to years
        T = days_to_maturity / 365.0
        
        # Handle edge cases
        if T <= 0 or volatility <= 0:
            return {
                'delta': 1.0 if option_type.lower() == 'call' and spot > strike else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        # Black-Scholes (Garman-Kohlhagen) formula
        d1 = (np.log(spot / strike) + (domestic_rate - foreign_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        
        # Normal probability density function
        n_d1 = norm.pdf(d1)
        
        # Delta
        if option_type.lower() == 'call':
            delta = np.exp(-foreign_rate * T) * norm.cdf(d1)
        else:
            delta = np.exp(-foreign_rate * T) * (norm.cdf(d1) - 1)
        
        # Gamma (same for call and put)
        gamma = np.exp(-foreign_rate * T) * n_d1 / (spot * volatility * np.sqrt(T))
        
        # Vega (same for call and put)
        vega = spot * np.exp(-foreign_rate * T) * n_d1 * np.sqrt(T) / 100  # Scaling by 100 for 1% change
        
        # Theta (time decay)
        term1 = -spot * np.exp(-foreign_rate * T) * n_d1 * volatility / (2 * np.sqrt(T))
        if option_type.lower() == 'call':
            theta = term1 + foreign_rate * spot * np.exp(-foreign_rate * T) * norm.cdf(d1) - domestic_rate * strike * np.exp(-domestic_rate * T) * norm.cdf(d2)
        else:
            theta = term1 - foreign_rate * spot * np.exp(-foreign_rate * T) * norm.cdf(-d1) + domestic_rate * strike * np.exp(-domestic_rate * T) * norm.cdf(-d2)
        
        # Convert theta to daily
        theta = theta / 365.0
        
        # Rho (sensitivity to domestic interest rate)
        if option_type.lower() == 'call':
            rho = strike * T * np.exp(-domestic_rate * T) * norm.cdf(d2) / 100  # Scaling by 100 for 1% change
        else:
            rho = -strike * T * np.exp(-domestic_rate * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def implied_volatility(self, market_price, spot, strike, days_to_maturity, 
                          domestic_rate, foreign_rate, option_type='call', 
                          precision=0.0001, max_iterations=100):
        """
        Calculate the implied volatility using the Black-Scholes (Garman-Kohlhagen) model.
        
        Args:
            market_price (float): Market price of the option.
            spot (float): Spot exchange rate.
            strike (float): Strike price.
            days_to_maturity (int): Number of days to maturity.
            domestic_rate (float): Domestic risk-free interest rate (annualized).
            foreign_rate (float): Foreign risk-free interest rate (annualized).
            option_type (str): 'call' or 'put'.
            precision (float): Desired precision for the implied volatility.
            max_iterations (int): Maximum number of iterations for the algorithm.
            
        Returns:
            float: Implied volatility.
        """
        # Convert days to years
        T = days_to_maturity / 365.0
        
        # Handle edge cases
        if T <= 0:
            return 0.0
        
        # Initial guess for volatility
        if spot > strike and option_type.lower() == 'call':
            # In-the-money call
            vol = 0.3
        elif spot < strike and option_type.lower() == 'put':
            # In-the-money put
            vol = 0.3
        else:
            # Out-of-the-money
            vol = 0.2
        
        # Newton-Raphson method for finding implied volatility
        for i in range(max_iterations):
            price = self.price_option(spot, strike, days_to_maturity, domestic_rate, foreign_rate, vol, option_type)
            vega = self.calculate_greeks(spot, strike, days_to_maturity, domestic_rate, foreign_rate, vol, option_type)['vega'] * 100  # Adjust vega back
            
            # Avoid division by very small vega
            if abs(vega) < 1e-10:
                if vol < 0.001:  # Try a higher vol if current is very low
                    vol = 0.2
                else:  # Otherwise, give up
                    break
            else:
                # Update volatility
                vol_new = vol + (market_price - price) / vega
                
                # Ensure volatility stays positive
                vol_new = max(0.001, vol_new)
                
                # Check for convergence
                if abs(vol_new - vol) < precision:
                    vol = vol_new
                    break
                
                vol = vol_new
        
        return vol
    
    def price_options_portfolio(self, options_data, market_data=None):
        """
        Price a portfolio of European FX options using the Black-Scholes (Garman-Kohlhagen) model.
        
        Args:
            options_data (list or pandas.DataFrame): Portfolio of options.
            market_data (tuple, optional): Tuple of (spot_rates, volatility, interest_rates) DataFrames.
                                           If not provided, the model will use the values in options_data.
            
        Returns:
            pandas.DataFrame: Options data with added pricing information.
        """
        if isinstance(options_data, list):
            df = pd.DataFrame(options_data)
        else:
            df = options_data.copy()
        
        # Add Black-Scholes price column if it doesn't exist
        if 'bs_price' not in df.columns:
            df['bs_price'] = np.nan
        
        # Add implied volatility column if it doesn't exist
        if 'implied_volatility' not in df.columns:
            df['implied_volatility'] = np.nan
        
        # Price each option
        for i, option in df.iterrows():
            try:
                # Get market data if provided
                if market_data is not None:
                    spot_rates, volatility, interest_rates = market_data
                    
                    # Get spot rate at issue date
                    issue_date = pd.to_datetime(option['issue_date'])
                    spot = spot_rates[spot_rates['date'] <= issue_date].iloc[-1]['EUR/TND']
                    
                    # Get volatility at issue date
                    vol = volatility[volatility['date'] <= issue_date].iloc[-1]['historical_vol']
                    
                    # Get interest rates at issue date
                    rates = interest_rates[interest_rates['date'] <= issue_date].iloc[-1]
                    domestic_rate = rates['EUR_rate']
                    foreign_rate = rates['TND_rate']
                else:
                    # Use the values from the options data
                    spot = option['spot_rate_at_issue']
                    vol = option.get('implied_volatility', self.market_config['volatility'])
                    domestic_rate = option['domestic_rate']
                    foreign_rate = option['foreign_rate']
                
                # Calculate option price
                price = self.price_option(
                    spot=spot,
                    strike=option['strike_price'],
                    days_to_maturity=option['days_to_maturity'],
                    domestic_rate=domestic_rate,
                    foreign_rate=foreign_rate,
                    volatility=vol,
                    option_type=option['type']
                )
                
                # Calculate implied volatility if not already provided
                if pd.isna(option['implied_volatility']):
                    implied_vol = self.implied_volatility(
                        market_price=price,  # Use the calculated price as a proxy
                        spot=spot,
                        strike=option['strike_price'],
                        days_to_maturity=option['days_to_maturity'],
                        domestic_rate=domestic_rate,
                        foreign_rate=foreign_rate,
                        option_type=option['type']
                    )
                    df.at[i, 'implied_volatility'] = implied_vol
                
                # Update option price
                df.at[i, 'bs_price'] = price
                
            except Exception as e:
                logger.error(f"Error pricing option {option['option_id']}: {e}")
        
        return df
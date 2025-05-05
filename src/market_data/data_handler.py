"""
Market Data Handler Module

This module handles the generation and management of market data for the simulation,
including EUR/TND spot rates, volatility, and interest rates.
"""

import os
import yaml
import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketDataHandler:
    """Handles market data for the simulation."""
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize the MarketDataHandler with configuration parameters.
        
        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config = self._load_config(config_path)
        self.market_config = self.config['market']
        self.time_config = self.config['time']
        self.simulation_config = self.config['simulation']
        
        # Set random seed for reproducibility
        np.random.seed(self.simulation_config['seed'])
        
        # Market data
        self.spot_rates = None
        self.volatility = None
        self.interest_rates = None
        
        logger.info("Market Data Handler initialized with configuration from %s", config_path)
    
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
    
    def _generate_time_series(self, start_date_str, end_date_str, initial_value, 
                              volatility, mean_reversion=0.03, jump_prob=0.05, 
                              jump_mean=-0.002, jump_std=0.01, trend=0.0001):
        """
        Generate a realistic time series with jumps and mean reversion.
        
        Args:
            start_date_str (str): Start date in 'YYYY-MM-DD' format.
            end_date_str (str): End date in 'YYYY-MM-DD' format.
            initial_value (float): Initial value of the time series.
            volatility (float): Daily volatility.
            mean_reversion (float): Strength of mean reversion.
            jump_prob (float): Probability of a jump on any given day.
            jump_mean (float): Mean of the jump size.
            jump_std (float): Standard deviation of the jump size.
            trend (float): Daily trend (drift).
            
        Returns:
            pandas.DataFrame: DataFrame with date and value columns.
        """
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        # Add buffer days before start_date for initialization and after end_date for future prices
        buffer_days_before = 30  # For historical volatility calculations
        buffer_days_after = 365  # For maturities in the following year
        
        extended_start_date = start_date - timedelta(days=buffer_days_before)
        extended_end_date = end_date + timedelta(days=buffer_days_after)
        
        # Create date range
        date_range = pd.date_range(start=extended_start_date, end=extended_end_date)
        
        # Initialize time series
        values = np.zeros(len(date_range))
        values[0] = initial_value
        
        # Generate time series with GBM, mean reversion, and jumps
        for i in range(1, len(values)):
            # Mean reversion component
            reversion = mean_reversion * (initial_value - values[i-1])
            
            # GBM component
            gbm = values[i-1] * (trend + volatility * np.random.normal(0, 1))
            
            # Jump component
            jump = 0
            if np.random.random() < jump_prob:
                jump = values[i-1] * np.random.normal(jump_mean, jump_std)
            
            # Combine components
            values[i] = values[i-1] + reversion + gbm + jump
            
            # Ensure values stay positive
            values[i] = max(values[i], 0.1 * initial_value)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': date_range,
            'value': values
        })
        
        return df
    
    def _calculate_historical_volatility(self, returns, window=30):
        """
        Calculate historical volatility from returns.
        
        Args:
            returns (pandas.Series): Series of returns.
            window (int): Rolling window for volatility calculation.
            
        Returns:
            pandas.Series: Historical volatility series.
        """
        # Calculate rolling standard deviation of returns
        rolling_std = returns.rolling(window=window).std()
        
        # Annualize volatility (assuming 252 trading days per year)
        historical_vol = rolling_std * np.sqrt(252)
        
        return historical_vol
    
    def generate_market_data(self, save=True):
        """
        Generate market data for the simulation.
        
        Args:
            save (bool): Whether to save the generated data to CSV files.
            
        Returns:
            tuple: (spot_rates, volatility, interest_rates) DataFrames.
        """
        # Extend the time range for future prices (for option maturities)
        start_date = self.time_config['start_date']
        end_date = self.time_config['end_date']
        initial_spot = self.market_config['spot_price']
        initial_vol = self.market_config['volatility']
        
        logger.info(f"Generating market data from {start_date} to {end_date}")
        
        # Generate EUR/TND spot rates
        self.spot_rates = self._generate_time_series(
            start_date, 
            end_date,
            initial_value=initial_spot,
            volatility=initial_vol / np.sqrt(252),  # Convert annual vol to daily
            mean_reversion=0.02,
            jump_prob=0.03,
            jump_mean=-0.003,
            jump_std=0.008,
            trend=0.0001  # Small upward trend
        )
        
        # Rename columns
        self.spot_rates.columns = ['date', 'EUR/TND']
        
        # Calculate returns
        self.spot_rates['returns'] = self.spot_rates['EUR/TND'].pct_change()
        
        # Calculate historical volatility
        vol_window = self.config['models']['black_scholes']['rolling_window']
        self.volatility = pd.DataFrame({
            'date': self.spot_rates['date'],
            'historical_vol': self._calculate_historical_volatility(self.spot_rates['returns'], window=vol_window)
        })
        
        # Generate interest rates (somewhat stable with occasional changes)
        dates = self.spot_rates['date']
        
        # EUR Interest Rate (more stable)
        eur_rate = np.ones(len(dates)) * self.market_config['eur_interest_rate']
        # Add some small random fluctuations and occasional changes
        for i in range(1, len(eur_rate)):
            if random.random() < 0.003:  # 0.3% chance of rate change
                change = random.choice([-0.0025, -0.0025, -0.0025, 0.0025, 0.0025])
                eur_rate[i:] = max(0.001, eur_rate[i-1] + change)
            else:
                # Small random fluctuation
                eur_rate[i] = eur_rate[i-1] + np.random.normal(0, 0.0001)
        
        # TND Interest Rate (more volatile)
        tnd_rate = np.ones(len(dates)) * self.market_config['tnd_interest_rate']
        # Add more significant fluctuations and changes
        for i in range(1, len(tnd_rate)):
            if random.random() < 0.005:  # 0.5% chance of rate change
                change = random.choice([-0.005, -0.0025, 0.0025, 0.005])
                tnd_rate[i:] = max(0.01, tnd_rate[i-1] + change)
            else:
                # Larger random fluctuation
                tnd_rate[i] = tnd_rate[i-1] + np.random.normal(0, 0.0002)
        
        self.interest_rates = pd.DataFrame({
            'date': dates,
            'EUR_rate': eur_rate,
            'TND_rate': tnd_rate
        })
        
        # Save data to CSV files if requested
        if save:
            self._save_market_data()
        
        logger.info("Market data generated successfully")
        return self.spot_rates, self.volatility, self.interest_rates
    
    def _save_market_data(self):
        """Save market data to CSV files."""
        # Create directory if it doesn't exist
        os.makedirs('data/market', exist_ok=True)
        
        # Save spot rates
        if self.spot_rates is not None:
            self.spot_rates.to_csv('data/market/spot_rates.csv', index=False)
            logger.info("Spot rates saved to data/market/spot_rates.csv")
        
        # Save volatility
        if self.volatility is not None:
            self.volatility.to_csv('data/market/volatility.csv', index=False)
            logger.info("Volatility data saved to data/market/volatility.csv")
        
        # Save interest rates
        if self.interest_rates is not None:
            self.interest_rates.to_csv('data/market/interest_rates.csv', index=False)
            logger.info("Interest rates saved to data/market/interest_rates.csv")
    
    def load_market_data(self):
        """
        Load market data from CSV files.
        
        Returns:
            tuple: (spot_rates, volatility, interest_rates) DataFrames.
        """
        try:
            self.spot_rates = pd.read_csv('data/market/spot_rates.csv')
            self.volatility = pd.read_csv('data/market/volatility.csv')
            self.interest_rates = pd.read_csv('data/market/interest_rates.csv')
            
            # Convert dates to datetime
            self.spot_rates['date'] = pd.to_datetime(self.spot_rates['date'])
            self.volatility['date'] = pd.to_datetime(self.volatility['date'])
            self.interest_rates['date'] = pd.to_datetime(self.interest_rates['date'])
            
            logger.info("Market data loaded successfully")
            return self.spot_rates, self.volatility, self.interest_rates
            
        except FileNotFoundError:
            logger.warning("Market data files not found. Generate market data first.")
            return None, None, None
    
    def get_rate_at_date(self, date_str, rate_type='spot'):
        """
        Get the rate at a specific date.
        
        Args:
            date_str (str): Date in 'YYYY-MM-DD' format.
            rate_type (str): Type of rate ('spot', 'eur_rate', 'tnd_rate', 'volatility').
            
        Returns:
            float: Rate at the specified date.
        """
        date = pd.to_datetime(date_str)
        
        if rate_type == 'spot':
            if self.spot_rates is None:
                self.load_market_data()
            
            # Find the closest date (exact or previous)
            closest = self.spot_rates[self.spot_rates['date'] <= date].iloc[-1]
            return closest['EUR/TND']
            
        elif rate_type == 'volatility':
            if self.volatility is None:
                self.load_market_data()
            
            # Find the closest date (exact or previous)
            closest = self.volatility[self.volatility['date'] <= date].iloc[-1]
            return closest['historical_vol']
            
        elif rate_type == 'eur_rate':
            if self.interest_rates is None:
                self.load_market_data()
            
            # Find the closest date (exact or previous)
            closest = self.interest_rates[self.interest_rates['date'] <= date].iloc[-1]
            return closest['EUR_rate']
            
        elif rate_type == 'tnd_rate':
            if self.interest_rates is None:
                self.load_market_data()
            
            # Find the closest date (exact or previous)
            closest = self.interest_rates[self.interest_rates['date'] <= date].iloc[-1]
            return closest['TND_rate']
            
        else:
            raise ValueError(f"Invalid rate type: {rate_type}")


def main():
    """Main function to generate market data."""
    handler = MarketDataHandler()
    handler.generate_market_data(save=True)


if __name__ == "__main__":
    main()
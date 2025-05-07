"""
Modified Option Contract Generator Module

This module generates synthetic European call option contracts on EUR/TND
with realistic parameters within the specified constraints.
All options now use the same fixed spot rate from config.
"""

import os
import csv
import yaml
import uuid
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptionGenerator:
    """Generates synthetic European call option contracts."""
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize the OptionGenerator with configuration parameters.
        
        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config = self._load_config(config_path)
        self.portfolio_config = self.config['portfolio']
        self.time_config = self.config['time']
        self.simulation_config = self.config['simulation']
        self.market_config = self.config['market']
        
        # Set random seed for reproducibility
        random.seed(self.simulation_config['seed'])
        np.random.seed(self.simulation_config['seed'])
        
        # Initialize empty portfolio
        self.options = []
        
        # Use fixed rates from config for all options
        self.fixed_spot_rate = self.market_config['spot_price']
        self.fixed_eur_rate = self.market_config['eur_interest_rate']
        self.fixed_tnd_rate = self.market_config['tnd_interest_rate']
        
        logger.info("Option Generator initialized with fixed spot rate: %s", self.fixed_spot_rate)
    
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
    
    def _generate_random_date(self, start_date_str, end_date_str):
        """
        Generate a random date between start_date and end_date.
        
        Args:
            start_date_str (str): Start date in 'YYYY-MM-DD' format.
            end_date_str (str): End date in 'YYYY-MM-DD' format.
            
        Returns:
            datetime: Random date between start_date and end_date.
        """
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        delta = (end_date - start_date).days
        random_days = random.randint(0, delta)
        
        return start_date + timedelta(days=random_days)
    
    def _generate_maturity_date(self, issue_date):
        """
        Generate a maturity date based on the issue date.
        
        Args:
            issue_date (datetime): Issue date of the option.
            
        Returns:
            datetime: Maturity date of the option.
        """
        min_days = self.time_config['min_maturity_days']
        max_days = self.time_config['max_maturity_days']
        
        maturity_days = random.randint(min_days, max_days)
        
        return issue_date + timedelta(days=maturity_days)
    
    def _generate_notional(self):
        """
        Generate a random notional amount within the configured range.
        
        Returns:
            float: Notional amount in EUR.
        """
        min_notional = self.portfolio_config['min_option_notional']
        max_notional = self.portfolio_config['max_option_notional']
        
        # Generate a random notional with a bias toward round numbers
        if random.random() < 0.7:  # 70% chance of round number
            round_values = [100000, 250000, 500000, 750000, 1000000]
            valid_values = [v for v in round_values if min_notional <= v <= max_notional]
            if valid_values:
                return random.choice(valid_values)
        
        # Otherwise, generate a random value and round to nearest 10,000
        notional = random.uniform(min_notional, max_notional)
        return round(notional / 10000) * 10000
    
    def _check_active_notional(self, date, proposed_notional=0):
        """
        Check if adding an option with the proposed notional would exceed 
        the maximum active notional at the given date.
        
        Args:
            date (datetime): Date to check active notional.
            proposed_notional (float): Notional of the proposed new option.
            
        Returns:
            bool: True if adding the option won't exceed maximum active notional.
        """
        active_notional = proposed_notional
        
        for option in self.options:
            issue_date = datetime.strptime(option['issue_date'], '%Y-%m-%d')
            maturity_date = datetime.strptime(option['maturity_date'], '%Y-%m-%d')
            
            if issue_date <= date <= maturity_date:
                active_notional += option['notional']
        
        max_notional = self.portfolio_config['max_total_notional']
        
        return active_notional <= max_notional
    
    def generate_option(self):
        """
        Generate a single option contract with random parameters within the constraints.
        
        Returns:
            dict: Option contract parameters.
        """
        # Generate issue date
        issue_date = self._generate_random_date(
            self.time_config['start_date'], 
            self.time_config['end_date']
        )
        
        # Generate maturity date
        maturity_date = self._generate_maturity_date(issue_date)
        
        # Generate notional
        notional = self._generate_notional()
        
        # Check if adding this option would exceed max active notional
        if not self._check_active_notional(issue_date, notional):
            # Try with a smaller notional
            notional = self.portfolio_config['min_option_notional']
            
            if not self._check_active_notional(issue_date, notional):
                logger.warning(f"Cannot add option at {issue_date.strftime('%Y-%m-%d')} - would exceed max active notional")
                return None
        
        # Use fixed spot rate and interest rates from config
        spot_rate = self.fixed_spot_rate
        eur_rate = self.fixed_eur_rate
        tnd_rate = self.fixed_tnd_rate
        
        # Create option contract
        option = {
            'option_id': str(uuid.uuid4())[:8],  # Generate a unique ID
            'currency_pair': self.portfolio_config['currency_pair'],
            'type': 'call',
            'style': 'european',
            'notional': notional,
            'issue_date': issue_date.strftime('%Y-%m-%d'),
            'maturity_date': maturity_date.strftime('%Y-%m-%d'),
            'days_to_maturity': (maturity_date - issue_date).days,
            'spot_rate_at_issue': spot_rate,  # Use fixed spot rate
            'domestic_rate': eur_rate,         # Use fixed interest rate
            'foreign_rate': tnd_rate,          # Use fixed interest rate
            # Strike price will be set slightly out of the money
            'strike_price': round(spot_rate * (1 + random.uniform(0.01, 0.1)), 4),
            'implied_volatility': None,  # Will be calculated by pricing models
            'bs_price': None,            # Black-Scholes price
            'jd_price': None,            # Jump-Diffusion price
            'sabr_price': None,          # SABR price
            'actual_payoff': None,       # Will be filled after maturity
        }
        
        return option
    
    def generate_portfolio(self):
        """
        Generate a portfolio of option contracts.
        
        Returns:
            list: List of option contracts.
        """
        num_options = self.simulation_config['num_options']
        logger.info(f"Generating portfolio with {num_options} option contracts using fixed spot rate {self.fixed_spot_rate}")
        
        self.options = []
        attempts = 0
        max_attempts = num_options * 3  # Allow for some failed attempts
        
        while len(self.options) < num_options and attempts < max_attempts:
            option = self.generate_option()
            
            if option:
                self.options.append(option)
                logger.debug(f"Added option {option['option_id']} to portfolio")
            
            attempts += 1
        
        logger.info(f"Generated {len(self.options)} option contracts in {attempts} attempts")
        return self.options
    
    def save_portfolio_to_csv(self, filepath="data/generated/option_contracts.csv"):
        """
        Save the generated portfolio to a CSV file.
        
        Args:
            filepath (str): Path to save the CSV file.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if not self.options:
            logger.warning("No options to save. Generate portfolio first.")
            return
        
        # Sort options by issue date
        sorted_options = sorted(self.options, key=lambda x: x['issue_date'])
        
        # Write to CSV
        with open(filepath, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=sorted_options[0].keys())
            writer.writeheader()
            writer.writerows(sorted_options)
        
        logger.info(f"Portfolio saved to {filepath}")
        
        # Create a summary of active notional over time
        self._save_notional_summary(filepath)
    
    def _save_notional_summary(self, option_filepath):
        """
        Create and save a summary of active notional over time.
        
        Args:
            option_filepath (str): Path to the option contracts CSV file.
        """
        summary_filepath = os.path.join(os.path.dirname(option_filepath), "notional_summary.csv")
        
        # Create a date range covering all options
        start_date = datetime.strptime(self.time_config['start_date'], '%Y-%m-%d')
        # End date is the latest maturity date
        end_date = max([datetime.strptime(option['maturity_date'], '%Y-%m-%d') for option in self.options])
        
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Calculate active notional for each date
        active_notional = []
        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            notional = 0
            active_options = 0
            
            for option in self.options:
                issue_date = datetime.strptime(option['issue_date'], '%Y-%m-%d')
                maturity_date = datetime.strptime(option['maturity_date'], '%Y-%m-%d')
                
                if issue_date <= date <= maturity_date:
                    notional += option['notional']
                    active_options += 1
            
            active_notional.append({
                'date': date_str,
                'active_notional': notional,
                'active_options': active_options
            })
        
        # Save to CSV
        with open(summary_filepath, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['date', 'active_notional', 'active_options'])
            writer.writeheader()
            writer.writerows(active_notional)
        
        logger.info(f"Notional summary saved to {summary_filepath}")


def main():
    """Main function to generate option contracts."""
    generator = OptionGenerator()
    generator.generate_portfolio()
    generator.save_portfolio_to_csv()


if __name__ == "__main__":
    main()
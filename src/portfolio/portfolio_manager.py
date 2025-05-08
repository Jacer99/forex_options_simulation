"""
Modified Portfolio Manager Module

This module manages a portfolio of European FX options and provides functions
for pricing, risk management, and performance evaluation.
The EGARCH model has been removed, and all options now use the same spot rate.
"""

import os
import pandas as pd
import numpy as np
import logging
import yaml
from datetime import datetime
import matplotlib.pyplot as plt

# Import pricing models
from src.models.black_scholes import BlackScholesModel
# EGARCH model has been removed
from src.models.jump_diffusion import JumpDiffusionModel
from src.models.sabr import SABRModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortfolioManager:
    """Manages a portfolio of European FX options."""
    
    def __init__(self, options_data=None, market_data=None, config_path='config.yaml'):
        """
        Initialize the PortfolioManager.
        
        Args:
            options_data (pandas.DataFrame or list, optional): Portfolio of options.
            market_data (tuple, optional): Tuple of (spot_rates, volatility, interest_rates) DataFrames.
            config_path (str): Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.output_config = self.config['output']
        self.market_config = self.config['market']
        
        # Initialize pricing models
        self.bs_model = BlackScholesModel()
        # EGARCH model has been removed
        self.jd_model = JumpDiffusionModel(config_path)
        self.sabr_model = SABRModel(config_path)
        
        # Set options and market data
        self.options_data = options_data
        self.market_data = market_data
        
        # Use fixed spot rate for all options
        self.fixed_spot_rate = self.market_config['spot_price']
        self.fixed_eur_rate = self.market_config['eur_interest_rate']
        self.fixed_tnd_rate = self.market_config['tnd_interest_rate']
        
        logger.info("Portfolio Manager initialized with fixed spot rate: %s", self.fixed_spot_rate)
    
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
    
    def load_options_data(self, filepath):
        """
        Load options data from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file.
            
        Returns:
            pandas.DataFrame: Options data.
        """
        try:
            df = pd.read_csv(filepath)
            
            # Set all options to use the fixed spot rate
            df['spot_rate_at_issue'] = self.fixed_spot_rate
            df['domestic_rate'] = self.fixed_eur_rate
            df['foreign_rate'] = self.fixed_tnd_rate
            
            self.options_data = df
            logger.info(f"Options data loaded from {filepath} with fixed spot rate {self.fixed_spot_rate}")
            return df
        except Exception as e:
            logger.error(f"Error loading options data: {e}")
            return None
    
    def set_market_data(self, market_data):
        """
        Set market data for the portfolio.
        
        Args:
            market_data (tuple): Tuple of (spot_rates, volatility, interest_rates) DataFrames.
        """
        self.market_data = market_data
        logger.info("Market data set")
    
    def price_portfolio(self):
        """
        Price the portfolio using all available pricing models (except EGARCH).
        
        Returns:
            pandas.DataFrame: Options data with added pricing information.
        """
        if self.options_data is None:
            logger.error("No options data. Load options data first.")
            return None
        
        if self.market_data is None:
            logger.error("No market data. Set market data first.")
            return None
        
        logger.info("Pricing portfolio using Black-Scholes, Jump-Diffusion, and SABR models")
        
        # Make a copy of the options data
        df = self.options_data.copy()
        
        # Set all options to use the fixed spot rate
        df['spot_rate_at_issue'] = self.fixed_spot_rate
        df['domestic_rate'] = self.fixed_eur_rate
        df['foreign_rate'] = self.fixed_tnd_rate
        
        # First, price with Black-Scholes to get implied volatilities
        df = self.bs_model.price_options_portfolio(df, self.market_data)
        
        # Then, price with other models (except EGARCH)
        df = self.jd_model.price_options_portfolio(df, self.market_data)
        df = self.sabr_model.price_options_portfolio(df, self.market_data)
        
        # Remove EGARCH price column if it exists
        if 'egarch_price' in df.columns:
            df = df.drop(columns=['egarch_price'])
            if 'egarch_pnl' in df.columns:
                df = df.drop(columns=['egarch_pnl'])
            if 'egarch_bs_spread' in df.columns:
                df = df.drop(columns=['egarch_bs_spread'])
            if 'egarch_abs_error' in df.columns:
                df = df.drop(columns=['egarch_abs_error'])
        
        # Calculate model spreads
        if all(col in df.columns for col in ['bs_price', 'jd_price', 'sabr_price']):
            df['jd_bs_spread'] = df['jd_price'] - df['bs_price']
            df['sabr_bs_spread'] = df['sabr_price'] - df['bs_price']
        
        # Update the options data
        self.options_data = df
        
        logger.info("Portfolio pricing completed")
        return df
    
    def calculate_risks(self):
        """
        Calculate risk metrics for the portfolio.
        
        Returns:
            pandas.DataFrame: Options data with added risk metrics.
        """
        if self.options_data is None:
            logger.error("No options data. Load options data first.")
            return None
        
        logger.info("Calculating portfolio risk metrics")
        
        # Make a copy of the options data
        df = self.options_data.copy()
        
        # Add columns for risk metrics if they don't exist
        for col in ['delta', 'gamma', 'vega', 'theta', 'rho']:
            if col not in df.columns:
                df[col] = np.nan
        
        # Calculate risks for each option using Black-Scholes model
        for i, option in df.iterrows():
            try:
                # Calculate the Greeks
                greeks = self.bs_model.calculate_greeks(
                    spot=self.fixed_spot_rate,
                    strike=option['strike_price'],
                    days_to_maturity=option['days_to_maturity'],
                    domestic_rate=self.fixed_eur_rate,
                    foreign_rate=self.fixed_tnd_rate,
                    volatility=option['implied_volatility'],
                    option_type=option['type']
                )
                
                # Adjust Greeks by notional
                for greek, value in greeks.items():
                    df.at[i, greek] = value * option['notional']
                
            except Exception as e:
                logger.error(f"Error calculating risks for option {option['option_id']}: {e}")
        
        # Calculate portfolio-level risks
        portfolio_risks = {
            'total_notional': df['notional'].sum(),
            'total_delta': df['delta'].sum(),
            'total_gamma': df['gamma'].sum(),
            'total_vega': df['vega'].sum(),
            'total_theta': df['theta'].sum(),
            'total_rho': df['rho'].sum(),
        }
        
        logger.info(f"Portfolio risks: {portfolio_risks}")
        
        # Update the options data
        self.options_data = df
        
        return df, portfolio_risks
    
    def calculate_actual_payoffs(self, future_spot_rates):
        """
        Calculate the actual payoffs of the options at maturity.
        
        Args:
            future_spot_rates (pandas.DataFrame): DataFrame with date and EUR/TND columns.
            
        Returns:
            pandas.DataFrame: Options data with added actual payoff information.
        """
        if self.options_data is None:
            logger.error("No options data. Load options data first.")
            return None
        
        logger.info("Calculating actual option payoffs")
        
        # Make a copy of the options data
        df = self.options_data.copy()
        
        # Add column for actual payoff if it doesn't exist
        if 'actual_payoff' not in df.columns:
            df['actual_payoff'] = np.nan
        
        # Calculate actual payoff for each option
        for i, option in df.iterrows():
            try:
                # Get maturity date
                maturity_date = pd.to_datetime(option['maturity_date'])
                
                # Find the spot rate at maturity
                spot_at_maturity = future_spot_rates[future_spot_rates['date'] <= maturity_date].iloc[-1]['EUR/TND']
                
                # Calculate payoff
                if option['type'].lower() == 'call':
                    payoff = max(0, spot_at_maturity - option['strike_price']) * option['notional']
                else:
                    payoff = max(0, option['strike_price'] - spot_at_maturity) * option['notional']
                
                # Update actual payoff
                df.at[i, 'actual_payoff'] = payoff
                
                # Record the spot rate at maturity
                df.at[i, 'spot_rate_at_maturity'] = spot_at_maturity
                
            except Exception as e:
                logger.error(f"Error calculating actual payoff for option {option['option_id']}: {e}")
        
        # Update the options data
        self.options_data = df
        
        logger.info("Actual option payoffs calculated")
        return df
    
    def calculate_pnl(self):
        """
        Calculate Profit and Loss (PnL) for each option and model.
        
        Returns:
            pandas.DataFrame: Options data with added PnL information.
        """
        if self.options_data is None:
            logger.error("No options data. Load options data first.")
            return None
        
        if 'actual_payoff' not in self.options_data.columns:
            logger.error("No actual payoff information. Calculate actual payoffs first.")
            return None
        
        logger.info("Calculating PnL for each option and model")
        
        # Make a copy of the options data
        df = self.options_data.copy()
        
        # Add columns for PnL if they don't exist
        for col in ['bs_pnl', 'jd_pnl', 'sabr_pnl']:  # EGARCH removed
            if col not in df.columns:
                df[col] = np.nan
        
        # Calculate PnL for each option and model
        for i, option in df.iterrows():
            try:
                # Skip options without actual payoff
                if pd.isna(option['actual_payoff']):
                    continue
                
                # PnL = Actual Payoff - Option Price
                if not pd.isna(option['bs_price']):
                    df.at[i, 'bs_pnl'] = option['actual_payoff'] - option['bs_price'] * option['notional']
                
                if not pd.isna(option['jd_price']):
                    df.at[i, 'jd_pnl'] = option['actual_payoff'] - option['jd_price'] * option['notional']
                
                if not pd.isna(option['sabr_price']):
                    df.at[i, 'sabr_pnl'] = option['actual_payoff'] - option['sabr_price'] * option['notional']
                
            except Exception as e:
                logger.error(f"Error calculating PnL for option {option['option_id']}: {e}")
        
        # Calculate total PnL for each model
        total_pnl = {
            'bs_total_pnl': df['bs_pnl'].sum(),
            'jd_total_pnl': df['jd_pnl'].sum(),
            'sabr_total_pnl': df['sabr_pnl'].sum(),
        }
        
        logger.info(f"Total PnL: {total_pnl}")
        
        # Update the options data
        self.options_data = df
        
        return df, total_pnl
    
    def evaluate_model_performance(self):
        """
        Evaluate the performance of each pricing model.
        
        Returns:
            dict: Performance metrics for each model.
        """
        if self.options_data is None:
            logger.error("No options data. Load options data first.")
            return None
        
        if 'actual_payoff' not in self.options_data.columns:
            logger.error("No actual payoff information. Calculate actual payoffs first.")
            return None
        
        logger.info("Evaluating model performance")
        
        # Make a copy of the options data
        df = self.options_data.copy()
        
        # Calculate performance metrics for each model
        models = ['bs', 'jd', 'sabr']  # EGARCH removed
        metrics = {}
        
        for model in models:
            # Skip models without price information
            if f'{model}_price' not in df.columns:
                continue
            
            # Get options with both price and actual payoff
            valid_options = df.dropna(subset=[f'{model}_price', 'actual_payoff'])
            
            if len(valid_options) == 0:
                continue
            
            # Calculate absolute errors
            valid_options[f'{model}_abs_error'] = abs(valid_options['actual_payoff'] - 
                                                     valid_options[f'{model}_price'] * valid_options['notional'])
            
            # Calculate percentage errors (for non-zero actual payoffs)
            non_zero_payoffs = valid_options[valid_options['actual_payoff'] > 0].copy()
            if len(non_zero_payoffs) > 0:
                # FIXED: Create a new column instead of trying to create a new row
                non_zero_payoffs[f'{model}_pct_error'] = (non_zero_payoffs[f'{model}_abs_error'] / 
                                                          non_zero_payoffs['actual_payoff']) * 100
            
            # Calculate metrics
            metrics[model] = {
                'mean_abs_error': valid_options[f'{model}_abs_error'].mean(),
                'median_abs_error': valid_options[f'{model}_abs_error'].median(),
                'max_abs_error': valid_options[f'{model}_abs_error'].max(),
                'rmse': np.sqrt((valid_options[f'{model}_abs_error'] ** 2).mean()),
                'total_pnl': valid_options[f'{model}_pnl'].sum(),
            }
            
            # Add percentage error metrics if available
            if len(non_zero_payoffs) > 0:
                metrics[model]['mean_pct_error'] = non_zero_payoffs[f'{model}_pct_error'].mean()
                metrics[model]['median_pct_error'] = non_zero_payoffs[f'{model}_pct_error'].median()
                metrics[model]['mape'] = abs(non_zero_payoffs[f'{model}_pct_error']).mean()
            
            # Add to the DataFrame
            df[f'{model}_abs_error'] = df.apply(
                lambda row: abs(row['actual_payoff'] - row[f'{model}_price'] * row['notional']) 
                if not pd.isna(row['actual_payoff']) and not pd.isna(row[f'{model}_price']) else np.nan, 
                axis=1
            )
        
        # Update the options data
        self.options_data = df
        
        logger.info("Model performance evaluation completed")
        return metrics
    
    def save_results(self, output_dir=None):
        """
        Save the portfolio analysis results to CSV files.
        
        Args:
            output_dir (str, optional): Directory to save the results. Defaults to the output directory in the configuration.
        """
        if self.options_data is None:
            logger.error("No options data. Nothing to save.")
            return
        
        # Get output directory
        if output_dir is None:
            output_dir = self.output_config['output_dir']
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save options data with pricing and risk information
        self.options_data.to_csv(os.path.join(output_dir, 'options_analysis.csv'), index=False)
        logger.info(f"Options analysis saved to {os.path.join(output_dir, 'options_analysis.csv')}")
        
        # Calculate and save portfolio summary
        if all(col in self.options_data.columns for col in ['bs_price', 'jd_price', 'sabr_price']):  # EGARCH removed
            summary = pd.DataFrame({
                'Model': ['Black-Scholes', 'Jump-Diffusion', 'SABR'],  # EGARCH removed
                'Total_Price': [
                    (self.options_data['bs_price'] * self.options_data['notional']).sum(),
                    (self.options_data['jd_price'] * self.options_data['notional']).sum(),
                    (self.options_data['sabr_price'] * self.options_data['notional']).sum()
                ]
            })
            
            # Add PnL information if available
            if all(col in self.options_data.columns for col in ['bs_pnl', 'jd_pnl', 'sabr_pnl']):  # EGARCH removed
                summary['Total_PnL'] = [
                    self.options_data['bs_pnl'].sum(),
                    self.options_data['jd_pnl'].sum(),
                    self.options_data['sabr_pnl'].sum()
                ]
            
            # Add error metrics if available
            error_columns = ['bs_abs_error', 'jd_abs_error', 'sabr_abs_error']  # EGARCH removed
            if all(col in self.options_data.columns for col in error_columns):
                summary['RMSE'] = [
                    np.sqrt((self.options_data['bs_abs_error'] ** 2).mean()),
                    np.sqrt((self.options_data['jd_abs_error'] ** 2).mean()),
                    np.sqrt((self.options_data['sabr_abs_error'] ** 2).mean())
                ]
            
            summary.to_csv(os.path.join(output_dir, 'model_summary.csv'), index=False)
            logger.info(f"Model summary saved to {os.path.join(output_dir, 'model_summary.csv')}")
        
        logger.info("All results saved successfully")
    
    def plot_model_comparison(self, output_dir=None):
        """
        Plot a comparison of model performance.
        
        Args:
            output_dir (str, optional): Directory to save the plot. Defaults to the output directory in the configuration.
        """
        if self.options_data is None:
            logger.error("No options data. Nothing to plot.")
            return
        
        # Get output directory
        if output_dir is None:
            output_dir = self.output_config['output_dir']
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if PnL information is available
        if not all(col in self.options_data.columns for col in ['bs_pnl', 'jd_pnl', 'sabr_pnl']):  # EGARCH removed
            logger.error("PnL information not available. Calculate PnL first.")
            return
        
        # Calculate total PnL for each model
        total_pnl = {
            'Black-Scholes': self.options_data['bs_pnl'].sum(),
            'Jump-Diffusion': self.options_data['jd_pnl'].sum(),
            'SABR': self.options_data['sabr_pnl'].sum()  # EGARCH removed
        }
        
        # Create figure for total PnL comparison
        plt.figure(figsize=(12, 6))
        models = list(total_pnl.keys())
        pnls = list(total_pnl.values())
        colors = ['blue', 'red', 'purple']  # EGARCH color removed
        
        bars = plt.bar(models, pnls, color=colors)
        plt.title('Total PnL by Model')
        plt.xlabel('Model')
        plt.ylabel('PnL (EUR)')
        plt.grid(True, axis='y')
        
        # Add PnL values on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}',
                    ha='center', va='bottom', rotation=0)
        
        # Save plot
        plt.savefig(os.path.join(output_dir, 'pnl_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"PnL comparison plot saved to {os.path.join(output_dir, 'pnl_comparison.png')}")
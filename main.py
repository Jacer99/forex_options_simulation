"""
Main script for the Forex Options Portfolio Simulation.

This script orchestrates the entire simulation process, from data generation
to portfolio pricing, evaluation, and visualization.
"""

import os
import argparse
import logging
import yaml
import pandas as pd
from datetime import datetime

# Import project modules
from src.data_generation.option_generator import OptionGenerator
from src.market_data.data_handler import MarketDataHandler
from src.portfolio.portfolio_manager import PortfolioManager
from src.evaluation.performance_metrics import calculate_model_comparison, rank_models
from src.visualization.plotting import create_dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("forex_options_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
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

def generate_data(config):
    """
    Generate option contracts and market data.
    
    Args:
        config (dict): Configuration parameters.
        
    Returns:
        tuple: (options_data, market_data)
    """
    logger.info("Starting data generation phase")
    
    # Generate option contracts
    generator = OptionGenerator(config_path='config.yaml')
    options_data = generator.generate_portfolio()
    generator.save_portfolio_to_csv()
    
    # Generate market data
    market_handler = MarketDataHandler(config_path='config.yaml')
    market_data = market_handler.generate_market_data(save=True)
    
    logger.info("Data generation phase completed")
    return options_data, market_data

def load_data():
    """
    Load previously generated option contracts and market data.
    
    Returns:
        tuple: (options_data, market_data)
    """
    logger.info("Loading existing data")
    
    # Load option contracts
    options_file = "data/generated/option_contracts.csv"
    if not os.path.exists(options_file):
        logger.error(f"Options file '{options_file}' not found. Generate data first.")
        return None, None
    
    options_data = pd.read_csv(options_file)
    
    # Load market data
    market_handler = MarketDataHandler(config_path='config.yaml')
    market_data = market_handler.load_market_data()
    
    if market_data[0] is None:
        logger.error("Market data not found. Generate data first.")
        return options_data, None
    
    logger.info("Data loading completed")
    return options_data, market_data

def price_portfolio(options_data, market_data):
    """
    Price the portfolio using all pricing models.
    
    Args:
        options_data (pandas.DataFrame): Options data.
        market_data (tuple): Market data.
        
    Returns:
        pandas.DataFrame: Options data with added pricing information.
    """
    logger.info("Starting portfolio pricing phase")
    
    # Initialize portfolio manager
    portfolio_manager = PortfolioManager(options_data, market_data)
    
    # Price portfolio using all models
    priced_options = portfolio_manager.price_portfolio()
    
    # Calculate risk metrics
    priced_options, portfolio_risks = portfolio_manager.calculate_risks()
    
    logger.info("Portfolio pricing phase completed")
    return priced_options, portfolio_manager

def calculate_payoffs_and_pnl(portfolio_manager, market_data):
    """
    Calculate actual payoffs and PnL for the portfolio.
    
    Args:
        portfolio_manager (PortfolioManager): Portfolio manager instance.
        market_data (tuple): Market data.
        
    Returns:
        pandas.DataFrame: Options data with added payoff and PnL information.
    """
    logger.info("Starting payoff and PnL calculation phase")
    
    # Calculate actual payoffs using spot rates
    spot_rates = market_data[0]
    options_with_payoffs = portfolio_manager.calculate_actual_payoffs(spot_rates)
    
    # Calculate PnL
    options_with_pnl, total_pnl = portfolio_manager.calculate_pnl()
    
    logger.info("Payoff and PnL calculation phase completed")
    return options_with_pnl

def evaluate_performance(portfolio_manager, options_data):
    """
    Evaluate the performance of different pricing models.
    
    Args:
        portfolio_manager (PortfolioManager): Portfolio manager instance.
        options_data (pandas.DataFrame): Options data with pricing and payoff information.
        
    Returns:
        dict: Performance metrics for each model.
    """
    logger.info("Starting performance evaluation phase")
    
    # Evaluate model performance
    metrics = portfolio_manager.evaluate_model_performance()
    
    # Calculate and rank model performance metrics
    model_comparison = calculate_model_comparison(options_data)
    ranked_models = rank_models(model_comparison)
    
    logger.info("Performance evaluation phase completed")
    return metrics, model_comparison, ranked_models

def visualize_results(portfolio_manager, options_data, market_data, metrics, output_dir=None):
    """
    Create visualizations of the results.
    
    Args:
        portfolio_manager (PortfolioManager): Portfolio manager instance.
        options_data (pandas.DataFrame): Options data with pricing and payoff information.
        market_data (tuple): Market data.
        metrics (dict): Performance metrics for each model.
        output_dir (str, optional): Output directory for visualizations.
        
    Returns:
        list: Paths to the generated visualizations.
    """
    logger.info("Starting visualization phase")
    
    # Create dashboard of visualizations
    figure_paths = create_dashboard(options_data, market_data, metrics, output_dir)
    
    # Save analysis results
    portfolio_manager.save_results(output_dir)
    
    logger.info("Visualization phase completed")
    return figure_paths

def main():
    """Main function to run the Forex Options Portfolio Simulation."""
    logger.info("Starting Forex Options Portfolio Simulation")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Forex Options Portfolio Simulation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--generate", action="store_true", help="Generate new data instead of loading existing data")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--skip-pricing", action="store_true", help="Skip portfolio pricing phase")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip performance evaluation phase")
    parser.add_argument("--skip-visualization", action="store_true", help="Skip visualization phase")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else config["output"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate or load data
    if args.generate:
        options_data, market_data = generate_data(config)
    else:
        options_data, market_data = load_data()
        
    if options_data is None or market_data is None:
        logger.error("Failed to load or generate data. Exiting.")
        return
    
    # Price portfolio
    if not args.skip_pricing:
        options_data, portfolio_manager = price_portfolio(options_data, market_data)
    else:
        # Initialize portfolio manager with loaded data
        portfolio_manager = PortfolioManager(options_data, market_data)
        logger.info("Skipping portfolio pricing phase")
    
    # Calculate payoffs and PnL
    options_data = calculate_payoffs_and_pnl(portfolio_manager, market_data)
    
    # Evaluate performance
    if not args.skip_evaluation:
        metrics, model_comparison, ranked_models = evaluate_performance(portfolio_manager, options_data)
    else:
        metrics = None
        logger.info("Skipping performance evaluation phase")
    
    # Visualize results
    if not args.skip_visualization:
        figure_paths = visualize_results(portfolio_manager, options_data, market_data, metrics, output_dir)
    else:
        logger.info("Skipping visualization phase")
    
    logger.info(f"Forex Options Portfolio Simulation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
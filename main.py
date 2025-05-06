"""
Optimized Main Script for Forex Options Portfolio Simulation

This script orchestrates the entire simulation process with significant performance improvements,
from data generation to portfolio pricing, evaluation, and visualization.
"""

import os
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_data(config):
    """Generate option contracts and market data with progress reporting."""
    logger.info("Starting data generation phase")
    start_time = time.time()
    
    # Generate option contracts
    generator = OptionGenerator(config_path='config.yaml')
    options_data = generator.generate_portfolio()
    generator.save_portfolio_to_csv()
    
    # Generate market data
    market_handler = MarketDataHandler(config_path='config.yaml')
    market_data = market_handler.generate_market_data(save=True)
    
    elapsed = time.time() - start_time
    logger.info(f"Data generation phase completed in {elapsed:.2f} seconds")
    return options_data, market_data

def load_data():
    """Load previously generated option contracts and market data with validation."""
    logger.info("Loading existing data")
    start_time = time.time()
    
    # Load option contracts
    options_file = "data/generated/option_contracts.csv"
    if not os.path.exists(options_file):
        logger.error(f"Options file '{options_file}' not found. Generate data first.")
        return None, None
    
    try:
        options_data = pd.read_csv(options_file)
        logger.info(f"Loaded {len(options_data)} option contracts")
        
        # Validate essential columns
        required_columns = ['option_id', 'strike_price', 'days_to_maturity', 'issue_date', 'maturity_date', 'notional', 'type']
        missing_columns = [col for col in required_columns if col not in options_data.columns]
        if missing_columns:
            logger.error(f"Options data missing required columns: {missing_columns}")
            return None, None
    except Exception as e:
        logger.error(f"Error loading options data: {e}")
        return None, None
    
    # Load market data
    market_handler = MarketDataHandler(config_path='config.yaml')
    market_data = market_handler.load_market_data()
    
    if market_data[0] is None:
        logger.error("Market data not found. Generate data first.")
        return options_data, None
    
    # Validate market data structure
    spot_rates, volatility, interest_rates = market_data
    
    if 'date' not in spot_rates.columns or 'EUR/TND' not in spot_rates.columns:
        logger.error("Spot rates data has invalid structure")
        return options_data, None
    
    if 'date' not in volatility.columns or 'historical_vol' not in volatility.columns:
        logger.error("Volatility data has invalid structure")
        return options_data, None
    
    if 'date' not in interest_rates.columns or 'EUR_rate' not in interest_rates.columns or 'TND_rate' not in interest_rates.columns:
        logger.error("Interest rates data has invalid structure")
        return options_data, None
    
    # Convert dates to datetime for consistency
    spot_rates['date'] = pd.to_datetime(spot_rates['date'])
    volatility['date'] = pd.to_datetime(volatility['date'])
    interest_rates['date'] = pd.to_datetime(interest_rates['date'])
    
    elapsed = time.time() - start_time
    logger.info(f"Data loading completed in {elapsed:.2f} seconds")
    return options_data, market_data

def price_portfolio(options_data, market_data, config, use_parallel=True):
    """
    Price the portfolio using all pricing models with optional parallelization.
    
    Args:
        options_data: Options data DataFrame
        market_data: Market data tuple
        config: Configuration dictionary
        use_parallel: Whether to use parallel processing
        
    Returns:
        Tuple of (priced_options, portfolio_manager)
    """
    logger.info("Starting portfolio pricing phase")
    start_time = time.time()
    
    # Initialize portfolio manager
    portfolio_manager = PortfolioManager(options_data, market_data)
    
    if use_parallel and len(options_data) > 20:  # Only parallelize for larger portfolios
        # Determine optimal batch size and number of processes
        num_options = len(options_data)
        num_processes = min(multiprocessing.cpu_count(), 4)  # Limit to 4 processes
        batch_size = max(5, num_options // (num_processes * 2))  # At least 5 options per batch
        
        logger.info(f"Using parallel processing with {num_processes} processes, {batch_size} options per batch")
        
        # Split options into batches
        batches = []
        for batch_start in range(0, num_options, batch_size):
            batch_end = min(batch_start + batch_size, num_options)
            batches.append(options_data.iloc[batch_start:batch_end].copy())
        
        # Define worker function for parallel processing
        def process_batch(batch):
            # Create a temporary portfolio manager for this batch
            local_manager = PortfolioManager(batch, market_data)
            # Price with all models
            priced_batch = local_manager.price_portfolio()
            return priced_batch
        
        # Process batches in parallel
        processed_batches = []
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            
            # Track progress
            for i, future in enumerate(as_completed(futures)):
                try:
                    priced_batch = future.result()
                    processed_batches.append(priced_batch)
                    logger.info(f"Completed batch {i+1}/{len(batches)}")
                except Exception as e:
                    logger.error(f"Error in batch {i+1}: {e}")
        
        # Combine all batches
        priced_options = pd.concat(processed_batches).sort_index()
        
        # Update options in portfolio manager
        portfolio_manager.options_data = priced_options
        
    else:
        # Serial processing
        logger.info("Using serial processing for portfolio")
        priced_options = portfolio_manager.price_portfolio()
    
    # Calculate risk metrics
    priced_options, portfolio_risks = portfolio_manager.calculate_risks()
    
    # Log some summary statistics
    price_columns = [col for col in priced_options.columns if col.endswith('_price')]
    for col in price_columns:
        if col in priced_options.columns:
            total_value = (priced_options[col] * priced_options['notional']).sum()
            avg_price = priced_options[col].mean()
            logger.info(f"{col.upper()}: Total value = €{total_value:,.2f}, Avg price = {avg_price:.4f}")
    
    elapsed = time.time() - start_time
    logger.info(f"Portfolio pricing phase completed in {elapsed:.2f} seconds")
    return priced_options, portfolio_manager

def calculate_payoffs_and_pnl(portfolio_manager, market_data):
    """Calculate actual payoffs and PnL for the portfolio."""
    logger.info("Starting payoff and PnL calculation phase")
    start_time = time.time()
    
    # Calculate actual payoffs using spot rates
    spot_rates = market_data[0]
    options_with_payoffs = portfolio_manager.calculate_actual_payoffs(spot_rates)
    
    # Calculate PnL
    options_with_pnl, total_pnl = portfolio_manager.calculate_pnl()
    
    # Log PnL summary
    for model, pnl in total_pnl.items():
        logger.info(f"{model}: Total PnL = €{pnl:,.2f}")
    
    elapsed = time.time() - start_time
    logger.info(f"Payoff and PnL calculation completed in {elapsed:.2f} seconds")
    return options_with_pnl

def evaluate_performance(portfolio_manager, options_data):
    """Evaluate the performance of different pricing models."""
    logger.info("Starting performance evaluation phase")
    start_time = time.time()
    
    # Evaluate model performance
    metrics = portfolio_manager.evaluate_model_performance()
    
    # Calculate and rank model performance metrics
    model_comparison = calculate_model_comparison(options_data)
    ranked_models = rank_models(model_comparison)
    
    # Log performance metrics
    if model_comparison is not None:
        for _, row in ranked_models.sort_values('overall_rank').iterrows():
            logger.info(f"Model: {row['model']}, Rank: {row['overall_rank']:.1f}, RMSE: {row['rmse']:.6f}")
    
    elapsed = time.time() - start_time
    logger.info(f"Performance evaluation completed in {elapsed:.2f} seconds")
    return metrics, model_comparison, ranked_models

def visualize_results(portfolio_manager, options_data, market_data, metrics, output_dir=None):
    """Create visualizations of the results."""
    logger.info("Starting visualization phase")
    start_time = time.time()
    
    # Create dashboard of visualizations
    figure_paths = create_dashboard(options_data, market_data, metrics, output_dir)
    
    # Save analysis results
    portfolio_manager.save_results(output_dir)
    
    elapsed = time.time() - start_time
    logger.info(f"Visualization phase completed in {elapsed:.2f} seconds")
    return figure_paths

def optimize_market_data(market_data):
    """Optimize market data for faster lookups and reduced memory usage."""
    if market_data is None or len(market_data) != 3:
        return market_data
    
    spot_rates, volatility, interest_rates = market_data
    
    # Sort all dataframes by date for faster lookups
    spot_rates = spot_rates.sort_values('date')
    volatility = volatility.sort_values('date')
    interest_rates = interest_rates.sort_values('date')
    
    # Convert to datetime if not already
    for df in [spot_rates, volatility, interest_rates]:
        if df['date'].dtype != 'datetime64[ns]':
            df['date'] = pd.to_datetime(df['date'])
    
    # Create lookup dictionaries for faster date-based access
    spot_rates_dict = {d: row for d, row in zip(spot_rates['date'], spot_rates.to_dict('records'))}
    vol_dict = {d: row for d, row in zip(volatility['date'], volatility.to_dict('records'))}
    rates_dict = {d: row for d, row in zip(interest_rates['date'], interest_rates.to_dict('records'))}
    
    # Attach lookup dictionaries as attributes
    spot_rates.lookup_dict = spot_rates_dict
    volatility.lookup_dict = vol_dict
    interest_rates.lookup_dict = rates_dict
    
    return spot_rates, volatility, interest_rates

def main():
    """Main function to run the optimized Forex Options Portfolio Simulation."""
    overall_start_time = time.time()
    logger.info("Starting Forex Options Portfolio Simulation")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Forex Options Portfolio Simulation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--generate", action="store_true", help="Generate new data instead of loading existing data")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--skip-pricing", action="store_true", help="Skip portfolio pricing phase")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip performance evaluation phase")
    parser.add_argument("--skip-visualization", action="store_true", help="Skip visualization phase")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
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
    
    # Optimize market data for faster lookups
    market_data = optimize_market_data(market_data)
    
    # Price portfolio
    if not args.skip_pricing:
        use_parallel = not args.no_parallel
        options_data, portfolio_manager = price_portfolio(options_data, market_data, config, use_parallel)
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
    
    overall_elapsed = time.time() - overall_start_time
    logger.info(f"Forex Options Portfolio Simulation completed in {overall_elapsed:.2f} seconds. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
"""
Performance Metrics Module

This module provides functions for evaluating the performance of different pricing models.
"""

import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_error_metrics(actual, predicted, weights=None):
    """
    Calculate error metrics between actual and predicted values.
    
    Args:
        actual (array-like): Actual values.
        predicted (array-like): Predicted values.
        weights (array-like, optional): Weights for each observation.
        
    Returns:
        dict: Dictionary containing error metrics.
    """
    # Convert to numpy arrays and ensure they're float type
    try:
        actual = np.array(actual, dtype=float)
        predicted = np.array(predicted, dtype=float)
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting to numerical arrays: {e}")
        logger.error(f"Actual type: {type(actual)}, Predicted type: {type(predicted)}")
        logger.error(f"Sample actual: {actual[:5] if hasattr(actual, '__getitem__') else actual}")
        logger.error(f"Sample predicted: {predicted[:5] if hasattr(predicted, '__getitem__') else predicted}")
        return {
            'mae': np.nan,
            'mse': np.nan,
            'rmse': np.nan,
            'mape': np.nan,
            'r_squared': np.nan
        }
    
    # Filter out NaN values
    valid_indices = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[valid_indices]
    predicted = predicted[valid_indices]
    
    if len(actual) == 0:
        logger.warning("No valid data points for error calculation")
        return {
            'mae': np.nan,
            'mse': np.nan,
            'rmse': np.nan,
            'mape': np.nan,
            'r_squared': np.nan
        }
    
    # Calculate errors
    errors = actual - predicted
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # Apply weights if provided
    if weights is not None:
        weights = np.array(weights)[valid_indices]
        weights = weights / np.sum(weights)  # Normalize weights
        mae = np.sum(weights * abs_errors)
        mse = np.sum(weights * squared_errors)
    else:
        mae = np.mean(abs_errors)
        mse = np.mean(squared_errors)
    
    rmse = np.sqrt(mse)
    
    # Calculate MAPE for non-zero actual values
    non_zero_indices = actual != 0
    if np.any(non_zero_indices):
        percentage_errors = abs_errors[non_zero_indices] / np.abs(actual[non_zero_indices])
        mape = np.mean(percentage_errors) * 100
    else:
        mape = np.nan
    
    # Calculate R-squared
    if np.var(actual) > 0:
        r_squared = 1 - (np.sum(squared_errors) / (np.var(actual) * len(actual)))
    else:
        r_squared = np.nan
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r_squared': r_squared
    }

def calculate_pnl_metrics(actual_payoffs, option_prices, notionals):
    """
    Calculate profit and loss (PnL) metrics.
    
    Args:
        actual_payoffs (array-like): Actual option payoffs at maturity.
        option_prices (array-like): Option prices at issuance.
        notionals (array-like): Option notionals.
        
    Returns:
        dict: Dictionary containing PnL metrics.
    """
    # Convert to numpy arrays
    try:
        actual_payoffs = np.array(actual_payoffs, dtype=float)
        option_prices = np.array(option_prices, dtype=float)
        notionals = np.array(notionals, dtype=float)
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting to numerical arrays in PnL calculation: {e}")
        return {
            'total_pnl': np.nan,
            'mean_pnl': np.nan,
            'pnl_std': np.nan,
            'win_rate': np.nan,
            'profit_factor': np.nan
        }
    
    # Filter out NaN values
    valid_indices = ~np.isnan(actual_payoffs) & ~np.isnan(option_prices) & ~np.isnan(notionals)
    actual_payoffs = actual_payoffs[valid_indices]
    option_prices = option_prices[valid_indices]
    notionals = notionals[valid_indices]
    
    if len(actual_payoffs) == 0:
        logger.warning("No valid data points for PnL calculation")
        return {
            'total_pnl': np.nan,
            'mean_pnl': np.nan,
            'pnl_std': np.nan,
            'win_rate': np.nan,
            'profit_factor': np.nan
        }
    
    # Calculate PnL for each option
    option_costs = option_prices * notionals
    pnl = actual_payoffs - option_costs
    
    # Calculate PnL metrics
    total_pnl = np.sum(pnl)
    mean_pnl = np.mean(pnl)
    pnl_std = np.std(pnl)
    
    # Calculate win rate (percentage of profitable options)
    win_rate = np.mean(pnl > 0) * 100
    
    # Calculate profit factor (sum of profits / sum of losses)
    profits = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    
    if len(losses) > 0 and np.sum(np.abs(losses)) > 0:
        profit_factor = np.sum(profits) / np.sum(np.abs(losses))
    else:
        profit_factor = np.inf if len(profits) > 0 else np.nan
    
    return {
        'total_pnl': total_pnl,
        'mean_pnl': mean_pnl,
        'pnl_std': pnl_std,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }

def calculate_model_comparison(options_data, models=None):
    """
    Calculate performance metrics for comparing different pricing models.
    
    Args:
        options_data (pandas.DataFrame): Options data with actual payoffs and model prices.
        models (list, optional): List of model names to compare. Defaults to ['bs', 'jd', 'sabr'].
        
    Returns:
        pandas.DataFrame: DataFrame containing performance metrics for each model.
    """
    if models is None:
        models = ['bs', 'jd', 'sabr']
    
    # Check if required columns exist
    required_columns = ['actual_payoff', 'notional']
    model_price_columns = [f'{model}_price' for model in models]
    
    missing_columns = [col for col in required_columns + model_price_columns if col not in options_data.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return None
    
    # Initialize results
    results = []
    
    # Calculate metrics for each model
    for model in models:
        price_col = f'{model}_price'
        
        # Get valid data points
        valid_data = options_data.dropna(subset=['actual_payoff', price_col, 'notional'])
        
        if len(valid_data) == 0:
            logger.warning(f"No valid data points for model {model}")
            continue
        
        # Calculate error metrics
        try:
            actual_costs = valid_data['actual_payoff'].astype(float)
            predicted_costs = (valid_data[price_col] * valid_data['notional']).astype(float)
            error_metrics = calculate_error_metrics(actual_costs, predicted_costs)
        except Exception as e:
            logger.error(f"Error calculating metrics for model {model}: {e}")
            error_metrics = {
                'mae': np.nan,
                'mse': np.nan,
                'rmse': np.nan,
                'mape': np.nan,
                'r_squared': np.nan
            }
        
        # Calculate PnL metrics
        try:
            pnl_metrics = calculate_pnl_metrics(
                valid_data['actual_payoff'],
                valid_data[price_col], 
                valid_data['notional']
            )
        except Exception as e:
            logger.error(f"Error calculating PnL metrics for model {model}: {e}")
            pnl_metrics = {
                'total_pnl': np.nan,
                'mean_pnl': np.nan,
                'pnl_std': np.nan,
                'win_rate': np.nan,
                'profit_factor': np.nan
            }
        
        # Combine metrics
        model_metrics = {
            'model': model.upper(),
            'num_options': len(valid_data),
            **error_metrics,
            **pnl_metrics
        }
        
        results.append(model_metrics)
    
    # Convert to DataFrame
    if results:
        return pd.DataFrame(results)
    else:
        logger.error("No valid models for comparison")
        return None

def rank_models(metrics_df, metrics=None, higher_is_better=None):
    """
    Rank models based on specified performance metrics.
    
    Args:
        metrics_df (pandas.DataFrame): DataFrame containing performance metrics.
        metrics (list, optional): List of metrics to use for ranking. Defaults to ['rmse', 'total_pnl', 'profit_factor'].
        higher_is_better (dict, optional): Dict indicating whether higher values are better for each metric.
            Defaults to {'rmse': False, 'total_pnl': True, 'profit_factor': True}.
        
    Returns:
        pandas.DataFrame: DataFrame with added ranking columns.
    """
    if metrics is None:
        metrics = ['rmse', 'total_pnl', 'profit_factor']
    
    if higher_is_better is None:
        higher_is_better = {
            'rmse': False,
            'mae': False,
            'mse': False,
            'mape': False,
            'total_pnl': True,
            'mean_pnl': True,
            'win_rate': True,
            'profit_factor': True,
            'r_squared': True
        }
    
    # Create a copy of the metrics DataFrame
    ranked_df = metrics_df.copy()
    
    # Rank models for each metric
    for metric in metrics:
        if metric not in ranked_df.columns:
            logger.warning(f"Metric {metric} not found in DataFrame")
            continue
        
        # Determine ranking order
        ascending = not higher_is_better.get(metric, True)
        
        # Rank models
        ranked_df[f'{metric}_rank'] = ranked_df[metric].rank(ascending=ascending, method='min')
    
    # Calculate average rank
    rank_columns = [f'{metric}_rank' for metric in metrics if f'{metric}_rank' in ranked_df.columns]
    if rank_columns:
        ranked_df['avg_rank'] = ranked_df[rank_columns].mean(axis=1)
        ranked_df['overall_rank'] = ranked_df['avg_rank'].rank(method='min')
    
    return ranked_df
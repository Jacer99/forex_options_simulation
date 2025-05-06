"""
Visualization Module

This module provides functions for visualizing the portfolio of options and model performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_plot_style():
    """Set a consistent style for all plots."""
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18

def plot_spot_rates(spot_rates, volatility=None, output_dir=None, filename='spot_rates.png'):
    """
    Plot the EUR/TND spot rates and optionally volatility.
    
    Args:
        spot_rates (pandas.DataFrame): DataFrame with date and EUR/TND columns.
        volatility (pandas.DataFrame, optional): DataFrame with date and historical_vol columns.
        output_dir (str, optional): Directory to save the plot.
        filename (str, optional): Filename for the saved plot.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    set_plot_style()
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots()
    
    # Format the date on the x-axis
    spot_rates['date'] = pd.to_datetime(spot_rates['date'])
    
    # Plot spot rates
    ax1.plot(spot_rates['date'], spot_rates['EUR/TND'], 'b-', label='EUR/TND Spot Rate')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('EUR/TND Rate', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Format the x-axis to show dates nicely
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Add volatility if provided
    if volatility is not None:
        volatility['date'] = pd.to_datetime(volatility['date'])
        
        # Create secondary axis for volatility
        ax2 = ax1.twinx()
        ax2.plot(volatility['date'], volatility['historical_vol'] * 100, 'r-', label='Volatility')
        ax2.set_ylabel('Volatility (%)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')
    
    plt.title('EUR/TND Spot Rate and Volatility')
    plt.tight_layout()
    
    # Save the figure if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Spot rates plot saved to {filepath}")
    
    return fig

def plot_active_notional(notional_data, output_dir=None, filename='active_notional.png'):
    """
    Plot the active notional over time.
    
    Args:
        notional_data (pandas.DataFrame): DataFrame with date, active_notional, and active_options columns.
        output_dir (str, optional): Directory to save the plot.
        filename (str, optional): Filename for the saved plot.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    set_plot_style()
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots()
    
    # Format the date on the x-axis
    notional_data['date'] = pd.to_datetime(notional_data['date'])
    
    # Plot active notional
    ax1.plot(notional_data['date'], notional_data['active_notional'], 'b-', label='Active Notional')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Active Notional (EUR)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Format y-axis to show currency values
    formatter = plt.FuncFormatter(lambda x, p: f'€{x:,.0f}')
    ax1.yaxis.set_major_formatter(formatter)
    
    # Format the x-axis to show dates nicely
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Create secondary axis for number of active options
    ax2 = ax1.twinx()
    ax2.plot(notional_data['date'], notional_data['active_options'], 'r-', label='Active Options')
    ax2.set_ylabel('Number of Active Options', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Active Notional and Number of Options')
    plt.tight_layout()
    
    # Save the figure if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Active notional plot saved to {filepath}")
    
    return fig

def plot_model_comparison(options_data, metric='price', output_dir=None, filename='model_comparison.png'):
    """
    Plot a comparison of different pricing models.
    
    Args:
        options_data (pandas.DataFrame): Options data with pricing information.
        metric (str, optional): Metric to compare ('price' or 'pnl'). Defaults to 'price'.
        output_dir (str, optional): Directory to save the plot.
        filename (str, optional): Filename for the saved plot.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    set_plot_style()
    
    # Check if required columns exist
    if metric == 'price':
        required_columns = ['bs_price', 'egarch_price', 'jd_price', 'sabr_price', 'notional']
        missing_columns = [col for col in required_columns if col not in options_data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Calculate total prices
        total_prices = {
            'Black-Scholes': (options_data['bs_price'] * options_data['notional']).sum(),
            'E-GARCH MC': (options_data['egarch_price'] * options_data['notional']).sum(),
            'Jump-Diffusion': (options_data['jd_price'] * options_data['notional']).sum(),
            'SABR': (options_data['sabr_price'] * options_data['notional']).sum()
        }
        
        # Create bar chart
        fig, ax = plt.subplots()
        models = list(total_prices.keys())
        values = list(total_prices.values())
        colors = ['blue', 'green', 'red', 'purple']
        
        bars = ax.bar(models, values, color=colors)
        ax.set_xlabel('Model')
        ax.set_ylabel('Total Option Value (EUR)')
        ax.set_title('Comparison of Model Prices')
        
        # Format y-axis to show currency values
        formatter = plt.FuncFormatter(lambda x, p: f'€{x:,.0f}')
        ax.yaxis.set_major_formatter(formatter)
        
        # Add values on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'€{height:,.0f}',
                    ha='center', va='bottom', rotation=0)
        
    elif metric == 'pnl':
        required_columns = ['bs_pnl', 'egarch_pnl', 'jd_pnl', 'sabr_pnl']
        missing_columns = [col for col in required_columns if col not in options_data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Calculate total PnL
        total_pnl = {
            'Black-Scholes': options_data['bs_pnl'].sum(),
            'E-GARCH MC': options_data['egarch_pnl'].sum(),
            'Jump-Diffusion': options_data['jd_pnl'].sum(),
            'SABR': options_data['sabr_pnl'].sum()
        }
        
        # Create bar chart
        fig, ax = plt.subplots()
        models = list(total_pnl.keys())
        values = list(total_pnl.values())
        colors = ['blue', 'green', 'red', 'purple']
        
        bars = ax.bar(models, values, color=colors)
        ax.set_xlabel('Model')
        ax.set_ylabel('Total PnL (EUR)')
        ax.set_title('Comparison of Model PnL')
        
        # Format y-axis to show currency values
        formatter = plt.FuncFormatter(lambda x, p: f'€{x:,.0f}')
        ax.yaxis.set_major_formatter(formatter)
        
        # Add values on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'€{height:,.0f}',
                    ha='center', va='bottom', rotation=0)
    
    else:
        logger.error(f"Invalid metric: {metric}")
        return None
    
    plt.tight_layout()
    
    # Save the figure if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {filepath}")
    
    return fig

def plot_error_metrics(metrics_df, output_dir=None, filename='error_metrics.png'):
    """
    Plot error metrics for each model.
    
    Args:
        metrics_df (pandas.DataFrame or dict): DataFrame or dictionary containing error metrics for each model.
        output_dir (str, optional): Directory to save the plot.
        filename (str, optional): Filename for the saved plot.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    set_plot_style()
    
    # Convert dict to DataFrame if necessary
    if isinstance(metrics_df, dict):
        # If it's a nested dict (model -> metric -> value)
        if any(isinstance(v, dict) for v in metrics_df.values()):
            metrics_list = []
            for model, model_metrics in metrics_df.items():
                model_metrics['model'] = model.upper()
                metrics_list.append(model_metrics)
            metrics_df = pd.DataFrame(metrics_list)
        else:
            # If it's a simple dict (metric -> value)
            metrics_df = pd.DataFrame([metrics_df])
    
    # Check if we have a valid DataFrame
    if metrics_df is None or len(metrics_df) == 0:
        logger.error("No error metrics available")
        return None
    
    # Define required metrics to look for
    required_metrics = ['rmse', 'mae', 'mape']
    
    # Find which metrics are available in the DataFrame
    if hasattr(metrics_df, 'columns'):
        available_metrics = [metric for metric in required_metrics if metric in metrics_df.columns]
    else:
        # If metrics_df doesn't have columns attribute (e.g., it's a Series or other object)
        available_metrics = []
        for metric in required_metrics:
            try:
                if metric in metrics_df:
                    available_metrics.append(metric)
            except:
                pass
    
    if not available_metrics:
        logger.error("No error metrics available in metrics_df")
        return None
    
    # Create subplots for each metric
    num_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 6))
    
    # If only one metric is available, axes is not a list
    if num_metrics == 1:
        axes = [axes]
    
    colors = ['blue', 'green', 'red', 'purple']
    
    # Plot each metric
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        
        try:
            # Sort by metric value (lower is better for error metrics)
            if 'model' in metrics_df.columns:
                sorted_df = metrics_df.sort_values(by=metric)
                
                # Plot as bar chart
                bars = ax.bar(sorted_df['model'], sorted_df[metric], color=colors[:len(sorted_df)])
                ax.set_xlabel('Model')
                ax.set_ylabel(metric.upper())
                ax.set_title(f'{metric.upper()} by Model')
                
                # Add values on top of the bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom', rotation=0)
            else:
                # Handle the case where we don't have a 'model' column
                labels = metrics_df.index if hasattr(metrics_df, 'index') else [f'Model {j+1}' for j in range(len(metrics_df))]
                values = metrics_df[metric] if hasattr(metrics_df, 'columns') else [metrics_df[metric]]
                
                bars = ax.bar(labels, values, color=colors[:len(labels)])
                ax.set_xlabel('Model')
                ax.set_ylabel(metric.upper())
                ax.set_title(f'{metric.upper()}')
                
                # Add values on top of the bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom', rotation=0)
        except Exception as e:
            logger.error(f"Error plotting metric {metric}: {e}")
            continue
        
        # Rotate x-axis labels if there are many models
        if hasattr(metrics_df, '__len__') and len(metrics_df) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the figure if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Error metrics plot saved to {filepath}")
    
    return fig

def plot_option_distribution(options_data, output_dir=None, filename='option_distribution.png'):
    """
    Plot the distribution of options by maturity and strike.
    
    Args:
        options_data (pandas.DataFrame): Options data.
        output_dir (str, optional): Directory to save the plot.
        filename (str, optional): Filename for the saved plot.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    set_plot_style()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot distribution of days to maturity
    sns.histplot(options_data['days_to_maturity'], bins=20, kde=True, ax=ax1)
    ax1.set_xlabel('Days to Maturity')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Option Maturities')
    
    # Plot distribution of strike prices
    sns.histplot(options_data['strike_price'], bins=20, kde=True, ax=ax2)
    ax2.set_xlabel('Strike Price')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Strike Prices')
    
    plt.tight_layout()
    
    # Save the figure if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Option distribution plot saved to {filepath}")
    
    return fig

def plot_price_vs_strike(options_data, output_dir=None, filename='price_vs_strike.png'):
    """
    Plot option prices vs. strike prices for each model.
    
    Args:
        options_data (pandas.DataFrame): Options data with pricing information.
        output_dir (str, optional): Directory to save the plot.
        filename (str, optional): Filename for the saved plot.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    set_plot_style()
    
    # Check if required columns exist
    required_columns = ['strike_price', 'bs_price', 'egarch_price', 'jd_price', 'sabr_price']
    missing_columns = [col for col in required_columns if col not in options_data.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return None
    
    # Create a figure
    fig, ax = plt.subplots()
    
    # Plot price vs. strike for each model
    ax.scatter(options_data['strike_price'], options_data['bs_price'], 
               label='Black-Scholes', color='blue', alpha=0.7)
    ax.scatter(options_data['strike_price'], options_data['egarch_price'], 
               label='E-GARCH MC', color='green', alpha=0.7)
    ax.scatter(options_data['strike_price'], options_data['jd_price'], 
               label='Jump-Diffusion', color='red', alpha=0.7)
    ax.scatter(options_data['strike_price'], options_data['sabr_price'], 
               label='SABR', color='purple', alpha=0.7)
    
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Option Price')
    ax.set_title('Option Prices vs. Strike Price')
    ax.legend()
    
    plt.tight_layout()
    
    # Save the figure if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Price vs. strike plot saved to {filepath}")
    
    return fig

def plot_volatility_smile(options_data, issue_date=None, output_dir=None, filename='volatility_smile.png'):
    """
    Plot the volatility smile for a specific issue date or for all options.
    
    Args:
        options_data (pandas.DataFrame): Options data with implied volatility information.
        issue_date (str, optional): Issue date to filter options. If None, all options are used.
        output_dir (str, optional): Directory to save the plot.
        filename (str, optional): Filename for the saved plot.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    set_plot_style()
    
    # Check if required columns exist
    if 'implied_volatility' not in options_data.columns or 'strike_price' not in options_data.columns:
        logger.error("Missing required columns: implied_volatility or strike_price")
        return None
    
    # Filter options by issue date if provided
    if issue_date is not None:
        filtered_data = options_data[options_data['issue_date'] == issue_date]
        if len(filtered_data) == 0:
            logger.error(f"No options found for issue date: {issue_date}")
            return None
    else:
        filtered_data = options_data
    
    # Create a figure
    fig, ax = plt.subplots()
    
    # Plot implied volatility vs. strike price
    ax.scatter(filtered_data['strike_price'], filtered_data['implied_volatility'] * 100, 
               color='blue', alpha=0.7)
    
    # Add a trendline
    if len(filtered_data) > 1:
        try:
            from scipy.stats import linregress
            from scipy.optimize import curve_fit
            
            # Define a quadratic function for the volatility smile
            def vol_smile(x, a, b, c):
                return a * (x - b)**2 + c
            
            # Fit the function to the data
            popt, _ = curve_fit(vol_smile, filtered_data['strike_price'], 
                               filtered_data['implied_volatility'] * 100)
            
            # Generate points for the trendline
            x_trend = np.linspace(filtered_data['strike_price'].min(), 
                                 filtered_data['strike_price'].max(), 100)
            y_trend = vol_smile(x_trend, *popt)
            
            # Plot the trendline
            ax.plot(x_trend, y_trend, 'r-', label='Volatility Smile')
            
        except Exception as e:
            logger.warning(f"Error fitting volatility smile: {e}")
    
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Implied Volatility (%)')
    title = 'Volatility Smile' if issue_date is None else f'Volatility Smile for {issue_date}'
    ax.set_title(title)
    
    if len(filtered_data) > 1:
        ax.legend()
    
    plt.tight_layout()
    
    # Save the figure if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Volatility smile plot saved to {filepath}")
    
    return fig

def create_dashboard(options_data, market_data, metrics_df=None, output_dir='output'):
    """
    Create a comprehensive dashboard of visualizations.
    
    Args:
        options_data (pandas.DataFrame): Options data with pricing information.
        market_data (tuple): Tuple of (spot_rates, volatility, interest_rates) DataFrames.
        metrics_df (pandas.DataFrame or dict, optional): DataFrame or dictionary containing performance metrics.
        output_dir (str, optional): Directory to save the plots.
        
    Returns:
        list: List of generated figure paths.
    """
    set_plot_style()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # List to store figure paths
    figure_paths = []
    
    # Extract market data
    spot_rates, volatility, interest_rates = market_data
    
    # Plot spot rates and volatility
    fig = plot_spot_rates(spot_rates, volatility, output_dir, 'spot_rates_volatility.png')
    figure_paths.append(os.path.join(output_dir, 'spot_rates_volatility.png'))
    
    # Plot option distribution
    fig = plot_option_distribution(options_data, output_dir, 'option_distribution.png')
    figure_paths.append(os.path.join(output_dir, 'option_distribution.png'))
    
    # Plot price vs. strike
    if all(col in options_data.columns for col in ['strike_price', 'bs_price', 'egarch_price', 'jd_price', 'sabr_price']):
        fig = plot_price_vs_strike(options_data, output_dir, 'price_vs_strike.png')
        figure_paths.append(os.path.join(output_dir, 'price_vs_strike.png'))
    
    # Plot volatility smile
    if 'implied_volatility' in options_data.columns:
        fig = plot_volatility_smile(options_data, None, output_dir, 'volatility_smile.png')
        figure_paths.append(os.path.join(output_dir, 'volatility_smile.png'))
    
    # Plot model comparison (prices)
    if all(col in options_data.columns for col in ['bs_price', 'egarch_price', 'jd_price', 'sabr_price', 'notional']):
        fig = plot_model_comparison(options_data, 'price', output_dir, 'model_comparison_price.png')
        figure_paths.append(os.path.join(output_dir, 'model_comparison_price.png'))
    
    # Plot model comparison (PnL)
    if all(col in options_data.columns for col in ['bs_pnl', 'egarch_pnl', 'jd_pnl', 'sabr_pnl']):
        fig = plot_model_comparison(options_data, 'pnl', output_dir, 'model_comparison_pnl.png')
        figure_paths.append(os.path.join(output_dir, 'model_comparison_pnl.png'))
    
    # Plot error metrics
    if metrics_df is not None:
        fig = plot_error_metrics(metrics_df, output_dir, 'error_metrics.png')
        figure_paths.append(os.path.join(output_dir, 'error_metrics.png'))
    
    logger.info(f"Dashboard created with {len(figure_paths)} plots in {output_dir}")
    return figure_paths
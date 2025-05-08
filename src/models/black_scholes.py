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
        # (unchanged content here...)

        # Convert days to years
        T = days_to_maturity / 365.0

        # Handle edge cases
        if T <= 0 or volatility <= 0:
            if option_type.lower() == 'call':
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)

        d1 = (np.log(spot / strike) + (domestic_rate - foreign_rate +
              0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)

        if option_type.lower() == 'call':
            price = spot * np.exp(-foreign_rate * T) * norm.cdf(d1) - \
                strike * np.exp(-domestic_rate * T) * norm.cdf(d2)
        else:
            price = strike * np.exp(-domestic_rate * T) * norm.cdf(-d2) - \
                spot * np.exp(-foreign_rate * T) * norm.cdf(-d1)

        return price

    def calculate_greeks(self, spot, strike, days_to_maturity, domestic_rate, foreign_rate,
                         volatility, option_type='call'):
        # (unchanged content...)

        T = days_to_maturity / 365.0

        if T <= 0 or volatility <= 0:
            return {
                'delta': 1.0 if option_type.lower() == 'call' and spot > strike else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }

        d1 = (np.log(spot / strike) + (domestic_rate - foreign_rate +
              0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        n_d1 = norm.pdf(d1)

        if option_type.lower() == 'call':
            delta = np.exp(-foreign_rate * T) * norm.cdf(d1)
        else:
            delta = np.exp(-foreign_rate * T) * (norm.cdf(d1) - 1)

        gamma = np.exp(-foreign_rate * T) * n_d1 / \
            (spot * volatility * np.sqrt(T))
        vega = spot * np.exp(-foreign_rate * T) * n_d1 * np.sqrt(T) / 100

        term1 = -spot * np.exp(-foreign_rate * T) * \
            n_d1 * volatility / (2 * np.sqrt(T))
        if option_type.lower() == 'call':
            theta = term1 + foreign_rate * spot * np.exp(-foreign_rate * T) * norm.cdf(
                d1) - domestic_rate * strike * np.exp(-domestic_rate * T) * norm.cdf(d2)
        else:
            theta = term1 - foreign_rate * spot * np.exp(-foreign_rate * T) * norm.cdf(
                -d1) + domestic_rate * strike * np.exp(-domestic_rate * T) * norm.cdf(-d2)

        theta = theta / 365.0

        if option_type.lower() == 'call':
            rho = strike * T * np.exp(-domestic_rate * T) * norm.cdf(d2) / 100
        else:
            rho = -strike * T * \
                np.exp(-domestic_rate * T) * norm.cdf(-d2) / 100

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
        # (unchanged content...)

        T = days_to_maturity / 365.0
        if T <= 0:
            return 0.0

        if spot > strike and option_type.lower() == 'call':
            vol = 0.3
        elif spot < strike and option_type.lower() == 'put':
            vol = 0.3
        else:
            vol = 0.2

        for i in range(max_iterations):
            price = self.price_option(
                spot, strike, days_to_maturity, domestic_rate, foreign_rate, vol, option_type)
            vega = self.calculate_greeks(
                spot, strike, days_to_maturity, domestic_rate, foreign_rate, vol, option_type)['vega'] * 100

            if abs(vega) < 1e-10:
                if vol < 0.001:
                    vol = 0.2
                else:
                    break
            else:
                vol_new = vol + (market_price - price) / vega
                vol_new = max(0.001, vol_new)

                if abs(vol_new - vol) < precision:
                    vol = vol_new
                    break

                vol = vol_new

        return vol

    def price_options_portfolio(self, options_data, market_data):
        """
        Price a portfolio of European FX options using the Black-Scholes model.
        """
        if isinstance(options_data, list):
            df = pd.DataFrame(options_data)
        else:
            df = options_data.copy()

        spot_rates, volatilities, interest_rates = market_data

        if 'bs_price' not in df.columns:
            df['bs_price'] = np.nan

        if 'implied_volatility' not in df.columns:
            df['implied_volatility'] = np.nan

        for i, option in df.iterrows():
            try:
                issue_date = pd.to_datetime(option['issue_date'])
                spot_data = spot_rates[spot_rates['date'] <= issue_date]
                if spot_data.empty:
                    logger.warning(
                        f"No spot rate data for {issue_date} for option {option['option_id']}")
                    continue
                spot = spot_data.iloc[-1]['EUR/TND']

                if abs(spot - option['spot_rate_at_issue']) > 0.0001:
                    logger.warning(f"Spot rate mismatch for option {option['option_id']}: "
                                   f"Stored: {option['spot_rate_at_issue']}, Found: {spot}")

                vol_data = volatilities[volatilities['date'] <= issue_date]
                if vol_data.empty:
                    logger.warning(
                        f"No volatility data for {issue_date} for option {option['option_id']}")
                    vol = self.market_config['volatility']  # Use default value
                else:
                    vol = vol_data.iloc[-1]['historical_vol']

                rates_data = interest_rates[interest_rates['date']
                                            <= issue_date]
                if rates_data.empty:
                    logger.warning(
                        f"No interest rate data for {issue_date} for option {option['option_id']}")
                    continue
                rates = rates_data.iloc[-1]
                domestic_rate = rates['EUR_rate']
                foreign_rate = rates['TND_rate']

                price = self.price_option(
                    spot=spot,
                    strike=option['strike_price'],
                    days_to_maturity=option['days_to_maturity'],
                    domestic_rate=domestic_rate,
                    foreign_rate=foreign_rate,
                    volatility=vol,
                    option_type=option['type']
                )

                implied_vol = self.implied_volatility(
                    market_price=price / option['notional'],
                    spot=spot,
                    strike=option['strike_price'],
                    days_to_maturity=option['days_to_maturity'],
                    domestic_rate=domestic_rate,
                    foreign_rate=foreign_rate,
                    option_type=option['type']
                )

                df.at[i, 'bs_price'] = price
                df.at[i, 'implied_volatility'] = implied_vol

            except Exception as e:
                logger.error(
                    f"Error pricing option {option.get('option_id', i)}: {e}")

        return df

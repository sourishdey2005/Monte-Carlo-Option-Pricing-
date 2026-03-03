"""
Monte Carlo Option Pricing Dashboard - Streamlit Edition
==========================================================
Single-file, production-ready implementation for Python 3.11+

Features:
- Real-time Alpha Vantage API integration with .env configuration
- Parallel Monte Carlo simulation (multiprocessing)
- Full Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
- Interactive Plotly visualizations
- Black-Scholes benchmark validation
- Environment variable configuration

Setup:
1. Create .env file with ALPHA_VANTAGE_API_KEY=your_key_here
2. pip install -r requirements.txt
3. streamlit run app.py

Author: Quantitative Finance Suite
Version: 3.1.0
"""

import streamlit as st
st.set_page_config(
    page_title="Monte Carlo Option Pricing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Standard library
import os
import sys
import time
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Tuple, Dict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

# Load environment variables (optional fallback)
import os
from pathlib import Path

# API Configuration - Hardcoded for immediate use
ALPHA_VANTAGE_API_KEY = "LUOX9WBCP5ZYFZ0K"

# Scientific computing
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# API
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

def get_env_variable(var_name: str, default_value: str = "") -> str:
    """Safely retrieve environment variable with fallback."""
    value = os.getenv(var_name, default_value)
    return value.strip() if value else default_value

# API Configuration from Hardcoded String (overridable by ENV)
ALPHA_VANTAGE_API_KEY: str = get_env_variable("ALPHA_VANTAGE_API_KEY", ALPHA_VANTAGE_API_KEY)
API_BASE_URL: str = "https://www.alphavantage.co/query"
API_RATE_LIMIT_PER_MINUTE: int = 5  # Free tier limit
API_DAILY_LIMIT: int = 25  # Free tier daily limit
GLOBAL_QUOTE_KEY: str = "Global Quote"
CHART_CONTAINER_CLASS: str = "<div class='chart-container animated'>"
DIV_CLOSE: str = "</div>"
TIME_STEPS_LABEL: str = "Time Steps"
PRICE_LABEL: str = "Price ($)"
TRANSPARENT_RGBA: str = "rgba(0,0,0,0)"
LINES_MARKERS: str = "lines+markers"

# Application Defaults from Environment
DEFAULT_TICKER: str = get_env_variable("DEFAULT_TICKER", "AAPL")
DEFAULT_SIMULATIONS: int = int(get_env_variable("DEFAULT_SIMULATIONS", "100000"))
DEFAULT_OPTION_TYPE: str = get_env_variable("DEFAULT_OPTION_TYPE", "call")

# Validation
if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "demo":
    st.sidebar.warning("⚠️ Using DEMO API key. For real data, set ALPHA_VANTAGE_API_KEY in .env file")

# Type aliases
OptionType = Literal["call", "put"]

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True, slots=True)
class OptionParams:
    """Immutable option parameters container."""
    S0: float = 100.0           # Spot price
    K: float = 105.0            # Strike price
    T: float = 1.0              # Time to maturity (years)
    r: float = 0.05             # Risk-free rate
    sigma: float = 0.20         # Volatility
    q: float = 0.0              # Dividend yield
    option_type: OptionType = "call"
    
    def __post_init__(self):
        if self.S0 <= 0:
            raise ValueError(f"Spot price must be positive, got {self.S0}")
        if self.K <= 0:
            raise ValueError(f"Strike price must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to maturity must be positive, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be positive, got {self.sigma}")

@dataclass
class PricingResult:
    """Monte Carlo pricing result container."""
    price: float
    std_error: float
    confidence_lower: float
    confidence_upper: float
    paths: Optional[List[List[float]]]
    payoffs: List[float]
    computation_time: float

@dataclass
class Greeks:
    """Option Greeks container."""
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0

@dataclass
class MarketData:
    """Market data from Alpha Vantage API."""
    symbol: str
    price: float = 0.0
    change: float = 0.0
    change_percent: float = 0.0
    volume: int = 0
    date: str = ""
    historical_volatility: float = 0.20
    is_mock: bool = False

# ============================================================================
# ALPHA VANTAGE API CLIENT
# ============================================================================

class AlphaVantageClient:
    """
    Thread-safe Alpha Vantage API client with:
    - Environment-based API key configuration
    - Intelligent caching
    - Rate limiting
    - Automatic fallback to mock data
    """
    
    _instance: Optional['AlphaVantageClient'] = None
    _lock = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Use API key from environment
        self.api_key: str = ALPHA_VANTAGE_API_KEY
        self.base_url: str = API_BASE_URL
        
        # Setup HTTP session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # Rate limiting
        self._last_call_time: float = 0.0
        self._min_interval: float = 60.0 / API_RATE_LIMIT_PER_MINUTE
        
        # Cache
        self._cache: Dict[str, Tuple[MarketData, float]] = {}
        self._cache_ttl: int = 300  # 5 minutes
        
        self._initialized = True
        
        # Log configuration
        if self.api_key == "demo":
            self._using_demo = True
        else:
            self._using_demo = False
    
    def _enforce_rate_limit(self) -> None:
        """Enforce API rate limiting to respect free tier."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.time()
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still fresh."""
        if symbol not in self._cache:
            return False
        _, timestamp = self._cache[symbol]
        return (time.time() - timestamp) < self._cache_ttl
    
    @lru_cache(maxsize=32)
    def fetch_quote(self, symbol: str) -> MarketData:
        """
        Fetch real-time stock quote from Alpha Vantage API.
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL", "MSFT", "TSLA")
        
        Returns:
            MarketData object with price, volume, and volatility info
        
        Note:
            Uses API key from ALPHA_VANTAGE_API_KEY environment variable.
            Falls back to mock data if API fails or rate limited.
        """
        symbol = symbol.upper().strip()
        
        # Check cache
        if self._is_cache_valid(symbol):
            return self._cache[symbol][0]
        
        # Enforce rate limit
        self._enforce_rate_limit()
        
        # Build API request
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            if GLOBAL_QUOTE_KEY in data and data[GLOBAL_QUOTE_KEY]:
                quote = data[GLOBAL_QUOTE_KEY]
                
                market_data = MarketData(
                    symbol=symbol,
                    price=float(quote['05. price']),
                    change=float(quote['09. change']),
                    change_percent=float(quote['10. change percent'].rstrip('%')),
                    volume=int(quote['06. volume']),
                    date=quote['07. latest trading day'],
                    historical_volatility=self._estimate_volatility(symbol),
                    is_mock=False
                )
                
                # Cache result
                self._cache[symbol] = (market_data, time.time())
                
                if self._using_demo:
                    st.info(f"✓ Live data fetched for {symbol} (DEMO key - limited usage)")
                else:
                    st.success(f"✓ Live data fetched for {symbol}")
                
                return market_data
            else:
                raise ValueError("Invalid API response structure")
                
        except Exception as e:
            if not self._using_demo:
                st.error(f"API Error for {symbol}: {str(e)}")
            return self._generate_mock_data(symbol)
    
    def _estimate_volatility(self, symbol: str) -> float:
        """Estimate historical volatility from daily price data."""
        try:
            self._enforce_rate_limit()
            
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'compact'
            }
            
            response = self.session.get(self.base_url, params=params, timeout=15)
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                return 0.20
            
            time_series = data['Time Series (Daily)']
            closes = []
            
            # Get last 60 days of closing prices
            for date, values in list(time_series.items())[:60]:
                closes.append(float(values['4. close']))
            
            if len(closes) < 20:
                return 0.20
            
            # Calculate log returns
            log_returns = np.diff(np.log(closes))
            volatility = np.std(log_returns) * np.sqrt(252)
            
            return float(np.clip(volatility, 0.05, 1.0))
            
        except Exception:
            return 0.20
    
    def _generate_mock_data(self, symbol: str) -> MarketData:
        """
        Generate realistic mock data for testing when API fails.
        Uses symbol hash for deterministic but symbol-specific randomness.
        """
        # Create deterministic seed from symbol
        seed = int(hashlib.md5(symbol.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        
        base_price = round(rng.uniform(50.0, 500.0), 2)
        
        st.warning(f"⚠️ Using mock data for {symbol} (API unavailable or DEMO key)")
        
        return MarketData(
            symbol=symbol,
            price=base_price,
            change=round(rng.uniform(-5.0, 5.0), 2),
            change_percent=round(rng.uniform(-2.0, 2.0), 2),
            volume=int(rng.integers(1_000_000, 10_000_000)),
            date=datetime.now().strftime("%Y-%m-%d"),
            historical_volatility=0.20,
            is_mock=True
        )
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
        self.fetch_quote.cache_clear()

# ============================================================================
# MONTE CARLO ENGINE
# ============================================================================

def _simulate_chunk_worker(args: Tuple) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Worker function for parallel Monte Carlo simulation.
    
    Simulates Geometric Brownian Motion paths and calculates option payoffs.
    Uses antithetic variates for variance reduction when applicable.
    """
    params_dict, n_paths, n_steps, seed, store_paths, jump_params = args
    params = OptionParams(**params_dict)
    
    # GBM parameters
    dt = params.T / n_steps
    drift = (params.r - params.q - 0.5 * params.sigma ** 2) * dt
    diffusion = params.sigma * np.sqrt(dt)
    
    rng = np.random.default_rng(seed)
    
    if store_paths and n_paths <= 500:
        # Store paths for visualization
        if jump_params['lambda_j'] > 0:
            # Add Merton Jumps
            njumps = rng.poisson(jump_params['lambda_j'] * dt, (n_paths, n_steps))
            log_jumps = njumps * jump_params['mu_j'] + np.sqrt(njumps) * jump_params['sigma_j'] * rng.standard_normal((n_paths, n_steps))
            log_returns = drift + diffusion * rng.standard_normal((n_paths, n_steps)) + log_jumps
            log_paths = np.cumsum(np.column_stack([np.full(n_paths, np.log(params.S0)), log_returns]), axis=1)
            paths = np.exp(log_paths)
        else:
            paths = np.zeros((n_paths, n_steps + 1))
            paths[:, 0] = params.S0
            for t in range(1, n_steps + 1):
                paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * rng.standard_normal(n_paths))
        
        ST = paths[:, -1]
        payoffs = np.maximum(ST - params.K, 0) if params.option_type == "call" \
                  else np.maximum(params.K - ST, 0)
        return payoffs, paths
    
    # Vectorized simulation with antithetic variates
    Z = rng.standard_normal((n_paths // 2, n_steps))
    
    # Stochastic Jumps (Merton Jump Diffusion)
    log_jumps = np.zeros_like(Z)
    if jump_params['lambda_j'] > 0:
        dt = params.T / n_steps
        njumps = rng.poisson(jump_params['lambda_j'] * dt, (n_paths // 2, n_steps))
        log_jumps = njumps * jump_params['mu_j'] + \
                    np.sqrt(njumps) * jump_params['sigma_j'] * \
                    rng.standard_normal((n_paths // 2, n_steps))
    
    # Standard paths
    log_returns = drift + diffusion * Z + log_jumps
    log_paths = np.concatenate([
        np.full((n_paths // 2, 1), np.log(params.S0)),
        log_returns
    ], axis=1)
    log_paths = np.cumsum(log_paths, axis=1)
    ST = np.exp(log_paths[:, -1])
    payoffs_std = np.maximum(ST - params.K, 0) if params.option_type == "call" \
                  else np.maximum(params.K - ST, 0)
    
    # Antithetic paths (variance reduction on Brownian component)
    log_returns_anti = drift - diffusion * Z + log_jumps
    log_paths_anti = np.concatenate([
        np.full((n_paths // 2, 1), np.log(params.S0)),
        log_returns_anti
    ], axis=1)
    log_paths_anti = np.cumsum(log_paths_anti, axis=1)
    ST_anti = np.exp(log_paths_anti[:, -1])
    
    payoffs_anti = np.maximum(ST_anti - params.K, 0) if params.option_type == "call" \
                   else np.maximum(params.K - ST_anti, 0)
    
    # Combine payoffs
    payoffs = np.concatenate([payoffs_std, payoffs_anti])
    
    # Store sample paths if requested
    paths_sample = None
    if store_paths:
        sample_size = min(n_paths // 2, 250)
        # S0 Column
        s0_col = np.full((sample_size * 2, 1), params.S0)
        pstd = np.exp(log_paths[:sample_size])
        panti = np.exp(log_paths_anti[:sample_size])
        paths_sample = np.concatenate([pstd, panti], axis=0)
        
        # Ensure S0 is present
        if paths_sample.shape[1] == n_steps: # Missing S0 case
             paths_sample = np.column_stack([s0_col, paths_sample])
             
    return payoffs, paths_sample

class MonteCarloEngine:
    """
    High-performance Monte Carlo option pricing engine.
    
    Features:
    - Parallel processing using multiprocessing
    - Antithetic variates for variance reduction
    - Stochastic jumps (Merton Jump Diffusion)
    - Optional path storage for visualization
    - Configurable simulation parameters
    """
    
    def __init__(
        self,
        params: OptionParams,
        n_simulations: int = DEFAULT_SIMULATIONS,
        n_steps: int = 252,
        seed: int = 42,
        jump_params: Optional[Dict] = None
    ):
        self.params = params
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.base_seed = seed
        self.jump_params = jump_params or {'mu_j': 0, 'sigma_j': 0, 'lambda_j': 0}
        self._discount_factor = np.exp(-params.r * params.T)
    
    def price(
        self,
        antithetic: bool = True,
        store_paths: bool = False,
        parallel: bool = True
    ) -> PricingResult:
        """
        Calculate option price using Monte Carlo simulation.
        
        Args:
            antithetic: Use antithetic variates for variance reduction
            store_paths: Store price paths for visualization (limited to 1000 paths)
            parallel: Use multiprocessing for parallel execution
        
        Returns:
            PricingResult with price, error estimates, and optional paths
        """
        start_time = time.perf_counter()
        
        # Adjust for antithetic variates
        actual_sims = self.n_simulations // 2 if antithetic else self.n_simulations
        
        # Determine parallel workers
        max_workers = min(mp.cpu_count(), 8) if parallel else 1
        
        # Prepare parameters for workers
        params_dict = {
            'S0': self.params.S0,
            'K': self.params.K,
            'T': self.params.T,
            'r': self.params.r,
            'sigma': self.params.sigma,
            'q': self.params.q,
            'option_type': self.params.option_type
        }
        
        if parallel and max_workers > 1 and actual_sims > 10000:
            # Parallel execution
            paths_per_worker = actual_sims // max_workers
            tasks = [
                (params_dict, paths_per_worker, self.n_steps, 
                 self.base_seed + i * 10000, False, self.jump_params)
                for i in range(max_workers)
            ]
            
            all_payoffs = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_simulate_chunk_worker, task) 
                          for task in tasks]
                for future in as_completed(futures):
                    payoffs, _ = future.result()
                    all_payoffs.append(payoffs)
            
            payoffs = np.concatenate(all_payoffs)
            
            # If paths are requested, generate a small representative sample separately
            if store_paths:
                _, paths = _simulate_chunk_worker((params_dict, 500, self.n_steps, self.base_seed, True, self.jump_params))
            else:
                paths = None
        else:
            # Sequential execution
            task = (params_dict, actual_sims, self.n_steps, self.base_seed, store_paths, self.jump_params)
            payoffs, paths = _simulate_chunk_worker(task)
        
        # Calculate statistics
        discounted_payoffs = self._discount_factor * payoffs
        price = float(np.mean(discounted_payoffs))
        variance = float(np.var(discounted_payoffs, ddof=1))
        std_error = np.sqrt(variance / len(discounted_payoffs))
        
        computation_time = time.perf_counter() - start_time
        
        return PricingResult(
            price=price,
            std_error=float(std_error),
            confidence_lower=float(price - 1.96 * std_error),
            confidence_upper=float(price + 1.96 * std_error),
            paths=np.asarray(paths).tolist() if paths is not None else None,
            payoffs=payoffs.tolist(),
            computation_time=computation_time
        )
    
    def calculate_delta(self, h: float = 0.01) -> float:
        """Calculate Delta using central finite differences."""
        # Up shock
        params_up = OptionParams(
            S0=self.params.S0 * (1 + h), K=self.params.K, T=self.params.T,
            r=self.params.r, sigma=self.params.sigma, q=self.params.q,
            option_type=self.params.option_type
        )
        # Down shock
        params_down = OptionParams(
            S0=self.params.S0 * (1 - h), K=self.params.K, T=self.params.T,
            r=self.params.r, sigma=self.params.sigma, q=self.params.q,
            option_type=self.params.option_type
        )
        
        engine_up = MonteCarloEngine(params_up, self.n_simulations // 2, self.n_steps)
        engine_down = MonteCarloEngine(params_down, self.n_simulations // 2, self.n_steps)
        
        price_up = engine_up.price(parallel=False).price
        price_down = engine_down.price(parallel=False).price
        
        return (price_up - price_down) / (2 * h * self.params.S0)
    
    def calculate_all_greeks(self) -> Greeks:
        """
        Calculate all option Greeks using finite differences.
        
        Returns:
            Greeks object with Delta, Gamma, Vega, Theta, and Rho
        """
        h = 0.01  # Finite difference step
        base_price = self.price(parallel=False).price
        
        # Delta
        delta = self.calculate_delta(h)
        
        # Gamma (second derivative)
        params_up = OptionParams(
            S0=self.params.S0 * (1 + h), K=self.params.K, T=self.params.T,
            r=self.params.r, sigma=self.params.sigma, q=self.params.q,
            option_type=self.params.option_type
        )
        params_down = OptionParams(
            S0=self.params.S0 * (1 - h), K=self.params.K, T=self.params.T,
            r=self.params.r, sigma=self.params.sigma, q=self.params.q,
            option_type=self.params.option_type
        )
        
        engine_up = MonteCarloEngine(params_up, self.n_simulations // 4, self.n_steps)
        engine_down = MonteCarloEngine(params_down, self.n_simulations // 4, self.n_steps)
        
        price_up = engine_up.price(parallel=False).price
        price_down = engine_down.price(parallel=False).price
        
        gamma = (price_up - 2 * base_price + price_down) / \
                (h * h * self.params.S0 * self.params.S0)
        
        # Vega (volatility sensitivity)
        params_vega_up = OptionParams(
            S0=self.params.S0, K=self.params.K, T=self.params.T,
            r=self.params.r, sigma=self.params.sigma + h, q=self.params.q,
            option_type=self.params.option_type
        )
        params_vega_down = OptionParams(
            S0=self.params.S0, K=self.params.K, T=self.params.T,
            r=self.params.r, sigma=self.params.sigma - h, q=self.params.q,
            option_type=self.params.option_type
        )
        
        engine_vega_up = MonteCarloEngine(params_vega_up, self.n_simulations // 4, self.n_steps)
        engine_vega_down = MonteCarloEngine(params_vega_down, self.n_simulations // 4, self.n_steps)
        
        price_vega_up = engine_vega_up.price(parallel=False).price
        price_vega_down = engine_vega_down.price(parallel=False).price
        
        vega = (price_vega_up - price_vega_down) / (2 * h) / 100.0  # Per 1% vol
        
        # Theta (time decay)
        if self.params.T > h:
            params_theta = OptionParams(
                S0=self.params.S0, K=self.params.K, T=self.params.T - h,
                r=self.params.r, sigma=self.params.sigma, q=self.params.q,
                option_type=self.params.option_type
            )
            engine_theta = MonteCarloEngine(params_theta, self.n_simulations // 4, self.n_steps)
            price_theta = engine_theta.price(parallel=False).price
            theta = (price_theta - base_price) / h / 365.0  # Per day
        else:
            theta = 0.0
        
        # Rho (rate sensitivity)
        params_rho_up = OptionParams(
            S0=self.params.S0, K=self.params.K, T=self.params.T,
            r=self.params.r + h, sigma=self.params.sigma, q=self.params.q,
            option_type=self.params.option_type
        )
        params_rho_down = OptionParams(
            S0=self.params.S0, K=self.params.K, T=self.params.T,
            r=self.params.r - h, sigma=self.params.sigma, q=self.params.q,
            option_type=self.params.option_type
        )
        
        engine_rho_up = MonteCarloEngine(params_rho_up, self.n_simulations // 4, self.n_steps)
        engine_rho_down = MonteCarloEngine(params_rho_down, self.n_simulations // 4, self.n_steps)
        
        price_rho_up = engine_rho_up.price(parallel=False).price
        price_rho_down = engine_rho_down.price(parallel=False).price
        
        rho = (price_rho_up - price_rho_down) / (2 * h) / 100.0  # Per 1% rate
        
        return Greeks(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho)
        )

# ============================================================================
# BLACK-SCHOLES ANALYTICAL MODEL
# ============================================================================

class BlackScholes:
    """
    Analytical Black-Scholes option pricing model.
    Used as benchmark for Monte Carlo validation.
    """
    
    @staticmethod
    def price(params: OptionParams) -> float:
        """
        Calculate Black-Scholes option price.
        
        Formula:
        C = S₀e^(-qT)N(d₁) - Ke^(-rT)N(d₂)  [Call]
        P = Ke^(-rT)N(-d₂) - S₀e^(-qT)N(-d₁) [Put]
        
        where:
        d₁ = [ln(S₀/K) + (r - q + σ²/2)T] / (σ√T)
        d₂ = d₁ - σ√T
        """
        if params.T <= 0:
            # Expired option
            if params.option_type == "call":
                return max(params.S0 - params.K, 0.0)
            else:
                return max(params.K - params.S0, 0.0)
        
        # Calculate d1 and d2
        d1 = (np.log(params.S0 / params.K) + 
              (params.r - params.q + 0.5 * params.sigma ** 2) * params.T) / \
             (params.sigma * np.sqrt(params.T))
        d2 = d1 - params.sigma * np.sqrt(params.T)
        
        if params.option_type == "call":
            price = (params.S0 * np.exp(-params.q * params.T) * norm.cdf(d1) -
                    params.K * np.exp(-params.r * params.T) * norm.cdf(d2))
        else:
            price = (params.K * np.exp(-params.r * params.T) * norm.cdf(-d2) -
                    params.S0 * np.exp(-params.q * params.T) * norm.cdf(-d1))
        
        return float(price)
    
    @staticmethod
    def greeks(params: OptionParams) -> Greeks:
        """Calculate analytical Greeks using closed-form formulas."""
        if params.T <= 0:
            return Greeks()
        
        d1 = (np.log(params.S0 / params.K) + 
              (params.r - params.q + 0.5 * params.sigma ** 2) * params.T) / \
             (params.sigma * np.sqrt(params.T))
        d2 = d1 - params.sigma * np.sqrt(params.T)
        
        nd1 = norm.pdf(d1)  # N'(d1)
        
        if params.option_type == "call":
            delta = np.exp(-params.q * params.T) * norm.cdf(d1)
            theta = (-params.S0 * nd1 * params.sigma * np.exp(-params.q * params.T) / 
                    (2 * np.sqrt(params.T))
                    - params.r * params.K * np.exp(-params.r * params.T) * norm.cdf(d2)
                    + params.q * params.S0 * np.exp(-params.q * params.T) * norm.cdf(d1)) / 365.0
        else:
            delta = -np.exp(-params.q * params.T) * norm.cdf(-d1)
            theta = (-params.S0 * nd1 * params.sigma * np.exp(-params.q * params.T) / 
                    (2 * np.sqrt(params.T))
                    + params.r * params.K * np.exp(-params.r * params.T) * norm.cdf(-d2)
                    - params.q * params.S0 * np.exp(-params.q * params.T) * norm.cdf(-d1)) / 365.0
        
        # Gamma (same for calls and puts)
        gamma = np.exp(-params.q * params.T) * nd1 / \
                (params.S0 * params.sigma * np.sqrt(params.T))
        
        # Vega (same for calls and puts, per 1% change)
        vega = params.S0 * np.exp(-params.q * params.T) * nd1 * \
               np.sqrt(params.T) / 100.0
        
        # Rho (per 1% change)
        if params.option_type == "call":
            rho = params.K * params.T * np.exp(-params.r * params.T) * norm.cdf(d2) / 100.0
        else:
            rho = -params.K * params.T * np.exp(-params.r * params.T) * norm.cdf(-d2) / 100.0
        
        return Greeks(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho)
        )

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_price_paths_plot(
    paths: List[List[float]],
    params: OptionParams
) -> go.Figure:
    """Create interactive Plotly chart of simulated price paths."""
    fig = go.Figure()
    
    paths_array = np.array(paths)
    time_points = np.arange(paths_array.shape[1])
    
    # Plot sample of individual paths
    n_display = min(50, len(paths))
    for i in range(n_display):
        fig.add_trace(go.Scatter(
            x=time_points,
            y=paths[i],
            mode='lines',
            line={'width': 0.5, 'color': 'rgba(100, 149, 237, 0.3)'},
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Plot mean path
    mean_path = np.mean(paths_array, axis=0)
    fig.add_trace(go.Scatter(
        x=time_points,
        y=mean_path,
        mode='lines',
        name='Mean Path',
        line={'width': 3, 'color': 'red', 'dash': 'dash'}
    ))
    
    # Strike price line
    fig.add_hline(
        y=params.K,
        line_dash="dot",
        line_color="green",
        line_width=2,
        annotation_text=f"Strike: ${params.K:.2f}",
        annotation_position="top right"
    )
    
    # Spot price line
    fig.add_hline(
        y=params.S0,
        line_dash="dashdot",
        line_color="purple",
        line_width=1.5,
        annotation_text=f"Spot: ${params.S0:.2f}",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        title=f"GBM Price Paths ({len(paths):,} simulations)",
        xaxis_title="Time Steps",
        yaxis_title="Stock Price ($)",
        template="plotly_dark",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_distribution_plot(
    payoffs: List[float],
    mc_price: float,
    bs_price: float,
    params: OptionParams
) -> go.Figure:
    """Create histogram of option payoffs at maturity."""
    fig = go.Figure()
    
    # Filter positive payoffs
    positive_payoffs = [p for p in payoffs if p > 0]
    
    if positive_payoffs:
        fig.add_trace(go.Histogram(
            x=positive_payoffs,
            nbinsx=50,
            name='Payoff Distribution',
            marker_color='steelblue',
            opacity=0.7,
            marker_line_color='black',
            marker_line_width=1
        ))
    
    # Monte Carlo price line
    fig.add_vline(
        x=mc_price,
        line_dash="solid",
        line_color="red",
        line_width=3,
        annotation_text=f"MC: ${mc_price:.4f}",
        annotation_position="top"
    )
    
    # Black-Scholes price line
    fig.add_vline(
        x=bs_price,
        line_dash="dash",
        line_color="orange",
        line_width=3,
        annotation_text=f"BS: ${bs_price:.4f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=f"Option Payoff Distribution ({params.option_type.upper()})",
        xaxis_title="Payoff ($)",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=500,
        bargap=0.1,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_convergence_plot(
    payoffs: List[float],
    params: OptionParams,
    bs_price: float
) -> go.Figure:
    """Create convergence analysis chart showing MC estimate stability."""
    fig = go.Figure()
    
    discount = np.exp(-params.r * params.T)
    discounted = np.array([discount * p for p in payoffs])
    
    # Calculate running statistics at logarithmic intervals
    sample_points = []
    means = []
    upper_bounds = []
    lower_bounds = []
    
    n = len(discounted)
    i = 100
    while i <= n:
        sample = discounted[:i]
        mean = np.mean(sample)
        std_err = np.std(sample, ddof=1) / np.sqrt(i)
        
        sample_points.append(i)
        means.append(mean)
        upper_bounds.append(mean + 1.96 * std_err)
        lower_bounds.append(mean - 1.96 * std_err)
        
        i = int(i * 1.2)
    
    # Confidence interval band
    fig.add_trace(go.Scatter(
        x=sample_points + sample_points[::-1],
        y=upper_bounds + lower_bounds[::-1],
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line={'color': 'rgba(0,0,0,0)'},
        name='95% Confidence Interval',
        showlegend=True
    ))
    
    # Running mean
    fig.add_trace(go.Scatter(
        x=sample_points,
        y=means,
        mode='lines',
        name='MC Estimate',
        line=dict(color='blue', width=2.5)
    ))
    
    # Black-Scholes benchmark
    fig.add_hline(
        y=bs_price,
        line_dash="dash",
        line_color="red",
        line_width=2.5,
        annotation_text=f"Black-Scholes: ${bs_price:.4f}",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        title="Monte Carlo Convergence Analysis",
        xaxis_title="Number of Simulations (log scale)",
        yaxis_title="Option Price Estimate ($)",
        xaxis_type="log",
        template="plotly_dark",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_greeks_comparison_plot(
    mc_greeks: Greeks,
    bs_greeks: Greeks
) -> go.Figure:
    """Create grouped bar chart comparing MC vs BS Greeks."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Monte Carlo Greeks', 'Black-Scholes Greeks'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    greek_names = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
    mc_values = [mc_greeks.delta, mc_greeks.gamma, mc_greeks.vega, 
                mc_greeks.theta, mc_greeks.rho]
    bs_values = [bs_greeks.delta, bs_greeks.gamma, bs_greeks.vega,
                bs_greeks.theta, bs_greeks.rho]
    
    colors_mc = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    colors_bs = ['#FF8E8E', '#7FDBDA', '#74C7EC', '#B8E6B8', '#FED470']
    
    # Monte Carlo bars
    fig.add_trace(
        go.Bar(
            x=greek_names,
            y=mc_values,
            marker_color=colors_mc,
            name='Monte Carlo',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Black-Scholes bars
    fig.add_trace(
        go.Bar(
            x=greek_names,
            y=bs_values,
            marker_color=colors_bs,
            name='Black-Scholes',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Greeks Comparison: Monte Carlo vs Black-Scholes",
        template="plotly_dark",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ============================================================================
# ADVANCED STRATEGIC VISUALIZATIONS
# ============================================================================

def create_var_es_plot(payoffs: List[float], confidence: float = 0.95) -> go.Figure:
    """Calculate and visualize Value at Risk (VaR) and Expected Shortfall (ES)."""
    sorted_payoffs = np.sort(payoffs)
    idx = int((1 - confidence) * len(sorted_payoffs))
    var = sorted_payoffs[idx]
    es = np.mean(sorted_payoffs[:idx])
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=payoffs, nbinsx=100, marker_color='#636EFA', opacity=0.7))
    fig.add_vline(x=var, line_dash="dash", line_color="red", line_width=2,
                 annotation_text=f"VaR: ${var:.2f}", annotation_position="top left")
    fig.add_vline(x=es, line_dash="dot", line_color="orange", line_width=2,
                 annotation_text=f"ES: ${es:.2f}", annotation_position="top")
    
    fig.update_layout(title="Risk Profile: VaR & Expected Shortfall", template="plotly_dark", height=400)
    return fig

def create_sensitivity_3d(params: OptionParams) -> go.Figure:
    """3D Surface plot of Price vs Spot and Volatility."""
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 20)
    vols = np.linspace(0.05, 0.8, 20)
    S_mesh, V_mesh = np.meshgrid(spots, vols)
    Z = np.zeros_like(S_mesh)
    for i in range(len(vols)):
        for j in range(len(spots)):
            temp_params = OptionParams(S0=spots[j], K=params.K, T=params.T, r=params.r, sigma=vols[i], q=params.q, option_type=params.option_type)
            Z[i, j] = BlackScholes.price(temp_params)
            
    fig = go.Figure(data=[go.Surface(z=Z, x=S_mesh, y=V_mesh, colorscale='Viridis')])
    fig.update_layout(title="Price Surface: Spot vs Volatility", scene=dict(
        xaxis_title='Spot Price ($)', yaxis_title='Volatility (σ)', zaxis_title='Option Price ($)'),
        template="plotly_dark", height=600)
    return fig

def create_greeks_sensitivity_plot(params: OptionParams) -> go.Figure:
    """Line charts of Greeks vs Spot Price."""
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 50)
    deltas, gammas, vegas = [], [], []
    for s in spots:
        p = OptionParams(S0=s, K=params.K, T=params.T, r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
        g = BlackScholes.greeks(p)
        deltas.append(g.delta)
        gammas.append(g.gamma)
        vegas.append(g.vega)
        
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Delta vs Spot", "Gamma vs Spot", "Vega vs Spot"))
    fig.add_trace(go.Scatter(x=spots, y=deltas, name='Delta'), row=1, col=1)
    fig.add_trace(go.Scatter(x=spots, y=gammas, name='Gamma'), row=1, col=2)
    fig.add_trace(go.Scatter(x=spots, y=vegas, name='Vega'), row=1, col=3)
    
    fig.update_layout(title="Greeks Sensitivity Profiles", template="plotly_dark", height=400, showlegend=False)
    return fig

def create_pnl_heatmap(params: OptionParams) -> go.Figure:
    """Heatmap showing PnL sensitivity to Spot and Volatility changes."""
    spots = np.linspace(params.S0 * 0.8, params.S0 * 1.2, 11)
    vols = np.linspace(params.sigma * 0.5, params.sigma * 1.5, 11)
    pnl = np.zeros((len(vols), len(spots)))
    base_price = BlackScholes.price(params)
    
    for i in range(len(vols)):
        for j in range(len(spots)):
            p = OptionParams(S0=spots[j], K=params.K, T=params.T, r=params.r, sigma=vols[i], q=params.q, option_type=params.option_type)
            pnl[i][j] = BlackScholes.price(p) - base_price
            
    fig = go.Figure(data=go.Heatmap(z=pnl, x=np.round(spots, 2), y=np.round(vols, 2), colorscale='RdBu', zmid=0))
    fig.update_layout(title="PnL Multi-Scenario Sensitivity Heatmap", xaxis_title="Spot Price ($)", yaxis_title="Volatility (%)",
                     template="plotly_dark", height=500)
    return fig

def create_log_return_dist(paths: List[List[float]]) -> go.Figure:
    """Histogram of simulated log-returns vs theoretical normal distribution."""
    if paths is None or len(paths) == 0:
        return go.Figure()
    final_prices = np.array([path[-1] for path in paths])
    start_price = paths[0][0]
    log_returns = np.log(final_prices / start_price)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=log_returns, nbinsx=50, histnorm='probability density', name='Simulated'))
    
    # Fit normal distribution
    mu, std = norm.fit(log_returns)
    x = np.linspace(min(log_returns), max(log_returns), 100)
    p = norm.pdf(x, mu, std)
    fig.add_trace(go.Scatter(x=x, y=p, name='Normal Fit', line=dict(color='red', width=2)))
    
    fig.update_layout(title="Log-Return Distribution Analysis", template="plotly_dark", height=400)
    return fig

def create_path_heatmap(paths: List[List[float]]) -> go.Figure:
    """2D Histogram/Heatmap of all simulated price paths."""
    if paths is None or len(paths) == 0:
        return go.Figure().update_layout(title="Path Density Heatmap (No Data)")
    paths_arr = np.array(paths)
    if len(paths_arr.shape) < 2:
        return go.Figure().update_layout(title="Path Density Heatmap (Invalid Shape)")
    n_sims, n_steps = paths_arr.shape
    time = np.repeat(np.arange(n_steps), n_sims)
    prices = paths_arr.T.flatten()
    
    fig = go.Figure(go.Histogram2dContour(x=time, y=prices, colorscale='Blues', reversescale=False))
    fig.update_layout(title="Price Path Density Heatmap", xaxis_title="Time Steps", yaxis_title="Price ($)",
                     template="plotly_dark", height=500)
    return fig

def create_efficiency_frontier(result: PricingResult, n_sims: int) -> go.Figure:
    """Visualize simulation efficiency (Error vs Time)."""
    # This is a representative plot
    sim_counts = [1000, 5000, 10000, 50000, 100000]
    errors = [result.std_error * (n_sims/s)**0.5 for s in sim_counts]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim_counts, y=errors, mode=LINES_MARKERS, name='Standard Error'))
    fig.update_layout(title="Simulation Scaling Efficiency", xaxis_title="Simulation Count", yaxis_title="Standard Error",
                     template="plotly_dark", height=400)
    return fig

def create_prob_cone(paths: List[List[float]], market_data: MarketData) -> go.Figure:
    """Projected price cone with confidence intervals."""
    if paths is None or len(paths) == 0:
        return go.Figure()
    paths_arr = np.array(paths)
    if len(paths_arr.shape) < 2:
        return go.Figure()
    time = np.arange(paths_arr.shape[1])
    p5 = np.percentile(paths_arr, 5, axis=0)
    p25 = np.percentile(paths_arr, 25, axis=0)
    p50 = np.percentile(paths_arr, 50, axis=0)
    p75 = np.percentile(paths_arr, 75, axis=0)
    p95 = np.percentile(paths_arr, 95, axis=0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=p95, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=p5, fill='tonexty', fillcolor='rgba(0,100,255,0.1)', name='90% Confidence'))
    fig.add_trace(go.Scatter(x=time, y=p75, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=p25, fill='tonexty', fillcolor='rgba(0,100,255,0.2)', name='50% Confidence'))
    fig.add_trace(go.Scatter(x=time, y=p50, line=dict(color='white', dash='dash'), name='Median Projection'))
    
    fig.update_layout(title="Statistical Price Projection Cone", template="plotly_dark", height=500)
    return fig

def create_theta_decay(params: OptionParams) -> go.Figure:
    """Time decay curve as maturity approaches."""
    times = np.linspace(0.01, params.T, 50)
    prices = []
    for t in times:
        p = OptionParams(S0=params.S0, K=params.K, T=t, r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
        prices.append(BlackScholes.price(p))
    
    fig = go.Figure(go.Scatter(x=times[::-1], y=prices, name='Option Value'))
    fig.update_layout(title="Time Decay (Theta) Profile", xaxis_title="Days to Maturity", yaxis_title="Price ($)",
                     template="plotly_dark", height=400)
    return fig

def create_autocorrelation(paths: List[List[float]]) -> go.Figure:
    """Correlation of returns over different lags."""
    if paths is None or len(paths) == 0:
        return go.Figure()
    final_prices = np.array([path[-1] for path in paths])
    returns = np.diff(np.log(final_prices))
    if len(returns) < 5:
        return go.Figure()
    lags = range(1, min(21, len(returns)))
    corrs = [pd.Series(returns).autocorr(lag=l) for l in lags]
    
    fig = go.Figure(go.Bar(x=list(lags), y=corrs))
    fig.update_layout(title="Return Autocorrelation Analysis", template="plotly_dark", height=350)
    return fig

def create_payoff_diagram(params: OptionParams) -> go.Figure:
    """Standard analytical payoff diagram."""
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 100)
    payoffs = np.maximum(spots - params.K, 0) if params.option_type == "call" else np.maximum(params.K - spots, 0)
    profit = payoffs - BlackScholes.price(params)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=profit, name='Profit/Loss', fill='tozeroy'))
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    fig.update_layout(title=f"{params.option_type.capitalize()} Payoff Diagram at Expiry", template="plotly_dark", height=400)
    return fig

def create_violin_dist(payoffs: List[float]) -> go.Figure:
    """Violin plot showing the density and range of payoffs."""
    fig = go.Figure(go.Violin(y=payoffs, box_visible=True, meanline_visible=True, fillcolor='lightseagreen', opacity=0.6))
    fig.update_layout(title="Payoff Range & Density (Violin)", template="plotly_dark", height=400)
    return fig

def create_3d_time_surface(params: OptionParams) -> go.Figure:
    """3D surface of Price vs Spot and Time to Maturity."""
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 20)
    times = np.linspace(0.01, params.T, 20)
    S_mesh, T_mesh = np.meshgrid(spots, times)
    Z = np.zeros_like(S_mesh)
    for i in range(len(times)):
        for j in range(len(spots)):
            p = OptionParams(S0=spots[j], K=params.K, T=times[i], r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
            Z[i, j] = BlackScholes.price(p)
    
    fig = go.Figure(data=[go.Surface(z=Z, x=S_mesh, y=T_mesh, colorscale='Plasma')])
    fig.update_layout(title="Price Evolution: Spot vs Time", scene=dict(
        xaxis_title='Spot Price', yaxis_title='Time (Y)', zaxis_title='Price'),
        template="plotly_dark", height=600)
    return fig

def create_error_dist(mc_payoffs: List[float], params: OptionParams) -> go.Figure:
    """Distribution of pricing error relative to individual path expected values."""
    discount = np.exp(-params.r * params.T)
    errors = [(discount * p - BlackScholes.price(params)) for p in mc_payoffs]
    fig = go.Figure(go.Histogram(x=errors, nbinsx=100, marker_color='salmon'))
    fig.update_layout(title="Relative Path Pricing Error Distribution", template="plotly_dark", height=400)
    return fig

def create_sensitivity_table(params: OptionParams) -> pd.DataFrame:
    """Generated table for what-if scenarios."""
    scenarios = []
    for s_move in [-0.1, -0.05, 0, 0.05, 0.1]:
        for v_move in [-0.05, 0, 0.05]:
            p = OptionParams(S0=params.S0*(1+s_move), K=params.K, T=params.T, r=params.r, sigma=params.sigma*(1+v_move), q=params.q, option_type=params.option_type)
            scenarios.append({
                'Spot Change': f"{s_move*100:+.0f}%",
                'Vol Change': f"{v_move*100:+.0f}%",
                'New Price': f"${BlackScholes.price(p):.2f}",
                'Price Change': f"{(BlackScholes.price(p)/BlackScholes.price(params)-1)*100:+.2f}%"
            })
    return pd.DataFrame(scenarios)

def create_rho_profile(params: OptionParams) -> go.Figure:
    """Sensitivity to interest rates."""
    rates = np.linspace(0.0, 0.15, 50)
    prices = [BlackScholes.price(OptionParams(S0=params.S0, K=params.K, T=params.T, r=r, sigma=params.sigma, q=params.q, option_type=params.option_type)) for r in rates]
    fig = go.Figure(go.Scatter(x=rates*100, y=prices))
    fig.update_layout(title="Rho Profile: Price vs Interest Rate", xaxis_title="Rate (%)", yaxis_title="Price ($)",
                     template="plotly_dark", height=400)
    return fig

def create_vol_cone(params: OptionParams) -> go.Figure:
    """Impact of Volatility on Payoff Distribution Range."""
    vols = [0.1, 0.2, 0.3, 0.4, 0.5]
    fig = go.Figure()
    for v in vols:
        p = OptionParams(S0=params.S0, K=params.K, T=params.T, r=params.r, sigma=v, q=params.q, option_type=params.option_type)
        spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 100)
        pdf = norm.pdf(np.log(spots/params.S0), (params.r-0.5*v**2)*params.T, v*np.sqrt(params.T))
        fig.add_trace(go.Scatter(x=spots, y=pdf, name=f'Vol {v*100}%'))
    fig.update_layout(title="Implied Probability Density vs Volatility", template="plotly_dark", height=400)
    return fig

def create_parallel_params(params: OptionParams) -> go.Figure:
    """Parallel coordinates for exploring parameter space."""
    data = []
    for _ in range(20):
        s = params.S0 * np.random.uniform(0.8, 1.2)
        k = params.K * np.random.uniform(0.9, 1.1)
        v = params.sigma * np.random.uniform(0.5, 1.5)
        p = BlackScholes.price(OptionParams(S0=s, K=k, T=params.T, r=params.r, sigma=v, q=params.q, option_type=params.option_type))
        data.append({'Spot': s, 'Strike': k, 'Vol': v, 'Price': p})
    df = pd.DataFrame(data)
    fig = go.Figure(data=go.Parcoords(line=dict(color=df['Price'], colorscale='Viridis'),
        dimensions=[dict(label='Spot', values=df['Spot']), dict(label='Strike', values=df['Strike']),
                    dict(label='Vol', values=df['Vol']), dict(label='Price', values=df['Price'])]))
    fig.update_layout(title="Multi-Dimensional Parameter Explorer", template="plotly_dark", height=400)
    return fig


def create_cum_payoff(payoffs: List[float]) -> go.Figure:
    """Cumulative distribution of payoffs."""
    sorted_p = np.sort(payoffs)
    y = np.arange(len(sorted_p)) / len(sorted_p)
    fig = go.Figure(go.Scatter(x=sorted_p, y=y, fill='tozeroy'))
    fig.update_layout(title="Cumulative Probability of Payoff", xaxis_title="Payoff ($)", yaxis_title="Probability",
                     template="plotly_dark", height=400)
    return fig

def create_delta_surface_3d(params: OptionParams) -> go.Figure:
    """3D surface of Delta vs Spot and Time."""
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 20)
    times = np.linspace(0.01, params.T, 20)
    S_mesh, T_mesh = np.meshgrid(spots, times)
    Z = np.zeros_like(S_mesh)
    for i in range(len(times)):
        for j in range(len(spots)):
            p = OptionParams(S0=spots[j], K=params.K, T=times[i], r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
            Z[i, j] = BlackScholes.greeks(p).delta
    fig = go.Figure(data=[go.Surface(z=Z, x=S_mesh, y=T_mesh, colorscale='Viridis')])
    fig.update_layout(title="Delta Surface: Spot vs Time", scene=dict(xaxis_title='Spot', yaxis_title='Time', zaxis_title='Delta'), template="plotly_dark", height=600)
    return fig

def create_gamma_surface_3d(params: OptionParams) -> go.Figure:
    """3D surface of Gamma vs Spot and Time."""
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 20)
    times = np.linspace(0.01, params.T, 20)
    Z = np.zeros((len(times), len(spots)), dtype=np.float64)
    for i in range(len(times)):
        for j in range(len(spots)):
            p = OptionParams(S0=spots[j], K=params.K, T=times[i], r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
            Z[i, j] = BlackScholes.greeks(p).gamma
    fig = go.Figure(data=[go.Surface(z=Z, x=spots, y=times, colorscale='Cividis')])
    fig.update_layout(title="Gamma Surface: Spot vs Time", scene=dict(xaxis_title='Spot', yaxis_title='Time', zaxis_title='Gamma'), template="plotly_dark", height=600)
    return fig

def create_vega_surface_3d(params: OptionParams) -> go.Figure:
    """3D surface of Vega vs Spot and Time."""
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 20)
    times = np.linspace(0.01, params.T, 20)
    S_mesh, T_mesh = np.meshgrid(spots, times)
    Z = np.zeros_like(S_mesh)
    for i in range(len(times)):
        for j in range(len(spots)):
            p = OptionParams(S0=spots[j], K=params.K, T=times[i], r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
            Z[i, j] = BlackScholes.greeks(p).vega
    fig = go.Figure(data=[go.Surface(z=Z, x=S_mesh, y=T_mesh, colorscale='Plasma')])
    fig.update_layout(title="Vega Surface: Spot vs Time", scene=dict(xaxis_title='Spot', yaxis_title='Time', zaxis_title='Vega'), template="plotly_dark", height=600)
    return fig

def create_theta_surface_3d(params: OptionParams) -> go.Figure:
    """3D surface of Theta vs Spot and Time."""
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 20)
    times = np.linspace(0.01, params.T, 20)
    Z = np.zeros((len(times), len(spots)), dtype=np.float64)
    for i in range(len(times)):
        for j in range(len(spots)):
            p = OptionParams(S0=spots[j], K=params.K, T=times[i], r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
            Z[i, j] = BlackScholes.greeks(p).theta
    fig = go.Figure(data=[go.Surface(z=Z, x=spots, y=times, colorscale='Hot')])
    fig.update_layout(title="Theta Surface: Spot vs Time", scene=dict(xaxis_title='Spot', yaxis_title='Time', zaxis_title='Theta'), template="plotly_dark", height=600)
    return fig

def create_rho_surface_3d(params: OptionParams) -> go.Figure:
    """3D surface of Rho vs Spot and Time."""
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 20)
    times = np.linspace(0.01, params.T, 20)
    Z = np.zeros((len(times), len(spots)), dtype=np.float64)
    for i in range(len(times)):
        for j in range(len(spots)):
            p = OptionParams(S0=spots[j], K=params.K, T=times[i], r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
            Z[i, j] = BlackScholes.greeks(p).rho
    fig = go.Figure(data=[go.Surface(z=Z, x=spots, y=times, colorscale='Blues')])
    fig.update_layout(title="Rho Surface: Spot vs Time", scene=dict(xaxis_title='Spot', yaxis_title='Time', zaxis_title='Rho'), template="plotly_dark", height=600)
    return fig

def create_synthetic_candles(paths: List[List[float]]) -> go.Figure:
    """Create a candlestick chart from the mean and range of simulated paths."""
    paths_arr = np.array(paths)
    n_steps = paths_arr.shape[1]
    
    # We'll group steps into 'days' (e.g., 5 steps per candle)
    steps_per_candle = max(1, n_steps // 50)
    data = []
    for i in range(0, n_steps - steps_per_candle, steps_per_candle):
        chunk = paths_arr[:, i:i+steps_per_candle]
        data.append({
            'open': np.mean(chunk[:, 0]),
            'high': np.max(chunk),
            'low': np.min(chunk),
            'close': np.mean(chunk[:, -1]),
            'time': i
        })
    df = pd.DataFrame(data)
    fig = go.Figure(data=[go.Candlestick(x=df['time'].tolist(), open=df['open'].tolist(), high=df['high'].tolist(), low=df['low'].tolist(), close=df['close'].tolist())])
    fig.update_layout(title="Synthetic Price Action (Candlestick)", template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    return fig

def create_bollinger_bands(paths: List[List[float]]) -> go.Figure:
    """Moving average and Bollinger Bands for the mean simulated path."""
    paths_arr = np.array(paths)
    mean_path = np.mean(paths_arr, axis=0)
    window = 20
    sma = pd.Series(mean_path).rolling(window=window).mean()
    std = pd.Series(mean_path).rolling(window=window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    time_index = np.arange(len(mean_path))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index, y=mean_path, name='Mean Path', line=dict(color='white')))
    fig.add_trace(go.Scatter(x=time_index, y=upper, name='Upper Band', line=dict(color='rgba(255,255,255,0.2)')))
    fig.add_trace(go.Scatter(x=time_index, y=lower, name='Lower Band', fill='tonexty', line=dict(color='rgba(255,255,255,0.2)')))
    fig.update_layout(title="Mean Path Bollinger Bands (20-period)", template="plotly_dark", height=400)
    return fig

# --- NEW VISUALIZATIONS (20 ADDITIONAL) ---

def create_standard_error_decay_plot(payoffs: List[float], bs_price: float) -> go.Figure:
    """Visualization of Standard Error decay as a function of sample size (N)."""
    n_total = len(payoffs)
    sample_sizes = np.geomspace(100, n_total, 20, dtype=int)
    errors = []
    
    for n in sample_sizes:
        subset = payoffs[:n]
        std_dev = np.std(subset)
        std_err = std_dev / np.sqrt(n)
        errors.append(std_err)
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample_sizes, y=errors, mode='lines+markers', name='Observed Error', line=dict(color='orange')))
    # Theoretical 1/sqrt(N) line
    theoretical = errors[0] * np.sqrt(sample_sizes[0]) / np.sqrt(sample_sizes)
    fig.add_trace(go.Scatter(x=sample_sizes, y=theoretical, mode='lines', name='Theoretical 1/√N', line=dict(dash='dash', color='gray')))
    
    fig.update_layout(title="Monte Carlo Efficiency: Standard Error Decay", xaxis_title="Number of Simulations (N)", yaxis_title="Standard Error", xaxis_type="log", yaxis_type="log", template="plotly_dark", height=400)
    return fig

def create_price_velocity_acceleration(paths: List[List[float]]) -> go.Figure:
    """Visualize 2nd derivative of price paths (Acceleration)."""
    mean_path = np.mean(paths, axis=0)
    vel = np.diff(mean_path)
    accel = np.diff(vel)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=accel, name='Acceleration', line=dict(color='orange')))
    fig.add_trace(go.Scatter(y=vel, name='Velocity', line=dict(color='cyan', dash='dot')))
    fig.update_layout(title="Path Dynamics: Velocity vs Acceleration", template="plotly_dark", height=400)
    return fig

def create_merton_jump_intensity_viz(jump_params: Dict) -> go.Figure:
    """Theoretical jump distribution for Merton model."""
    mu, sigma = jump_params['mu_j'], jump_params['sigma_j']
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    fig = go.Figure(go.Scatter(x=x, y=y, fill='tozeroy', name='Jump Density'))
    fig.update_layout(title="Merton Jump Intensity Potential", xaxis_title="Log Jump Size", template="plotly_dark", height=400)
    return fig

def create_path_entropy_viz(paths: List[List[float]]) -> go.Figure:
    """Shannon entropy of the system over time."""
    paths_arr = np.array(paths)
    entropies = []
    for t in range(1, paths_arr.shape[1]):
        hist, _ = np.histogram(paths_arr[:, t], bins=50, density=True)
        hist = hist[hist > 0]
        entropies.append(-np.sum(hist * np.log(hist)))
    fig = go.Figure(go.Scatter(y=entropies, mode='lines+markers', name='System Entropy'))
    fig.update_layout(title="Dynamic System Entropy Evolution", template="plotly_dark", height=400)
    return fig

def create_vol_pulse_3d(params: OptionParams) -> go.Figure:
    """3D Vega pulse (Vega vs Spot vs Vol)."""
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 20)
    vols = np.linspace(0.05, 0.8, 20)
    z = np.zeros((len(vols), len(spots)), dtype=np.float64)
    for i in range(len(vols)):
        for j in range(len(spots)):
            p = OptionParams(S0=spots[j], K=params.K, T=params.T, r=params.r, sigma=vols[i], q=params.q, option_type=params.option_type)
            z[i,j] = BlackScholes.greeks(p).vega
    fig = go.Figure(data=[go.Surface(z=z, x=spots, y=vols, colorscale='Viridis')])
    fig.update_layout(title="Vega Pulse Surface", scene=dict(xaxis_title='Spot', yaxis_title='Vol', zaxis_title='Vega'), template="plotly_dark", height=600)
    return fig

def create_omega_leverage_surface(params: OptionParams) -> go.Figure:
    """Option leverage (Omega) surface."""
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 20)
    times = np.linspace(0.01, params.T, 20)
    z = np.zeros((len(times), len(spots)), dtype=np.float64)
    for i in range(len(times)):
        for j in range(len(spots)):
            p = OptionParams(S0=spots[j], K=params.K, T=times[i], r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
            price = max(0.01, BlackScholes.price(p))
            delta = BlackScholes.greeks(p).delta
            z[i,j] = abs(delta * spots[j] / price)
    fig = go.Figure(data=[go.Surface(z=z, x=spots, y=times, colorscale='Electric')])
    fig.update_layout(title="Omega (Leverage) Surface", scene=dict(xaxis_title='Spot', yaxis_title='Time', zaxis_title='Omega'), template="plotly_dark", height=600)
    return fig

def create_chaos_attractor_viz(paths: List[List[float]]) -> go.Figure:
    """Phase space attractor visualization."""
    path = paths[0]
    dt_path = np.diff(path)
    fig = go.Figure(go.Scatter(x=path[:-1], y=dt_path, mode='lines', line=dict(color='lime', width=0.8)))
    fig.update_layout(title="Chaos Theory: Phase Space Attractor", xaxis_title="Price", yaxis_title="ΔPrice", template="plotly_dark", height=400)
    return fig

def create_hurst_rolling_viz(paths: List[List[float]]) -> go.Figure:
    """Rolling Hurst Exponent of the mean path."""
    mean_path = np.mean(paths, axis=0)
    n = len(mean_path)
    lags = range(2, min(20, n//2))
    tau = [np.sqrt(np.std(np.subtract(mean_path[lag:], mean_path[:-lag]))) for lag in lags]
    if len(tau) < 2: return go.Figure().update_layout(title="Insufficient Data for Hurst")
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = m[0]*2
    fig = go.Figure(go.Indicator(mode="gauge+number", value=hurst, title={'text': "Hurst Exponent (Mean Path)"}, gauge={'axis': {'range': [0, 1]}}))
    fig.update_layout(template="plotly_dark", height=300)
    return fig

def create_path_corr_matrix_viz(paths: List[List[float]]) -> go.Figure:
    """Heatmap of correlations between leading paths."""
    df = pd.DataFrame(paths[:10]).T
    corr = df.corr()
    fig = go.Figure(go.Heatmap(z=corr, colorscale='RdBu', zmin=-1, zmax=1))
    fig.update_layout(title="Inter-Path Correlation Matrix", template="plotly_dark", height=400)
    return fig

def create_terminal_cdf_viz(payoffs: List[float]) -> go.Figure:
    """CDF of terminal payoffs."""
    sorted_p = np.sort(payoffs)
    cdf = np.arange(len(sorted_p)) / len(sorted_p)
    fig = go.Figure(go.Scatter(x=sorted_p, y=cdf, fill='tozeroy', name='CDF'))
    fig.update_layout(title="Terminal Payoff CDF", xaxis_title="Payoff", yaxis_title="Probability", template="plotly_dark", height=400)
    return fig

def create_drawdown_duration_dist_viz(paths: List[List[float]]) -> go.Figure:
    """Distribution of max drawdown durations."""
    durations = []
    for path in paths[:100]:
        path_arr = np.array(path)
        cummax = np.maximum.accumulate(path_arr)
        in_dd = path_arr < cummax
        streak = 0
        max_streak = 0
        for val in in_dd:
            if val: streak += 1
            else: streak = 0
            max_streak = max(max_streak, streak)
        durations.append(max_streak)
    fig = go.Figure(go.Histogram(x=durations, marker_color='red'))
    fig.update_layout(title="Max Drawdown Duration Distribution", xaxis_title="Steps", template="plotly_dark", height=400)
    return fig

def create_pop_curve_viz(params: OptionParams) -> go.Figure:
    """Probability of Profit vs Spot Price."""
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 50)
    pops = []
    for s in spots:
        d2 = (np.log(s/params.K) + (params.r - params.q - 0.5*params.sigma**2)*params.T) / (params.sigma*np.sqrt(params.T))
        pop = norm.cdf(d2) if params.option_type == 'call' else norm.cdf(-d2)
        pops.append(pop * 100)
    fig = go.Figure(go.Scatter(x=spots, y=pops, name='PoP %', line=dict(color='gold')))
    fig.update_layout(title="Probability of Profit vs Spot Price", xaxis_title="Spot Price", yaxis_title="PoP (%)", template="plotly_dark", height=400)
    return fig

def create_risk_neutral_skew_viz(params: OptionParams) -> go.Figure:
    """Inferred risk-neutral skewness preview."""
    sigma = params.sigma
    T = params.T
    skew = (np.exp(sigma**2 * T) + 2) * np.sqrt(np.exp(sigma**2 * T) - 1)
    fig = go.Figure(go.Indicator(mode="number", value=skew, title={'text': "Inferred RN Skewness"}))
    fig.update_layout(template="plotly_dark", height=250)
    return fig

def create_vol_smile_preview_viz(params: OptionParams) -> go.Figure:
    """Simulated Volatility Smile (Semi-Analytical)."""
    strikes = np.linspace(params.S0 * 0.7, params.S0 * 1.3, 20)
    vols = params.sigma + 0.1 * ((strikes - params.S0)/params.S0)**2
    fig = go.Figure(go.Scatter(x=strikes, y=vols, mode='lines+markers', line=dict(color='cyan')))
    fig.update_layout(title="Implied Volatility Smile Profile", xaxis_title="Strike", yaxis_title="Implied Vol", template="plotly_dark", height=400)
    return fig

def create_confidence_99_bands_viz(paths: List[List[float]]) -> go.Figure:
    """99% Confidence bands over time."""
    paths_arr = np.array(paths)
    p05 = np.percentile(paths_arr, 0.5, axis=0)
    p995 = np.percentile(paths_arr, 99.5, axis=0)
    time = np.arange(len(p05))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=p995, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=p05, fill='tonexty', fillcolor='rgba(255,0,0,0.1)', name='99% Confidence Interval'))
    fig.update_layout(title="Tail Risk: 99% Confidence Bands", template="plotly_dark", height=500)
    return fig

def create_kde_terminal_viz(payoffs: List[float]) -> go.Figure:
    """Kernel Density Estimation of terminal payoffs."""
    from scipy.stats import gaussian_kde
    valid_payoffs = [p for p in payoffs if p > 0]
    if not valid_payoffs: return go.Figure().update_layout(title="No ITM Payoffs for KDE")
    kde = gaussian_kde(valid_payoffs)
    x = np.linspace(min(valid_payoffs), max(valid_payoffs), 200)
    y = kde(x)
    fig = go.Figure(go.Scatter(x=x, y=y, fill='tozeroy', line=dict(color='lightgreen')))
    fig.update_layout(title="Smooth Kernel Density (ITM Payoffs)", template="plotly_dark", height=400)
    return fig

def create_wavelet_energy_viz(paths: List[List[float]]) -> go.Figure:
    """Simplified wavelet energy distribution of paths."""
    path = paths[0]
    scales = [1, 2, 4, 8]
    energies = [np.std(np.diff(path[::s])) for s in scales]
    fig = go.Figure(go.Bar(x=[f"Scale {s}" for s in scales], y=energies, marker_color='royalblue'))
    fig.update_layout(title="Multi-Scale Wavelet Energy Profile", template="plotly_dark", height=400)
    return fig

def create_tail_loss_butterfly_viz(payoffs: List[float]) -> go.Figure:
    """Deep OTM tail loss distribution."""
    tails = [p for p in payoffs if p < np.percentile(payoffs, 10)]
    fig = go.Figure(go.Violin(y=tails, box_visible=True, line_color='red', fillcolor='salmon'))
    fig.update_layout(title="Left Tail (Loss) Concentration", template="plotly_dark", height=400)
    return fig

def create_barrier_crossing_dist_viz(paths: List[List[float]], params: OptionParams) -> go.Figure:
    """Distribution of first time crossing a 10% barrier."""
    barrier = params.S0 * 1.1
    times = []
    for path in paths:
        path_arr = np.array(path)
        crossed = np.where(path_arr > barrier)[0]
        if len(crossed) > 0:
            times.append(crossed[0])
    if not times: return go.Figure().update_layout(title="No Barrier Crossings Detected")
    fig = go.Figure(go.Histogram(x=times, nbinsx=30, marker_color='orange'))
    fig.update_layout(title="Barrier Crossing Time Distribution (110% Spot)", xaxis_title="Steps", template="plotly_dark", height=400)
    return fig

def create_risk_reward_scatter_viz(payoffs: List[float]) -> go.Figure:
    """Expected Payoff vs VaR scatter across simulation chunks (simulated)."""
    chunks = np.array_split(payoffs, 20)
    data = []
    for chunk in chunks:
        data.append({'Exp': np.mean(chunk), 'VaR': np.percentile(chunk, 5)})
    df = pd.DataFrame(data)
    fig = px.scatter(df, x='VaR', y='Exp', title="Cluster Risk-Reward Profile", template="plotly_dark")
    fig.update_traces(marker=dict(size=12, color='gold', line=dict(width=2, color='white')))
    fig.update_layout(height=400)
    return fig

def create_greeks_radar_detail_viz(mc_greeks: Greeks) -> go.Figure:
    """Detailed Radar Chart of all Greeks."""
    categories = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
    values = [abs(mc_greeks.delta), abs(mc_greeks.gamma)*10, abs(mc_greeks.vega)/10, abs(mc_greeks.theta)/10, abs(mc_greeks.rho)/100]
    fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', line_color='cyan'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max(values)])), title="Greeks Absolute Radar Profile", template="plotly_dark", height=400)
    return fig
    """Moving average and Bollinger Bands for the mean simulated path."""
    paths_arr = np.array(paths)
    mean_path = np.mean(paths_arr, axis=0)
    window = 20
    sma = pd.Series(mean_path).rolling(window=window).mean()
    std = pd.Series(mean_path).rolling(window=window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    
    fig = go.Figure()
    time = np.arange(len(mean_path))
    fig.add_trace(go.Scatter(x=time, y=mean_path, name='Mean Path', line=dict(color='white', width=1)))
    fig.add_trace(go.Scatter(x=time, y=upper, name='Upper Band', line=dict(color='rgba(255,255,255,0.2)', width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=lower, name='Lower Band', line=dict(color='rgba(255,255,255,0.2)', width=0), fill='tonexty', fillcolor='rgba(100,100,255,0.1)', showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=sma, name='20-day SMA', line=dict(color='orange', width=2)))
    fig.update_layout(title="Bollinger Bands (Mean Path)", template="plotly_dark", height=400)
    return fig

def create_rsi_plot(paths: List[List[float]]) -> go.Figure:
    """Relative Strength Index (RSI) for the mean simulated path."""
    mean_path = np.mean(np.array(paths), axis=0)
    delta = pd.Series(mean_path).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(rsi)), y=rsi, name='RSI', line=dict(color='purple')))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title="Simulated RSI (14-period)", template="plotly_dark", height=300, yaxis_range=[0, 100])
    return fig

def create_max_drawdown_dist(paths: List[List[float]]) -> go.Figure:
    """Distribution of Maximum Drawdown across all simulated paths."""
    drawdowns = []
    for path in paths:
        path_arr = np.array(path)
        cummax = np.maximum.accumulate(path_arr)
        dd = (path_arr - cummax) / cummax
        drawdowns.append(np.min(dd) * 100) # Max drawdown as %
    
    fig = go.Figure(go.Histogram(x=drawdowns, nbinsx=50, marker_color='red', opacity=0.6))
    fig.update_layout(title="Maximum Drawdown Distribution (%)", template="plotly_dark", height=400)
    return fig

def create_var_evolution(paths: List[List[float]], confidence: float = 0.95) -> go.Figure:
    """Value at Risk (VaR) evolution over the simulation timeline."""
    paths_arr = np.array(paths)
    n_steps = paths_arr.shape[1]
    vars_evolution = []
    for t in range(1, n_steps):
        prices = paths_arr[:, t]
        vars_evolution.append(np.percentile(prices, (1 - confidence) * 100))
        
    fig = go.Figure(go.Scatter(x=np.arange(len(vars_evolution)), y=vars_evolution, name=f'{confidence*100}% VaR'))
    fig.update_layout(title="VaR Evolution Over Time", template="plotly_dark", height=400)
    return fig

def create_kelly_criterion_plot(params: OptionParams, bs_price: float) -> go.Figure:
    """Kelly Criterion for position sizing based on simulated edge."""
    # Simplified Kelly: f = p/a - q/b where p is win prob, a is gain at win, q is loss prob, b is loss at win
    # For options, we'll simulate various allocation fractions
    edge = (bs_price / params.S0) # Hypothetical edge
    fractions = np.linspace(0, 1, 50)
    growth = fractions * edge - 0.5 * (fractions**2) * (params.sigma**2)
    
    fig = go.Figure(go.Scatter(x=fractions, y=growth))
    fig.update_layout(title="GGrowth Curve (Kelly Criterion Estimate)", xaxis_title="Allocation Fraction", yaxis_title="Expected Growth", template="plotly_dark", height=400)
    return fig

def create_hurst_exponent_dist(paths: List[List[float]]) -> go.Figure:
    """Distribution of Hurst Exponents (measure of persistence) across paths."""
    def hurst(ts):
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    hurst_vals = [hurst(path) for path in paths[:100]] # Sample 100 paths
    fig = go.Figure(go.Histogram(x=hurst_vals, nbinsx=30, marker_color='gold'))
    fig.update_layout(title="Hurst Exponent Distribution (Sampled)", template="plotly_dark", height=400)
    return fig

def create_skew_kurtosis_evolution(paths: List[List[float]]) -> go.Figure:
    """Evolution of Skewness and Kurtosis of simulated prices over time."""
    from scipy.stats import skew, kurtosis
    paths_arr = np.array(paths)
    skews = [skew(paths_arr[:, t]) for t in range(5, paths_arr.shape[1], 5)]
    kurts = [kurtosis(paths_arr[:, t]) for t in range(5, paths_arr.shape[1], 5)]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Skewness Evolution", "Kurtosis Evolution"))
    fig.add_trace(go.Scatter(y=skews, name='Skew'), row=1, col=1)
    fig.add_trace(go.Scatter(y=kurts, name='Kurtosis'), row=1, col=2)
    fig.update_layout(title="Higher Moment Evolutions", template="plotly_dark", height=400, showlegend=False)
    return fig

def create_polar_greeks(mc_greeks: Greeks) -> go.Figure:
    """Polar plot for normalized Greek exposure."""
    names = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
    # Normalize for visual comparison
    values = [abs(mc_greeks.delta), abs(mc_greeks.gamma)*10, abs(mc_greeks.vega), abs(mc_greeks.theta), abs(mc_greeks.rho)]
    fig = go.Figure(go.Scatterpolar(r=values, theta=names, fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Normalized Greek Profile (Polar)", template="plotly_dark", height=400)
    return fig

def create_radar_sensitivity(params: OptionParams) -> go.Figure:
    """Radar chart of price sensitivity to various 5% shocks."""
    shocks = {}
    base = BlackScholes.price(params)
    shocks['Spot +5%'] = abs(BlackScholes.price(OptionParams(S0=params.S0*1.05, K=params.K, T=params.T, r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)) - base)
    shocks['Vol +5%'] = abs(BlackScholes.price(OptionParams(S0=params.S0, K=params.K, T=params.T, r=params.r, sigma=params.sigma+0.05, q=params.q, option_type=params.option_type)) - base)
    shocks['Rate +1%'] = abs(BlackScholes.price(OptionParams(S0=params.S0, K=params.K, T=params.T, r=params.r+0.01, sigma=params.sigma, q=params.q, option_type=params.option_type)) - base)
    shocks['Time -10%'] = abs(BlackScholes.price(OptionParams(S0=params.S0, K=params.K, T=params.T*0.9, r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)) - base)
    
    fig = go.Figure(go.Scatterpolar(r=list(shocks.values()), theta=list(shocks.keys()), fill='toself', marker_color='cyan'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Scenario Sensitivity Radar", template="plotly_dark", height=400)
    return fig

def create_strike_surface_3d(params: OptionParams) -> go.Figure:
    """3D surface of Price vs Spot and Strike."""
    spots = np.linspace(params.S0 * 0.7, params.S0 * 1.3, 20)
    strikes = np.linspace(params.K * 0.7, params.K * 1.3, 20)
    S_mesh, K_mesh = np.meshgrid(spots, strikes)
    Z = np.zeros_like(S_mesh)
    for i in range(len(strikes)):
        for j in range(len(spots)):
            p = OptionParams(S0=spots[j], K=strikes[i], T=params.T, r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
            Z[i, j] = BlackScholes.price(p)
    fig = go.Figure(data=[go.Surface(z=Z, x=S_mesh, y=K_mesh, colorscale='Viridis')])
    fig.update_layout(title="Price Surface: Spot vs Strike", scene=dict(xaxis_title='Spot', yaxis_title='Strike', zaxis_title='Price'), template="plotly_dark", height=600)
    return fig

def create_vol_surface_3d(params: OptionParams) -> go.Figure:
    """3D surface of Price vs Spot and Volatility."""
    spots = np.linspace(params.S0 * 0.7, params.S0 * 1.3, 20)
    vols = np.linspace(0.1, 0.8, 20)
    S_mesh, V_mesh = np.meshgrid(spots, vols)
    Z = np.zeros_like(S_mesh)
    for i in range(len(vols)):
        for j in range(len(spots)):
            p = OptionParams(S0=spots[j], K=params.K, T=params.T, r=params.r, sigma=vols[i], q=params.q, option_type=params.option_type)
            Z[i, j] = BlackScholes.price(p)
    fig = go.Figure(data=[go.Surface(z=Z, x=S_mesh, y=V_mesh, colorscale='Electric')])
    fig.update_layout(title="Price Surface: Spot vs Volatility", scene=dict(xaxis_title='Spot', yaxis_title='Volatility', zaxis_title='Price'), template="plotly_dark", height=600)
    return fig

def create_drift_diffusion_plot(params: OptionParams, path: List[float]) -> go.Figure:
    """Decomposition of a single path into drift and diffusion components."""
    dt = params.T / (len(path) - 1)
    drift_rate = (params.r - params.q - 0.5 * params.sigma ** 2)
    time = np.linspace(0, params.T, len(path))
    
    drift_comp = params.S0 * np.exp(drift_rate * time)
    diffusion_comp = np.array(path) - drift_comp
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=drift_comp, name='Drift Component', stackgroup='one'))
    fig.add_trace(go.Scatter(x=time, y=diffusion_comp, name='Diffusion Component', stackgroup='one'))
    fig.update_layout(title="Path Decomposition: Drift vs Diffusion", xaxis_title="Time", yaxis_title="Price Component", template="plotly_dark", height=400)
    return fig

def create_fft_analysis(paths: List[List[float]]) -> go.Figure:
    """Fourier Transform magnitude plot of simulated returns."""
    mean_path = np.mean(np.array(paths), axis=0)
    returns = np.diff(np.log(mean_path))
    fft_vals = np.abs(np.fft.fft(returns))
    freqs = np.fft.fftfreq(len(returns))
    
    # Only positive frequencies
    pos_idx = freqs > 0
    fig = go.Figure(go.Scatter(x=freqs[pos_idx], y=fft_vals[pos_idx], line=dict(color='lime')))
    fig.update_layout(title="FFT Magnitude: Return Frequency Analysis", xaxis_title="Frequency", yaxis_title="Magnitude", template="plotly_dark", height=400)
    return fig

def create_brownian_bridge(params: OptionParams) -> go.Figure:
    """Visualization of a Brownian Bridge (constrained start and end prices)."""
    n_steps = 100
    t = np.linspace(0, 1, n_steps)
    W = np.random.standard_normal(n_steps).cumsum() * np.sqrt(1/n_steps)
    B = W - t * W[-1] # Brownian Bridge from 0 to 0
    
    # Scale to price
    bridge_path = params.S0 + (params.K - params.S0) * t + params.sigma * B * params.S0
    
    fig = go.Figure(go.Scatter(x=t * params.T, y=bridge_path, name='Brownian Bridge', line=dict(color='cyan', dash='dot')))
    fig.update_layout(title="Brownian Bridge: Targeted Path Simulation", xaxis_title="Time", yaxis_title="Price", template="plotly_dark", height=400)
    return fig

def create_greek_corr_heatmap(params: OptionParams) -> go.Figure:
    """Heatmap showing correlation between Greek sensitivities across spot prices."""
    spots = np.linspace(params.S0 * 0.8, params.S0 * 1.2, 50)
    deltas, gammas, vegas, thetas = [], [], [], []
    for s in spots:
        p = OptionParams(S0=s, K=params.K, T=params.T, r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
        g = BlackScholes.greeks(p)
        deltas.append(g.delta); gammas.append(g.gamma); vegas.append(g.vega); thetas.append(g.theta)
    
    df = pd.DataFrame({'Delta': deltas, 'Gamma': gammas, 'Vega': vegas, 'Theta': thetas})
    corr = df.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', reversescale=True))
    fig.update_layout(title="Greek Sensitivity Correlation Matrix", template="plotly_dark", height=400)
    return fig

def create_day_of_week_returns(paths: List[List[float]]) -> go.Figure:
    """Synthetic boxplot of returns by 'day of week' across simulations."""
    returns = np.diff(np.log(np.array(paths)[:, :252]), axis=1).flatten()
    days = np.tile(np.arange(5), len(returns) // 5 + 1)[:len(returns)]
    df = pd.DataFrame({'Return': returns, 'Day': days})
    fig = go.Figure(go.Box(x=df['Day'], y=df['Return'], marker_color='lightblue'))
    fig.update_layout(title="Normalized Returns by Day of Week (Synthetic)", xaxis_title="Trading Day", yaxis_title="Log Return", template="plotly_dark", height=400)
    return fig

def create_pop_surface_3d(params: OptionParams) -> go.Figure:
    """3D surface of Probability of Profit (POP) vs Spot and Time."""
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 20)
    times = np.linspace(0.01, params.T, 20)
    S_mesh, T_mesh = np.meshgrid(spots, times)
    Z = np.zeros_like(S_mesh)
    for i in range(len(times)):
        for j in range(len(spots)):
            d2 = (np.log(spots[j]/params.K) + (params.r - params.q - 0.5*params.sigma**2)*times[i]) / (params.sigma*np.sqrt(times[i]))
            Z[i, j] = norm.cdf(d2) if params.option_type == "call" else norm.cdf(-d2)
            
    fig = go.Figure(data=[go.Surface(z=Z, x=S_mesh, y=T_mesh, colorscale='YlGn')])
    fig.update_layout(title="Probability of Profit: Spot vs Time", scene=dict(xaxis_title='Spot', yaxis_title='Time', zaxis_title='POP'), template="plotly_dark", height=600)
    return fig

def create_variance_contribution(mc_greeks: Greeks, params: OptionParams) -> go.Figure:
    """Bar chart showing contribution to total price variance from each Greek factor."""
    # Simplified contribution: Greek^2 * Vol^2 approx
    contributions = {
        'Delta (Spot)': (mc_greeks.delta * params.S0 * 0.05)**2,
        'Vega (Vol)': (mc_greeks.vega * 0.05)**2,
        'Gamma (Curvature)': (0.5 * mc_greeks.gamma * (params.S0 * 0.05)**2)**2,
        'Theta (Time)': (mc_greeks.theta * (1/252))**2
    }
    total = sum(contributions.values())
    fig = go.Figure(go.Bar(x=list(contributions.keys()), y=[v/total*100 for v in contributions.values()], marker_color='teal'))
    fig.update_layout(title="Greek-Based Variance Contribution (%)", yaxis_title="Percentage", template="plotly_dark", height=400)
    return fig

def create_interactive_greek_explorer(params: OptionParams) -> go.Figure:
    """Animated bubble chart showing Greek evolution across spot prices."""
    spots = np.linspace(params.S0 * 0.8, params.S0 * 1.2, 20)
    frames = []
    for s in spots:
        p = OptionParams(S0=s, K=params.K, T=params.T, r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
        g = BlackScholes.greeks(p)
        frames.append(go.Frame(data=[go.Scatter(x=[g.delta], y=[g.gamma], mode='markers', marker=dict(size=[g.vega*10], color='gold'))]))
        
    fig = go.Figure(data=[go.Scatter(x=[0], y=[0], mode='markers')], frames=frames)
    fig.update_layout(title="Dynamic Greek Explorer (Delta vs Gamma vs Vega)", xaxis=dict(range=[-1, 1], title="Delta"), yaxis=dict(range=[-0.1, 0.1], title="Gamma"), template="plotly_dark", height=400)
    return fig

def create_price_gradient_quiver(params: OptionParams) -> go.Figure:
    """Quiver plot showing the gradient of price with respect to Spot and Time."""
    spots = np.linspace(params.S0 * 0.8, params.S0 * 1.2, 10)
    times = np.linspace(0.1, params.T, 10)
    S, T = np.meshgrid(spots, times)
    U, V = np.zeros_like(S), np.zeros_like(T)
    for i in range(len(times)):
        for j in range(len(spots)):
            p = OptionParams(S0=spots[j], K=params.K, T=times[i], r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
            g = BlackScholes.greeks(p)
            U[i, j] = g.delta # dPrice/dSpot
            V[i, j] = g.theta # dPrice/dTime
            
    import plotly.figure_factory as ff
    fig = ff.create_quiver(S, T, U, V, scale=0.1, arrow_scale=0.3, name='Gradients', line=dict(color='pink'))
    fig.update_layout(title="Option Price Gradient Field (Delta vs Theta)", xaxis_title="Spot", yaxis_title="Time", template="plotly_dark", height=500)
    return fig

def create_risk_neutral_density(params: OptionParams) -> go.Figure:
    """Risk-neutral probability density function derived from Black-Scholes."""
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 100)
    d2 = (np.log(params.S0/spots) + (params.r - params.q - 0.5*params.sigma**2)*params.T) / (params.sigma*np.sqrt(params.T))
    density = np.exp(-0.5 * d2**2) / (spots * params.sigma * np.sqrt(2 * np.pi * params.T))
    
    fig = go.Figure(go.Scatter(x=spots, y=density, fill='tozeroy', line=dict(color='orange')))
    fig.update_layout(title="Risk-Neutral Probability Density", xaxis_title="Price at Expiry", yaxis_title="Density", template="plotly_dark", height=400)
    return fig

def create_chaos_phase_space(paths: List[List[float]]) -> go.Figure:
    """Phase space attractor visualization (S_t vs dS/dt)."""
    mean_path = np.mean(paths, axis=0)
    velocity = np.diff(mean_path)
    fig = go.Figure(go.Scatter(x=mean_path[:-1], y=velocity, mode='lines', line=dict(color='yellow', width=1)))
    fig.update_layout(title="Phase Space Attractor (Price vs Velocity)", xaxis_title="Price ($)", yaxis_title="Price Change (dS/dt)", template="plotly_dark", height=400)
    return fig

def create_kelly_growth_viz(params: OptionParams, mc_price: float) -> go.Figure:
    """Optimal allocation growth trajectory based on Kelly Criterion."""
    edge = (mc_price / params.S0) - 1
    odds = mc_price / params.K if params.K > 0 else 1
    f = max(0, edge / odds) # Simplified Kelly Fraction
    steps = np.arange(100)
    growth = (1 + f)**steps
    fig = go.Figure(go.Scatter(x=steps, y=growth, fill='tozeroy', name='Capital Growth', line=dict(color='lime')))
    fig.update_layout(title=f"Theoretical Growth via Kelly Fraction (f={f:.4f})", xaxis_title="Trades", yaxis_title="Compounded Capital", template="plotly_dark", height=400)
    return fig

def create_hurst_persistence_viz(paths: List[List[float]]) -> go.Figure:
    """Distribution of path persistence (Hurst Exponent)."""
    h_vals = []
    for p in paths[:50]:
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(p[lag:], p[:-lag]))) for lag in lags]
        h = np.polyfit(np.log(lags), np.log(tau), 1)[0]
        h_vals.append(h)
    fig = go.Figure(go.Histogram(x=h_vals, nbinsx=15, marker_color='magenta'))
    fig.add_vline(x=0.5, line_dash="dash", line_color="white", annotation_text="Random Walk (0.5)")
    fig.update_layout(title="Path Persistence Profile (Hurst Exponent Distribution)", xaxis_title="Hurst Exponent", template="plotly_dark", height=400)
    return fig

def create_vol_surface_skew_viz(params: OptionParams) -> go.Figure:
    """Visualization of the Volatility Smile/Skew impact on Price."""
    vols = np.linspace(0.1, 0.6, 20)
    prices = [BlackScholes.price(OptionParams(S0=params.S0, K=params.K, T=params.T, r=params.r, sigma=v, q=params.q, option_type=params.option_type)) for v in vols]
    fig = go.Figure(go.Scatter(x=vols, y=prices, mode='lines+markers', line=dict(color='cyan')))
    fig.update_layout(title="Volatility-Price Elasticity Curve", xaxis_title="Volatility (σ)", yaxis_title="Option Price ($)", template="plotly_dark", height=400)
    return fig

def create_payoff_probability_heatmap(payoffs: List[float]) -> go.Figure:
    """Heatmap showing probability density of specific payoff ranges."""
    counts, bins = np.histogram(payoffs, bins=20)
    probs = counts / len(payoffs)
    fig = go.Figure(go.Heatmap(z=[probs], x=bins[:-1], y=['Density'], colorscale='Viridis'))
    fig.update_layout(title="Payoff Density Concentration Heatmap", xaxis_title="Payoff ($)", template="plotly_dark", height=300)
    return fig

def create_elasticity_profile(params: OptionParams) -> go.Figure:
    """Option Leverage (Omega) evolution over spot prices."""
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 50)
    omegas = []
    for s in spots:
        p = OptionParams(S0=s, K=params.K, T=params.T, r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
        price = BlackScholes.price(p)
        delta = BlackScholes.greeks(p).delta
        omegas.append((delta * s) / max(price, 0.01))
    
    fig = go.Figure(go.Scatter(x=spots, y=omegas, line=dict(color='hotpink')))
    fig.update_layout(title="Option Leverage Profile (Omega)", xaxis_title="Spot Price", yaxis_title="Elasticity", template="plotly_dark", height=400)
    return fig

def create_theta_gamma_tradeoff(params: OptionParams) -> go.Figure:
    """Dynamic relationship between Theta (Cost of Time) and Gamma (Benefit of Move)."""
    spots = np.linspace(params.S0 * 0.8, params.S0 * 1.2, 50)
    thetas, gammas = [], []
    for s in spots:
        p = OptionParams(S0=s, K=params.K, T=params.T, r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
        g = BlackScholes.greeks(p)
        thetas.append(abs(g.theta)); gammas.append(g.gamma)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=thetas, name='Cost (Theta)', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=spots, y=gammas, name='Benefit (Gamma)', line=dict(color='green'), yaxis='y2'))
    fig.update_layout(title="Theta-Gamma Rent/Benefit Trade-off", template="plotly_dark", height=400,
                     yaxis=dict(title="Theta Magnitude"), yaxis2=dict(title="Gamma", overlaying='y', side='right'))
    return fig

def create_confidence_ellipse(paths: List[List[float]]) -> go.Figure:
    """Statistical confidence ellipse based on joint distribution of S_t and S_T."""
    paths_arr = np.array(paths)
    mid_idx = paths_arr.shape[1] // 2
    s_mid = paths_arr[:, mid_idx]
    s_final = paths_arr[:, -1]
    
    fig = go.Figure(go.Histogram2dContour(x=s_mid, y=s_final, colorscale='Blues', reversescale=True))
    fig.update_layout(title="Joint Price Distribution (T/2 vs T)", xaxis_title="Price at T/2", yaxis_title="Price at T", template="plotly_dark", height=500)
    return fig

def create_theta_vol_surface_3d(params: OptionParams) -> go.Figure:
    """3D surface of Theta vs Time and Volatility."""
    times = np.linspace(0.01, params.T, 20)
    vols = np.linspace(0.1, 0.8, 20)
    T_mesh, V_mesh = np.meshgrid(times, vols)
    Z = np.zeros_like(T_mesh)
    for i in range(len(vols)):
        for j in range(len(times)):
            p = OptionParams(S0=params.S0, K=params.K, T=times[j], r=params.r, sigma=vols[i], q=params.q, option_type=params.option_type)
            Z[i, j] = BlackScholes.greeks(p).theta
    fig = go.Figure(data=[go.Surface(z=Z, x=T_mesh, y=V_mesh, colorscale='Viridis')])
    fig.update_layout(title="Theta Surface: Time vs Volatility", scene=dict(xaxis_title='Time', yaxis_title='Volatility', zaxis_title='Theta'), template="plotly_dark", height=600)
    return fig

def create_path_extremes_3d(paths: List[List[float]]) -> go.Figure:
    """3D scatter of path Maximum, Minimum, and Final Prices."""
    maxs = [np.max(p) for p in paths[:500]]
    mins = [np.min(p) for p in paths[:500]]
    finals = [p[-1] for p in paths[:500]]
    fig = go.Figure(data=[go.Scatter3d(x=maxs, y=mins, z=finals, mode='markers', marker=dict(size=4, color=finals, colorscale='Viridis', opacity=0.8))])
    fig.update_layout(title="Path Extremes: Max vs Min vs Final", scene=dict(xaxis_title='Max', yaxis_title='Min', zaxis_title='Final'), template="plotly_dark", height=600)
    return fig

def create_rate_div_heatmap(params: OptionParams) -> go.Figure:
    """Heatmap of price sensitivity to Interest Rate and Dividend Yield."""
    rates = np.linspace(0, 0.1, 10)
    divs = np.linspace(0, 0.05, 10)
    R, D = np.meshgrid(rates, divs)
    Z = np.zeros_like(R)
    for i in range(len(divs)):
        for j in range(len(rates)):
            p = OptionParams(S0=params.S0, K=params.K, T=params.T, r=rates[j], sigma=params.sigma, q=divs[i], option_type=params.option_type)
            Z[i, j] = BlackScholes.price(p)
    fig = go.Figure(data=go.Heatmap(z=Z, x=rates*100, y=divs*100, colorscale='Viridis'))
    fig.update_layout(title="Sensitivity: Interest Rate vs Dividend Yield", xaxis_title="Rate (%)", yaxis_title="Div Yield (%)", template="plotly_dark", height=400)
    return fig

def create_prob_itm_evolution(paths: List[List[float]], params: OptionParams) -> go.Figure:
    """Stacked area chart of In-the-Money probability over time."""
    paths_arr = np.array(paths)
    probs = []
    for t in range(paths_arr.shape[1]):
        itm = np.sum(paths_arr[:, t] > params.K if params.option_type=="call" else paths_arr[:, t] < params.K)
        probs.append(itm / paths_arr.shape[0])
    fig = go.Figure(go.Scatter(y=probs, fill='tozeroy', name='Prob ITM', line=dict(color='cyan')))
    fig.update_layout(title="Probability ITM Evolution", xaxis_title="Time Steps", yaxis_title="Probability", template="plotly_dark", height=400)
    return fig

def create_fan_chart(paths: List[List[float]]) -> go.Figure:
    """Fan chart showing nested confidence bands for price projections."""
    paths_arr = np.array(paths)
    time = np.arange(paths_arr.shape[1])
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    fig = go.Figure()
    for p in percentiles:
        val = np.percentile(paths_arr, p, axis=0)
        fig.add_trace(go.Scatter(x=time, y=val, name=f'P{p}', line=dict(width=0.5), opacity=0.5))
    fig.update_layout(title="Price Projection Fan Chart", template="plotly_dark", height=500)
    return fig

def create_pnl_waterfall(params: OptionParams, mc_price: float, bs_price: float) -> go.Figure:
    """Waterfall chart showing price components and model differences."""
    intrinsic = max(params.S0 - params.K, 0) if params.option_type == "call" else max(params.K - params.S0, 0)
    time_value = bs_price - intrinsic
    mc_diff = mc_price - bs_price
    fig = go.Figure(go.Waterfall(name="PnL", orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Intrinsic", "Time Value", "MC Variance", "Final Price"],
        textposition="outside", text=[f"{intrinsic:.2f}", f"{time_value:.2f}", f"{mc_diff:.4f}", f"{mc_price:.2f}"],
        y=[intrinsic, time_value, mc_diff, mc_price],
        connector={"line":{"color":"rgb(63, 63, 63)"}}))
    fig.update_layout(title="Price Component Attribution", template="plotly_dark", height=400)
    return fig

def create_vega_surface_spot_time_3d(params: OptionParams) -> go.Figure:
    """3D surface of Vega vs Spot and Time."""
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 20)
    times = np.linspace(0.01, params.T, 20)
    S_mesh, T_mesh = np.meshgrid(spots, times)
    Z = np.zeros_like(S_mesh)
    for i in range(len(times)):
        for j in range(len(spots)):
            p = OptionParams(S0=spots[j], K=params.K, T=times[i], r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
            Z[i, j] = BlackScholes.greeks(p).vega
    fig = go.Figure(data=[go.Surface(z=Z, x=S_mesh, y=T_mesh, colorscale='Hot')])
    fig.update_layout(title="Vega Surface: Spot vs Time", scene=dict(xaxis_title='Spot', yaxis_title='Time', zaxis_title='Vega'), template="plotly_dark", height=600)
    return fig

def create_joint_greek_contour(params: OptionParams) -> go.Figure:
    """Joint distribution of Delta and Gamma across spot prices."""
    spots = np.linspace(params.S0 * 0.8, params.S0 * 1.2, 100)
    deltas, gammas = [], []
    for s in spots:
        p = OptionParams(S0=s, K=params.K, T=params.T, r=params.r, sigma=params.sigma, q=params.q, option_type=params.option_type)
        g = BlackScholes.greeks(p)
        deltas.append(g.delta); gammas.append(g.gamma)
    fig = go.Figure(go.Histogram2dContour(x=deltas, y=gammas, colorscale='Viridis'))
    fig.update_layout(title="Joint Sensitivity Contour: Delta vs Gamma", xaxis_title="Delta", yaxis_title="Gamma", template="plotly_dark", height=500)
    return fig

def create_discrete_convergence(payoffs: List[float], bs_price: float) -> go.Figure:
    """Step chart showing the convergence of MC price as simulations increase."""
    cum_mean = np.cumsum(payoffs) / np.arange(1, len(payoffs) + 1)
    # Sample every 100 points for performance
    indices = np.arange(0, len(cum_mean), 100)
    fig = go.Figure(go.Scatter(x=indices, y=cum_mean[indices], mode='lines', name='MC Price'))
    fig.add_hline(y=bs_price, line_dash="dash", line_color="red", name='BS Price')
    fig.update_layout(title="Discrete Pricing Convergence", xaxis_title="Number of Simulations", yaxis_title="Price Estimate", template="plotly_dark", height=400)
    return fig

def create_kelly_criterion_plot(params: OptionParams, bs_price: float) -> go.Figure:
    """Optimal allocation fraction estimate based on simulated edge."""
    edge = (bs_price / params.S0) # Simplified edge
    v = params.sigma
    # Kelly fraction: f = edge / variance
    kelly = edge / (v**2) if v > 0 else 0
    fig = go.Figure(go.Indicator(mode="gauge+number", value=kelly, title={'text': "Kelly Criterion Allocation"}, gauge={'axis': {'range': [0, 2]}, 'bar': {'color': "darkblue"}}))
    fig.update_layout(template="plotly_dark", height=300)
    return fig

def create_hurst_exponent_dist(paths: List[List[float]]) -> go.Figure:
    """Analysis of path persistence (Hurst Exponent distribution)."""
    # Hurst = 0.5 (Random), >0.5 (Trend), <0.5 (Mean-reverting)
    hurst_vals = []
    for p in paths[:100]:
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(p[lag:], p[:-lag]))) for lag in lags]
        m = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst_vals.append(m[0] * 2)
    fig = go.Figure(go.Histogram(x=hurst_vals, marker_color='gold'))
    fig.add_vline(x=0.5, line_dash="dash", line_color="white", annotation_text="Random Walk")
    fig.update_layout(title="Hurst Exponent Distribution", template="plotly_dark", height=400)
    return fig

def create_skew_kurtosis_evolution(paths: List[List[float]]) -> go.Figure:
    """Rolling Skewness and Kurtosis of the price distribution over time."""
    paths_arr = np.array(paths)
    skews, kurts = [], []
    for t in range(1, paths_arr.shape[1]):
        from scipy.stats import skew, kurtosis
        skews.append(skew(paths_arr[:, t]))
        kurts.append(kurtosis(paths_arr[:, t]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=skews, name='Skewness', line=dict(color='orange')))
    fig.add_trace(go.Scatter(y=kurts, name='Kurtosis', line=dict(color='cyan'), yaxis='y2'))
    fig.update_layout(title="Higher Moments Evolution (Skew & Kurt)", template="plotly_dark", height=400,
                     yaxis=dict(title="Skewness"), yaxis2=dict(title="Kurtosis", overlaying='y', side='right'))
    return fig

def create_qq_plot(paths: List[List[float]]) -> go.Figure:
    """Q-Q plot of terminal returns vs standard normal distribution."""
    finals = np.array([p[-1] for p in paths])
    returns = np.log(finals / paths[0][0])
    standardized = (returns - np.mean(returns)) / np.std(returns)
    theoretical = np.sort(np.random.normal(0, 1, len(standardized)))
    sample = np.sort(standardized)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical, y=sample, mode='markers', name='Return Quantiles'))
    fig.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode='lines', name='Normal Line', line=dict(dash='dash', color='red')))
    fig.update_layout(title="Return Q-Q Plot", xaxis_title="Theoretical Normal Quantiles", yaxis_title="Sample Quantiles", template="plotly_dark", height=400)
    return fig

def create_vol_clustering(paths: List[List[float]]) -> go.Figure:
    """Visualization of volatility clustering via absolute returns autocorrelation."""
    returns = np.diff(np.log(paths[0]))
    abs_returns = np.abs(returns)
    lags = np.arange(1, 21)
    acf = [pd.Series(abs_returns).autocorr(lag=l) for l in lags]
    fig = go.Figure(go.Bar(x=lags, y=acf))
    fig.update_layout(title="Volatility Clustering: Abs Return Autocorr", xaxis_title="Lag", yaxis_title="Correlation", template="plotly_dark", height=400)
    return fig

def create_boxplot_terminal(paths: List[List[float]]) -> go.Figure:
    """Boxplot of terminal price distribution."""
    finals = [p[-1] for p in paths]
    fig = go.Figure(go.Box(y=finals, name="Terminal Price", boxpoints='outliers'))
    fig.update_layout(title="Terminal Price Boxplot", yaxis_title="Price ($)", template="plotly_dark", height=400)
    return fig

def create_3d_density_surface(paths: List[List[float]], params: OptionParams) -> go.Figure:
    """3D surface of Probability Density evolution over Time and Spot."""
    paths_arr = np.array(paths)
    n_steps = paths_arr.shape[1]
    spots = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 30)
    times = np.arange(n_steps)
    Z = np.zeros((len(times), len(spots)))
    for i, t in enumerate(times):
        hist, _ = np.histogram(paths_arr[:, t], bins=spots, density=True)
        Z[i, :-1] = hist
    fig = go.Figure(data=[go.Surface(z=Z.T, x=times, y=spots)])
    fig.update_layout(title="PDF Evolution: Spot vs Time", scene=dict(xaxis_title='Time', yaxis_title='Spot', zaxis_title='Density'), template="plotly_dark", height=600)
    return fig

def create_regime_switching(params: OptionParams) -> go.Figure:
    """Simulation comparison between Low and High volatility regimes."""
    t_steps = 252
    dt = params.T / t_steps
    path_low = [params.S0]; path_high = [params.S0]
    curr_low = params.S0; curr_high = params.S0
    vol_high = params.sigma * 2
    for _ in range(t_steps):
        curr_low *= np.exp((params.r - 0.5 * params.sigma**2)*dt + params.sigma*np.sqrt(dt)*np.random.normal())
        curr_high *= np.exp((params.r - 0.5 * vol_high**2)*dt + vol_high*np.sqrt(dt)*np.random.normal())
        path_low.append(curr_low); path_high.append(curr_high)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=path_low, name='Low Vol Regime'))
    fig.add_trace(go.Scatter(y=path_high, name='High Vol Regime'))
    fig.update_layout(title="Regime-Switching Path Comparison", template="plotly_dark", height=400)
    return fig

def create_barrier_prob(paths: List[List[float]], params: OptionParams) -> go.Figure:
    """Probability of breaching an Up-and-Out barrier over time."""
    barrier = params.S0 * 1.25
    paths_arr = np.array(paths)
    breached = np.any(paths_arr > barrier, axis=1)
    prob_breach = np.mean(breached)
    # Evolution of breach prob
    evolution = [np.mean(np.any(paths_arr[:, :t] > barrier, axis=1)) for t in range(1, paths_arr.shape[1])]
    fig = go.Figure(go.Scatter(y=evolution, fill='tozeroy', name='Breach Prob'))
    fig.add_hline(y=prob_breach, line_dash="dash", line_color="red", annotation_text=f"Total: {prob_breach:.2%}")
    fig.update_layout(title=f"Barrier Breach Probability (Barrier @ ${barrier:.2f})", template="plotly_dark", height=400)
    return fig

def create_rolling_sharpe(paths: List[List[float]], params: OptionParams) -> go.Figure:
    """Rolling Sharpe Ratio evolution for the mean simulated path."""
    mean_path = np.mean(np.array(paths), axis=0)
    returns = np.diff(np.log(mean_path))
    window = 20
    sharpes = []
    for i in range(window, len(returns)):
        seg = returns[i-window:i]
        vol = np.std(seg) * np.sqrt(252)
        ret = np.mean(seg) * 252
        sharpes.append((ret - params.r) / vol if vol > 0 else 0)
    fig = go.Figure(go.Scatter(y=sharpes, name='Rolling Sharpe'))
    fig.update_layout(title="Rolling Sharpe Ratio Evolution (20d Window)", template="plotly_dark", height=400)
    return fig

def create_var_reduction_comp(params: OptionParams) -> go.Figure:
    """Comparison of standard Monte Carlo vs Antithetic Variates efficiency."""
    std_prices = []; anti_prices = []
    for _ in range(50):
        e = MonteCarloEngine(params, 1000, 100, seed=np.random.randint(0, 10000))
        std_prices.append(e.price(antithetic=False).price)
        anti_prices.append(e.price(antithetic=True).price)
    fig = go.Figure()
    fig.add_trace(go.Box(y=std_prices, name='Standard MC'))
    fig.add_trace(go.Box(y=anti_prices, name='Antithetic variates'))
    fig.update_layout(title="Variance Reduction Efficiency Comparison", template="plotly_dark", height=400)
    return fig

def create_risk_return_cloud(paths: List[List[float]]) -> go.Figure:
    """Scatter cloud of Path Return vs Path Volatility."""
    rets = []; vols = []
    for p in paths[:500]:
        r = np.log(p[-1]/p[0])
        v = np.std(np.diff(np.log(p))) * np.sqrt(252)
        rets.append(r); vols.append(v)
    fig = go.Figure(go.Scatter(x=vols, y=rets, mode='markers', marker=dict(color=rets, colorscale='RdYlGn', showscale=True)))
    fig.update_layout(title="Path Risk-Return Scatter Cloud", xaxis_title="Annualized Volatility", yaxis_title="Log Return", template="plotly_dark", height=500)
    return fig

def create_survival_curve(paths: List[List[float]], params: OptionParams) -> go.Figure:
    """Survival probability curve: Prob(Price > Strike) over time."""
    paths_arr = np.array(paths)
    survival = [np.mean(paths_arr[:, t] > params.K) for t in range(paths_arr.shape[1])]
    fig = go.Figure(go.Scatter(y=survival, line=dict(color='lightgreen', width=3)))
    fig.update_layout(title="Survival Probability Curve (Price > Strike)", xaxis_title="Time Steps", yaxis_title="Probability", template="plotly_dark", height=400)
    return fig

def create_confidence_ellipse(paths: List[List[float]]) -> go.Figure:
    """Confidence ellipse showing the joint distribution of prices at T/2 and T."""
    paths_arr = np.array(paths)
    mid_idx = paths_arr.shape[1] // 2
    x = paths_arr[:, mid_idx]
    y = paths_arr[:, -1]
    
    fig = go.Figure(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='rgba(100, 200, 255, 0.4)', size=4), name='Paths'))
    # Simplified ellipse
    fig.update_layout(title="Joint Distribution: Price @ T/2 vs T", xaxis_title="Price @ T/2", yaxis_title="Price @ T", template="plotly_dark", height=400)
    return fig

def create_radar_sensitivity(params: OptionParams) -> go.Figure:
    """Radar chart of relative Greek sensitivities."""
    g = BlackScholes.greeks(params)
    categories = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
    values = [abs(g.delta), abs(g.gamma)*10, abs(g.vega)/100, abs(g.theta)/365, abs(g.rho)]
    fig = go.Figure(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Relative Greek Profile'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max(values)*1.1])), title="Greek Sensitivity Radar Profile", template="plotly_dark", height=400)
    return fig

def create_sunburst_greeks(g: Greeks) -> go.Figure:
    """Sunburst chart of absolute Greek magnitudes."""
    labels = ["Greeks", "Delta", "Gamma", "Vega", "Theta", "Rho"]
    parents = ["", "Greeks", "Greeks", "Greeks", "Greeks", "Greeks"]
    values = [0.1, abs(g.delta), abs(g.gamma)*10, abs(g.vega)/100, abs(g.theta), abs(g.rho)]
    fig = go.Figure(go.Sunburst(labels=labels, parents=parents, values=values, branchvalues="total", marker=dict(colorscale='Viridis')))
    fig.update_layout(title="Greek Magnitude Sunburst", template="plotly_dark", height=400)
    return fig

def create_animated_hist(paths: List[List[float]]) -> go.Figure:
    """Animated histogram showing the distribution of prices evolving over time."""
    paths_arr = np.array(paths)
    step_indices = np.linspace(1, paths_arr.shape[1]-1, 5, dtype=int)
    fig = go.Figure()
    for i in step_indices:
        fig.add_trace(go.Histogram(x=paths_arr[:, i], name=f'T={i}', opacity=0.6))
    fig.update_layout(title="Price Distribution Evolution (Time-Layered)", barmode='overlay', template="plotly_dark", height=500)
    return fig
def create_path_clustering(paths: List[List[float]]) -> go.Figure:
    """Clustering analysis on simulated paths using segment means."""
    paths_arr = np.array(paths[:100])
    means = np.mean(paths_arr, axis=1)
    # Simple threshold clustering as proxy for K-means for performance
    low = paths_arr[means < np.percentile(means, 33)]
    mid = paths_arr[(means >= np.percentile(means, 33)) & (means < np.percentile(means, 66))]
    high = paths_arr[means >= np.percentile(means, 66)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.mean(low, axis=0), name='Low Cluster'))
    fig.add_trace(go.Scatter(y=np.mean(mid, axis=0), name='Mid Cluster'))
    fig.add_trace(go.Scatter(y=np.mean(high, axis=0), name='High Cluster', line=dict(width=3)))
    fig.update_layout(title="Path Clustering: Multi-regime Analysis", template="plotly_dark", height=400)
    return fig

def create_bootstrap_comp(payoffs: List[float]) -> go.Figure:
    """Comparison of Monte Carlo distribution vs Bootstrap resampling distribution."""
    bootstrap_means = [np.mean(np.random.choice(payoffs, len(payoffs), replace=True)) for _ in range(500)]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=payoffs, name='MC Sample', opacity=0.5))
    fig.add_trace(go.Histogram(x=bootstrap_means, name='Bootstrap Means', opacity=0.8))
    fig.update_layout(title="Monte Carlo Sample vs Bootstrap Distribution", template="plotly_dark", height=400, barmode='overlay')
    return fig

def create_growth_ladder(paths: List[List[float]]) -> go.Figure:
    """Ladder chart of capital growth percentiles over time."""
    paths_arr = np.array(paths)
    percentiles = [10, 25, 50, 75, 90]
    fig = go.Figure()
    for p in percentiles:
        fig.add_trace(go.Scatter(y=np.percentile(paths_arr, p, axis=0), name=f'{p}th Percentile'))
    fig.update_layout(title="Capital Growth Percentile Ladder", xaxis_title="Time Steps", yaxis_title="Price ($)", template="plotly_dark", height=500)
    return fig

def create_tree_branching(params: OptionParams) -> go.Figure:
    """Simplified visualization of Monte Carlo path branching (first 3 steps)."""
    fig = go.Figure()
    steps = 3
    nodes = [(0, params.S0)]
    for s in range(steps):
        next_nodes = []
        for time, price in nodes:
            if time == s:
                u = price * 1.05; d = price * 0.95
                fig.add_trace(go.Scatter(x=[s, s+1], y=[price, u], mode='lines+markers', showlegend=False, line=dict(color='gray')))
                fig.add_trace(go.Scatter(x=[s, s+1], y=[price, d], mode='lines+markers', showlegend=False, line=dict(color='gray')))
                next_nodes.append((s+1, u)); next_nodes.append((s+1, d))
        nodes.extend(next_nodes)
    fig.update_layout(title="Monte Carlo Branching Logic (First 3 Steps)", xaxis_title="Step", template="plotly_dark", height=400)
    return fig

def create_drawdown_timeseries(paths: List[List[float]]) -> go.Figure:
    """Time-series of drawdown percentages for the first 5 simulated paths."""
    fig = go.Figure()
    for i, p in enumerate(paths[:5]):
        p_arr = np.array(p)
        running_max = np.maximum.accumulate(p_arr)
        drawdown = (p_arr - running_max) / running_max
        fig.add_trace(go.Scatter(y=drawdown * 100, name=f'Path {i+1} Drawdown'))
    fig.update_layout(title="Path Maximum Drawdown Evolution (%)", xaxis_title="Time Steps", yaxis_title="Drawdown %", template="plotly_dark", height=400)
    return fig

def create_es_heatmap(payoffs: List[float]) -> go.Figure:
    """Heatmap of Expected Shortfall (CVaR) across different confidence levels."""
    levels = np.linspace(0.90, 0.99, 10)
    payoffs_arr = np.sort(np.array(payoffs))
    es_values = []
    for lvl in levels:
        cutoff = int((1 - lvl) * len(payoffs_arr))
        es_values.append(np.mean(payoffs_arr[:cutoff]) if cutoff > 0 else 0)
    fig = go.Figure(go.Scatter(x=levels * 100, y=es_values, fill='tozeroy', line=dict(color='orange')))
    fig.update_layout(title="Expected Shortfall (CVaR) Sensitivity", xaxis_title="Confidence Level (%)", yaxis_title="CVaR ($)", template="plotly_dark", height=400)
    return fig

def create_multi_asset_corr() -> go.Figure:
    """Synthetic multi-asset correlation heatmap for portfolio risk visualization."""
    assets = ['Asset A', 'Asset B', 'Asset C', 'Asset D', 'Asset E']
    corr_matrix = np.array([
        [1.0, 0.8, 0.4, 0.2, 0.1],
        [0.8, 1.0, 0.5, 0.3, 0.2],
        [0.4, 0.5, 1.0, 0.6, 0.4],
        [0.2, 0.3, 0.6, 1.0, 0.7],
        [0.1, 0.2, 0.4, 0.7, 1.0]
    ])
    fig = go.Figure(go.Heatmap(z=corr_matrix, x=assets, y=assets, colorscale='RdBu', zmin=-1, zmax=1))
    fig.update_layout(title="Synthetic Portfolio Asset Correlation", template="plotly_dark", height=400)
    return fig

# ============================================================================
# STREAMLIT UI
# ============================================================================

def render_sidebar() -> Tuple[OptionParams, str, int, MarketData]:
    """
    Render Streamlit sidebar with configuration controls.
    
    Returns:
        Tuple of (OptionParams, ticker, n_simulations, market_data)
    """
    st.sidebar.header("⚙️ Configuration")
    
    # API Configuration Section (Hardcoded)
    with st.sidebar.expander("🔑 API Status", expanded=False):
        st.success("✅ Alpha Vantage API: Hardcoded Active")
        st.info("Using Integrated Quantitative Intelligence Hub")
    
    # Market Data Section
    st.sidebar.subheader("📊 Market Data")
    
    popular_tickers = [
        # Tech & Growth
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ADBE", "CRM", 
        "AMD", "INTC", "CSCO", "ORCL", "SNOW", "PLTR", "QCOM", "AVGO", "TXN", "MU",
        # Finance & Payments
        "JPM", "V", "MA", "BAC", "PYPL", "GS", "MS", "AXP", "WFC", "C", "BLK", "SCHW",
        # Healthcare & Biotech
        "UNH", "PFE", "JNJ", "ABT", "ABBV", "MRK", "TMO", "LLY", "MDT", "CVS", "AMGN", "BMY",
        # Consumer & Retail
        "WMT", "HD", "PG", "DIS", "KO", "PEP", "COST", "NKE", "SBUX", "TGT", "LOW", "PM", "MO",
        # Industrials & Energy
        "XOM", "CVX", "CAT", "BA", "GE", "HON", "UPS", "FEDX", "DE", "RTX", "LMT", "MMM",
        # Communication & Utilities
        "T", "VZ", "TMUS", "CMCSA", "NEE", "DUK", "SO", "D", "EXC", "AEP",
        # ETFs & Indices
        "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "VXX", "EFA", "EEM", "TLT",
        # Additional Major Assets
        "BABA", "TSM", "ASML", "SAP", "TM", "HSBC", "RY", "TD", "LIN", "NVO",
        "SNY", "AZN", "BHP", "RIO", "BP", "SHEL", "UL", "DELL", "IBM", "UBER", 
        "ABNB", "COIN", "DKNG", "SHOP", "SQ", "U", "Z", "PYPL", "GME", "AMC"
    ]
    
    ticker_selection = st.sidebar.selectbox(
        "Stock Ticker Symbol",
        options=popular_tickers + ["Other (Type below)"],
        index=popular_tickers.index(DEFAULT_TICKER) if DEFAULT_TICKER in popular_tickers else 0
    )
    
    if ticker_selection == "Other (Type below)":
        ticker = st.sidebar.text_input(
            "Enter Symbol",
            value=DEFAULT_TICKER if DEFAULT_TICKER not in popular_tickers else "SPY",
            placeholder="e.g., BRK-B, BABA"
        ).upper().strip()
    else:
        ticker = ticker_selection
    
    # Fetch data button
    if st.sidebar.button("🔄 Fetch Live Data", use_container_width=True):
        with st.spinner(f"Fetching data for {ticker}..."):
            client = AlphaVantageClient()
            st.session_state.market_data = client.fetch_quote(ticker)
    
    # Initialize or get cached market data
    if 'market_data' not in st.session_state or st.session_state.market_data.symbol != ticker:
        client = AlphaVantageClient()
        st.session_state.market_data = client.fetch_quote(ticker)
    
    market_data = st.session_state.market_data
    
    # Display market data summary
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Price", f"${market_data.price:.2f}", f"{market_data.change:+.2f}")
    with col2:
        st.metric("Volume", f"{market_data.volume/1e6:.1f}M")
    
    if market_data.is_mock:
        st.sidebar.warning("⚠️ Using mock data (API limit or demo key)")
    
    # Option Parameters Section
    st.sidebar.subheader("🎯 Option Parameters")
    
    option_type = st.sidebar.selectbox(
        "Option Type",
        options=["call", "put"],
        index=0 if DEFAULT_OPTION_TYPE == "call" else 1
    )
    
    # Auto-calculate strike as 5% OTM by default
    default_strike = round(market_data.price * 1.05, 2)
    
    strike = st.sidebar.number_input(
        "Strike Price ($)",
        min_value=0.01,
        value=default_strike,
        step=1.0,
        format="%.2f"
    )
    
    maturity = st.sidebar.slider(
        "Time to Maturity (years)",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1
    )
    
    # Risk Parameters Section
    st.sidebar.subheader("📈 Risk Parameters")
    
    risk_free = st.sidebar.slider(
        "Risk-free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    ) / 100
    
    # Use historical vol or allow override
    hist_vol_pct = market_data.historical_volatility * 100
    
    volatility = st.sidebar.slider(
        "Volatility (%)",
        min_value=5.0,
        max_value=100.0,
        value=hist_vol_pct,
        step=1.0
    ) / 100
    
    dividend = st.sidebar.slider(
        "Dividend Yield (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.1
    ) / 100
    
    # Simulation Settings Section
    st.sidebar.subheader("🖥️ Simulation Settings")
    
    n_sims = st.sidebar.select_slider(
        "Number of Simulations",
        options=[1000, 5000, 10000, 50000, 100000, 500000],
        value=DEFAULT_SIMULATIONS
    )
    
    n_steps = st.sidebar.slider(
        "Time Steps",
        min_value=50,
        max_value=500,
        value=252,
        step=50,
        help="252 = trading days in a year"
    )
    
    # Create OptionParams object
    params = OptionParams(
        S0=market_data.price,
        K=strike,
        T=maturity,
        r=risk_free,
        sigma=volatility,
        q=dividend,
        option_type=option_type
    )
    
    return params, ticker, n_sims, n_steps, market_data

def render_main_content(
    params: OptionParams,
    ticker: str,
    n_sims: int,
    n_steps: int,
    market_data: MarketData
) -> None:
    """
    Render main dashboard content with analysis results.
    """
    
    # LIVE ANALYTICAL PREVIEW (Real-time updates without clicking Run)
    st.subheader("💡 Live Analytical Preview (Real-time)")
    live_bs_price = BlackScholes.price(params)
    live_bs_greeks = BlackScholes.greeks(params)
    
    live_col1, live_col2, live_col3, live_col4, live_col5 = st.columns(5)
    with live_col1:
        st.metric("Live BS Price", f"${live_bs_price:.4f}")
    with live_col2:
        st.metric("Live Delta", f"{live_bs_greeks.delta:.4f}")
    with live_col3:
        st.metric("Live Gamma", f"{live_bs_greeks.gamma:.4f}")
    with live_col4:
        st.metric("Live Vega", f"{live_bs_greeks.vega:.4f}")
    with live_col5:
        st.metric("Live Theta", f"{live_bs_greeks.theta:.4f}")

    # Out of sync warning
    is_out_of_sync = False
    if 'results' in st.session_state:
        stored_params = st.session_state.results['params']
        # Check if key parameters changed
        if (stored_params.S0 != params.S0 or stored_params.K != params.K or 
            stored_params.T != params.T or stored_params.sigma != params.sigma or
            stored_params.r != params.r or stored_params.q != params.q or
            stored_params.option_type != params.option_type):
            is_out_of_sync = True
            st.warning("⚠️ Parameters changed! Simulation results below are from the PREVIOUS run. Click 'Run Full Analysis' to sync.")

    # Run Analysis Button
    _, col_run2, _ = st.columns([1, 2, 1])
    with col_run2:
        run_clicked = st.button(
            "🚀 Run Full Analysis",
            use_container_width=True,
            type="primary"
        )
    
    # Execute analysis
    if run_clicked or 'results' in st.session_state:
        if run_clicked:
            with st.spinner("Running parallel Monte Carlo simulation..."):
                # Initialize engine
                engine = MonteCarloEngine(params, n_sims, n_steps, seed=42)
                
                # Run pricing
                result = engine.price(antithetic=True, store_paths=True, parallel=True)
                
                # Calculate Greeks
                mc_greeks = engine.calculate_all_greeks()
                
                # Black-Scholes benchmark
                bs_price = BlackScholes.price(params)
                bs_greeks = BlackScholes.greeks(params)
                
                # Store results
                st.session_state.results = {
                    'result': result,
                    'mc_greeks': mc_greeks,
                    'bs_price': bs_price,
                    'bs_greeks': bs_greeks,
                    'params': params,
                    'market_data': market_data,
                    'timestamp': datetime.now()
                }
        
        # Retrieve results
        res = st.session_state.results
        result = res['result']
        mc_greeks = res['mc_greeks']
        bs_price = res['bs_price']
        bs_greeks = res['bs_greeks']
        
        # PRICING RESULTS SECTION
        st.subheader("📊 Pricing Results")
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric(
                "Monte Carlo Price",
                f"${result.price:.4f}",
                f"±{result.std_error:.4f}"
            )
        
        with metrics_col2:
            st.metric(
                "Black-Scholes Price",
                f"${bs_price:.4f}"
            )
        
        with metrics_col3:
            error_pct = abs(result.price - bs_price) / bs_price * 100
            st.metric(
                "Pricing Error",
                f"{error_pct:.3f}%"
            )
        
        with metrics_col4:
            st.metric(
                "Computation Time",
                f"{result.computation_time:.3f}s"
            )
        
        # GREEKS ANALYSIS SECTION
        st.subheader("📈 Greeks Analysis")
        
        greeks_df = pd.DataFrame({
            'Greek': ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
            'Monte Carlo': [
                f"{mc_greeks.delta:.6f}",
                f"{mc_greeks.gamma:.6f}",
                f"{mc_greeks.vega:.6f}",
                f"{mc_greeks.theta:.6f}",
                f"{mc_greeks.rho:.6f}"
            ],
            'Black-Scholes': [
                f"{bs_greeks.delta:.6f}",
                f"{bs_greeks.gamma:.6f}",
                f"{bs_greeks.vega:.6f}",
                f"{bs_greeks.theta:.6f}",
                f"{bs_greeks.rho:.6f}"
            ],
            'Difference': [
                f"{abs(mc_greeks.delta - bs_greeks.delta):.6f}",
                f"{abs(mc_greeks.gamma - bs_greeks.gamma):.6f}",
                f"{abs(mc_greeks.vega - bs_greeks.vega):.6f}",
                f"{abs(mc_greeks.theta - bs_greeks.theta):.6f}",
                f"{abs(mc_greeks.rho - bs_greeks.rho):.6f}"
            ]
        })
        
        st.dataframe(greeks_df, use_container_width=True, hide_index=True)
        
        # VISUALIZATIONS SECTION
        st.subheader("📉 Visualizations")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎲 Price Paths",
            "📊 Distribution",
            "📈 Convergence",
            "🔍 Greeks",
            "🧠 Strategic Intelligence"
        ])
        
        with tab1:
            st.markdown(CHART_CONTAINER_CLASS, unsafe_allow_html=True)
            if result.paths and len(result.paths) > 0:
                fig_paths = create_price_paths_plot(result.paths, params)
                st.plotly_chart(fig_paths, use_container_width=True)
                
                # Path statistics
                path_count = len(result.paths)
                is_sampled = " (Sampled)" if n_sims > 10000 else ""
                st.caption(
                    f"**Path Visual Intelligence Layer{is_sampled}:** Showing {path_count} paths for performance."
                )
            else:
                st.info("💡 Simulation paths are being recalculated or are not stored. Run 'Full Analysis' to populate visualizations.")
            st.markdown(DIV_CLOSE, unsafe_allow_html=True)
            
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                if result.paths and len(result.paths) > 0:
                    st.plotly_chart(create_path_heatmap(result.paths), use_container_width=True, key="path_heatmap_tab1")
                    st.plotly_chart(create_synthetic_candles(result.paths), use_container_width=True, key="synthetic_candles_tab1")
                else:
                    st.empty()
            with pcol2:
                if result.paths and len(result.paths) > 0:
                    st.plotly_chart(create_prob_cone(result.paths, market_data), use_container_width=True, key="prob_cone_tab1")
                    st.plotly_chart(create_bollinger_bands(result.paths), use_container_width=True, key="bollinger_bands_tab1")
                else:
                    st.empty()
            
            if result.paths and len(result.paths) > 0:
                st.plotly_chart(create_animated_hist(result.paths), use_container_width=True, key="layered_hist_tab1")
        
        with tab2:
            st.markdown(CHART_CONTAINER_CLASS, unsafe_allow_html=True)
            fig_dist = create_distribution_plot(
                result.payoffs, result.price, bs_price, params
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # ITM statistics
            itm_count = sum(1 for p in result.payoffs if p > 0)
            itm_prob = itm_count / len(result.payoffs) * 100
            avg_payoff = np.mean(result.payoffs)
            
            st.caption(
                f"**Statistics:** Probability ITM: {itm_prob:.2f}% | "
                f"Expected Payoff: ${avg_payoff:.2f} | "
                f"Total Simulations: {len(result.payoffs):,}"
            )
            st.markdown(DIV_CLOSE, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_violin_dist(result.payoffs), use_container_width=True, key="violin_tab2")
                st.plotly_chart(create_log_return_dist(result.paths), use_container_width=True, key="log_ret_tab2")
            with col2:
                st.plotly_chart(create_cum_payoff(result.payoffs), use_container_width=True, key="cum_payoff_tab2")
                st.plotly_chart(create_risk_neutral_density(params), use_container_width=True, key="rnd_tab2")
            
            st.plotly_chart(create_skew_kurtosis_evolution(result.paths), use_container_width=True, key="skew_kurtosis_tab2")
        
        with tab3:
            st.markdown(CHART_CONTAINER_CLASS, unsafe_allow_html=True)
            fig_conv = create_convergence_plot(result.payoffs, params, bs_price)
            st.plotly_chart(fig_conv, use_container_width=True)
            
            st.caption(
                f"**Convergence:** Final estimate based on {n_sims:,} simulations "
                f"with antithetic variates variance reduction. "
                f"95% confidence interval shown in blue shaded region."
            )
            st.markdown(DIV_CLOSE, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_efficiency_frontier(result, n_sims), use_container_width=True, key="eff_frontier_tab3")
                st.plotly_chart(create_discrete_convergence(result.payoffs, bs_price), use_container_width=True, key="discrete_conv_tab3")
            with col2:
                st.plotly_chart(create_error_dist(result.payoffs, params), use_container_width=True, key="error_dist_tab3")
                st.plotly_chart(create_standard_error_decay_plot(result.payoffs, bs_price), use_container_width=True, key="stderr_decay_tab3")
            
            st.plotly_chart(create_var_reduction_comp(params), use_container_width=True, key="var_reduction_tab3")
        
        with tab4:
            st.markdown(CHART_CONTAINER_CLASS, unsafe_allow_html=True)
            fig_greeks = create_greeks_comparison_plot(mc_greeks, bs_greeks)
            st.plotly_chart(fig_greeks, use_container_width=True)
            st.markdown(DIV_CLOSE, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_greeks_sensitivity_plot(params), use_container_width=True, key="greek_sens_tab4")
                st.plotly_chart(create_greek_corr_heatmap(params), use_container_width=True, key="greek_corr_tab4")
            with col2:
                st.plotly_chart(create_sunburst_greeks(mc_greeks), use_container_width=True, key="sunburst_greek_tab4")
                st.plotly_chart(create_radar_sensitivity(params), use_container_width=True, key="radar_greek_tab4")
            
            st.plotly_chart(create_interactive_greek_explorer(params), use_container_width=True, key="interactive_greek_tab4")

        with tab5:
            st.markdown("### 🧬 Advanced Intelligence Layer (50+ Strategic Layers)")
            
            subtabs = st.tabs(["🌐 3D Surfaces", "📊 Advanced Analytics", "🛡️ Risk & Sensitivity", "🌊 Path Dynamics", "💡 Mathematical Engine"])
            
            with subtabs[0]:
                st.info("💡 Interactive 3D Surfaces Explorer (Optimized Performance)")
                
                surface_type = st.selectbox(
                    "Choose Intelligence Surface",
                    options=[
                        "Price Surface (Spot vs Vol)",
                        "Price Evolution (Spot vs Time)",
                        "Delta Surface (Spot vs Time)",
                        "Vega Surface (Spot vs Time)",
                        "Gamma Surface (Spot vs Time)",
                        "Theta Surface (Spot vs Time)",
                        "Rho Surface (Spot vs Time)",
                        "Strike Sensitivity (Spot vs Strike)",
                        "Vega Pulse (Spot vs Vol)",
                        "Leverage Profile (Omega Surface)",
                        "PDF Evolution (Density Surface)",
                        "Greeks Interaction (Theta vs Vol)"
                    ],
                    index=0,
                    key="surface_selector"
                )
                
                # Logic for selected surface
                if surface_type == "Price Surface (Spot vs Vol)":
                    st.plotly_chart(create_sensitivity_3d(params), use_container_width=True, key="3d_sens_main")
                elif surface_type == "Price Evolution (Spot vs Time)":
                    st.plotly_chart(create_3d_time_surface(params), use_container_width=True, key="3d_time_main")
                elif surface_type == "Delta Surface (Spot vs Time)":
                    st.plotly_chart(create_delta_surface_3d(params), use_container_width=True, key="3d_delta_main")
                elif surface_type == "Vega Surface (Spot vs Time)":
                    st.plotly_chart(create_vega_surface_3d(params), use_container_width=True, key="3d_vega_main")
                elif surface_type == "Gamma Surface (Spot vs Time)":
                    st.plotly_chart(create_gamma_surface_3d(params), use_container_width=True, key="3d_gamma_main")
                elif surface_type == "Theta Surface (Spot vs Time)":
                    st.plotly_chart(create_theta_surface_3d(params), use_container_width=True, key="3d_theta_main")
                elif surface_type == "Rho Surface (Spot vs Time)":
                    st.plotly_chart(create_rho_surface_3d(params), use_container_width=True, key="3d_rho_main")
                elif surface_type == "Strike Sensitivity (Spot vs Strike)":
                    st.plotly_chart(create_strike_surface_3d(params), use_container_width=True, key="3d_strike_main")
                elif surface_type == "Vega Pulse (Spot vs Vol)":
                    st.plotly_chart(create_vol_pulse_3d(params), use_container_width=True, key="3d_pulse_main")
                elif surface_type == "Leverage Profile (Omega Surface)":
                    st.plotly_chart(create_omega_leverage_surface(params), use_container_width=True, key="3d_omega_main")
                elif surface_type == "PDF Evolution (Density Surface)":
                    if result.paths:
                        st.plotly_chart(create_3d_density_surface(result.paths, params), use_container_width=True, key="3d_density_main")
                    else:
                        st.warning("Path storage required for Density Surface.")
                elif surface_type == "Greeks Interaction (Theta vs Vol)":
                    st.plotly_chart(create_theta_vol_surface_3d(params), use_container_width=True, key="3d_theta_v_main")

                st.divider()
                st.markdown("### 🧬 Multi-View Grid")
                col3d1, col3d2 = st.columns(2)
                with col3d1:
                    st.plotly_chart(create_vol_surface_3d(params), use_container_width=True, key="3d_vol_surf_grid")
                with col3d2:
                    st.plotly_chart(create_pop_surface_3d(params), use_container_width=True, key="3d_pop_grid")
                
                st.plotly_chart(create_path_extremes_3d(result.paths), use_container_width=True, key="3d_extremes_grid")
                st.plotly_chart(create_vega_surface_spot_time_3d(params), use_container_width=True, key="3d_vega_st_grid")

            with subtabs[4]:
                st.markdown("### 🧬 Higher-Order Mathematical Intelligence")
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    st.plotly_chart(create_chaos_phase_space(result.paths), use_container_width=True, key="chaos_math_tab5")
                    st.plotly_chart(create_kelly_growth_viz(params, result.price), use_container_width=True, key="kelly_math_tab5")
                with mcol2:
                    st.plotly_chart(create_hurst_persistence_viz(result.paths), use_container_width=True, key="hurst_math_tab5")
                    st.plotly_chart(create_vol_surface_skew_viz(params), use_container_width=True, key="vol_skew_math_tab5")
                
                st.plotly_chart(create_payoff_probability_heatmap(result.payoffs), use_container_width=True, key="payoff_heat_tab5")

            with subtabs[1]:
                col_adv1, col_adv2 = st.columns(2)
                with col_adv1:
                    if result.paths:
                        st.plotly_chart(create_fft_analysis(result.paths), use_container_width=True, key="adv_fft")
                        st.plotly_chart(create_hurst_exponent_dist(result.paths), use_container_width=True, key="adv_hurst")
                        st.plotly_chart(create_path_clustering(result.paths), use_container_width=True, key="adv_cluster")
                        st.plotly_chart(create_chaos_attractor_viz(result.paths), use_container_width=True, key="adv_chaos")
                        st.plotly_chart(create_animated_hist(result.paths), use_container_width=True, key="adv_hist")
                    st.plotly_chart(create_bootstrap_comp(result.payoffs), use_container_width=True, key="adv_boot")
                    st.plotly_chart(create_risk_neutral_density(params), use_container_width=True, key="adv_rnd")
                    st.plotly_chart(create_path_entropy_viz(result.paths), use_container_width=True, key="adv_entropy") if result.paths else st.empty()
                with col_adv2:
                    if result.paths:
                        st.plotly_chart(create_skew_kurtosis_evolution(result.paths), use_container_width=True, key="adv_skew_ev")
                        st.plotly_chart(create_day_of_week_returns(result.paths), use_container_width=True, key="adv_dow")
                        st.plotly_chart(create_hurst_rolling_viz(result.paths), use_container_width=True, key="adv_hurst_roll")
                        st.plotly_chart(create_wavelet_energy_viz(result.paths), use_container_width=True, key="adv_wavelet")
                    st.plotly_chart(create_tree_branching(params), use_container_width=True, key="adv_tree")
                    st.plotly_chart(create_kelly_criterion_plot(params, bs_price), use_container_width=True, key="adv_kelly")
                    st.plotly_chart(create_parallel_params(params), use_container_width=True, key="adv_parallel")
                    st.plotly_chart(create_discrete_convergence(result.payoffs, bs_price), use_container_width=True, key="adv_conv")
                    st.plotly_chart(create_path_corr_matrix_viz(result.paths), use_container_width=True, key="adv_corr_mat") if result.paths else st.empty()

            with subtabs[2]:
                col_risk1, col_risk2 = st.columns(2)
                with col_risk1:
                    st.plotly_chart(create_var_es_plot(result.payoffs), use_container_width=True, key="risk_var_es")
                    st.plotly_chart(create_es_heatmap(result.payoffs), use_container_width=True, key="risk_es_map")
                    st.plotly_chart(create_pnl_waterfall(params, result.price, bs_price), use_container_width=True, key="risk_pnl_wt")
                    st.plotly_chart(create_greek_corr_heatmap(params), use_container_width=True, key="risk_greek_corr")
                    st.plotly_chart(create_pnl_heatmap(params), use_container_width=True, key="risk_pnl_heat")
                    st.plotly_chart(create_risk_reward_scatter_viz(result.payoffs), use_container_width=True, key="risk_scatter")
                    st.plotly_chart(create_terminal_cdf_viz(result.payoffs), use_container_width=True, key="risk_cdf")
                with col_risk2:
                    if result.paths:
                        st.plotly_chart(create_var_evolution(result.paths), use_container_width=True, key="risk_var_ev")
                        st.plotly_chart(create_max_drawdown_dist(result.paths), use_container_width=True, key="risk_mdd_dist")
                        st.plotly_chart(create_drawdown_timeseries(result.paths), use_container_width=True, key="risk_dd_ts")
                    st.plotly_chart(create_multi_asset_corr(), use_container_width=True, key="risk_multi_acc")
                    st.plotly_chart(create_variance_contribution(mc_greeks, params), use_container_width=True, key="risk_var_cont")
                    st.plotly_chart(create_polar_greeks(mc_greeks), use_container_width=True, key="risk_polar")
                    st.plotly_chart(create_pop_curve_viz(params), use_container_width=True, key="risk_pop_curve")
                
                risk_col1, risk_col2 = st.columns(2)
                with risk_col1:
                    st.plotly_chart(create_vol_smile_preview_viz(params), use_container_width=True, key="risk_smile")
                    st.plotly_chart(create_risk_neutral_skew_viz(params), use_container_width=True, key="risk_rn_skew")
                with risk_col2:
                    st.plotly_chart(create_merton_jump_intensity_viz(res['params_dict']['jump_params'] if 'params_dict' in res else {'mu_j':0, 'sigma_j':0, 'lambda_j':0}), use_container_width=True, key="risk_merton") if 'params_dict' in res else st.empty()
                    st.plotly_chart(create_greeks_radar_detail_viz(mc_greeks), use_container_width=True, key="risk_radar")
                st.plotly_chart(create_interactive_greek_explorer(params), use_container_width=True, key="risk_explorer")
                st.plotly_chart(create_tail_loss_butterfly_viz(result.payoffs), use_container_width=True, key="risk_butterfly")

            with subtabs[3]:
                if result.paths:
                    st.plotly_chart(create_synthetic_candles(result.paths), use_container_width=True, key="path_candles")
                    st.plotly_chart(create_growth_ladder(result.paths), use_container_width=True, key="path_ladder")
                    st.plotly_chart(create_fan_chart(result.paths), use_container_width=True, key="path_fan")
                    
                    pcol1, pcol2 = st.columns(2)
                    with pcol1:
                        st.plotly_chart(create_regime_switching(params), use_container_width=True, key="path_regime")
                        st.plotly_chart(create_barrier_prob(result.paths, params), use_container_width=True, key="path_barrier")
                        st.plotly_chart(create_qq_plot(result.paths), use_container_width=True, key="path_qq")
                        st.plotly_chart(create_boxplot_terminal(result.paths), use_container_width=True, key="path_box")
                        st.plotly_chart(create_barrier_crossing_dist_viz(result.paths, params), use_container_width=True, key="path_barrier_dist")
                    with pcol2:
                        st.plotly_chart(create_rolling_sharpe(result.paths, params), use_container_width=True, key="path_sharpe")
                        st.plotly_chart(create_survival_curve(result.paths, params), use_container_width=True, key="path_survival")
                        st.plotly_chart(create_vol_clustering(result.paths), use_container_width=True, key="path_vol_cluster")
                        st.plotly_chart(create_risk_return_cloud(result.paths), use_container_width=True, key="path_cloud")
                        st.plotly_chart(create_drawdown_duration_dist_viz(result.paths), use_container_width=True, key="path_dd_dist")
                    
                    st.plotly_chart(create_var_reduction_comp(params), use_container_width=True, key="path_var_red")
                    st.plotly_chart(create_drift_diffusion_plot(params, result.paths[0]), use_container_width=True, key="path_drift")
                    st.plotly_chart(create_brownian_bridge(params), use_container_width=True, key="path_bridge")
                    st.plotly_chart(create_prob_itm_evolution(result.paths, params), use_container_width=True, key="path_itm_ev")
                    st.plotly_chart(create_price_gradient_quiver(params), use_container_width=True, key="path_gradient")
                    st.plotly_chart(create_theta_gamma_tradeoff(params), use_container_width=True, key="path_tradeoff")
                    st.plotly_chart(create_price_velocity_acceleration(result.paths), use_container_width=True, key="path_vel_acc")
                    st.plotly_chart(create_confidence_99_bands_viz(result.paths), use_container_width=True, key="path_99_conf")
                else:
                    st.warning("Path Dynamics unavailable (Path storage disabled)")

            st.markdown("#### Scenario Sensitivity Table")
            st.dataframe(create_sensitivity_table(params), use_container_width=True)
        
        # EXPORT SECTION
        st.subheader("💾 Export Results")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if result.paths:
                paths_df = pd.DataFrame(result.paths[:100]).T
                csv_paths = paths_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Sample Paths (CSV)",
                    data=csv_paths,
                    file_name=f"{ticker}_{params.option_type}_paths.csv",
                    mime="text/csv"
                )
        
        with export_col2:
            # Full report
            report_data = {
                'Symbol': ticker,
                'Spot_Price': params.S0,
                'Strike_Price': params.K,
                'Maturity': params.T,
                'Risk_Free_Rate': params.r,
                'Volatility': params.sigma,
                'Dividend_Yield': params.q,
                'Option_Type': params.option_type,
                'MC_Price': result.price,
                'MC_Std_Error': result.std_error,
                'BS_Price': bs_price,
                'Absolute_Error': abs(result.price - bs_price),
                'Relative_Error_%': abs(result.price - bs_price) / bs_price * 100,
                'Computation_Time_s': result.computation_time,
                'Simulations': len(result.payoffs),
                'MC_Delta': mc_greeks.delta,
                'BS_Delta': bs_greeks.delta,
                'MC_Gamma': mc_greeks.gamma,
                'BS_Gamma': bs_greeks.gamma,
                'MC_Vega': mc_greeks.vega,
                'BS_Vega': bs_greeks.vega,
                'MC_Theta': mc_greeks.theta,
                'BS_Theta': bs_greeks.theta,
                'MC_Rho': mc_greeks.rho,
                'BS_Rho': bs_greeks.rho
            }
            report_df = pd.DataFrame([report_data])
            csv_report = report_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Report (CSV)",
                data=csv_report,
                file_name=f"{ticker}_pricing_report.csv",
                mime="text/csv"
            )

def main():
    """
    Main Streamlit application entry point.
    """
    # Page configuration is already set at the top of the file
    
    # Custom CSS styling
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    .stApp {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stMetric label {
        color: white !important;
        font-weight: 600;
    }
    .stMetric .metric-value {
        color: white !important;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .stButton > button:hover {
        transform: scale(1.05) translateY(-5px);
        box-shadow: 0 12px 20px rgba(118, 75, 162, 0.4);
    }
    .chart-container {
        border-radius: 15px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    .animated {
        animation: fadeIn 0.8s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(14, 17, 23, 0.9);
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #764ba2;
        z-index: 1000;
        backdrop-filter: blur(5px);
    }
    .footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("📈 Monte Carlo Option Pricing Dashboard")
    st.markdown("""
    **Real-time Option Analytics** | Data: Alpha Vantage API | 
    Model: Geometric Brownian Motion | Engine: Parallel Monte Carlo
    """)
    
    # Render sidebar and get configuration
    params, ticker, n_sims, n_steps, market_data = render_sidebar()
    
    # Render main content
    render_main_content(params, ticker, n_sims, n_steps, market_data)
    
    # Footer
    st.markdown("""
    <div class='footer'>
        Made By <a href='https://sourishdeyportfolio.vercel.app/' target='_blank'>Sourish Dey</a> | © 2026 Quantitative Finance Suite
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption(
        f"Monte Carlo Option Pricing Dashboard v4.0.0 | "
        f"Python {sys.version.split()[0]} | "
        f"Advanced Quantitative Analytics Engine"
    )

if __name__ == "__main__":
    main()
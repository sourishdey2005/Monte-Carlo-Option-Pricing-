---
title: Monte Carlo Option Pricing
emoji: 📈
colorFrom: pink
colorTo: red
sdk: docker
app_port: 7860
---

# 🎯 Monte Carlo Option Pricing Dashboard

A high-performance, professional-grade quantitative finance tool for pricing European options using parallel Monte Carlo simulations. This dashboard provides deep insights into option sensitivities (Greeks), path dynamics, and statistical distributions with over 70+ advanced visualizations.

![Dashboard Preview](https://img.shields.io/badge/Status-Live-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-orange)

## 🚀 Key Features

### 1. **High-Performance Simulation**
- **Parallel Processing**: Utilizes `concurrent.futures` for blazing-fast multi-threaded simulations.
- **Variance Reduction**: Implements **Antithetic Variates** to decrease standard error and improve pricing accuracy.
- **Real-Time Benchmarking**: Compares Monte Carlo results against the analytical **Black-Scholes-Merton** model instantly.

### 2. **3D Intelligence Explorer**
- **Interactive Surfaces**: Visualize Price, Delta, Vega, and Gamma across Spot, Time, and Volatility dimensions.
- **Optimized Rendering**: Uses selected WebGL contexts to maintain high frame rates even with complex 3D lattices.

### 3. **Strategic Analytics Layer (70+ Visualizations)**
- **Path Dynamics**: Synthetic Candlesticks, Brownian Bridges, and Regime Switching analysis.
- **Risk Metrics**: Value at Risk (VaR), Expected Shortfall (CVaR), and Maximum Drawdown distributions.
- **Advanced Math**: Hurst Exponents (Persistence), Chaos Attractors, and Fourier Transform (FFT) frequency analysis of price returns.

### 4. **Greeks Profile**
- Real-time calculation of **Delta, Gamma, Vega, Theta, and Rho**.
- Interactive Greek Explorer and Sensitivity Heatmaps.

## 🛠️ Tech Stack
- **Engine**: NumPy, SciPy (Scientific Computing)
- **Frontend**: Streamlit (Reactive Web UI)
- **Visuals**: Plotly Graph Objects (Interactive 3D/2D Charts)
- **Data**: Pandas (Time-series & Tabular analysis)
- **API**: Alpha Vantage Integration for live ticker data

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sourishdey2005/Monte-Carlo-Option-Pricing-.git
   cd Monte-Carlo-Option-Pricing-
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 📊 Sample Visualizations
- **Price Evolution**: 3D Spot vs. Time Lattice.
- **PnL Waterfall**: Attribution of option value to intrinsic and time components.
- **Probability Cone**: 90% and 50% confidence intervals for future price projections.

---
**Disclaimer**: This tool is for educational and research purposes only. Not financial advice.

**Made by Sourish Dey**

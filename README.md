# Multi-Step Investment Pipeline
**MGMT 490 Capstone Project**

## Overview
This integrated financial command center runs a three-step quantitative pipeline to evaluate equity portfolios. It moves beyond standard technical analysis by combining balance sheet health, intrinsic valuation, and modern portfolio theory into a single workflow. 

## The Pipeline
1. **Credit Risk Screener:** Filters out over-leveraged equities based on dynamically adjustable Debt-to-Equity thresholds.
2. **DCF Valuation & Monte Carlo:** Calculates intrinsic value using a Discounted Cash Flow model and stress-tests the asset via a 1,000-scenario Monte Carlo simulation.
3. **Portfolio Optimization:** Identifies the Efficient Frontier and generates optimal capital weights to maximize the Sharpe Ratio or minimize variance.

## Setup & Execution
To run this application locally:
1. Clone this repository.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Launch the app: `streamlit run app.py`

*AI Usage Disclosure: In accordance with course guidelines, the DRIVER framework was utilized alongside AI to architect the financial logic and optimize Streamlit UI components.*

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.graph_objects as go
import datetime

st.set_page_config(page_title="Multi-Step Investment Pipeline", layout="wide")



# ==========================================
# CACHED FUNCTIONS (To keep UI reactive and fast)
# ==========================================

@st.cache_data(show_spinner=False)
def get_d_e_ratio(ticker):
    """Fetch or calculate Debt-to-Equity Ratio."""
    stock = yf.Ticker(ticker)
    info = stock.info
    # Try using info dictionary first
    if 'debtToEquity' in info and info['debtToEquity'] is not None:
        return info['debtToEquity'] / 100.0  # Usually returned as percentage (e.g. 150 = 1.5)
    
    # Fallback to balance sheet
    try:
        balance = stock.balance_sheet
        if balance.empty:
            return None
        
        debt = balance.loc["Total Debt"].iloc[0] if "Total Debt" in balance.index else 0
        
        equity = None
        if "Stockholders Equity" in balance.index:
            equity = balance.loc["Stockholders Equity"].iloc[0]
        elif "Total Equity Gross Minority Interest" in balance.index:
            equity = balance.loc["Total Equity Gross Minority Interest"].iloc[0]
            
        if equity and equity > 0:
            return debt / equity
        return None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_financials(ticker):
    """Fetch financial data for DCF model."""
    stock = yf.Ticker(ticker)
    cashflow = stock.cashflow
    balance = stock.balance_sheet
    info = stock.info

    if cashflow.empty or balance.empty:
        return None, None, None, None, None

    # Free Cash Flow
    if "Free Cash Flow" in cashflow.index and not pd.isna(cashflow.loc["Free Cash Flow"].iloc[0]):
        fcf = cashflow.loc["Free Cash Flow"].iloc[0]
    else:
        ocf = cashflow.loc["Total Cash From Operating Activities"].iloc[0] if "Total Cash From Operating Activities" in cashflow.index else 0
        capex = cashflow.loc["Capital Expenditures"].iloc[0] if "Capital Expenditures" in cashflow.index else 0
        fcf = ocf - capex

    debt = balance.loc["Total Debt"].iloc[0] if "Total Debt" in balance.index else 0
    cash = balance.loc["Cash And Cash Equivalents"].iloc[0] if "Cash And Cash Equivalents" in balance.index else 0
    shares = info.get("sharesOutstanding", None)
    price = info.get("currentPrice", None)

    return fcf, debt, cash, shares, price

def dcf_model(fcf, growth, wacc, terminal_growth, years=5):
    """Calculate Enterprise Value using DCF."""
    projected_fcfs = []
    for i in range(1, years + 1):
        projected = fcf * (1 + growth) ** i
        discounted = projected / (1 + wacc) ** i
        projected_fcfs.append(discounted)

    terminal_value = (fcf * (1 + growth) ** years * (1 + terminal_growth)) / (wacc - terminal_growth)
    terminal_discounted = terminal_value / (1 + wacc) ** years
    enterprise_value = sum(projected_fcfs) + terminal_discounted
    return enterprise_value

@st.cache_data(show_spinner=False)
def get_historical_prices(tickers):
    """Fetch 1-year historical data for portfolio optimization."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        if 'Adj Close' in data:
            data = data['Adj Close']
        elif 'Close' in data:
            data = data['Close']
        else:
            return pd.DataFrame()
            
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
            
        data.dropna(inplace=True)
        return data
    except Exception:
        return pd.DataFrame()

def portfolio_performance(weights, mean_returns, cov_matrix):
    port_returns = np.sum(mean_returns * weights)
    port_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_returns, port_std_dev

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_std

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

# ==========================================
# UI & PIPELINE LOGIC
# ==========================================

st.title("📈 Multi-Step Investment Pipeline")
st.markdown("A seamless pipeline integrating **Credit Risk Screening**, **DCF Valuation**, and **Portfolio Optimization**.")

# ==========================================
# SIDEBAR: EXECUTIVE SUMMARY LAYOUT
# ==========================================
st.sidebar.header("Executive Inputs & Controls")

tickers_input = st.sidebar.text_input(
    "1. Enter Stock Tickers (comma separated)", 
    "AAPL, MSFT, GOOGL, AMZN, BAC"
)

max_de_ratio = st.sidebar.slider("2. Max Debt-to-Equity Ratio", 0.0, 5.0, 2.0, 0.1)

with st.sidebar.expander("ℹ️ Guide: What is a good Debt-to-Equity Ratio?"):
    st.markdown("""
    **General Rule**: A D/E ratio under 1.0 is generally considered safe, but "good" varies heavily by industry.
    
    **Tech Sector (e.g., AAPL, GOOGL)**: Typically have low D/E ratios (often under 1.0 or 1.5).
    
    **Financial Sector (e.g., BAC, JPM)**: Inherently have much higher D/E ratios (often 1.5 to 3.0+).
    
    ⚠️ **Warning**: Setting the slider too strictly will safely filter out perfectly healthy banks.
    """)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# -----------------------------
# STEP 1: CREDIT RISK SCREENER
# -----------------------------
st.header("Step 1: Credit Risk Screener")
st.write("Filter out highly leveraged companies before running valuation.")

if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

with st.spinner("Fetching D/E Ratios..."):
    de_results = []
    surviving_step1 = []
    
    for ticker in tickers:
        de = get_d_e_ratio(ticker)
        if de is not None:
            passed = de <= max_de_ratio
            if passed:
                surviving_step1.append(ticker)
            de_results.append({
                "Ticker": ticker, 
                "D/E Ratio": round(de, 2), 
                "Status": "✅ Pass" if passed else "❌ Fail"
            })
        else:
            de_results.append({
                "Ticker": ticker, 
                "D/E Ratio": "N/A", 
                "Status": "⚠️ Unknown"
            })

    df_de = pd.DataFrame(de_results)
    
    # Visual Summaries (Metrics)
    m1, m2 = st.columns(2)
    m1.metric("Total Stocks Screened", len(tickers))
    m2.metric("Number of Stocks Passed", len(surviving_step1))
    
    # Bar Chart: D/E Ratios
    st.subheader("Debt-to-Equity Analysis")
    if not df_de.empty:
        chart_data = df_de[df_de["D/E Ratio"] != "N/A"].copy()
        if not chart_data.empty:
            chart_data["D/E Ratio"] = chart_data["D/E Ratio"].astype(float)
            fig_de = go.Figure(data=[
                go.Bar(x=chart_data["Ticker"], y=chart_data["D/E Ratio"], marker_color='royalblue')
            ])
            fig_de.add_hline(y=max_de_ratio, line_dash="dash", line_color="red", annotation_text="Max Allowed")
            fig_de.update_layout(title="D/E Ratio by Ticker", yaxis_title="Debt-to-Equity Ratio", template="plotly_white", height=400)
            st.plotly_chart(fig_de, use_container_width=True)

    st.dataframe(df_de, width='stretch')

if not surviving_step1:
    st.error("No stocks survived Step 1. Please increase the maximum D/E ratio or add different stocks.")
    st.stop()
st.divider()

# -----------------------------
# STEP 2: VALUATION INTEGRATION
# -----------------------------
st.header("Step 2: Valuation Integration")

# Layout Cleanup: Place inputs side-by-side with informational text
col_info, v_col1, v_col2 = st.columns([2, 1, 1])
with col_info:
    st.write("Run a DCF Model to identify undervalued stocks among those that passed Step 1.")
    metric_placeholder = st.empty()

with v_col1:
    wacc = st.number_input("WACC (Discount Rate)", value=0.09, step=0.01, format="%.3f")
with v_col2:
    growth = st.number_input("Growth Rate (Years 1-5)", value=0.04, step=0.01, format="%.3f")

terminal_growth = 0.025
years = 5

with st.spinner("Running Valuation Models..."):
    val_results = []
    chart_val_data = []
    surviving_step2 = []
    
    for ticker in surviving_step1:
        fcf, debt, cash, shares, price = get_financials(ticker)
        
        if fcf is None or not shares or not price:
            val_results.append({
                "Ticker": ticker,
                "Current Price": price if price else "N/A",
                "Intrinsic Value": "N/A",
                "Status": "⚠️ Data Error"
            })
            continue
            
        ev = dcf_model(fcf, growth, wacc, terminal_growth, years)
        equity_val = ev - debt + cash
        intrinsic_val = equity_val / shares
        
        undervalued = intrinsic_val > price
        if undervalued:
            surviving_step2.append(ticker)
            
        val_results.append({
            "Ticker": ticker,
            "Current Price": f"${price:,.2f}",
            "Intrinsic Value": f"${intrinsic_val:,.2f}",
            "Status": "✅ Undervalued" if undervalued else "❌ Overvalued"
        })
        
        chart_val_data.append({
            "Ticker": ticker,
            "Current Price": price,
            "Intrinsic Value": intrinsic_val
        })

    # Display Visual Summaries (Metrics)
    metric_placeholder.metric("Undervalued Stocks Found", len(surviving_step2))

    df_val = pd.DataFrame(val_results)
    
    # Grouped Bar Chart: Current Price vs Intrinsic Value
    st.subheader("Price vs. Intrinsic Value")
    if chart_val_data:
        df_chart_val = pd.DataFrame(chart_val_data)
        fig_val = go.Figure(data=[
            go.Bar(name='Current Price', x=df_chart_val['Ticker'], y=df_chart_val['Current Price'], marker_color='lightslategray'),
            go.Bar(name='Intrinsic Value', x=df_chart_val['Ticker'], y=df_chart_val['Intrinsic Value'], marker_color='mediumseagreen')
        ])
        fig_val.update_layout(barmode='group', template='plotly_white', height=400, yaxis_title="Price ($)")
        st.plotly_chart(fig_val, use_container_width=True)

    st.dataframe(df_val, use_container_width=True)
    
    # --- MONTE CARLO SIMULATION ---
    st.divider()
    st.subheader("Monte Carlo Valuation Simulation")
    st.write("Stress test your valuation by adding random noise to the WACC and Growth Rate inputs.")
    
    if len(surviving_step2) > 0:
        mc_ticker = st.selectbox("Select Stock to Simulate", surviving_step2)
        if st.button(f"Run 1,000 Scenario Simulations for {mc_ticker}"):
            with st.spinner(f"Running 1,000 scenarios for {mc_ticker}..."):
                fcf, debt, cash, shares, price = get_financials(mc_ticker)
                
                sim_values = []
                for _ in range(1000):
                    # Add random noise
                    sim_wacc = max(0.01, np.random.normal(wacc, 0.015)) # 1.5% std dev noise
                    sim_growth = np.random.normal(growth, 0.005) # 0.5% std dev noise
                    
                    ev_sim = dcf_model(fcf, sim_growth, sim_wacc, terminal_growth, years)
                    eq_sim = ev_sim - debt + cash
                    iv_sim = eq_sim / shares
                    sim_values.append(iv_sim)
                
                # Display histogram using st.bar_chart as requested
                hist, bin_edges = np.histogram(sim_values, bins=40)
                # Ensure index are strings or floats for bar chart
                hist_df = pd.DataFrame(hist, index=np.round(bin_edges[:-1], 2), columns=['Frequency'])
                
                st.write(f"**Distribution of Possible Intrinsic Values ({mc_ticker})**")
                st.bar_chart(hist_df)

if not surviving_step2:
    st.error("No stocks were found to be undervalued in Step 2. Please adjust valuation inputs or add different stocks.")
    st.stop()
st.divider()

# -----------------------------
# STEP 3: PORTFOLIO OPTIMIZATION
# -----------------------------
st.header("Step 3: Portfolio Optimization Integration")
st.write("Generate optimal weights for the undervalued stocks.")

if len(surviving_step2) < 2:
    st.warning(f"Only {len(surviving_step2)} stock(s) reached this step. Portfolio optimization requires at least 2 assets. Consider adjusting the D/E slider or Valuation inputs.")
    st.stop()

p_col1, p_col2, p_col3 = st.columns(3)
with p_col1:
    optimization_goal = st.selectbox("Optimization Goal", ["Maximize Sharpe Ratio", "Minimize Variance"])
with p_col2:
    max_weight = st.slider("Max Weight per Stock", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
with p_col3:
    min_weight = st.slider("Min Weight per Stock", min_value=0.0, max_value=0.15, value=0.05, step=0.01)

num_assets = len(surviving_step2)

if min_weight * num_assets > 1.0:
    st.error(f"Mathematical Error: Minimum weight ({min_weight:.0%}) * {num_assets} stocks = {min_weight * num_assets:.0%}, which exceeds 100%. Please lower the minimum weight limit.")
    st.stop()
if max_weight * num_assets < 1.0:
    st.error(f"Mathematical Error: Maximum weight ({max_weight:.0%}) * {num_assets} stocks = {max_weight * num_assets:.0%}, which is less than 100%. Please raise the maximum weight limit.")
    st.stop()

with st.spinner("Downloading 1-year data and optimizing portfolio..."):
    data = get_historical_prices(surviving_step2)
    
    if data.empty or len(data.columns) < 2:
        st.error("Failed to download enough historical data for optimization.")
        st.stop()
        
    actual_tickers = list(data.columns)
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    risk_free_rate = 0.02
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]
    
    if optimization_goal == "Maximize Sharpe Ratio":
        opt_result = sco.minimize(neg_sharpe_ratio, init_guess, 
                                  args=(mean_returns, cov_matrix, risk_free_rate),
                                  method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        opt_result = sco.minimize(portfolio_volatility, init_guess, 
                                  args=(mean_returns, cov_matrix),
                                  method='SLSQP', bounds=bounds, constraints=constraints)

    if opt_result.success:
        opt_weights = opt_result.x
        opt_returns, opt_std = portfolio_performance(opt_weights, mean_returns, cov_matrix)
        opt_sharpe = (opt_returns - risk_free_rate) / opt_std
        
        # Display Metrics
        r_col1, r_col2, r_col3 = st.columns(3)
        r_col1.metric("Expected Annual Return", f"{opt_returns:.2%}")
        r_col2.metric("Expected Volatility (Risk)", f"{opt_std:.2%}")
        r_col3.metric("Sharpe Ratio", f"{opt_sharpe:.2f}")
        
        st.subheader("Optimal Weight Allocation")
        weights_df = pd.DataFrame({"Ticker": actual_tickers, "Weight": opt_weights})
        weights_df = weights_df.sort_values(by="Weight", ascending=False)
        weights_df_display = weights_df.copy()
        weights_df_display['Weight'] = weights_df_display['Weight'].apply(lambda x: f"{x:.2%}")
        st.dataframe(weights_df_display, hide_index=True)
        
        # Visualize efficiently frontier
        st.subheader("Efficient Frontier")
        with st.spinner("Generating Frontier..."):
            num_portfolios = 2000
            results = np.zeros((3, num_portfolios))
            
            for i in range(num_portfolios):
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)
                pret, pstd = portfolio_performance(weights, mean_returns, cov_matrix)
                results[0,i] = pstd
                results[1,i] = pret
                results[2,i] = (pret - risk_free_rate) / pstd
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=results[0,:],
                y=results[1,:],
                mode='markers',
                marker=dict(
                    color=results[2,:],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Sharpe Ratio', thickness=15),
                    size=4,
                    opacity=0.6
                ),
                name='Simulated Portfolios',
                hoverinfo='text',
                text=[f"Return: {r:.2%}<br>Volatility: {v:.2%}<br>Sharpe: {s:.2f}" 
                      for r, v, s in zip(results[1,:], results[0,:], results[2,:])]
            ))
            
            fig.add_trace(go.Scatter(
                x=[opt_std],
                y=[opt_returns],
                mode='markers',
                marker=dict(color='crimson', size=16, symbol='star', line=dict(width=2, color='DarkSlateGrey')),
                name=f'Optimal: {optimization_goal}',
                hoverinfo='text',
                text=[f"Target Return: {opt_returns:.2%}<br>Target Volatility: {opt_std:.2%}"]
            ))
            
            fig.update_layout(
                xaxis_title='Expected Volatility (Risk)',
                yaxis_title='Expected Annual Return',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Optimization failed to converge. Please relax constraints (e.g. increase max weight).")

# ==========================================
# FOOTER
# ==========================================
st.divider()
st.markdown("#### System Status: DRIVER Framework Compliant ✅")
st.markdown("<small>Integrated Multi-Step FinTech Application. Built in accordance with Capstone structural guidelines.</small>", unsafe_allow_html=True)

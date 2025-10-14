import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import datetime as dt
from io import BytesIO
import xlsxwriter

# Set page configuration
st.set_page_config(
    page_title="投资组合回测分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure matplotlib for Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Application title
st.title("📈 投资组合回测分析系统")
st.markdown("---")

# Sidebar configuration
st.sidebar.header("⚙️ 参数设置")

# Stock selection
stocks = ["VOO", "VEA", "QQQ", "FNGS", "VWO", "IAU", "BNDX"]
selected_stocks = st.sidebar.multiselect(
    "选择股票/ETF",
    stocks,
    default=stocks,
    help="选择要分析的股票或ETF"
)

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "开始日期",
        value=dt.date(2023, 5, 1),
        help="回测开始日期"
    )
with col2:
    end_date = st.date_input(
        "结束日期",
        value=dt.date(2025, 9, 30),
        help="回测结束日期"
    )

# Risk-free rate
risk_free_rate = st.sidebar.slider(
    "无风险利率",
    min_value=0.0,
    max_value=0.1,
    value=0.02,
    step=0.001,
    format="%.3f",
    help="用于计算夏普比率的无风险利率"
)

# Monte Carlo simulations
n_simulations = st.sidebar.slider(
    "蒙特卡罗模拟次数",
    min_value=1000,
    max_value=50000,
    value=20000,
    step=1000,
    help="投资组合优化的模拟次数"
)

# Data refresh options
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 数据管理")
if st.sidebar.button("清除缓存并刷新数据", type="secondary"):
    st.cache_data.clear()
    st.rerun()

# Function to fetch data from Stooq
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stooq_data(symbol):
    """Fetch stock data from Stooq"""
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
        df = pd.read_csv(url)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        st.error(f"获取 {symbol} 数据失败: {str(e)}")
        return None

# Adjustment function (simplified for demo)
def auto_adjust_prices(df):
    """Simple price adjustment (no dividend data available from Stooq)"""
    if df is None:
        return None
    df = df.sort_index(ascending=True)
    df['Adj_Factor'] = 1.0
    df['Adj_Close'] = df['Close'] * df['Adj_Factor']
    df['Adj_Open'] = df['Open'] * df['Adj_Factor']
    df['Adj_High'] = df['High'] * df['Adj_Factor']
    df['Adj_Low'] = df['Low'] * df['Adj_Factor']
    return df

# Performance metrics calculation
def performance_metrics(ret_series, name="Stock"):
    """Calculate performance metrics for a return series"""
    if len(ret_series) == 0:
        return pd.Series({
            "CAGR": np.nan,
            "Volatility": np.nan,
            "Sharpe": np.nan,
            "Max Drawdown": np.nan
        }, name=name)
    
    cagr = (1 + ret_series).prod() ** (252 / len(ret_series)) - 1
    volatility = ret_series.std() * np.sqrt(252)
    sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 else np.nan
    cum_ret = (1 + ret_series).cumprod()
    drawdown = (cum_ret / cum_ret.cummax() - 1).min()
    return pd.Series({
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": drawdown
    }, name=name)

# Risk analysis functions
def calculate_var_cvar(returns, confidence_level=0.95):
    """Calculate Value at Risk (VaR) and Conditional VaR (CVaR)"""
    if len(returns) == 0:
        return np.nan, np.nan
    
    var = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar

# Portfolio optimization functions
def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate portfolio performance metrics"""
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, vol

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """Negative Sharpe ratio for optimization (we minimize)"""
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - risk_free_rate) / vol

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate):
    """Optimize portfolio for maximum Sharpe ratio"""
    n_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess (equal weights)
    x0 = np.array([1/n_assets] * n_assets)
    
    result = minimize(
        neg_sharpe_ratio,
        x0,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

# Excel export function
def create_excel_report(metrics_df, optimal_weights, stock_columns, portfolio_metrics, voo_metrics=None):
    """Create Excel report with analysis results"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Sheet 1: Individual stock metrics
        metrics_df.to_excel(writer, sheet_name='个股表现')
        
        # Sheet 2: Optimal portfolio weights
        weights_df = pd.DataFrame({
            '股票': stock_columns,
            '权重': optimal_weights
        })
        weights_df.to_excel(writer, sheet_name='最优组合权重', index=False)
        
        # Sheet 3: Portfolio metrics
        portfolio_metrics.to_frame(name='最优组合').to_excel(writer, sheet_name='组合表现')
        
        # Sheet 4: Comparison (if VOO data available)
        if voo_metrics is not None:
            comparison = pd.concat([portfolio_metrics, voo_metrics], axis=1)
            comparison.to_excel(writer, sheet_name='组合vs基准对比')
        
        # Format workbook
        workbook = writer.book
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1
        })
        
    output.seek(0)
    return output

# Main analysis
if st.sidebar.button("🚀 开始分析", type="primary"):
    if not selected_stocks:
        st.error("请至少选择一个股票或ETF")
    else:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data
        status_text.text("正在下载股票数据...")
        raw_data = {}
        
        for i, stock in enumerate(selected_stocks):
            progress_bar.progress((i + 1) / len(selected_stocks))
            df = get_stooq_data(stock)
            if df is not None:
                raw_data[stock] = auto_adjust_prices(df)
                status_text.text(f"已下载 {stock} 数据...")
        
        if not raw_data:
            st.error("未能获取任何股票数据，请检查网络连接或稍后重试")
        else:
            # Combine adjusted close prices
            data = pd.DataFrame({s: df['Adj_Close'] for s, df in raw_data.items() if df is not None})
            data = data.loc[(data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))]
            data = data.dropna(axis=1)
            
            if data.empty:
                st.error("所选日期范围内没有可用数据")
            else:
                # Calculate returns
                returns = data.pct_change().dropna()
                
                # Pre-calculate optimized portfolio weights (used across multiple tabs)
                mean_returns = returns.mean() * 252
                cov_matrix = returns.cov() * 252
                
                if len(data.columns) >= 2:
                    optimal_weights = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)
                else:
                    optimal_weights = None
                
                status_text.text("计算中...")
                progress_bar.progress(100)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results with data update time
                current_time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.success(f"✅ 成功分析 {len(data.columns)} 只股票，数据期间：{data.index.min().strftime('%Y-%m-%d')} 至 {data.index.max().strftime('%Y-%m-%d')}")
                st.caption(f"📊 数据更新时间：{current_time} | 💾 数据已缓存1小时")
                
                # Tab layout
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 单只表现", "🎯 蒙特卡罗优化", "📈 均值-方差优化", "⚖️ 组合对比", "🎲 风险分析"])
                
                with tab1:
                    st.header("📊 个股表现分析")
                    
                    # Performance metrics table
                    metrics_list = []
                    for stock in data.columns:
                        metrics = performance_metrics(returns[stock], stock)
                        metrics_list.append(metrics)
                    
                    metrics_df = pd.concat(metrics_list, axis=1).T
                    
                    # Format the dataframe for display
                    display_df = metrics_df.copy()
                    display_df['CAGR'] = display_df['CAGR'].apply(lambda x: f"{x:.2%}")
                    display_df['Volatility'] = display_df['Volatility'].apply(lambda x: f"{x:.2%}")
                    display_df['Sharpe'] = display_df['Sharpe'].apply(lambda x: f"{x:.2f}")
                    display_df['Max Drawdown'] = display_df['Max Drawdown'].apply(lambda x: f"{x:.2%}")
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Add download button for individual metrics
                    csv = display_df.to_csv(index=True).encode('utf-8-sig')
                    st.download_button(
                        label="📥 下载个股指标 (CSV)",
                        data=csv,
                        file_name=f"individual_stocks_{dt.datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # Cumulative returns chart
                    st.subheader("累计收益率图表")
                    cum_returns = (1 + returns).cumprod()
                    
                    fig = go.Figure()
                    for stock in cum_returns.columns:
                        fig.add_trace(go.Scatter(
                            x=cum_returns.index,
                            y=cum_returns[stock],
                            mode='lines',
                            name=stock,
                            line=dict(width=2)
                        ))
                    
                    fig.update_layout(
                        title="股票累计表现 (2023-2025)",
                        xaxis_title="日期",
                        yaxis_title="累计收益率",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.header("🎯 蒙特卡罗投资组合优化")
                    
                    if len(data.columns) < 2:
                        st.warning("需要至少2只股票才能进行投资组合优化")
                    else:
                        status_text2 = st.empty()
                        status_text2.text("正在进行蒙特卡罗模拟...")
                        
                        # Monte Carlo simulation using pre-calculated mean_returns and cov_matrix
                        n_assets = len(data.columns)
                        results = np.zeros((3, n_simulations))
                        weights_record = []
                        
                        for i in range(n_simulations):
                            weights = np.random.random(n_assets)
                            weights /= np.sum(weights)
                            weights_record.append(weights)
                            
                            port_ret = np.dot(mean_returns, weights)
                            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                            sharpe = (port_ret - risk_free_rate) / port_vol
                            
                            results[0, i] = port_ret
                            results[1, i] = port_vol
                            results[2, i] = sharpe
                        
                        status_text2.empty()
                        
                        # Find optimal portfolio
                        max_sharpe_idx = np.argmax(results[2])
                        max_sharpe_ret = results[0, max_sharpe_idx]
                        max_sharpe_vol = results[1, max_sharpe_idx]
                        max_sharpe_weights = weights_record[max_sharpe_idx]
                        
                        # Display optimal portfolio
                        st.subheader("最优夏普比率组合")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            weights_df = pd.DataFrame({
                                '股票': data.columns,
                                '权重': [f"{w:.2%}" for w in max_sharpe_weights]
                            })
                            st.dataframe(weights_df, use_container_width=True, hide_index=True)
                        
                        with col2:
                            st.metric("年化收益率", f"{max_sharpe_ret:.2%}")
                            st.metric("波动率", f"{max_sharpe_vol:.2%}")
                            st.metric("夏普比率", f"{results[2, max_sharpe_idx]:.2f}")
                        
                        # Scatter plot of portfolios
                        st.subheader("风险-收益散点图")
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=results[1],
                            y=results[0],
                            mode='markers',
                            marker=dict(
                                size=4,
                                color=results[2],
                                colorscale='Viridis',
                                colorbar=dict(title="夏普比率"),
                                opacity=0.6
                            ),
                            name='投资组合'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=[max_sharpe_vol],
                            y=[max_sharpe_ret],
                            mode='markers',
                            marker=dict(
                                size=15,
                                color='red',
                                symbol='star'
                            ),
                            name='最优组合'
                        ))
                        
                        fig.update_layout(
                            title="蒙特卡罗模拟结果",
                            xaxis_title="波动率",
                            yaxis_title="年化收益率",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    st.header("📈 均值-方差优化")
                    
                    if len(data.columns) < 2:
                        st.warning("需要至少2只股票才能进行投资组合优化")
                    else:
                        # Use pre-calculated optimal weights
                        opt_ret, opt_vol = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
                        opt_sharpe = (opt_ret - risk_free_rate) / opt_vol
                        
                        st.subheader("最优夏普比率组合")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            weights_df = pd.DataFrame({
                                '股票': data.columns,
                                '权重': [f"{w:.2%}" for w in optimal_weights]
                            })
                            st.dataframe(weights_df, use_container_width=True, hide_index=True)
                        
                        with col2:
                            st.metric("年化收益率", f"{opt_ret:.2%}")
                            st.metric("波动率", f"{opt_vol:.2%}")
                            st.metric("夏普比率", f"{opt_sharpe:.2f}")
                        
                        # Efficient frontier
                        st.subheader("有效前沿")
                        
                        # Generate efficient frontier
                        target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 100)
                        efficient_vols = []
                        
                        for target in target_returns:
                            constraints = [
                                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                                {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target}
                            ]
                            bounds = tuple((0, 1) for _ in range(len(mean_returns)))
                            x0 = np.array([1/len(mean_returns)] * len(mean_returns))
                            
                            try:
                                result = minimize(
                                    lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))),
                                    x0,
                                    method='SLSQP',
                                    bounds=bounds,
                                    constraints=constraints
                                )
                                if result.success:
                                    efficient_vols.append(result.fun)
                                else:
                                    efficient_vols.append(np.nan)
                            except:
                                efficient_vols.append(np.nan)
                        
                        # Plot efficient frontier
                        fig = go.Figure()
                        
                        valid_points = ~np.isnan(efficient_vols)
                        fig.add_trace(go.Scatter(
                            x=np.array(efficient_vols)[valid_points],
                            y=target_returns[valid_points],
                            mode='lines',
                            name='有效前沿',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=[opt_vol],
                            y=[opt_ret],
                            mode='markers',
                            marker=dict(
                                size=15,
                                color='red',
                                symbol='star'
                            ),
                            name='最优组合'
                        ))
                        
                        # Add individual assets
                        individual_vols = np.sqrt(np.diag(cov_matrix))
                        fig.add_trace(go.Scatter(
                            x=individual_vols,
                            y=mean_returns,
                            mode='markers',
                            marker=dict(
                                size=10,
                                color='green'
                            ),
                            text=data.columns,
                            name='个股'
                        ))
                        
                        fig.update_layout(
                            title="有效前沿图",
                            xaxis_title="波动率",
                            yaxis_title="年化收益率",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    st.header("⚖️ 投资组合 vs 基准对比")
                    
                    if "VOO" not in data.columns:
                        st.warning("需要包含VOO才能进行基准对比")
                    elif len(data.columns) < 2:
                        st.warning("需要至少2只股票才能进行投资组合优化")
                    else:
                        # Calculate portfolio daily returns using pre-calculated weights
                        portfolio_daily = (returns * optimal_weights).sum(axis=1)
                        portfolio_cum = (1 + portfolio_daily).cumprod()
                        voo_cum = (1 + returns["VOO"]).cumprod()
                        
                        # Performance comparison
                        port_metrics = performance_metrics(portfolio_daily, "最优组合")
                        voo_metrics = performance_metrics(returns["VOO"], "VOO")
                        
                        comparison_df = pd.concat([port_metrics, voo_metrics], axis=1)
                        
                        # Format for display
                        display_comparison = comparison_df.copy()
                        for col in display_comparison.columns:
                            if col in display_comparison.index:
                                continue
                            display_comparison.loc['CAGR', col] = f"{display_comparison.loc['CAGR', col]:.2%}"
                            display_comparison.loc['Volatility', col] = f"{display_comparison.loc['Volatility', col]:.2%}"
                            display_comparison.loc['Sharpe', col] = f"{display_comparison.loc['Sharpe', col]:.2f}"
                            display_comparison.loc['Max Drawdown', col] = f"{display_comparison.loc['Max Drawdown', col]:.2%}"
                        
                        st.subheader("表现对比")
                        st.dataframe(display_comparison, use_container_width=True)
                        
                        # Cumulative returns comparison chart
                        st.subheader("累计收益对比")
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=portfolio_cum.index,
                            y=portfolio_cum,
                            mode='lines',
                            name='最优夏普比率组合',
                            line=dict(width=3, color='blue')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=voo_cum.index,
                            y=voo_cum,
                            mode='lines',
                            name='VOO',
                            line=dict(width=2, dash='dash', color='red')
                        ))
                        
                        fig.update_layout(
                            title="最优夏普比率组合 vs VOO",
                            xaxis_title="日期",
                            yaxis_title="累计收益率",
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download Excel report
                        st.subheader("📥 下载分析报告")
                        excel_data = create_excel_report(
                            metrics_df, 
                            optimal_weights, 
                            data.columns, 
                            port_metrics, 
                            voo_metrics
                        )
                        st.download_button(
                            label="下载 Excel 报告",
                            data=excel_data,
                            file_name=f"portfolio_analysis_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                
                with tab5:
                    st.header("🎲 风险分析")
                    
                    if len(data.columns) < 2:
                        st.warning("需要至少2只股票才能进行投资组合风险分析")
                    else:
                        # Use pre-calculated optimal weights for risk analysis
                        portfolio_daily = (returns * optimal_weights).sum(axis=1)
                        
                        # VaR and CVaR analysis
                        st.subheader("风险价值分析 (VaR & CVaR)")
                        
                        confidence_levels = [0.90, 0.95, 0.99]
                        risk_metrics = []
                        
                        for conf in confidence_levels:
                            var, cvar = calculate_var_cvar(portfolio_daily, conf)
                            risk_metrics.append({
                                '置信水平': f"{conf:.0%}",
                                'VaR (日)': f"{var:.2%}",
                                'CVaR (日)': f"{cvar:.2%}",
                                'VaR (年化)': f"{var * np.sqrt(252):.2%}",
                                'CVaR (年化)': f"{cvar * np.sqrt(252):.2%}"
                            })
                        
                        risk_df = pd.DataFrame(risk_metrics)
                        st.dataframe(risk_df, use_container_width=True, hide_index=True)
                        
                        st.info("""
                        **📌 指标说明：**
                        
                        **VaR (Value at Risk - 风险价值)**：在给定置信水平下，投资组合在一定时期内可能遭受的最大损失。
                        - 例如：95% VaR = -2% 表示有95%的把握日损失不会超过2%
                        
                        **CVaR (Conditional VaR - 条件风险价值)**：当损失超过VaR阈值时的平均损失，更能反映极端风险。
                        - CVaR总是大于等于VaR，体现"最坏情况"下的平均损失
                        
                        💡 负值表示损失，数值越大表示风险越高
                        """)
                        
                        # Returns distribution
                        st.subheader("收益率分布")
                        
                        fig = go.Figure()
                        
                        # Histogram
                        fig.add_trace(go.Histogram(
                            x=portfolio_daily,
                            nbinsx=50,
                            name='收益率分布',
                            marker_color='lightblue',
                            opacity=0.7
                        ))
                        
                        # Add VaR lines
                        var_95, cvar_95 = calculate_var_cvar(portfolio_daily, 0.95)
                        fig.add_vline(x=var_95, line_dash="dash", line_color="red", 
                                     annotation_text=f"VaR 95%: {var_95:.2%}")
                        fig.add_vline(x=cvar_95, line_dash="dash", line_color="darkred", 
                                     annotation_text=f"CVaR 95%: {cvar_95:.2%}")
                        
                        fig.update_layout(
                            title="投资组合日收益率分布",
                            xaxis_title="日收益率",
                            yaxis_title="频数",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Rolling volatility
                        st.subheader("滚动波动率分析")
                        
                        rolling_window = st.slider("滚动窗口 (天)", 20, 120, 60, 10)
                        rolling_vol = portfolio_daily.rolling(window=rolling_window).std() * np.sqrt(252)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=rolling_vol.index,
                            y=rolling_vol,
                            mode='lines',
                            name=f'{rolling_window}日滚动波动率',
                            line=dict(color='purple', width=2)
                        ))
                        
                        fig.update_layout(
                            title=f"投资组合滚动波动率 ({rolling_window}日窗口)",
                            xaxis_title="日期",
                            yaxis_title="年化波动率",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Individual stock risk contribution
                        st.subheader("个股风险贡献")
                        
                        # Calculate marginal contribution to risk
                        portfolio_var = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
                        marginal_contrib = np.dot(cov_matrix, optimal_weights)
                        risk_contrib = optimal_weights * marginal_contrib / np.sqrt(portfolio_var)
                        
                        contrib_df = pd.DataFrame({
                            '股票': data.columns,
                            '权重': [f"{w:.2%}" for w in optimal_weights],
                            '风险贡献': [f"{rc:.2%}" for rc in risk_contrib]
                        })
                        
                        st.dataframe(contrib_df, use_container_width=True, hide_index=True)
                        
                        # Risk contribution pie chart
                        fig = go.Figure(data=[go.Pie(
                            labels=data.columns,
                            values=risk_contrib,
                            hole=0.4
                        )])
                        
                        fig.update_layout(
                            title="风险贡献占比",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

else:
    # Instructions
    st.info("👈 请在侧边栏设置参数并点击'开始分析'按钮来开始投资组合回测分析")
    
    # Features overview
    st.markdown("### 📋 功能特色")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **数据获取**
        - 🌐 实时从 Stooq 获取股票数据
        - 📅 自定义时间区间分析
        - 🔄 自动复权价格调整
        
        **投资组合优化**
        - 🎯 蒙特卡罗模拟优化
        - 📊 均值-方差优化
        - ⚡ 夏普比率最大化
        """)
    
    with col2:
        st.markdown("""
        **表现分析**
        - 📈 CAGR、波动率、夏普比率
        - 📉 最大回撤分析
        - 📊 有效前沿可视化
        
        **风险分析与报告**
        - 🎲 VaR/CVaR 风险价值分析
        - 📊 收益率分布与滚动波动率
        - 📥 Excel 分析报告下载
        """)
    
    st.markdown("### 🎯 支持的ETF")
    st.markdown("VOO (Vanguard S&P 500), VEA (Vanguard FTSE Developed Markets), QQQ (Invesco QQQ Trust), FNGS (MicroSectors FANG+), VWO (Vanguard Emerging Markets), IAU (iShares Gold Trust), BNDX (Vanguard Total International Bond)")

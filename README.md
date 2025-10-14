# Overview

This is a comprehensive portfolio backtesting and analysis system built with Streamlit. The application provides interactive tools for analyzing investment portfolios consisting of stocks and ETFs (VOO, VEA, QQQ, FNGS, VWO, IAU, BNDX). It enables users to perform backtesting analysis, visualize portfolio performance, optimize asset allocation strategies, analyze risk metrics, and export detailed reports through an intuitive web interface.

## Recent Updates (October 2025)
- ✅ Added Excel multi-sheet report download functionality
- ✅ Implemented comprehensive risk analysis (VaR & CVaR with multiple confidence levels)
- ✅ Added rolling volatility analysis with customizable windows
- ✅ Implemented risk contribution analysis for individual assets
- ✅ Added CSV download for individual stock metrics
- ✅ Performance optimization: centralized calculation of mean returns and covariance matrix
- ✅ Added data cache management with manual refresh capability
- ✅ Enhanced user experience with data update timestamps and detailed metric explanations

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
**Framework**: Streamlit-based web application
- **Rationale**: Streamlit provides rapid development of data-driven applications with minimal boilerplate code, ideal for financial analysis dashboards
- **UI Components**: Sidebar for parameter configuration, multi-column layouts for organized data presentation
- **Visualization Libraries**: 
  - Plotly Express and Plotly Graph Objects for interactive charts
  - Matplotlib for static visualizations with Chinese character support (SimHei, DejaVu Sans fonts)
- **Pros**: Fast prototyping, built-in interactive widgets, seamless Python integration
- **Cons**: Limited customization compared to traditional web frameworks, server-side rendering only

## Data Processing
**Core Libraries**: Pandas and NumPy
- **Purpose**: Time-series financial data manipulation and numerical computations
- **Data Structure**: DataFrame-based storage for historical stock/ETF price data
- **Rationale**: Industry-standard libraries for financial data analysis with extensive built-in functions

## Portfolio Optimization
**Optimization Engine**: SciPy's minimize function
- **Purpose**: Asset allocation optimization and portfolio construction
- **Rationale**: Robust numerical optimization for modern portfolio theory calculations (Sharpe ratio, efficient frontier analysis)

## Application Configuration
**Page Setup**: Wide layout with expanded sidebar
- **Localization**: Chinese language interface (投资组合回测分析系统)
- **Date Handling**: Python datetime module for temporal range selection
- **Default Portfolio**: Seven pre-configured assets (US equities, international equities, emerging markets, gold, bonds)

## Export Functionality
**Output Formats**: Multi-format export capability
- **Excel Reports**: Multi-sheet workbooks containing individual stock metrics, optimal portfolio weights, performance comparisons, and benchmark analysis
- **CSV Downloads**: Individual stock performance metrics in CSV format
- **Implementation**: xlsxwriter and BytesIO for in-memory spreadsheet generation
- **Purpose**: Enable users to download comprehensive backtesting results and analysis data for offline use

## Risk Analysis
**Risk Metrics**: VaR (Value at Risk) and CVaR (Conditional Value at Risk)
- **Confidence Levels**: 90%, 95%, and 99% confidence intervals
- **Visualization**: Returns distribution histograms with VaR/CVaR markers, rolling volatility charts
- **Risk Contribution**: Marginal risk contribution analysis for each asset in the optimized portfolio
- **Purpose**: Provide comprehensive downside risk assessment and extreme event analysis

# External Dependencies

## Python Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Static visualization library
- **plotly**: Interactive charting (express and graph_objects modules)
- **scipy**: Scientific computing and optimization (minimize function)
- **xlsxwriter**: Excel file generation
- **openpyxl**: Excel file reading/writing support

## Performance Optimizations
- **Computation Caching**: Pre-calculation of mean returns and covariance matrix to avoid redundant computation across tabs
- **Data Caching**: 1-hour TTL cache for fetched stock data from Stooq API
- **Optimized Monte Carlo**: Reuse of pre-computed statistics in simulation loops
- **Single Optimization**: Portfolio weights calculated once and shared across all analysis tabs

## Data Sources
The application fetches real-time financial market data from Stooq API for the following instruments:
- VOO (Vanguard S&P 500 ETF)
- VEA (Vanguard FTSE Developed Markets ETF)
- QQQ (Invesco QQQ Trust - NASDAQ-100)
- FNGS (MicroSectors FANG+ ETN)
- VWO (Vanguard FTSE Emerging Markets ETF)
- IAU (iShares Gold Trust)
- BNDX (Vanguard Total International Bond ETF)

**Data Management**: 
- Automatic price adjustment for historical data
- 1-hour caching mechanism to reduce API calls
- Manual cache refresh button for on-demand data updates
- Data timestamp display for transparency

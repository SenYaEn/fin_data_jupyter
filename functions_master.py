import pandas as pd
import numpy as np
import scipy.stats
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import statistics
import datetime as dt

def get_returns_per_company (company_id, df_name):
    """
    Applies the pct_change() method Open, High, and Close Series per single CompanyOd
    converting each datapoint into a percentage change in relation to its preceding datepoint.
    
    Returns a time series of asset returns DataFrame for 'Open', 'High', 'Close' data.
    """
    filtered_company = (df_name
                        .where(df_name['CompanyId']==company_id)
                        .dropna()
                        .sort_values(by=['Date']))
    
    filtered_company = filtered_company.set_index('Date')
    
    filtered_company['YearMonth'] = pd.to_datetime(filtered_company.index).to_period('M')
    
    filtered_company['Open'] = filtered_company['Open'].pct_change()
    filtered_company['High'] = filtered_company['High'].pct_change()
    filtered_company['Close'] = filtered_company['Close'].pct_change()
    
    returns = (filtered_company[['CompanyId', 'CompanyName', 'Open', 'High', 'Close', 'YearMonth']]
                   .dropna())
                   
    returns = returns.reset_index()
    
    return returns
    
    
def get_returns_single_metric (company_id, df, start_date, end_date, metric):
    """
    Filters time series of asset returns DataFrame by date and pivots the data.
    
    Returns a Series with date as an index, Cpmpany Name as a column name,
    and percent change as data.
    """
    df = get_returns_per_company (company_id, df)
    df = df.where((df['Date'] >= start_date) & (df['Date'] <= end_date)).dropna()
    df = df.set_index('Date')
    df = df[['CompanyName', metric]]
    df = df.rename(columns={metric: df['CompanyName'][0]})
    df = df[df['CompanyName'][0]]
    
    return df

    
    
def get_drawdown (return_series):
    """
    Takes a time series of asset returns and returns a 
    DataFrame with columns for the wealth index, 
    the previous peaks, and the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth" : wealth_index,
        "Peaks" : previous_peaks,
        "Drawdown" : drawdown
    })


def plot_wealth_peaks_drawdown (returns_df):
    if len(returns_df.shape) == 1:
        get_drawdown(returns_df)[['Wealth', 'Peaks']].plot(figsize=(15, 4), title='Wealth & Peaks: ' + returns_df.name)
        get_drawdown(returns_df)[['Drawdown']].plot(figsize=(15, 4), title='Drawdown: ' + returns_df.name)
    else:
        columns = get_portfolio_columns (returns_df)
        
        get_drawdown(returns_df[columns[0]])[['Wealth', 'Peaks']].plot(figsize=(15, 4), title='Wealth & Peaks: ' + columns[0])
        get_drawdown(returns_df[columns[0]])[['Drawdown']].plot(figsize=(15, 4), title='Drawdown: ' + columns[0])
        stocks = columns[1:]
        for i in stocks:
            get_drawdown(returns_df[i])[['Wealth', 'Peaks']].plot(figsize=(15, 4), title='Wealth & Peaks: ' + i)
            get_drawdown(returns_df[i])[['Drawdown']].plot(figsize=(15, 4), title='Drawdown: ' + i)
    
    
def get_returns (dataframe):
    """
    Takes a DataFrame with asset prices for multiple companies,
    separates data per company with date as index, converts asser prices
    into time series of asset returns and concatenates individual DataFrames
    into a single DataFrame with returns on assets for all companies.
    """
    companies = dataframe['CompanyId'].unique()
    
    returns = pd.DataFrame(columns = ['Date', 'CompanyId', 'CompanyName', 'Open', 'High', 'Close'])
    
    for company in companies:
        df = get_returns_per_company (company, dataframe)
        returns = pd.concat([returns, df])
    
    return returns


def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1   
    
    
def get_stats_per_company (df, company_id, metric, start_date = '', end_date = ''):
    """
    Calculates key metrics per returns time series for a single company at a time.
    
    Returns a DataFrame with key metrics for a company for a specified time period.
    """
    if start_date != '' and end_date != '':
        df = df.where((df['Date'] >= start_date) & (df['Date'] <= end_date)).dropna()
    
    elif start_date != '' and end_date == '':
        df = df.where(df['Date'] >= start_date).dropna()
    
    elif start_date == '' and end_date != '':
        df = df.where(df['Date'] <= end_date).dropna()
         
    else:
        df = df.where(df['CompanyId']==company_id).dropna()
      
    df = df.where(df['CompanyId']==company_id).dropna()
    df = df.reset_index()
    
    if df.shape[0] > 0:
        company_id = df['CompanyId'][0]
        company_name = df['CompanyName'][0]
        
        period_start = df['YearMonth'].min()
        period_end = df['YearMonth'].max()
        months_in_period = df['YearMonth'].nunique()

        mean = round(df[metric].mean(), 3)
        median = round(df[metric].median(), 3)  
        skew = round(scipy.stats.skew(df[metric]), 3)
        kurtosis = round(scipy.stats.kurtosis(df[metric]), 3)    
        volatility = round(df[metric].std(ddof=0), 3)
        semideviation = round(df[metric][df[metric]<0].std(ddof=0), 3)
        annualized_volatility = round(volatility*np.sqrt(12), 3)
        return_per_month = round((df[metric]+1).prod()**(1/months_in_period) - 1, 3)
        annualized_return = round((return_per_month+1)**12 - 1, 3)
        
        return pd.DataFrame({
                 "CompanyId" : company_id,
                 "CompanyName" : company_name,
                 "PeriodStart" : period_start,
                 "PeriodEnd" : period_end,
                 "MonthsInPeriod" : months_in_period,
                 "Mean" : mean,
                 "Median" : median,
                 "Skew" : skew,
                 "Kurtosis" : kurtosis,
                 "Volatility" : volatility,
                 "Semideviation" : semideviation,
                 "AnnualizedVolatility" : annualized_volatility,
                 "ReturnPerMonth" : return_per_month,
                 "AnnualizedReturn" : annualized_return
             }, index=[0])
             
             

def get_stats (input_df, metric, start_date = '', end_date = ''):
    """
    Concatenates DataFrames with key metrics generated by get_stats_per_company.
    
    Returns a DataFrame with key metrics for multiple companies.
    """
    companies = input_df['CompanyId'].unique()
    
    stats = pd.DataFrame(columns = ['CompanyId', 'CompanyName', 'PeriodStart', 'PeriodEnd', 'MonthsInPeriod', 'Mean', 'Median', 'Skew', 'Kurtosis', 'Volatility', 'Semideviation', 'AnnualizedVolatility', 'ReturnPerMonth', 'AnnualizedReturn'])
    
    for company in companies:
    
        df = get_stats_per_company (input_df, company, metric, start_date, end_date)
        stats = pd.concat([stats, df])
        
    stats = stats.set_index('CompanyId')
    
    return stats
    
    
    
def compare_returns (companies, input_df, metric, start_date, end_date):
    """
    Joins returns time series for multiple companies.
    
    Returns a DataFrame with time series returns for multiple companies.
    """
    start_df = pd.DataFrame(input_df['Date'].where((input_df['Date'] >= start_date) & (input_df['Date'] <= end_date)).dropna().unique(), columns = ['Date'])

    for company in companies:
        df = get_returns_single_metric (company, input_df, start_date, end_date, metric)
        start_df = pd.merge(start_df,df, on='Date', how='left')
        
    output_df = start_df.sort_values(by=['Date'])
    output_df = output_df.set_index('Date')
    output_df = output_df.dropna()
    
    return output_df
    
    
def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns
    
    
def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5
    
    
def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x


def optimal_weights(n_points, er, cov):
    """
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights   



def get_msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x
  
  
def get_gmv (portfolios_df, output_format):
    
    top_portfolio = portfolios_df[portfolios_df['Volatility'] == min(portfolios_df['Volatility'])]
    top_portfolio = top_portfolio[top_portfolio['Returns'] == max(top_portfolio['Returns'])]
    
    if output_format == "dataframe":
        return top_portfolio
    
    elif output_format == "piechart":
        columns = top_portfolio.columns.tolist()
        columns.remove('Volatility')
        columns.remove('Returns')
        
        portfolio_volatility = top_portfolio['Volatility'].iloc[0]
        portfolio_return = top_portfolio['Returns'].iloc[0]
        
        split = top_portfolio[columns].T
        split.columns = ['Portfolio Split']
        return split.plot.pie(y='Portfolio Split', title='Portfolio volatility: ' + str(portfolio_volatility) + ' | Portfolio return: ' + str(portfolio_return), figsize=(10, 10)).legend(bbox_to_anchor=(0.95, 0.85))
  
  
def get_portfolio_details(n_points, er, cov, riskfree_rate, output_format="plot", show_cml=False, show_ew=False, show_gmv=False):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    
    df_output = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols,
        "Weights": weights
        })
    
    df_weights = pd.DataFrame(df_output['Weights'].to_list(), columns = cov.columns.tolist())
    df_output = pd.concat([df_output, df_weights], axis=1) 
    df_output = df_output.drop('Weights', axis=1)
    df_output = df_output.round(decimals = 3)
    df_output.columns = ['Returns', 'Volatility'] + cov.columns.tolist()
    
    if output_format == "plot":    
        df = pd.DataFrame({
            "Returns": rets,    
            "Volatility": vols
        })
        ax = df.plot.line(x="Volatility", y="Returns", style='.-', legend=False, figsize=(20, 10))
        
        if show_cml:
            ax.set_xlim(left = 0)
            # get MSR
            w_msr = get_msr(riskfree_rate, er, cov)
            r_msr = portfolio_return(w_msr, er)
            vol_msr = portfolio_vol(w_msr, cov)
            # add CML
            cml_x = [0, vol_msr]
            cml_y = [riskfree_rate, r_msr]
            ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10, label='Capital Market Line')
            
            xmean = sum(i for i in cml_x) / float(len(cml_x))
            ymean = sum(i for i in cml_y) / float(len(cml_y))
            
            ax.annotate('Capital Market Line', xy=(xmean,ymean), xycoords='data', ha="center")
            
        if show_ew:
            n = er.shape[0]
            w_ew = np.repeat(1/n, n)
            r_ew = portfolio_return(w_ew, er)
            vol_ew = portfolio_vol(w_ew, cov)
            # add EW
            ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
            ax.text(vol_ew, r_ew, 'Volatility: ' + str(round(vol_ew, 3)) + '; Return: ' + str(round(r_ew, 3)) + ' (Equal Weight)')
            
        if show_gmv:          
            gmv = get_gmv (df_output, output_format="dataframe")
            vol_gmv = gmv['Volatility'].iloc[0]
            r_gmv = gmv['Returns'].iloc[0]
            # add EW
            ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
            ax.text(vol_gmv, r_gmv, 'Volatility: ' + str(round(vol_gmv, 3)) + '; Return: ' + str(round(r_gmv, 3)) + ' (Global Minimum Volatility)')
            
        return ax
        
    elif output_format == "dataframe":
        return df_output
    

def get_portfolio_columns (portfolio_df):
    columns = portfolio_df.columns.tolist()
    try:
        columns.remove('Volatility')
    except: pass
    try:
        columns.remove('Returns')
    except: pass
    return columns
    
    
def get_portfolio_weights (portfolio_df):
    columns = portfolio_df.columns.tolist()
    columns.remove('Volatility')
    columns.remove('Returns')
    weights = portfolio_df[columns].values.tolist()[0]
    return weights
    
    
def plot_portfolio_value (portfolio_df, returns, returns_type="weighed", rolling_window=31):

    columns = get_portfolio_columns (portfolio_df)
    weights = get_portfolio_weights (portfolio_df)
    
    weighed_returns = returns * weights
    
    if returns_type == "raw":
        returns_df = returns
    elif returns_type == "weighed":
        returns_df = weighed_returns
    elif returns_type == "total":
        returns_df = weighed_returns.sum(axis=1)
        
    if returns_type == "raw" or returns_type == "weighed":
        ax = get_drawdown (returns_df[columns[0]])['Wealth'].plot.line(figsize=(20, 10))
        companies = columns[1:]
        for i in companies:
            get_drawdown (returns_df[i])['Wealth'].plot.line()
        output = ax.legend(columns) 
    elif returns_type == "total":
        output = get_drawdown (returns_df)['Wealth'].plot.line(legend=True, label="Portfolio returns", figsize=(20, 10))
        get_drawdown (returns_df)['Wealth'].rolling(window=rolling_window).mean().plot(legend=True, label="Portfolio returns - rolling average", figsize=(20, 10))
        
    return output
    
    
def get_portfolio_returns (portfolio_df, returns, returns_type="weighed"):

    columns = get_portfolio_columns (portfolio_df)
    weights = get_portfolio_weights (portfolio_df)
    
    weighed_returns = returns * weights
    
    if returns_type == "raw":
        returns_df = returns
    elif returns_type == "weighed":
        returns_df = weighed_returns
    elif returns_type == "total":
        returns_df = weighed_returns.sum(axis=1)
        
    return returns_df
    
    
    
def plot_portfolio_returns_and_correlations (portfolio_df, returns, rolling_window=30):
    
    total_portfolio_returns = get_portfolio_returns (portfolio_df, returns, returns_type="total")
    raw_portfolio_returns = get_portfolio_returns (portfolio_df, returns, returns_type="raw")
    
    correlations = raw_portfolio_returns.rolling(window=rolling_window).corr().groupby(level='Date').apply(lambda cormat: cormat.values.mean())
    returns = total_portfolio_returns.rolling(window=rolling_window).aggregate(annualize_rets, periods_per_year=12)
    
    ax = returns.plot(secondary_y=True, legend=True, label="Rolling average returns", figsize=(20, 10))
    correlations.plot(legend=True, label="Rolling average correlations")
    
    return ax
    
    
def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = account_value
    if isinstance(risky_r, pd.Series): 
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 # fast way to set all values to a number
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        # recompute the new account value at the end of this step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history
    }
    return backtest_result
    
    
def plot_cppi_wealth (risky_returns, safe_returns=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=0.25):
    output = run_cppi(risky_returns, safe_r=None, m=3, start=start, floor=floor, riskfree_rate=riskfree_rate, drawdown=drawdown)
    ax = output['Wealth'].rename(columns={'R': 'Wealth'}).plot(figsize=(20, 10), legend=True)
    output['Risky Wealth'].rename(columns={'R': 'Risky Wealth'}).plot(ax=ax, style="k--", figsize=(20, 10), legend=True)
    return ax
import pandas as pd
import numpy as np
import scipy.stats
from scipy.optimize import minimize

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
  
  
def plot_ef(n_points, er, cov):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style='.-', legend=False, figsize=(20, 10))
    
    
def get_portfolio_details(n_points, er, cov):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols,
        "Weights": weights
    })
    return ef
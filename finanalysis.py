import pandas as pd
import numpy as np
import scipy.stats
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import statistics
import datetime as dt

class ReturnsAnalyzer:
    def __init__(self, dataframe, metric, start_date = '', end_date = ''):
        self.dataframe = dataframe
        self.metric = metric
        self.start_date = start_date
        self.end_date = end_date
        
    def get_returns_per_company (self, company_id):
        """
        Applies the pct_change() method Open, High, and Close Series per single CompanyOd
        converting each datapoint into a percentage change in relation to its preceding datepoint.
        
        Returns a time series of asset returns DataFrame for 'Open', 'High', 'Close' data.
        """
        filtered_company = (self.dataframe
                                .where(self.dataframe['CompanyId']==company_id)
                                .dropna()
                                .sort_values(by=['Date']))
        
        filtered_company = filtered_company.set_index('Date')
        
        filtered_company['YearMonth'] = pd.to_datetime(filtered_company.index).to_period('M')
        
        filtered_company['Open'] = filtered_company['Open'].pct_change()
        filtered_company['High'] = filtered_company['High'].pct_change()
        filtered_company['Close'] = filtered_company['Close'].pct_change()
        
        returns_per_company = (filtered_company[['CompanyId', 'CompanyName', 'Open', 'High', 'Close', 'YearMonth']]
                    .dropna())
                    
        returns_per_company = returns_per_company.reset_index()
        
        return returns_per_company

        
    def get_drawdown (self, return_series):
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

    def plot_wealth_peaks_drawdown (self, returns_df):
        if len(returns_df.shape) == 1:
            self.get_drawdown(returns_df)[['Wealth', 'Peaks']].plot(figsize=(15, 4), title='Wealth & Peaks: ' + returns_df.name)
            self.get_drawdown(returns_df)[['Drawdown']].plot(figsize=(15, 4), title='Drawdown: ' + returns_df.name)
        else:
            columns = returns_df.columns.tolist()
            
            self.get_drawdown(returns_df[columns[0]])[['Wealth', 'Peaks']].plot(figsize=(15, 4), title='Wealth & Peaks: ' + columns[0])
            self.get_drawdown(returns_df[columns[0]])[['Drawdown']].plot(figsize=(15, 4), title='Drawdown: ' + columns[0])
            stocks = columns[1:]
            for i in stocks:
                self.get_drawdown(returns_df[i])[['Wealth', 'Peaks']].plot(figsize=(15, 4), title='Wealth & Peaks: ' + i)
                self.get_drawdown(returns_df[i])[['Drawdown']].plot(figsize=(15, 4), title='Drawdown: ' + i)
            
        
    def get_returns (self):
        """
        Takes a DataFrame with asset prices for multiple companies,
        separates data per company with date as index, converts asser prices
        into time series of asset returns and concatenates individual DataFrames
        into a single DataFrame with returns on assets for all companies.
        """
        companies = self.dataframe['CompanyId'].unique()
        
        returns = pd.DataFrame(columns = ['Date', 'CompanyId', 'CompanyName', 'Open', 'High', 'Close'])
        
        for company in companies:
            df = self.get_returns_per_company (company)
            returns = pd.concat([returns, df])
        
        return returns
        
    def get_stats_per_company (self, df, company_id):
        """
        Calculates key metrics per returns time series for a single company at a time.
        
        Returns a DataFrame with key metrics for a company for a specified time period.
        """
        if self.start_date != '' and self.end_date != '':
            df = df.where((df['Date'] >= self.start_date) & (df['Date'] <= self.end_date)).dropna()
        
        elif self.start_date != '' and self.end_date == '':
            df = df.where(df['Date'] >= self.start_date).dropna()
        
        elif self.start_date == '' and self.end_date != '':
            df = df.where(df['Date'] <= self.end_date).dropna()
            
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
    
            mean = round(df[self.metric].mean(), 3)
            median = round(df[self.metric].median(), 3)  
            skew = round(scipy.stats.skew(df[self.metric]), 3)
            kurtosis = round(scipy.stats.kurtosis(df[self.metric]), 3)    
            volatility = round(df[self.metric].std(ddof=0), 3)
            semideviation = round(df[self.metric][df[self.metric]<0].std(ddof=0), 3)
            annualized_volatility = round(volatility*np.sqrt(12), 3)
            return_per_month = round((df[self.metric]+1).prod()**(1/months_in_period) - 1, 3)
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

    def get_stats (self, returns_df):
        """
        Concatenates DataFrames with key metrics generated by get_stats_per_company.
        
        Returns a DataFrame with key metrics for multiple companies.
        """
        companies = returns_df['CompanyId'].unique()
        
        stats = pd.DataFrame(columns = ['CompanyId', 'CompanyName', 'PeriodStart', 'PeriodEnd', 'MonthsInPeriod', 'Mean', 'Median', 'Skew', 'Kurtosis', 'Volatility', 'Semideviation', 'AnnualizedVolatility', 'ReturnPerMonth', 'AnnualizedReturn'])
        
        for company in companies:
        
            df = self.get_stats_per_company (returns_df, company)
            stats = pd.concat([stats, df])
            
        stats = stats.set_index('CompanyId')
        
        return stats

    def get_returns_single_metric (self, company_id):
        """
        Filters time series of asset returns DataFrame by date and pivots the data.
        
        Returns a Series with date as an index, Cpmpany Name as a column name,
        and percent change as data.
        """
        df = self.get_returns_per_company (company_id)
        df = df.where((df['Date'] >= self.start_date) & (df['Date'] <= self.end_date)).dropna()
        df = df.set_index('Date')
        df = df[['CompanyName', self.metric]]
        df = df.rename(columns={self.metric: df['CompanyName'][0]})
        df = df[df['CompanyName'][0]]
        
        return df
     
    def compare_returns (self, companies):
        """
        Joins returns time series for multiple companies.
        
        Returns a DataFrame with time series returns for multiple companies.
        """
        start_df = pd.DataFrame(self.dataframe['Date'].where((self.dataframe['Date'] >= self.start_date) & (self.dataframe['Date'] <= self.end_date)).dropna().unique(), columns = ['Date'])
    
        for company in companies:
            df = self.get_returns_single_metric (company)
            start_df = pd.merge(start_df, df, on='Date', how='left')
            
        output_df = start_df.sort_values(by=['Date'])
        output_df = output_df.set_index('Date')
        output_df = output_df.dropna()
        
        return output_df
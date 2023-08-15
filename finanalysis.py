import pandas as pd
import numpy as np
import scipy.stats
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import statistics
import datetime as dt

class ReturnsAnalyzer:
    def __init__(self, 
                 dataframe,
                 metric, 
                 start_date = '', 
                 end_date = ''):
        """
        Initializes a ReturnsAnalyzer instance.
        
        Parameters:
        - dataframe: DataFrame containing asset price data for multiple companies.
        - metric: The metric to analyze (e.g., 'Open', 'High', 'Close').
        - start_date: Optional start date for data filtering.
        - end_date: Optional end date for data filtering.
        """
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
        """
        Plots wealth index, peaks, and drawdown for a returns DataFrame.
        """
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

            
            
class PortfolioAnalyzer:
    def __init__(self, 
                 returns_object, 
                 portfolio_size, 
                 n_points, 
                 riskfree_rate, 
                 show_cml=True, 
                 show_ew=True, 
                 show_gmv=True):
        """
        Initializes a PortfolioAnalyzer instance.

        Parameters:
        - returns_object: An instance of the ReturnsAnalyzer class containing asset return data.
        - portfolio_size: Number of top assets to consider in the portfolio.
        - n_points: Number of points on the efficient frontier.
        - riskfree_rate: Risk-free rate for calculating Sharpe ratio.
        - show_cml: Whether to show Capital Market Line in plots.
        - show_ew: Whether to show Equal Weight portfolio in plots.
        - show_gmv: Whether to show Global Minimum Volatility portfolio in plots.
        """
        self.returns_object = returns_object
        self.portfolio_size = portfolio_size
        self.n_points = n_points
        self.riskfree_rate = riskfree_rate
        self.returns = returns_object.get_returns()
        self.annualized_rets = returns_object.get_stats(self.returns).sort_values(by=['MonthsInPeriod', 'ReturnPerMonth', 'AnnualizedReturn'], ascending=False)['AnnualizedReturn'][0:self.portfolio_size]
        self.stocks = self.annualized_rets.index.tolist()
        self.risky_returns = returns_object.compare_returns(self.stocks)
        self.cov_matrix = self.risky_returns.cov()
        self.show_cml = show_cml
        self.show_ew = show_ew
        self.show_gmv = show_gmv

    def portfolio_return(self, weights):
        """
        Computes the return on a portfolio from constituent returns and weights
        weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
        """
        return weights.T @ self.annualized_rets

    def portfolio_vol(self, weights):
        """
        Computes the vol of a portfolio from a covariance matrix and constituent weights
        weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
        """
        return (weights.T @ self.cov_matrix @ weights)**0.5

    def minimize_vol(self, target_return):
        """
        Returns the optimal weights that achieve the target return
        given a set of expected returns and a covariance matrix
        """
        n = self.annualized_rets.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((0.0, 1.0),) * n
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
        }
        return_is_target = {'type': 'eq',
                            'fun': lambda weights: target_return - self.portfolio_return(weights)
        }
        weights = minimize(fun=self.portfolio_vol, x0=init_guess,
                           method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1, return_is_target),
                           bounds=bounds)
        return weights.x

    def optimal_weights(self):
        """
        Generates optimal portfolio weights for various target returns.
        Returns a list of optimal portfolio weights.
        """
        target_rs = np.linspace(self.annualized_rets.min(), self.annualized_rets.max(), self.n_points)
        weights = [self.minimize_vol(target_return) for target_return in target_rs]
        return weights

    def get_msr(self):
        """
        Returns the weights of the portfolio that gives you the maximum sharpe ratio
        given the riskfree rate and expected returns and a covariance matrix
        """
        n = self.annualized_rets.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
        }
        def neg_sharpe(weights):
            """
            Returns the negative of the sharpe ratio
            of the given portfolio
            """
            r = self.portfolio_return(weights)
            vol = self.portfolio_vol(weights)
            return -(r - self.riskfree_rate)/vol
        
        weights = minimize(fun=neg_sharpe, x0=init_guess,
                        method='SLSQP',
                        options={'disp': False},
                        constraints=(weights_sum_to_1,),
                        bounds=bounds)
        return weights.x

    def get_gmv (self, portfolios_df, gmv_output_format):
        """
        Retrieves the portfolio details for the Global Minimum Volatility portfolio.

        Parameters:
        - portfolios_df: DataFrame containing portfolio details.
        - gmv_output_format: Output format ("dataframe" or "piechart").

        Returns either a DataFrame or a pie chart representation of the GMV portfolio.
        """       
        top_portfolio = portfolios_df[portfolios_df['Volatility'] == min(portfolios_df['Volatility'])]
        top_portfolio = top_portfolio[top_portfolio['Returns'] == max(top_portfolio['Returns'])]
        
        if gmv_output_format == "dataframe":
            return top_portfolio
        
        elif gmv_output_format == "piechart":
            columns = top_portfolio.columns.tolist()
            columns.remove('Volatility')
            columns.remove('Returns')
            
            portfolio_volatility = top_portfolio['Volatility'].iloc[0]
            portfolio_return = top_portfolio['Returns'].iloc[0]
            
            split = top_portfolio[columns].T
            split.columns = ['Portfolio Split']
            return split.plot.pie(y='Portfolio Split', title='Portfolio volatility: ' + str(portfolio_volatility) + ' | Portfolio return: ' + str(portfolio_return), figsize=(10, 10)).legend(bbox_to_anchor=(0.95, 0.85))


    def get_portfolio_details(self, output_format):
        """
        Generates and plots the multi-asset efficient frontier.

        Parameters:
        - output_format: Output format ("plot" or "dataframe").

        Returns either a plot or a DataFrame with efficient frontier details.
        """
        weights = self.optimal_weights()
        rets = [self.portfolio_return(w) for w in weights]
        vols = [self.portfolio_vol(w) for w in weights]

        df_output = pd.DataFrame({
            "Returns": rets, 
            "Volatility": vols,
            "Weights": weights
            })
        
        df_weights = pd.DataFrame(df_output['Weights'].to_list(), columns = self.cov_matrix.columns.tolist())
        df_output = pd.concat([df_output, df_weights], axis=1) 
        df_output = df_output.drop('Weights', axis=1)
        df_output = df_output.round(decimals = 3)
        df_output.columns = ['Returns', 'Volatility'] + self.cov_matrix.columns.tolist()

        if output_format == "plot":    
            df = pd.DataFrame({
                "Returns": rets,    
                "Volatility": vols
            })
            ax = df.plot.line(x="Volatility", y="Returns", style='.-', legend=False, figsize=(20, 10))
            
            if self.show_cml:
                ax.set_xlim(left = 0)
                # get MSR
                w_msr = self.get_msr()
                r_msr = self.portfolio_return(w_msr)
                vol_msr = self.portfolio_vol(w_msr)
                # add CML
                cml_x = [0, vol_msr]
                cml_y = [self.riskfree_rate, r_msr]
                ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10, label='Capital Market Line')
                
                xmean = sum(i for i in cml_x) / float(len(cml_x))
                ymean = sum(i for i in cml_y) / float(len(cml_y))
                
                ax.annotate('Capital Market Line', xy=(xmean,ymean), xycoords='data', ha="center")
                
            if self.show_ew:
                n = self.annualized_rets.shape[0]
                w_ew = np.repeat(1/n, n)
                r_ew = self.portfolio_return(w_ew)
                vol_ew = self.portfolio_vol(w_ew)
                # add EW
                ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
                ax.text(vol_ew, r_ew, 'Volatility: ' + str(round(vol_ew, 3)) + '; Return: ' + str(round(r_ew, 3)) + ' (Equal Weight)')
                
            if self.show_gmv:          
                gmv = self.get_gmv(df_output, gmv_output_format = "dataframe")
                vol_gmv = gmv['Volatility'].iloc[0]
                r_gmv = gmv['Returns'].iloc[0]
                # add EW
                ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
                ax.text(vol_gmv, r_gmv, 'Volatility: ' + str(round(vol_gmv, 3)) + '; Return: ' + str(round(r_gmv, 3)) + ' (Global Minimum Volatility)')
                
            return ax
            
        elif output_format == "dataframe":
            return df_output

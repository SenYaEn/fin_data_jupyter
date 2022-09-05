import pandas as pd

def get_returns_open_high_close (company_id, df_name):

    filtered_company = (df_name
                        .where(df_name['CompanyId']==company_id)
                        .dropna()
                        .sort_values(by=['Date']))
    
    filtered_company = filtered_company.set_index('Date')
    
    filtered_company['open'] = filtered_company['Open'].pct_change()
    filtered_company['high'] = filtered_company['High'].pct_change()
    filtered_company['close'] = filtered_company['Close'].pct_change()
    
    returns = (filtered_company[['CompanyId', 'CompanyName', 'open', 'high', 'close']]
                   .dropna())
    
    return returns
    
    
def get_returns_single_metric (company_id, df_name, start_date, end_date, metric):

    returns = get_returns_open_high_close (company_id, df_name)[['CompanyName', metric]]
    returns = returns.loc[start_date : end_date]
    returns = returns.rename(columns={metric: returns['CompanyName'][0]})
    returns = returns[returns['CompanyName'][0]]
    
    return returns
    
    
def compare_returns (company_id_1, company_id_2, df_name, start_date, end_date, metric):

    set_1 = get_returns_single_metric (company_id_1, df_name, start_date, end_date, metric)
    set_2 = get_returns_single_metric (company_id_2, df_name, start_date, end_date, metric)
    output = pd.merge(set_1, set_2, left_index=True, right_index=True)
    
    return output
    
    
def get_drawdown (return_series):
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth" : wealth_index,
        "Peaks" : previous_peaks,
        "Drawdown" : drawdown
    })
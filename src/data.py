import streamlit as st
import psycopg2
from psycopg2 import sql
import pandas as pd
import numpy as np
import copy
import datetime
from src import api


@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None, 'builtins.weakref': lambda _: None})
def init_connection():
    return psycopg2.connect(**st.secrets["gcp"])


@st.cache(ttl=600,allow_output_mutation=True, hash_funcs={psycopg2.extensions.connection: lambda _: None})
def run_query(query, conn):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()


def get_filters():
    path = './data/filter.csv'
    filter_pd = pd.read_csv(path)

    return filter_pd

@st.cache(allow_output_mutation=True, hash_funcs={psycopg2.extensions.connection: lambda _: None})
def initialise_data(asset_class, exchange, conn):
    results = pd.read_sql(api.initialise(asset_class, exchange), con=conn)
    
    return results


@st.cache(allow_output_mutation=True, hash_funcs={psycopg2.extensions.connection: lambda _: None})
def search(text, conn):
    txt = text.split()
    search_str = []

    if len(txt) > 2:
        for i in range(len(txt), 0, -1):
            for j in range(0, len(txt) - i + 1):
                search_str += [' '.join(txt[j:j+i])]
    else:
        search_str = [st.session_state.search.lower()]

    results = pd.read_sql(api.search(search_str), con=conn)
    return results


def get_etfs_lst(conn, exchange=None):
    etfs = pd.read_sql(api.get_etfs_list(exchange), con = conn)
    return etfs.to_records()

@st.cache(allow_output_mutation=True, hash_funcs={psycopg2.extensions.connection: lambda _: None})
def get_etfs_data(isins, exchange ,conn):
    query_overview = api.get_by_isin(isins, "gcp_funds", ["ISINCode", "Ticker", "ExchangeTicker", "FundName", "Exchange", "TradeCurrency", "Price", "1DChange",
            "Volume3M", "FundCurrency", "NAV", "NAV_1DChange", "AUM","AUM_USD", "AUM_EUR", "AUM_GBP",
            "DistributionIndicator", "TotalExpenseRatio", "ETPIssuerName", "IndexName", "Rank5"])

    query_overview += """ AND "Exchange" = '""" + exchange + """' """
    data_overview = pd.read_sql(query_overview, con = conn)

    query_perf = api.get_by_isin(isins, "calc_return_period", ["ISINCode", "Type", "Description", "Return"])
    data_perf = pd.read_sql(query_perf, con = conn)

    data_fundflow = pd.read_sql(api.get_fundflows(isins), con = conn)

    data_div = pd.read_sql(api.get_div(isins), con = conn)
    data_div['exDivDate'] = pd.to_datetime(data_div['exDivDate'], utc=True).apply(lambda x: str(x.date()))

    return data_overview, data_perf, data_fundflow, data_div


def table_format(data, display, currency, return_type=None):
    names = data['overview'][['ISINCode','ExchangeTicker','FundName','Exchange']]   

    output_data = pd.DataFrame()
    if display == 'Overview':
        if currency == 'USD':
            currency_col = 'AUM_USD'
        elif currency == 'GBP':
            currency_col = 'AUM_GBP'
        elif currency == 'EUR':
            currency_col = 'AUM_EUR'
        else:
            currency_col = 'AUM'

        col_names = ['ISINCode','ExchangeTicker', 'FundName', 'Exchange', 'TradeCurrency', 'Price', '1DChange', 'Volume3M', 'FundCurrency', 'NAV', 'NAV_1DChange',\
              currency_col, 'DistributionIndicator', 'TotalExpenseRatio', 'ETPIssuerName', 'IndexName', 'Rank5' ]
        output_data = data['overview'][col_names]
        output_data = output_data.rename(columns = {'1DChange': 'Price_1DChange', currency_col: 'AUM'})


    elif display == 'Performance':
        des = return_type
        des = 'Calendar' if return_type == 'Calendar Year' else des
        output_data = data['performance']
        output_data = output_data[output_data['Type'] == des]
        output_data = output_data.pivot(index='ISINCode', columns='Description', values='Return').reset_index()
        if return_type == 'Calendar Year':
            year_rename = 'Year' + output_data.columns
            output_data.columns = year_rename
            output_data = output_data[year_rename[::-1]]
            output_data = output_data.rename(columns={'YearISINCode': 'ISINCode', 'YearYTD': 'YTD'})
        output_data = names.merge(output_data, how='left', on='ISINCode')

    

    elif display == 'Fund Flow':
        if currency == 'USD':
            flow_col = 'flow_USD'
            aum_col = 'AUM_USD'
        elif currency == 'EUR':
            flow_col = 'flow_EUR'
            aum_col = 'AUM_EUR'
        elif currency == 'GBP':
            flow_col = 'flow_GBP'
            aum_col = 'AUM_GBP'
        else:
            flow_col = 'flow_local'
            aum_col = 'AUM'

        col_names = ["ISINCode", "Description", "Currency", aum_col, flow_col]
        output_data = data['flow'][col_names]
        output_data = output_data.rename(columns={aum_col: 'AUM',flow_col: 'Flow'})
        output_flow = output_data.pivot(index='ISINCode', columns='Description', values='Flow').reset_index()
        output_data = output_flow.merge(output_data[['ISINCode', 'Currency', 'AUM']].drop_duplicates(), how='left', on='ISINCode')
        output_data = names.merge(output_data, how='left', on='ISINCode')


    elif display == 'Income':
        output_data = data['div']
        output_data['exDivDate'] = output_data['exDivDate'].apply(lambda x: '' if x=='NaT' else x)
        output_data = names.merge(output_data, how='left', on='ISINCode')

    else: 
        return

    output_data = output_data.fillna('')   #fill NA value as '' in fees column, otherwise react has json error
    output_data.insert(loc=0, column='id', value = np.arange(0, len(output_data)))

    return output_data

@st.cache
def num_format(num, currency=None, digits=1):
    num_abs = abs(num)
    sign = '-' if num<0 else ''
    output = ''
    if num_abs > 999999999:
        output = str(round(num_abs / 1000000000,digits)) +'B'
    elif num_abs > 999999:
        output = str(round(num_abs / 1000000,digits)) +'M'
    elif num_abs > 999:
        output = str(round(num_abs / 1000,digits)) +'K'
    else:
        output = str(round(num_abs,digits))

    if currency is not None:
        output = currency + (' ' if len(currency) == 3 else '') + output
    return sign + output

@st.cache
def rank_format(rank):
    output = ''
    if rank == 5:
        output = '⭐⭐⭐⭐⭐'
    elif rank ==4:
        output = '⭐⭐⭐⭐'
    elif rank ==3:
        output = '⭐⭐⭐'
    elif rank ==2:
        output = '⭐⭐'
    else:
        output = '⭐'
    return output


@st.cache
def currency_revert(currency):
    return 'USD' if currency == '$' else 'EUR' if currency == '€' else 'GBP' if currency == '£' else currency

@st.cache
def currency_format(currency):
    return '$' if currency == 'USD' else '€' if currency == 'EUR' else '£' if currency == 'GBP' else 'GBp ' if currency == 'GBX' else str(currency) + ' '


@st.cache(allow_output_mutation=True, hash_funcs={psycopg2.extensions.connection: lambda _: None})
def get_exchanges(conn):
    return pd.read_sql(api.get_exchanges(), con=conn).to_records()


#@st.cache Cannot use cache here
def get_overview(ticker, overview_pd, conn):
    # if ticker doesn't exist in display_pd, need to extract from db
    info = {}
    data_overview = overview_pd[overview_pd['ExchangeTicker'] == ticker]
    if len(data_overview)>0:
        data_overview = data_overview.to_dict(orient='records')[0]
    else:
        data_overview = pd.read_sql(api.get_by_ticker([ticker], 'gcp_funds', ["ISINCode", "Ticker", "ExchangeTicker", "FundName",\
                             "Exchange", "TradeCurrency", "Price", "Volume3M", "FundCurrency", "NAV", "AUM",
                            "DistributionIndicator", "TotalExpenseRatio", "IndexName", "Rank5"]), con=conn)
        data_overview = data_overview.to_dict(orient='records')[0]
        

    isin = data_overview['ISINCode']
    info['ISINCode'] = isin


    info['ExchangeTicker'] = data_overview['ExchangeTicker']
    info['Name'] = data_overview['FundName']
    info['Exchange'] = data_overview['Exchange']
    info['ExchangePrice'] = data_overview['TradeCurrency'] + (' ' if len(data_overview['TradeCurrency']) ==3 else '') + str(round(data_overview['Price'],2))
    info['TradingCurrency'] = currency_revert(data_overview['TradeCurrency'])
    info['Volume'] = num_format(data_overview['Volume3M'])
    info['NAV'] = data_overview['FundCurrency'] + (' ' if len(data_overview['FundCurrency']) ==3 else '') + str(round(data_overview['NAV'],2))
    info['FundCurrency'] = currency_revert(data_overview['FundCurrency'])

    info['AUM'] = data_overview['FundCurrency'] + (' ' if len(data_overview['FundCurrency']) ==3 else '') + num_format(data_overview['AUM'])
    info['Cost'] = data_overview['TotalExpenseRatio']
    info['Rank'] = rank_format(data_overview['Rank5'])
    info['IndexName'] = data_overview['IndexName']
    info['Distribution'] = data_overview['DistributionIndicator']

    return info

def get_return(isin, perf_pd, conn):
    perf = perf_pd[perf_pd['ISINCode'] == isin]
    if len(perf) >0 :
        perf = perf[perf['Type'] == 'Cumulative']
    else:
        perf = pd.read_sql(api.get_by_isin([isin], 'calc_return_period', ['Type', 'Description', 'Return']), con=conn)
        perf = perf[perf['Type'] == 'Cumulative']
    perf_dict = perf[['Description', 'Return']].set_index('Description').to_dict()
    return perf_dict['Return']


@st.cache(allow_output_mutation=True, hash_funcs={psycopg2.extensions.connection: lambda _: None})
def get_etf_overview(ticker, data_dict, conn):

    etf_info = get_overview(ticker, data_dict['overview'], conn)
    #etf_info = copy.deepcopy(overview)
    isin = etf_info['ISINCode']
        
    
    etf_info['Return'] = get_return(isin, data_dict['performance'], conn)
    etf_info['NAV_1MChange'] = round(etf_info['Return']['1M'],2)


    if '3Y' in etf_info['Return']:
        y3 = etf_info['Return']['3Y']
        etf_info['NAV_3YChange_Cum'] = str(round(y3,2))+'%'
        etf_info['NAV_3YChange_Ann'] = str(round(np.power(y3+1,1/3)-1,2))+'%'
    else:

        etf_info['NAV_3YChange_Cum'] = '-'
        etf_info['NAV_3YChange_Ann'] = '-'


    funds = pd.read_sql(api.get_by_isin([isin], 'gcp_funds', ['DateLatest','Objective', 'AssetClass', 'Sector'], 1), con=conn)
    funds = funds.to_dict(orient='records')[0]
    etf_info['DateLatest'] = pd.to_datetime(funds['DateLatest']).strftime('%Y-%m-%d')
    etf_info['Objective'] = funds['Objective']
    etf_info['AssetClass'] = funds['AssetClass']
    etf_info['Sector'] = funds['Sector']

    rank = pd.read_sql(api.get_by_isin([isin], 'calc_rank', ['CostRank', 'ReturnRank', 'AUMRank', 'TrackingErrorRank', 'VolumeRank', 'Rank', 
                        'sector_CostRank', 'sector_ReturnRank', 'sector_AUMRank', 'sector_TrackingErrorRank', 'sector_VolumeRank', 'sectorRank']), con = conn)
    etf_info['Ranks'] = rank.to_dict(orient='records')[0]
    

    flow = pd.read_sql(api.get_by_isin([isin], 'calc_fundflow_period', ['Description','flow_local','flow_USD', 'flow_EUR', 'flow_GBP', 'sectorflow_USD', 'sectorflow_EUR', 'sectorflow_GBP']), con=conn)
    etf_info['Flow'] = flow.set_index('Description').to_dict(orient='dict')
    etf_info['AUM_1MChange'] = num_format(float(etf_info['Flow']['flow_local']['1M']), etf_info['FundCurrency'])

    similar_etfs= pd.read_sql(api.get_similar_etfs([isin]), con = conn)
    similar_etfs = similar_etfs[~similar_etfs['ISINCode'].duplicated()]

    similar_top = pd.read_sql(api.get_1y_perf(similar_etfs['ISINCode'].head(5)), con=conn)

    etf_info['Similar_ETFs'] = similar_etfs.groupby('Description')['ISINCode'].apply(list).to_dict()
    etf_info['Similar_ETFs_top'] = similar_top.drop_duplicates().to_dict(orient='records')


    holding = pd.read_sql(api.get_top5_holding([isin]), con=conn)
    holding = holding.rename(columns={'InstrumentDescription': 'Name'})
    etf_info['Top10'] = holding.to_dict(orient='records')

    navs = pd.read_sql(api.get_navs_weekly([isin]), con = conn)
    navs["TimeStamp"] = pd.to_datetime(navs['TimeStamp'], utc=True)
    navs['Dates'] = navs["TimeStamp"].dt.strftime('%Y-%m-%d')
    return etf_info, navs[["TimeStamp",'Dates', 'NAV']]


def get_etf_details(isin, conn):
    data = pd.read_sql(api.get_details(isin), con = conn)
    data['LaunchDate'] = pd.to_datetime(data['LaunchDate'], utc=True).dt.strftime('%Y-%m-%d')
    if data['exDivDate'] is not None:
        data['exDivDate'] = pd.to_datetime(data['exDivDate'], utc=True).dt.strftime('%Y-%m-%d')
    
    data = data.to_dict(orient='records')[0]
    if data['Dividend'] is not None:
        data['Dividend'] = currency_format(data['Currency']) + str(round(data['Dividend'],2))
    data['UCITS'] = 'Yes' if data['UCITS'] == True else 'No'

    data['Volatility'] = str(round(data['Volatility'],2))+'%' if not (data['Volatility'] is None) else '-'
    data['Drawdown'] = str(round(data['Drawdown'],2))+'%' if not(data['Drawdown'] is None) else '-'
    data['TrackingError3Y'] = str(round(data['TrackingError3Y'],4)) if not(data['TrackingError3Y'] is None) else '-'
    return data

@st.cache
def format_flow(fundflow, currency, flow_period):
    currency_key = 'flow_USD' if currency == 'USD' else 'flow_GBP' if currency == 'GBP' else 'flow_EUR'

    flow = pd.DataFrame(fundflow[currency_key].items())
    flow = flow.rename(columns={0: 'Period', 1: 'Fund Flow'})
    flow['Type'] = 'ETF'

    flow_sector = pd.DataFrame(fundflow['sector'+currency_key].items())
    flow_sector = flow_sector.rename(columns={0: 'Period', 1: 'Fund Flow'})
    flow_sector['Type'] = 'Sector Average'
    flow = pd.concat([flow, flow_sector], axis=0)
    
    flow = flow[flow['Period'].isin(flow_period)]

    return flow



def get_listing(isin, conn):
    data = pd.read_sql(api.get_listings(isin), con = conn)
    #data.insert(loc=0, column='id', value = np.arange(0, len(data)))
    #data['Price'] = data[['TradingCurrency', 'Price']]currency_format(data['TradingCurrency']) + str(round(data['Price'],2))
    data['Volume3M'] = data['Volume3M'].apply(lambda x: num_format(x))
    data['Price'] = data['TradingCurrency'].apply(lambda x: currency_format(x)) + data['Price'].apply(lambda x: str(round(x,2)))
    
    data = data.to_dict(orient='records')
    
    #st.write(data['TradingCurrency'])
    return data


def get_price(isins, conn, start_date=None, end_date=None ,price_type = 'NAV'):
    if price_type == 'Total Return':
        data = pd.read_sql(api.get_tr(isins, start_date, end_date), con=conn) 
        data = data.rename(columns={'TotalReturn': 'Price'})
    else:
        data = pd.read_sql(api.get_nav(isins, start_date, end_date), con=conn) 
        data = data.rename(columns={'NAV': 'Price'})
    data['TimeStamp'] = pd.to_datetime(data['TimeStamp'], utc=True)
    data['Dates'] = data['TimeStamp'].apply(lambda x: x.date())
    return data[['ISINCode', 'Dates', 'Price']]

def get_equity_indices_names(conn):

    names = pd.read_sql(api.get_equity_indices_names(), con=conn)
    return names.to_records()

def get_equity_indices(tickers, conn, start_date=None, end_date=None):
    prices = pd.read_sql(api.get_equity_indices(tickers, start_date, end_date), con=conn)
    prices = prices.rename(columns={'Ticker': 'ISINCode', 'Timestamp':'Dates' ,'Close': 'Price'})
    return prices


def normalise(data, names_pd):
    data_norm = data.pivot(index='Dates', columns='ISINCode', values='Price')
    data_norm = data_norm.fillna(method='ffill')
    data_norm = data_norm.dropna(axis=0, how='any')
    

    data_initial = 100/data_norm[:1]
    data_norm = data_norm * data_initial.values

    data_norm = data_norm.melt(ignore_index=False).reset_index()

    data_norm = data_norm.merge(names_pd, how='left', on='ISINCode')
    data_norm = data_norm.rename(columns={'value': 'Price'})
    #data_norm['Dates'] = data_norm['TimeStamp'].dt.strftime('%Y-%m-%d')
    return data_norm


def calc_date_range(period_select, date_from, date_to, date_min, date_max):
    if period_select != 'Custom period':
        if period_select != 'All':
            year_select = 1 if period_select == '1Y' else 3 if period_select=='3Y' else 5 if period_select=='5Y' else 10 
            date_start = date_max - datetime.timedelta(days=365*year_select)
        else:
            date_start = date_min
        date_end = date_max
    else:
        date_start = pd.to_datetime(date_from).date()
        date_end = pd.to_datetime(date_to).date()

    return date_start, date_end

def add_compare_prices(compare_select, compare_type, original_price, original_name, start_date, end_date, price_type,conn):

    if price_type == 'Total Return':
        original_price = get_price([original_name[0][0]], conn, start_date, end_date , price_type)
    
    if len(compare_select) > 0:
        compare_tickers = [i[1] for i in compare_select]
        compare_names = [(i[1],i[3]) for i in compare_select]
        if compare_type == 'Equity indices':
            compare_prices = get_equity_indices(compare_tickers, conn, start_date, end_date)
        else:
            compare_prices = get_price(compare_tickers, conn, start_date, end_date, price_type)
        
        prices = pd.concat([original_price, compare_prices], axis=0, ignore_index=True)
        names = original_name + compare_names
        names_pd = pd.DataFrame(names, columns = ['ISINCode', 'Name'])
    else:
        prices = original_price
        prices['Name'] = original_name[0][1]
        names_pd = pd.DataFrame(original_name, columns = ['ISINCode', 'Name'])

    return prices, names_pd


def calc_cum_returns(prices, date_end, names_pd):

    periods = ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', '10Y']
    periods_valid = []
    dates = [date_end - datetime.timedelta(30), date_end - datetime.timedelta(91), \
            date_end - datetime.timedelta(182), datetime.date(date_end.year,1,1), \
            date_end - datetime.timedelta(365), date_end - datetime.timedelta(365*3 + 1), \
            date_end - datetime.timedelta(365*5+1), date_end - datetime.timedelta(365*10 + 2)]

    price_last = prices.groupby('ISINCode').last()
    for i, date in enumerate(dates):
        price_beg = prices[(prices['Dates']>=date)&(prices['Dates'] <= (date+datetime.timedelta(4)))].groupby('ISINCode').first()
        
        if len(price_beg[price_beg.notna()]) > 0:
            price_beg = price_beg.rename(columns = {'Dates': 'Dates' + periods[i], 'Price': periods[i]})
            price_last = pd.concat([price_last, price_beg], axis=1)
            periods_valid += [periods[i]]

    return_cum = pd.DataFrame()
    for i in periods_valid:
        return_cum[i] = (price_last['Price'] / price_last[i] - 1 )

    return_cum = names_pd.merge(return_cum.reset_index(), how='left', on='ISINCode')
    return_cum = return_cum.to_dict(orient='records')

    return return_cum, periods_valid

def calc_ann_returns(cum_returns, cum_periods):
    cum_pd = pd.DataFrame(cum_returns)
    periods = ['1Y', '3Y', '5Y', '10Y']
    periods_num = [1, 3, 5, 10]
    periods_valid = [i for i in periods if i in cum_periods]

    if len(periods_valid) > 0:
        return_ann = cum_pd[['ISINCode', 'Name', '1Y']]
        for i, period in enumerate(periods_valid):
            if period == '1Y':
                return_ann[period] = cum_pd[period]
            else:
                return_ann[period] = np.power(cum_pd[period] + 1, 1/periods_num[i]) -1

    else:
        return_ann = None 

    return return_ann, periods_valid


def calc_cal_returns(prices, date_beg, date_end, names_pd):
    years = list(range(date_end.year, date_beg.year-1, -1))
    periods_valid = []
    dates = []
    for i in years:
        dates += [datetime.date(i, 1,1)]

    periods = [str(years[0]) + ' YTD'] + [str(i) for i in years[1:]]
    price_last = prices.groupby('ISINCode').last()
    for i, date in enumerate(dates):
        price_beg = prices[(prices['Dates']>=date)&(prices['Dates'] <= (date+datetime.timedelta(4)))].groupby('ISINCode').first()
        
        if len(price_beg[price_beg.notna()]) > 0:

            price_beg = price_beg.rename(columns = {'Dates': 'Dates' + str(periods[i]), 'Price': periods[i]})
            price_last = pd.concat([price_last, price_beg], axis=1)
            periods_valid += [periods[i]]

    return_ann = pd.DataFrame()
    for i in periods_valid:
        return_ann[i] = (price_last['Price'] / price_last[i] - 1 )

    return_ann = names_pd.merge(return_ann.reset_index(), how='left', on='ISINCode')

    return return_ann, periods_valid



def volatility(data, names_pd):
    data_pivot = data.pivot(index='Dates', columns='ISINCode', values='Price')
    data_pivot = data_pivot.fillna(method='ffill')
    data_pivot = data_pivot.dropna(axis=0, how='any')

    ret = data_pivot / data_pivot.shift(1) - 1

    vol = ret.rolling(30).std(ddof=0) * np.sqrt(252)
    vol = vol.dropna(how='all')
    vol = vol.melt(ignore_index=False).reset_index()
    vol = vol.merge(names_pd, how='left', on='ISINCode')
    vol = vol.rename(columns={'value': 'Volatility'})

    return vol


def get_monthly(prices, date_end):
    date_month_end = datetime.date(date_end.year, date_end.month, 1)
    prices_month = prices[prices['Dates'] <date_month_end]

    prices_month['Month'] = prices_month['Dates'].apply(lambda x: datetime.date(x.year, x.month,1))
    prices_month = prices_month.groupby(['ISINCode', 'Month']).last().reset_index()
    prices_month = prices_month.pivot_table(index='Month', columns='ISINCode', values='Price')

    return prices_month


def calc_vol_period(prices, date_end ,names_pd):
    periods = ['1Y', '3Y', '5Y', '10Y']
    periods_num = [12, 36, 60, 120]
    periods_valid = []
    
    prices_month = get_monthly(prices, date_end)
    vol = pd.DataFrame()
    ret = prices_month / prices_month.shift(1) -1 
    for i, num in enumerate(periods_num):
        vol_period = ret[-num:].rolling(num).std(ddof=0).iloc[-1]*np.sqrt(12)

        if len(vol_period[vol_period.notna()]) >0 :
            vol_period.name = periods[i]
            vol = pd.concat([vol, vol_period], axis=1)
            periods_valid += [periods[i]]

    vol = vol.reset_index().rename(columns={'index': 'ISINCode'})
    vol = names_pd.merge(vol, how='left',  on='ISINCode')
    return vol, periods_valid


def drawdown(prices, names_pd):
    data_pivot = prices.pivot(index='Dates', columns='ISINCode', values='Price')
    data_pivot = data_pivot.fillna(method='ffill')
    data_pivot = data_pivot.dropna(axis=0, how='any')

    max_pd = data_pivot.rolling(len(data_pivot), min_periods=1).max()
    drawdown = data_pivot / max_pd -1 

    drawdown = drawdown.melt(ignore_index=False).reset_index()
    drawdown = drawdown.merge(names_pd, how='left', on='ISINCode')
    drawdown = drawdown.rename(columns={'value': 'Drawdown'})

    return drawdown


def calc_drawdown_period(prices, date_end ,names_pd):
    periods = ['1M', '3M', '6M', 'YTD','1Y', '3Y', '5Y', '10Y']
    periods_valid = []
    dates = [date_end - datetime.timedelta(30), date_end - datetime.timedelta(91), \
            date_end - datetime.timedelta(182), datetime.date(date_end.year,1,1), \
            date_end - datetime.timedelta(365), date_end - datetime.timedelta(365*3 + 1), \
            date_end - datetime.timedelta(365*5+1), date_end - datetime.timedelta(365*10 + 2)]

    prices_pivot = prices.pivot_table(index='Dates', columns='ISINCode', values='Price')
    prices_pivot = prices_pivot.fillna(method='ffill')

    dd = pd.DataFrame()
    dd_dates = pd.DataFrame()

    for i, date in enumerate(dates):
        if prices_pivot.index[0] <= date:

            price_period = prices_pivot[prices_pivot.index >= date]
            col_valid = price_period.columns[~price_period.iloc[0,:].isna()]
            price_period = price_period[col_valid]

            price_max = price_period.rolling(len(price_period), min_periods=1).max()
            ret = price_period / price_max - 1
            
            dd_period = ret.min()
            dd_period.name = periods[i]
            dd = pd.concat([dd, dd_period], axis=1)

            dd_period_dates = ret.idxmin()
            dd_period_dates.name = periods[i]
            dd_dates = pd.concat([dd_dates, dd_period_dates], axis=1)

    dd = dd.reset_index().rename(columns = {'index': 'ISINCode'})
    dd_dates = dd_dates.reset_index().rename(columns = {'index': 'ISINCode'})
    
    dd = names_pd.merge(dd, how='left', on='ISINCode')
    dd_dates = names_pd.merge(dd_dates, how='left', on='ISINCode')
    dd_dates = dd_dates.fillna('')

    return dd.to_dict(orient='records'), dd_dates.to_dict(orient='records'), dd.columns[2:].to_list()



def get_holding(isin, conn):
    holding_all = pd.read_sql(api.get_holding_all(isin), con=conn)
    holding_all = holding_all.rename(columns={'InstrumentDescription': 'Holding'})
    if len(isin) == 1:
        top = holding_all[['ISINCode','Holding', 'Weight']].head(10)
        top['HoldingType'] = 'Top10'

    holding_type = pd.read_sql(api.get_holding_type(isin), con=conn)
    holding_type = holding_type.rename(columns={'HoldingName': 'Holding'})
    if len(isin) == 1:
        holding_type = pd.concat([holding_type, top], axis=0)

    holding_all['Weight'] = holding_all['Weight']/100
    holding_type['Weight'] = holding_type['Weight']/100

    return holding_type, holding_all


def get_fundflow(isins, flow_currency, conn, date_start=None, date_end=None):
    flow_monthly = pd.read_sql(api.get_fundflow(isins, flow_currency, date_start, date_end), con=conn)
    flow_monthly['TimeStamp'] = pd.to_datetime(flow_monthly['TimeStamp'], utc=True)
    flow_monthly['Dates'] = flow_monthly['TimeStamp'].apply(lambda x: x.date())

    return flow_monthly[['ISINCode', 'Dates', flow_currency]]


def add_compare_flow(compare_select, currency_select , original_flow, original_name, date_start, date_end, conn):

    currency_col = 'flow_USD' if currency_select == 'USD' else 'flow_GBP' if currency_select == 'GBP' else 'flow_EUR'

    if currency_col != 'flow_USD':
        original_flow = get_fundflow([original_name[0][0]], currency_col, conn, date_start, date_end)
    else:
        if date_start is not None:
            original_flow = original_flow[original_flow['Dates'] >= date_start]
        if date_end is not None:
            original_flow = original_flow[original_flow['Dates'] <= date_end]


    if len(compare_select) > 0:
        compare_isins = [i[1] for i in compare_select]
        compare_names = [(i[1],i[3]) for i in compare_select]

        compare_flow = get_fundflow(compare_isins, currency_col, conn, date_start, date_end)

        flows = pd.concat([original_flow, compare_flow], axis=0, ignore_index=True)

        names = original_name + compare_names
        names_pd = pd.DataFrame(names, columns = ['ISINCode', 'Name'])

    else:
        flows = original_flow
        flows['Name'] = original_name[0][1]
        names_pd = pd.DataFrame(original_name, columns = ['ISINCode', 'Name'])

    flows_pivot = flows.pivot(index='Dates', columns='ISINCode', values=currency_col)
    flows_pivot = flows_pivot[names_pd['ISINCode'].to_list()]
    flows_pivot = flows_pivot.fillna(0)

    return flows_pivot, names_pd
    

def get_flow_period(isins, currency_select, names_pd, conn):
    currency_col = 'flow_USD' if currency_select == 'USD' else 'flow_GBP' if currency_select == 'GBP' else 'flow_EUR'
    flow_period = pd.read_sql(api.get_by_isin(isins, 'calc_fundflow_period', ["ISINCode", "Description", currency_col]), con=conn)
    flow_period[currency_col] = flow_period[currency_col].apply(lambda x: num_format(x))
    period_names = ['1M','3M','6M','YTD','1Y', '3Y', '5Y']
    
    flow_pivot = flow_period.pivot(index='ISINCode', columns='Description', values=currency_col)

    valid_period_names = period_names[:len(flow_pivot.columns)-1]
    flow_pivot = pd.concat([names_pd.set_index('ISINCode'), flow_pivot], axis=1)
    
    flow_pivot = flow_pivot[['Name'] + list(valid_period_names)].reindex(names_pd['ISINCode'])
    #flow_pivot = flow_pivot.fillna(nan_convert)
    return flow_pivot.to_dict(orient='records'), valid_period_names



def get_dividend(isin, conn):
    div = pd.read_sql(api.get_by_isin(isin, 'calc_div_yield', ["ISINCode","TimeStamp", "Currency", "Dividend", "Yield"]), con=conn)
    div['TimeStamp'] = pd.to_datetime(div['TimeStamp'], utc=True)
    div['Dates'] = div['TimeStamp'].apply(lambda x: x.date())
    div['Yield'] = div['Yield'] / 100
    div['Dividend'] = div['Dividend'].apply(lambda x: round(x,2))

    return div[['ISINCode','Dates', 'Currency', 'Dividend', 'Yield']]

def get_div_latest(isins, names_pd, conn):
    div = pd.read_sql(api.get_by_isin(isins, 'latest_div', ["ISINCode", "exDivDate", "Currency", "Dividend", "Yield", "DivGrowth", "DivGrowthPct"]), con=conn)
    div = div.merge(names_pd, how='left', on='ISINCode')
    div = div.set_index('ISINCode')

    div_isins = [i for i in isins if i in div.index]
    div = div.loc[div_isins]

    div = div.fillna(np.nan) #occasionally NA value appears None

    div['exDivDate'] = pd.to_datetime(div['exDivDate'], utc=True).dt.strftime('%Y-%m-%d')
    div['Currency'] = div['Currency'].apply(lambda x: currency_format(x))
    div['Dividend'] = div[['Currency', 'Dividend']].apply(lambda x: num_format(x[1], x[0], digits=2) if ~np.isnan(x[1]) else '-', axis=1)
    div['DivGrowth'] = div[['Currency', 'DivGrowth']].apply(lambda x: num_format(x[1], x[0], digits=2) if ~np.isnan(x[1]) else '-', axis=1)
    div['Yield'] = div['Yield'].apply(lambda x: '{:.1%}'.format(x/100))
    div['DivGrowthPct'] = div['DivGrowthPct'].apply(lambda x:  str(round(x,1))+'%' if ~np.isnan(x) else '-')

    div['DivGrowth'] = div[['DivGrowth', 'DivGrowthPct']].apply(lambda x: (x[0] + ' ({})'.format(x[1])) if x[0] != '-' else '-', axis=1)
    div = div[['Name', 'exDivDate', 'Dividend', 'Yield', 'DivGrowth']]
    div = div.rename(columns = {'exDivDate': 'Ex-Dividend Date', 'Yield': 'Yield (%)', 'DivGrowth': 'Dividend Growth'})


    return div.to_dict(orient='records')


def get_similar_etfs_detail(isins, conn):
    similar_etfs = pd.read_sql(api.get_similar_etfs_details(isins), con=conn)

    similar_etfs['Currency'] = similar_etfs['Currency'].apply(lambda x: currency_format(x))
    similar_etfs['AUM'] = similar_etfs[['Currency', 'AUM']].apply(lambda x: num_format(x[1], x[0]), axis=1)
    similar_etfs['Return'] = similar_etfs['Return']/100
    similar_etfs['TotalExpenseRatio'] = similar_etfs['TotalExpenseRatio'].apply(lambda x: str(round(x,2)))

    similar_etfs = similar_etfs[['FundName', 'DistributionIndicator', 'TotalExpenseRatio', 'AUM', 'Return']]
    similar_etfs = similar_etfs.rename(columns = {'FundName': 'Name', 'DistributionIndicator': 'Distribution', 'TotalExpenseRatio': 'Cost (%)', 'Return': 'YTD (%)' })
    col_names = similar_etfs.columns

    return similar_etfs.to_dict(orient='records'), col_names



def init_data(conn):
    etfs_all = run_query(api.get_unique_fund_list(), conn)
    return etfs_all








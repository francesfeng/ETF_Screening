import streamlit as st
import psycopg2
from psycopg2 import sql
import pandas as pd
import numpy as np
import copy
from src import api


@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None, 'builtins.weakref': lambda _: None})
def init_connection():
    return psycopg2.connect(**st.secrets["local"])


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
def initialise_data(asset_class, conn):
    if asset_class == 'Equity':
        results = pd.read_sql(api.init_equity(), con=conn)
    elif asset_class == 'Fixed Income':
        results = pd.read_sql(api.init_fi(), con=conn)
    elif asset_class == 'Commodity':
        results = pd.read_sql(api.init_commodity(), con=conn)
    elif asset_class == 'Currency':
        results = pd.read_sql(api.init_currency(), con=conn)
    elif asset_class == 'Structured':
        results = pd.read_sql(api.init_structured(), con=conn)
    elif asset_class == 'Alternative & Multi-Assets':
        results = pd.read_sql(api.init_alt(), con=conn)
    elif asset_class == 'Thematic':
        results = pd.read_sql(api.init_thematic(), con=conn) 

    else:
        return

    return results

@st.cache(allow_output_mutation=True, hash_funcs={psycopg2.extensions.connection: lambda _: None})
def get_etfs_data(isins, conn):
    query_overview = api.get_by_isin(isins, "gcp_funds", ["ISINCode", "Ticker", "ExchangeTicker", "FundName", "Exchange", "TradeCurrency", "Price", "1DChange",
            "Volume3M", "FundCurrency", "NAV", "NAV_1DChange", "AUM","AUM_USD", "AUM_EUR", "AUM_GBP",
            "DistributionIndicator", "TotalExpenseRatio", "ETPIssuerName", "IndexName", "Rank5"])
    data_overview = pd.read_sql(query_overview, con = conn)

    query_perf = api.get_by_isin(isins, "calc_return_period", ["ISINCode", "Type", "Description", "Return"])
    data_perf = pd.read_sql(query_perf, con = conn)

    data_fundflow = pd.read_sql(api.get_fundflows(isins), con = conn)

    return data_overview, data_perf, data_fundflow


def table_format(data, display, exchange=None ,currency=None, return_type=None):

    names = data['overview'][['ISINCode','ExchangeTicker','FundName','Exchange']]
    if exchange != 'All':
        names = names[names['Exchange'] == exchange]        

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
        if exchange != 'All':
            output_data = output_data[output_data['Exchange'] == exchange]


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


    else:
        return

    output_data = output_data.fillna('')   #fill NA value as '' in fees column, otherwise react has json error
    output_data.insert(loc=0, column='id', value = np.arange(0, len(output_data)))

    return output_data

@st.cache
def num_format(num, currency=None):
    num_abs = abs(num)
    sign = '-' if num<0 else ''
    output = ''
    if num_abs > 999999999:
        output = str(round(num_abs / 1000000000,1)) +'B'
    elif num_abs > 999999:
        output = str(round(num_abs / 1000000,1)) +'M'
    elif num_abs > 999:
        output = str(round(num_abs / 1000,1)) +'K'
    else:
        output = str(round(num_abs,1))

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
def get_etf_overview(ticker, data_dict, conn):
    etf_info = {}
    data_overview = data_dict['overview']
    
    data_overview = data_overview[data_overview['ExchangeTicker'] == ticker].to_dict(orient='records')[0]
    isin = data_overview['ISINCode']
    etf_info['ISINCode'] = isin
    data_perf = data_dict['performance']
    data_perf = data_perf[data_perf['ISINCode'] == isin]

    etf_info['ExchangeTicker'] = data_overview['ExchangeTicker']
    etf_info['Name'] = data_overview['FundName']
    etf_info['Exchange'] = data_overview['Exchange']
    etf_info['ExchangePrice'] = data_overview['TradeCurrency'] + (' ' if len(data_overview['TradeCurrency']) ==3 else '') + str(round(data_overview['Price'],2))
    etf_info['TradingCurrency'] = currency_revert(data_overview['TradeCurrency'])
    etf_info['Volume'] = num_format(data_overview['Volume3M'])
    etf_info['NAV'] = data_overview['FundCurrency'] + (' ' if len(data_overview['FundCurrency']) ==3 else '') + str(round(data_overview['NAV'],2))
    etf_info['FundCurrency'] = currency_revert(data_overview['FundCurrency'])
    etf_info['NAV_1MChange'] = round(float(data_perf.loc[(data_perf['ISINCode'] == isin) & (data_perf['Description'] == '1M'),'Return']),2)

    perf_3y = data_perf.loc[(data_perf['Description'] == '3Y') & (data_perf['Type'] == 'Cumulative'), 'Return']
    if len(perf_3y) == 0:
        etf_info['NAV_3YChange_Cum'] = '-'
        etf_info['NAV_3YChange_Ann'] = '-'
    else:
        etf_info['NAV_3YChange_Cum'] = str(round(perf_3y,2))+'%'
        perf_3m_ann = data_perf.loc[(data_perf['Description'] == '3Y') & (data_perf['Type'] == 'Annualised'), 'Return']
        etf_info['NAV_3YChange_Ann'] = str(round(perf_3m_ann,2))+'%'

    etf_info['AUM'] = data_overview['FundCurrency'] + (' ' if len(data_overview['FundCurrency']) ==3 else '') + num_format(data_overview['AUM'])
    etf_info['Cost'] = data_overview['TotalExpenseRatio']
    etf_info['Rank'] = rank_format(data_overview['Rank5'])
    etf_info['IndexName'] = data_overview['IndexName']
    etf_info['Distribution'] = data_overview['DistributionIndicator']


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
    etf_info['AUM_1MChange'] = num_format(float(etf_info['Flow']['flow_local']['1M']), data_overview['FundCurrency'])

    similar_etfs= pd.read_sql(api.get_similar_etfs([isin]), con = conn)
    similar_etfs = similar_etfs[~similar_etfs['ISINCode'].duplicated()]
    similar_top = pd.read_sql(api.get_1m_perf(similar_etfs['ISINCode'].head(5)), con=conn)
    etf_info['Similar_ETFs'] = similar_etfs.groupby('Description')['ISINCode'].apply(list).to_dict()
    etf_info['Similar_ETFs_top'] = similar_top.drop_duplicates().to_dict(orient='records')


    holding = pd.read_sql(api.get_top5_holding([isin]), con=conn)
    holding = holding.rename(columns={'InstrumentDescription': 'Name'})
    etf_info['Top5'] = holding.to_dict(orient='records')

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


def get_listing(isin, conn):
    data = pd.read_sql(api.get_listings(isin), con = conn)
    #data.insert(loc=0, column='id', value = np.arange(0, len(data)))
    #data['Price'] = data[['TradingCurrency', 'Price']]currency_format(data['TradingCurrency']) + str(round(data['Price'],2))
    data['Volume3M'] = data['Volume3M'].apply(lambda x: num_format(x))
    data['Price'] = data['TradingCurrency'].apply(lambda x: currency_format(x)) + data['Price'].apply(lambda x: str(round(x,2)))
    
    data = data.to_dict(orient='records')
    
    #st.write(data['TradingCurrency'])
    return data


def get_tr(isins, start_date, conn):
    data_tr = pd.read_sql(api.get_tr(isins, start_date), con=conn)  
    data = data_tr
    data['TimeStamp'] = pd.to_datetime(data['TimeStamp'], utc=True)
    data['Dates'] = data['TimeStamp'].dt.strftime('%Y-%m-%d')
    return data


def normalise(data, names_pd):
    data_norm = data.pivot(index='TimeStamp', columns='ISINCode', values='TotalReturn')
    data_norm = data_norm.fillna(method='ffill')
    data_norm = data_norm.dropna(axis=0, how='any')
    

    data_initial = 100/data_norm[:1]
    data_norm = data_norm * data_initial.values

    data_norm = data_norm.melt(ignore_index=False).reset_index()

    data_norm = data_norm.merge(names_pd, how='left', on='ISINCode')
    data_norm = data_norm.rename(columns={'value': 'TotalReturn'})
    data_norm['Dates'] = data_norm['TimeStamp'].dt.strftime('%Y-%m-%d')
    return data_norm


def __convert_period_stats(data, period_names, names, value='Return', is_num = True, nan_convert = 0):
    data_pivot = data.pivot(index='ISINCode', columns='Description', values=value)
    if is_num == True:
        data_pivot = data_pivot/100
    valid_period_names = period_names[:len(data_pivot.columns)-1]
    names_pd = names.set_index('ISINCode')
    data_pivot = pd.concat([names_pd, data_pivot], axis=1)
    data_pivot = data_pivot[['FundName'] + list(valid_period_names)].reindex(names['ISINCode'])
    data_pivot = data_pivot.fillna(nan_convert)
    return data_pivot.to_dict(orient='records'), valid_period_names



def get_perf_period(isins, names_pd, conn):
    data = pd.read_sql(api.get_by_isin(isins, 'calc_return_period', ["ISINCode", "Type", "Description", "Return" ]), con=conn)
    #data = data.merge(names_pd, how='left', on="ISINCode")
    cumulative = data.loc[data['Type'] == 'Cumulative', ['ISINCode', 'Description', 'Return']]
    #cumulative = cumulative.pivot_table(index='FundName', columns='Description', values='Return')
    period_names = ['1M', '3M', '6M','YTD','1Y', '3Y', '5Y','10Y']
    cumulative, cum_periods = __convert_period_stats(cumulative, period_names, names_pd)

    annualised = data.loc[data['Type'] == 'Annualised', ['ISINCode', 'Description', 'Return']]
    annualised, ann_periods = __convert_period_stats(annualised, ['1Y', '3Y', '5Y', '10Y'], names_pd)


    calendar = data.loc[data['Type'] == 'Calendar', ['ISINCode', 'Description', 'Return']]
    cal_names = calendar['Description'].unique()
    calendar, cal_periods = __convert_period_stats(calendar, cal_names, names_pd)
    
    return cumulative, cum_periods, annualised, ann_periods, calendar, cal_periods


def volatility(data, names_pd):
    data_pivot = data.pivot(index='TimeStamp', columns='ISINCode', values='TotalReturn')
    data_pivot = data_pivot.fillna(method='ffill')
    data_pivot = data_pivot.dropna(axis=0, how='any')

    ret = data_pivot / data_pivot.shift(1) - 1

    vol = ret.rolling(30).std(ddof=0) * np.sqrt(252)
    vol = vol.dropna(how='all')
    vol = vol.melt(ignore_index=False).reset_index()
    vol = vol.merge(names_pd, how='left', on='ISINCode')
    vol = vol.rename(columns={'value': 'Volatility'})
    vol['Dates'] = vol['TimeStamp'].dt.strftime('%Y-%m-%d')

    return vol

def get_vol_period(isins, names_pd, conn):
    vol = pd.read_sql(api.get_vol_period(isins), con = conn)
    vol = vol.merge(names_pd, how='left', on="ISINCode")

    vol_pivot, vol_names = __convert_period_stats(vol, ['1Y', '3Y', '5Y', '10Y'], names_pd, 'Volatility')
    
    return vol_pivot, vol_names


def drawdown(data, names_pd):
    data_pivot = data.pivot(index='TimeStamp', columns='ISINCode', values='TotalReturn')
    data_pivot = data_pivot.fillna(method='ffill')
    data_pivot = data_pivot.dropna(axis=0, how='any')

    max_pd = data_pivot.rolling(len(data_pivot), min_periods=1).max()
    drawdown = data_pivot / max_pd -1 

    drawdown = drawdown.melt(ignore_index=False).reset_index()
    drawdown = drawdown.merge(names_pd, how='left', on='ISINCode')
    drawdown = drawdown.rename(columns={'value': 'Drawdown'})
    drawdown['Dates'] = drawdown['TimeStamp'].dt.strftime('%Y-%m-%d')

    return drawdown


def get_drawdown_period(isins, names_pd, conn):
    dd = pd.read_sql(api.get_drawdown_period(isins), con = conn)
    dd['DateDrawdown'] = pd.to_datetime(dd['DateDrawdown'], utc=True).dt.strftime('%Y-%m-%d')

    dd_pivot, dd_names = __convert_period_stats(dd, ['1Y', '3Y', '5Y', '10Y'], names_pd, 'Drawdown')
    #st.write(dd_pivot)

    dd_date_pivot, _ = __convert_period_stats(dd, ['1Y', '3Y', '5Y', '10Y'], names_pd, 'DateDrawdown', is_num=False, nan_convert='')
    return dd_pivot, dd_date_pivot, dd_names

def get_holding(isin, conn):
    holding_all = pd.read_sql(api.get_holding_all(isin), con=conn)
    holding_all = holding_all.rename(columns={'InstrumentDescription': 'Holding'})
    top = holding_all[['Holding', 'Weight']].head(10)
    top['HoldingType'] = 'Top10'

    holding_type = pd.read_sql(api.get_holding_type(isin), con=conn)
    holding_type = holding_type.rename(columns={'HoldingName': 'Holding'})
    holding_type = pd.concat([holding_type, top], axis=0)

    holding_all['Weight'] = holding_all['Weight']/100
    holding_type['Weight'] = holding_type['Weight']/100

    return holding_type, holding_all


def get_fundflow(isins, flow_currency, names_pd, conn):
    flow_monthly = pd.read_sql(api.get_by_isin(isins, 'calc_fundflow_monthly', ['ISINCode', 'TimeStamp', flow_currency]), con=conn)
    flow_monthly = flow_monthly.merge(names_pd, how='left', on="ISINCode")
    flow_monthly['TimeStamp'] = pd.to_datetime(flow_monthly['TimeStamp'], utc=True)
    flow_monthly['Dates'] = flow_monthly['TimeStamp'].dt.strftime('%Y-%m-%d')

    flow_period = pd.read_sql(api.get_by_isin(isins, 'calc_fundflow_period', ["ISINCode", "Description", "flow_USD", "flow_EUR", "flow_GBP"]), con=conn)

    return flow_monthly

def get_flow_period(isins, flow_currency, names_pd, conn):
    flow_period = pd.read_sql(api.get_by_isin(isins, 'calc_fundflow_period', ["ISINCode", "Description", flow_currency]), con=conn)
    flow_period[flow_currency] = flow_period[flow_currency].apply(lambda x: num_format(x))
    flow_pivot, flow_names = __convert_period_stats(flow_period, ['1M','3M','6M','YTD','1Y', '3Y', '5Y'], names_pd, flow_currency, is_num=False)
    
    return flow_pivot, flow_names


def get_dividend(isin, conn):
    div = pd.read_sql(api.get_by_isin(isin, 'calc_div_yield', ["TimeStamp", "Currency", "Dividend", "Yield"]), con=conn)
    div['TimeStamp'] = pd.to_datetime(div['TimeStamp'], utc=True)
    div['Dates'] = div['TimeStamp'].dt.strftime('%Y-%m-%d')
    div['Yield'] = div['Yield'] / 100
    div['Dividend'] = div['Dividend'].apply(lambda x: round(x,2))

    return div

def get_div_latest(isins, names_pd, conn):
    div = pd.read_sql(api.get_by_isin(isins, 'latest_div', ["ISINCode", "exDivDate", "Currency", "Dividend", "Yield", "DivGrowth", "DivGrowthPct"]), con=conn)
    div = div.merge(names_pd, how='left', on='ISINCode')
    div['exDivDate'] = pd.to_datetime(div['exDivDate']).dt.strftime('%Y-%m-%d')
    div['Currency'] = div['Currency'].apply(lambda x: currency_format(x))
    div['Dividend'] = div[['Currency', 'Dividend']].apply(lambda x: num_format(x[1], x[0]), axis=1)
    div['DivGrowth'] = div[['Currency', 'DivGrowth']].apply(lambda x: num_format(x[1], x[0]), axis=1)
    div['Yield'] = div['Yield'].apply(lambda x: '{:.1%}'.format(x/100))
    div['DivGrowthPct'] = div['DivGrowthPct'] / 100
    div['DivGrowth'] = div[['DivGrowth', 'DivGrowthPct']].apply(lambda x: x[0] + ' ({:.1%})'.format(x[1]), axis=1)
    div = div[['FundName', 'exDivDate', 'Dividend', 'Yield', 'DivGrowth']]
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













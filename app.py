import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import psycopg2
import datetime

from src import style
from react_table import selectable_data_table
from src.data import init_connection
from src.data import get_filters
from src.data import initialise_data
from src.data import get_etfs_data
from src.data import table_format
from src.data import get_etf_overview
from src.data import get_etf_details
from src.data import get_listing
from src.data import get_tr
from src.data import normalise
from src.data import get_perf_period
from src.data import volatility
from src.data import get_vol_period
from src.data import drawdown
from src.data import get_drawdown_period
from src.data import get_holding
from src.data import get_fundflow
from src.data import get_flow_period
from src.data import get_dividend
from src.data import get_div_latest
from src.data import get_similar_etfs_detail

from src.viz import performance_line_simple
from src.viz import draw_radar_graph
from src.viz import draw_grouped_bar_vertical
from src.viz import draw_top_holding_graph
from src.viz import performance_graph
from src.viz import performance_grouped_bar_graph
from src.viz import legend_graph
from src.viz import draw_holding_type
from src.viz import draw_full_holding_graph
from src.viz import fundflow_graph
from src.viz import dividend_graph


from src.components import metric
from src.components import table_num
from src.components import performance_table
from src.components import div_table
from src.components import similar_etfs_table

st.set_page_config(layout="wide")

alt.themes.register("lab_theme", style.lab_theme)
alt.themes.enable("lab_theme")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

filter_pd = get_filters()
asset_classes = filter_pd['AssetClass'].unique()
conn = init_connection()

if 'filter_dict' not in st.session_state:
  st.session_state.filter_dict = {}
  for asset_class in asset_classes:
    st.session_state.filter_dict[asset_class] = {}

if 'asset_class' not in st.session_state:
  st.session_state.asset_class = 'Equity'


if 'filter_data' not in st.session_state:
  st.session_state.filter_data = initialise_data(st.session_state.asset_class, conn)

if 'display_data' not in st.session_state:
  st.session_state.display_data = {}
  data_overview, data_perf, data_fundflow = get_etfs_data(st.session_state.filter_data['ISINCode'], conn)
  st.session_state.display_data['overview'] = data_overview
  st.session_state.display_data['performance'] = data_perf
  st.session_state.display_data['flow'] = data_fundflow

if 'last_etf' not in st.session_state:
  st.session_state.last_etf = ''

if 'selected_etf' not in st.session_state:
  st.session_state.selected_etf = []

if 'etf_info' not in st.session_state:
  st.session_state.etf_info = {}
#-------------------------------
def change_asset_class():
  st.session_state.filter_data = initialise_data(st.session_state.asset_class, conn)
  #st.write(st.session_state.asset_class)
  #st.write(st.session_state.filter_data)
  return


def run_filter():
  data = st.session_state.filter_data
  filter_dict = {}
  for k1, column_dict in st.session_state.filter_dict[st.session_state.asset_class].items():
    for k2, v in column_dict.items():
      if k2 not in filter_dict:
        filter_dict[k2] = []
      filter_dict[k2] += v

  for k, v in filter_dict.items():
    if len(v)>0:
      data = data[data[k].isin(v)]
    
  return data

def add_filter(asset_class, column, label, label_key):
  asset_class= st.session_state.asset_class
  st.session_state.filter_dict[asset_class][label] = {column: st.session_state[label_key]}
  #st.write(st.session_state.filter_dict)
  return


def get_etfs(isins):
  data_overview, data_perf, data_fundflow = get_etfs_data(isins, conn)
  st.session_state.display_data['overview'] = data_overview
  st.session_state.display_data['performance'] = data_perf
  st.session_state.display_data['flow'] = data_fundflow
  return


def update_selections(selected_ticker):
  if len(selected_ticker) > len(st.session_state.selected_etf):
    for i in selected_ticker:
      if i not in st.session_state.selected_etf:
        st.session_state.last_etf = i 
  st.session_state.selected_etf = selected_ticker 
  return 


with st.sidebar:
  asset_class = st.selectbox('Asset Class', asset_classes, key='asset_class', on_change=change_asset_class)
  

  filter_group = filter_pd[filter_pd['AssetClass'] == asset_class]
  filter_labels = filter_group[['Column','Label']].drop_duplicates().to_records(index=False)
  results = run_filter()
  for column,label in filter_labels:
    filter_items = filter_group.loc[filter_group['Label'] == label,'Item']
    
    
    filter_stats = results[results[column].isin(filter_items)].groupby(column)['ISINCode'].count()
    filter_items = filter_items[filter_items.isin(filter_stats.index)]
    filter_stats = filter_stats[filter_items].to_dict()
    filter_key = (asset_class + '_' + label).replace(' ','_')
    default_value = []
    if label in st.session_state.filter_dict[asset_class]:
      val = st.session_state.filter_dict[asset_class][label][column]
      if len(val) > 0:
        default_value = val
    st.multiselect(label, filter_stats, default = default_value,format_func = lambda x:x+' ('+str(filter_stats[x]) + ')', key=filter_key ,on_change=add_filter, args=(asset_class, column, label, filter_key, ))
    
  st.button("Show filter results (" + str(len(results)) + ")", on_click=get_etfs, args=(results["ISINCode"],))  

exchanges = st.session_state.display_data['overview'][['Exchange', 'ISINCode']].groupby('Exchange').count().sort_values("ISINCode", ascending=False)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
col_header1, col_header2, col_header3 = st.columns([3,1,1])
with col_header1:
  st.write('')
  st.write('')
  st.radio('',['Overview', 'Performance', 'Fund Flow', 'Income'], key='display')
with col_header2:
  st.selectbox('Exchange', ['All'] + list(exchanges.index), key='exchange' )
with col_header3:
  if st.session_state.display == 'Performance':
    st.selectbox('Select Fund Currency', ['Cumulative','Annualised', 'Calendar Year'], key='return_type')
  else: 
    st.selectbox('Select Fund Currency', ['USD', 'EUR', 'GBP', 'Fund currency'], key='currency')

data = table_format(st.session_state.display_data, st.session_state.display, \
              st.session_state.exchange if 'exchange' in st.session_state else None, \
              st.session_state.currency if 'currency' in st.session_state else None, \
              st.session_state.return_type if 'return_type' in st.session_state else None)

rows = selectable_data_table(data.to_dict(orient='records'), st.session_state.display, \
                             st.session_state.currency if 'currency' in st.session_state else None, \
                              st.session_state.return_type if 'return_type' in st.session_state else None)


if len(rows)>0:
  update_selections(list(data.loc[data['id'].isin(rows), 'ExchangeTicker']))

  st.session_state.etf_info, navs = get_etf_overview(st.session_state.last_etf, st.session_state.display_data, conn)
  
  

col_select1, col_select2, _ = st.columns([1,1,2])
with col_select1:
  st.button('Compare selected ETFs (' + str(len(rows)) + ')')

#----------------------------------------------------------------------------------- Overview

if st.session_state.last_etf != '':
  st.header(st.session_state.etf_info['Name'])
  st.markdown(st.session_state.etf_info['ExchangeTicker'] + """     -      """ + st.session_state.etf_info['ISINCode'])
  st.markdown("***")
  cols_headers = st.columns([1,1,1,1,1,1,1,1])
  cols_headers[0].metric(label="NAV (1M %)", value=st.session_state.etf_info['NAV'], delta=str(st.session_state.etf_info['NAV_1MChange'])+'%')
  cols_headers[1].metric(label="AUM (1M Chg.)", value=st.session_state.etf_info['AUM'], delta=st.session_state.etf_info['AUM_1MChange'] )
  cols_headers[2].metric(label="Cost per Annum", value=str(st.session_state.etf_info['Cost']) + '%',)
  cols_headers[3].metric(label="Rating", value=st.session_state.etf_info['Rank'],)

  cols_headers[5].metric(label="Exchange Price", value=st.session_state.etf_info['ExchangePrice'],)
  cols_headers[6].metric(label="3M Avg. Volume", value=st.session_state.etf_info['Volume'],)
  cols_headers[7].metric(label="Exchange", value=st.session_state.etf_info['Exchange'],)

  with st.expander('Overview', expanded=True if len(rows)>0 else False):

    col_overview_lines1 = st.columns(3)
    col_overview_lines1[0].markdown('##### Description')
    col_overview_lines1[1].markdown('##### Performance')
    col_overview_lines1[2].markdown('##### Rating')

    col_overview_lines2 = st.columns([5,5,1,1,1,2])
    col_overview_lines2[0].write('')
    col_overview_lines2[0].markdown('**Investment objective**')
    col_overview_lines2[1].write('')
    periods = np.array(['1Y', '3Y', '5Y', 'All'])
    period_select = col_overview_lines2[1].radio('', periods)
    col_overview_lines2[3].metric(label='ETF', value=st.session_state.etf_info['Ranks']['Rank'])
    col_overview_lines2[4].markdown(" &nbsp; &nbsp;  |")
    col_overview_lines2[5].metric(label='Sector Average', value=st.session_state.etf_info['Ranks']['sectorRank'])

    col_overview_lines3 = st.columns(3)
    with col_overview_lines3[0]:
      st.write(st.session_state.etf_info['Objective'])

      st.write('**Asset Class:**', st.session_state.etf_info['AssetClass'])
      st.write('**Sector**:', st.session_state.etf_info['Sector'])
      st.write('**Index**:', st.session_state.etf_info['IndexName'])
      st.write('**Dividend Treatment**:', st.session_state.etf_info['Distribution'])

    with col_overview_lines3[1]:
      period_length = np.array([-52, -52*3, -52*5, 0])
      period_start = period_length[periods == period_select].item(0)
      alt_perf = performance_line_simple(navs[period_start:], 'NAV', '.2f')
      st.altair_chart(alt_perf, use_container_width=True)

    with col_overview_lines3[2]:
      rank_categories = ['Cost', 'Return', 'AUM', 'TrackingError', 'Volume']
      data1 = [st.session_state.etf_info['Ranks'].get(k+'Rank') for k in rank_categories]
      data2 = [st.session_state.etf_info['Ranks'].get('sector_' +k+'Rank') for k in rank_categories]
      
      fig_radar = draw_radar_graph(
            data1, 
            data2, 
            rank_categories, 
            'ETF',
            'Sector')  
      st.plotly_chart(fig_radar, use_container_width=True)

    st.write('')
    col_overview_lines4 = st.columns(3)
    with col_overview_lines4[0]:
      similar_etfs = st.session_state.etf_info['Similar_ETFs_top']
      st.write('##### Similar ETFs')  
      header = """|  | 1M (%) |
| --- | ----------- |
|""" + similar_etfs[0]['FundName']  + """ | """ + str(round(similar_etfs[0]['Return'],2)) + '%' + """ |
|""" + similar_etfs[1]['FundName']  + """ | """ + str(round(similar_etfs[1]['Return'],2)) + '%' + """ |
|""" + similar_etfs[2]['FundName']  + """ | """ + str(round(similar_etfs[2]['Return'],2)) + '%' + """ |
|""" + similar_etfs[3]['FundName']  + """ | """ + str(round(similar_etfs[3]['Return'],2)) + '%' + """ |
|""" + similar_etfs[4]['FundName']  + """ | """ + str(round(similar_etfs[4]['Return'],2)) + '%' + """ | """

      st.write(header)

    with col_overview_lines4[1]:
      st.write('##### Fund Flow')
      currency_select = st.radio('', ['USD', 'GBP', 'EUR'])
      currency_key = 'flow_USD' if currency_select == 'USD' else 'flow_GBP' if currency_select == 'GBP' else 'flow_EUR'
      
      flow = pd.DataFrame(st.session_state.etf_info['Flow'][currency_key].items())
      flow = flow.rename(columns={0: 'Period', 1: 'FundFlow'})
      flow['Type'] = 'ETF'
      flow_sector = pd.DataFrame(st.session_state.etf_info['Flow']['sector'+currency_key].items())
      flow_sector = flow_sector.rename(columns={0: 'Period', 1: 'FundFlow'})
      flow_sector['Type'] = 'Sector Average'
      flow = pd.concat([flow, flow_sector], axis=0)

      alt_flow = draw_grouped_bar_vertical(flow, 'FundFlow', 'Type', 'Period', ['ETF', 'Sector Average'], ['1M', '3M','6M', 'YTD', '1Y','3Y','5Y'])
      st.altair_chart(alt_flow, use_container_width=True)

    with col_overview_lines4[2]:
      st.write('##### Top 5 Holdings')
      top5 = pd.DataFrame(st.session_state.etf_info['Top5'])
      alt_top5 = draw_top_holding_graph(top5)
      st.write(alt_top5, use_container_width=True)

  with st.expander('Fund Details', expanded=False):
    st.session_state.etf_info['Details'] = get_etf_details([st.session_state.etf_info["ISINCode"]], conn)
    etf_info = st.session_state.etf_info
    st.write('##### Investment objective ****')
    st.write(st.session_state.etf_info['Objective'])
    st.write('##### Index Description')
    st.write(st.session_state.etf_info['Details']['IndexDescription'])
    
    st.write('##### Listing')
    col_listings = st.columns(5)
    col_listings[0].markdown(metric('ISIN', etf_info["ISINCode"]), unsafe_allow_html=True)
    col_listings[1].markdown(metric('Exchange Ticker', etf_info["ExchangeTicker"].split(' ')[1]), unsafe_allow_html=True)
    col_listings[2].markdown(metric('Trading / Fund Currency', etf_info["TradingCurrency"] + ' / '+etf_info["FundCurrency"]),unsafe_allow_html=True)
    col_listings[3].markdown(metric('Launch Date', etf_info["Details"]['LaunchDate']), unsafe_allow_html=True)
    col_listings[4].markdown(metric('Exchange', etf_info["Exchange"]), unsafe_allow_html=True)

    st.write('')
    st.write('')
    st.write('##### Key Facts')
    col_facts = st.columns(5)
    col_facts[0].markdown(metric('TER', str(etf_info["Cost"])+'%'), unsafe_allow_html=True)
    col_facts[1].markdown(metric('NAV', etf_info["NAV"]), unsafe_allow_html=True)
    col_facts[2].markdown(metric('AUM', etf_info["AUM"]),unsafe_allow_html=True)
    col_facts[3].markdown(metric('Shares Outstanding', '{:,}'.format(int(etf_info["Details"]['Shares']))), unsafe_allow_html=True)
    col_facts[4].markdown(metric('3M Average Volume', etf_info["Volume"]), unsafe_allow_html=True)

    st.write('')
    st.write('')
    st.write('##### Benchmark')
    col_bench = st.columns(5)
    col_bench[0].markdown(metric('Index', etf_info["IndexName"]), unsafe_allow_html=True)
    col_bench[1].markdown(metric('Index Provider', etf_info['Details']["IndexProvider"]), unsafe_allow_html=True)
    col_bench[2].markdown(metric('Index Rebalance', etf_info['Details']["RebalanceFrequency"]),unsafe_allow_html=True)
    col_bench[3].markdown(metric('Replication Method', etf_info['Details']['Replication']), unsafe_allow_html=True)
    col_bench[4].markdown(metric('Currency Hedged', etf_info['Details']["CurrencyHedge"]), unsafe_allow_html=True)

    if etf_info['Distribution'] == 'Distributing':
      st.write('')
      st.write('')
      st.write('##### Dividend')
      col_divs = st.columns(5)
      col_divs[0].markdown(metric('Last Dividend', etf_info['Details']["Dividend"]), unsafe_allow_html=True)
      col_divs[1].markdown(metric('Dividend Yield', str(round(etf_info['Details']["Yield"],2))+'%'), unsafe_allow_html=True)
      col_divs[2].markdown(metric('Frequency', etf_info['Details']["CashFlowFrequency"]),unsafe_allow_html=True)
      col_divs[3].markdown(metric('Ex-Dividend Date', etf_info['Details']['exDivDate']), unsafe_allow_html=True)
      
    st.write('')
    st.write('')
    st.write('##### Structure')
    col_str = st.columns(5)
    col_str[0].markdown(metric('Fund Manager', etf_info['Details']["FundCompany"]), unsafe_allow_html=True)
    col_str[1].markdown(metric('Castodian', etf_info['Details']["Custodian"]), unsafe_allow_html=True)
    col_str[2].markdown(metric('Domicile', etf_info['Details']["Domicile"]),unsafe_allow_html=True)
    col_str[3].markdown(metric('Legal Structure', etf_info['Details']['LegalStructure']), unsafe_allow_html=True)
    col_str[4].markdown(metric('UCITS', etf_info['Details']["UCITS"]), unsafe_allow_html=True)

    st.write('')
    st.write('')
    st.write('##### Statistics')
    col_stats = st.columns(5)
    col_stats[0].markdown(metric('3-Year Return (Cumulative)', etf_info["NAV_3YChange_Cum"]), unsafe_allow_html=True)
    col_stats[1].markdown(metric('3-Year Return (Annualised)', etf_info["NAV_3YChange_Ann"]), unsafe_allow_html=True)
    col_stats[2].markdown(metric('3-Year Volatility', etf_info['Details']["Volatility"]),unsafe_allow_html=True)
    col_stats[3].markdown(metric('3-Year Max Drawdown', etf_info['Details']['Drawdown']), unsafe_allow_html=True)
    col_stats[4].markdown(metric('Tracking Error', etf_info['Details']["TrackingError3Y"]), unsafe_allow_html=True)

    st.write('')
    st.write('')

  with st.expander('Listing', expanded=False):
    st.session_state.etf_info['Listing'] = get_listing([st.session_state.etf_info["ISINCode"]], conn)
    listing = st.session_state.etf_info['Listing']
    #st.table(st.session_state.etf_info['Listing'])
    col_listing_header = st.columns(7)
    col_listing_header[0].write('**Exchange**')
    col_listing_header[1].write('**Exchange Ticker**')
    col_listing_header[2].write('**Trading Currency**')
    col_listing_header[3].write('**Price**')
    col_listing_header[4].write('**3M Avg. Volume**')
    col_listing_header[5].write('**SEDOL**')
    col_listing_header[6].write('**Trading Hour**')

    for i, value in enumerate(st.session_state.etf_info['Listing']):
      col_listing = 'col_listing'+str(i)
      col_listing = st.columns(7)
      col_listing[0].write(value['Exchange'])
      col_listing[1].write(value['ExchangeTicker'])
      col_listing[2].write(value['TradingCurrency'])
      col_listing[3].write(value['Price'])
      col_listing[4].write(value['Volume3M'])
      col_listing[5].write(value['SEDOL'])
      col_listing[6].write(value['ExchangeHour'])

  with st.expander('Performance', expanded=False):
    perf_select = st.radio('', ['Cumulative', 'Annualised', 'Calendar', 'Volatility', 'Max Drawdown'])
    top_similar = [[i["ISINCode"],i['FundName']] for i in st.session_state.etf_info['Similar_ETFs_top']]
    isins = [st.session_state.etf_info['ISINCode']] + [i[0] for i in top_similar]
    perf_cum = get_tr(isins, navs['Dates'][0],conn)
    names_pd = pd.DataFrame([[st.session_state.etf_info['ISINCode'], st.session_state.etf_info['Name']]] + top_similar, columns=['ISINCode', 'FundName'])

    cumulative, cum_col, annualised, ann_col, calendar, cal_col = get_perf_period(isins, names_pd, conn)
    legend = legend_graph(names_pd, 'FundName',names_pd['FundName'].to_list() )
    
    isins_selected = [st.session_state.etf_info['ISINCode']]
    names_selected = []


    col_perf_header1, col_perf_header2 = st.columns(2)
    col_perf_header1.empty()
    col_perf_header1.empty()

    if perf_select == 'Cumulative' or perf_select == 'Volatility' or perf_select == 'Max Drawdown': 
      with col_perf_header2:
        st.write('')
        cum_selected = st.multiselect('Compare to similar ETFs', top_similar, format_func = lambda x: x[1])

      isins_selected += ([i[0] for i in cum_selected] if len(cum_selected)>0 else [])
      names_selected = names_pd.loc[names_pd['ISINCode'].isin(isins_selected), 'FundName'].to_list()

      
    if perf_select == 'Cumulative': 
      with col_perf_header1:
        st.write("**Cumulative Return**")
        st.caption('Total returns are shown on a Net Asset Value (NAV) basis, with gross income reinvested for dividend paying ETFs.')
      
      
      perf_cum_norm = normalise(perf_cum[perf_cum['ISINCode'].isin(isins_selected)], names_pd)

      alt_cum = performance_graph(perf_cum_norm, 'TotalReturn',names_selected , names_pd, '.1f', 'FundName')
      st.altair_chart(alt_cum, use_container_width=True)

      performance_table(cumulative, cum_col)
      

    #-------------------------------------------------------------------------------------- Annualised
    if perf_select == 'Annualised':
      with col_perf_header1:
        st.write("**Cumulative Return**")
        st.caption('Total returns are shown on a Net Asset Value (NAV) basis, with gross income reinvested for dividend paying ETFs.')

      annualised_pd = pd.DataFrame(annualised)
      ann_min = np.nanmin(annualised_pd[ann_col].values)
      ann_max = np.nanmax(annualised_pd[ann_col].values)

      
      st.altair_chart(legend, use_container_width=True)

      for i, col in enumerate(st.columns(len(ann_col))):
        with col:
          alt_ann = performance_grouped_bar_graph(annualised_pd,'FundName',ann_col[i], ann_col[i], \
                                          names_pd['FundName'].to_list(), [ann_min, ann_max], True if i == 0 else False)
          st.altair_chart(alt_ann, use_container_width=True)

      performance_table(annualised, ann_col)

    #-------------------------------------------------------------------------------------- Annualised
    if perf_select == 'Calendar':
      calendar_pd = pd.DataFrame(calendar)
      cal_min = np.nanmin(calendar_pd[cal_col].values)
      cal_max = np.nanmax(calendar_pd[cal_col].values)

      st.altair_chart(legend, use_container_width=True)

      with col_perf_header1:
        st.write("**Calendar Return**")
        st.caption('Total returns are shown on a Net Asset Value (NAV) basis, with gross income reinvested for dividend paying ETFs.')
      with col_perf_header2:
        if len(cal_col) >=5:
          cal_selected= st.select_slider('', cal_col, value=[cal_col[0], cal_col[4]])
          left_idx = np.where(cal_col == cal_selected[0])[0][0]
          right_idx = np.where(cal_col == cal_selected[1])[0][0] + 1
          cal_col = cal_col[left_idx:right_idx]
      
      for i, col in enumerate(st.columns(len(cal_col))):
        with col:
          alt_cal = performance_grouped_bar_graph(calendar_pd, 'FundName', cal_col[i], cal_col[i], \
                                        names_pd['FundName'].to_list(), [cal_min, cal_max], True if i==0 else False)
          st.altair_chart(alt_cal, use_container_width=True)

      calendar = calendar_pd[np.insert(cal_col, 0, 'FundName')].to_dict(orient='records')

      performance_table(calendar, cal_col)

    #-------------------------------------------------------------------------------------- Volatility
    if perf_select == 'Volatility':


      with col_perf_header1:
        st.write("**Volatility**")
        st.caption('Daily Volatility based on Total returns.')

      vol = volatility(perf_cum[perf_cum['ISINCode'].isin(isins_selected)], names_pd)

      alt_vol = performance_graph(vol, 'Volatility',names_selected , names_pd, '.1%', 'FundName')
      st.altair_chart(alt_vol, use_container_width=True)


      vol_period, vol_names = get_vol_period(isins, names_pd, conn)

      performance_table(vol_period, vol_names)

    #-------------------------------------------------------------------------------------- Max drawdown
    if perf_select == 'Max Drawdown':

      with col_perf_header1:
        st.write("**Max drawdown**")
        st.caption('Daily Drawdown based on Total returns.')

      drawdown = drawdown(perf_cum[perf_cum['ISINCode'].isin(isins_selected)], names_pd)

      alt_dd = performance_graph(drawdown, 'Drawdown',names_selected , names_pd, '.1%', 'FundName')
      st.altair_chart(alt_dd, use_container_width=True)
    
      dd_period, dd_date_period, dd_names = get_drawdown_period(isins, names_pd, conn)

      performance_table(dd_period, dd_names, dd_date_period)

    st.write('')
    st.write('')

  with st.expander('Holdings'):
    holding_type, holding_all = get_holding([st.session_state.etf_info['ISINCode']], conn)
    types = holding_type['HoldingType'].unique()
    holding_select = st.radio('', ['Main', 'Other Characteristics', 'Detailed Holding'])

    main_types = ['Top10', 'Sector', 'Geography']
    
    if holding_select == 'Main':
      col_holdings_main = st.columns(3)
      for i, col in enumerate(st.columns(3)):
        with col:
          label = main_types[i]
          label_width = 25 if label == 'Geography' else 120
          
          holding_data = holding_type[holding_type['HoldingType'] == label]
          holding_title = ('Top 10 ' if len(holding_data) == 10 else '') + label + ' ({:.0%})'.format(sum(holding_data['Weight']))
          alt_holding = draw_holding_type(holding_data, 'Flag' if label == 'Geography' else 'Holding', holding_title, label_width )
          st.altair_chart(alt_holding, use_container_width=True)

    elif holding_select == 'Other Characteristics':
      other_types = list(set(types) - set(main_types))
      for i, col in enumerate(st.columns(len(other_types))):
        with col:
          label = other_types[i]
          label_width = 50 if label == 'Credit Rating' else 80 if label == 'Maturity' else 50 if label == 'Currency' else 120

          holding_data = holding_type[holding_type['HoldingType'] == label]
          holding_title = ('Top 10 ' if len(holding_data) == 10 else '') + ('Asset Class' if label == 'AssetClass' else label) + ' ({:.0%})'.format(sum(holding_data['Weight']))

          alt_holding = draw_holding_type(holding_data, 'Holding',holding_title, label_width)
          st.altair_chart(alt_holding, use_container_width=True)

    else:
      alt_full_holding = draw_full_holding_graph(holding_all)
      st.altair_chart(alt_full_holding, use_container_width=True)

  with st.expander('Fund Flow', expanded=False):
    flow_isins = [st.session_state.etf_info['ISINCode']]
    flow_names = [st.session_state.etf_info['Name']]
    flow_monthly = get_fundflow(flow_isins, 'flow_USD' ,names_pd, conn)

    flow_header = st.columns([3,2,1,1])
    
    flow_select = flow_header[0].multiselect('Compare to similar ETFs', top_similar, format_func = lambda x: x[1], key='flow_select')
    flow_currency_select = flow_header[1].radio('Currency', ['USD', 'GBP', 'EUR'], key='flow2')

    date_min = min(flow_monthly['TimeStamp'])
    date_max = max(flow_monthly['TimeStamp'])

    flow_from = flow_header[2].date_input('from', value=date_min ,min_value=date_min, max_value=date_max )
    flow_to = flow_header[3].date_input('to', value=date_max ,min_value=date_min, max_value=date_max)


    currency_col = 'flow_USD' if flow_currency_select == 'USD' else 'flow_GBP' if flow_currency_select == 'GBP' else 'flow_EUR'
    if currency_col != 'flow_USD':
      flow_monthly = get_fundflow(flow_isins, currency_col ,names_pd, conn)
    if len(flow_select) > 0:
      flow_monthly = pd.concat([flow_monthly, get_fundflow([i[0] for i in flow_select], currency_col, names_pd, conn)], axis=0)
      flow_isins += [i[0] for i in flow_select]

    flow_monthly = flow_monthly[(flow_monthly['TimeStamp'] >= pd.to_datetime(flow_from, utc=True)) &(flow_monthly['TimeStamp'] <= pd.to_datetime(flow_to, utc=True))]
    flow_pivot = flow_monthly.pivot(index='Dates', columns='ISINCode', values=currency_col)
    
    flow_pivot = flow_pivot[[i for i in flow_isins if i in flow_pivot.columns]]

    flow_pivot = flow_pivot.fillna(0)
    
    pp_flow = fundflow_graph(flow_pivot, names_pd)
    st.plotly_chart(pp_flow, use_container_width=True)

    flow_period, flow_period_names = get_flow_period(isins, currency_col, names_pd, conn)
  
    performance_table(flow_period, flow_period_names, header_suffix=' (' + flow_currency_select + ')', is_num=False)

  if st.session_state.etf_info['Distribution'] == 'Distributing':
    with st.expander('Dividend', expanded=False):
      div = get_dividend([st.session_state.etf_info['ISINCode']], conn)

      div_header = st.columns([5,1,1])
      div_date_min = div['TimeStamp'].iloc[0]
      div_date_max = div['TimeStamp'].iloc[-1]

      div_default_date = max(div_date_max - datetime.timedelta(days=3652), div_date_min)

      div_from = div_header[1].date_input('From', value=div_default_date, min_value=div_date_min, max_value = div_date_max)
      div_to = div_header[2].date_input('To', value=div_date_max, min_value=div_from, max_value = div_date_max)

      alt_div = dividend_graph(div[(div['TimeStamp'] >= pd.to_datetime(div_from, utc=True)) & (div['TimeStamp'] <= pd.to_datetime(div_to, utc=True))])
      st.altair_chart(alt_div, use_container_width=True)


      div_latest = get_div_latest(isins, names_pd, conn)
      div_table(div_latest, ['Ex-Dividend Date', 'Dividend', 'Yield (%)', 'Dividend Growth'])
      st.write('')
      st.write('')

  with st.expander('Similar ETFs', expanded=False):

    for k, v in st.session_state.etf_info['Similar_ETFs'].items():
      st.write('**'+k+'**')
      similar_etfs_data, similar_etfs_header = get_similar_etfs_detail(v, conn)

      similar_etfs_table(similar_etfs_data, similar_etfs_header)
      st.write('')
   

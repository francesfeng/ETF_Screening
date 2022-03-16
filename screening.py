import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import psycopg2
import datetime

from streamlit_option_menu import option_menu

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


from src.data import get_fundflow
from src.data import add_compare_flow
from src.data import get_flow_period


from src.data import get_similar_etfs_detail
from src.data import get_exchanges
from src.data import search
from src.data import get_equity_indices_names
from src.data import get_equity_indices
from src.data import get_etfs_lst

from src.data import calc_date_range


from src.viz import performance_line_simple
from src.viz import draw_radar_graph
from src.viz import draw_grouped_bar_vertical
from src.viz import draw_top_holding_graph
from src.viz import performance_graph
from src.viz import performance_grouped_bar_graph
from src.viz import legend_graph


from src.components.com_components import metric
from src.components.com_components import table_num
from src.components.com_components import performance_table

from src.components.com_components import similar_etfs_table

from src.components.display import display_listing
from src.components.display import display_detail
from src.components.display import display_header
from src.components.display import display_overview
from src.components.display import display_performance
from src.components.display import display_holding
from src.components.display import display_fundflow
from src.components.display import display_dividend


#-------------------------------------------------------Initialise

st.set_page_config(layout="wide")
alt.themes.register("lab_theme", style.lab_theme)
alt.themes.enable("lab_theme")

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

conn = init_connection()

filter_pd = get_filters()
asset_classes = filter_pd['AssetClass'].unique()
exchanges = get_exchanges(conn)

#-------------------------------------------------------Menu


if 'default_page' not in st.session_state:
  st.session_state.default_page = 0


if 'main_menu' not in st.session_state:
  st.session_state.main_menu = ['ETF Screening']

if 'filter_dict' not in st.session_state:
  st.session_state.filter_dict = {}
  for asset_class in asset_classes:
    st.session_state.filter_dict[asset_class] = {}

if 'filter_data' not in st.session_state:
  # get all classification data by asset class and exchange
  st.session_state.filter_data = initialise_data(asset_classes[0], exchanges[0][3] ,conn)

if 'display_data' not in st.session_state:
  # get all display data, including overview, performance, fund flow
  st.session_state.display_data = {}
  data_overview, data_perf, data_fundflow, data_div = get_etfs_data(st.session_state.filter_data['ISINCode'], exchanges[0][3], conn)
  st.session_state.display_data['overview'] = data_overview
  st.session_state.display_data['performance'] = data_perf
  st.session_state.display_data['flow'] = data_fundflow
  st.session_state.display_data['div'] = data_div


if 'etf_info' not in st.session_state:
  st.session_state.etf_info = {}

if 'search_default' not in st.session_state:
  st.session_state.search_default = 'Search ETFs'

if 'msg' not in st.session_state:
  st.session_state.msg = ('None', '')

if 'pre_select' not in st.session_state:
  st.session_state.pre_select = []

#-------------------------------
def change_scope(conn):
  asset_class = st.session_state.asset_class
  exchange = st.session_state.exchange[3]
  st.session_state.filter_data = initialise_data(asset_class, exchange, conn)
  data_overview, data_perf, data_fundflow, data_div = get_etfs_data(st.session_state.filter_data['ISINCode'], exchange, conn)
  st.session_state.display_data['overview'] = data_overview
  st.session_state.display_data['performance'] = data_perf
  st.session_state.display_data['flow'] = data_fundflow
  st.session_state.display_data['div'] = data_div
  
  st.session_state.msg = ('none', '')
  return


def run_filter(asset_class, exchange, conn):
  data = st.session_state.filter_data
  filter_dict = {}
  for k1, column_dict in st.session_state.filter_dict[asset_class].items():
    for k2, v in column_dict.items():
      if k2 not in filter_dict:
        filter_dict[k2] = []
      filter_dict[k2] += v

  for k, v in filter_dict.items():
    if len(v)>0:
      data = data[data[k].isin(v)]
    
  return data

def add_filter(asset_class, column, label, label_key):
  st.session_state.filter_dict[asset_class][label] = {column: st.session_state[label_key]}
  results = run_filter(st.session_state.asset_class, st.session_state.exchange, conn)
  get_etfs(results['ISINCode'], conn)
  st.session_state.search = ""

  st.session_state.msg = ('none', '')
  return


def get_etfs(isins, conn):
  # click Show ETF button
  data_overview, data_perf, data_fundflow, data_div = get_etfs_data(isins, st.session_state.exchange[3] ,conn)
  st.session_state.display_data['overview'] = data_overview
  st.session_state.display_data['performance'] = data_perf
  st.session_state.display_data['flow'] = data_fundflow
  st.session_state.display_data['div'] = data_div
  st.session_state.msg = ('none', '')

  return


def update_selections(selected_etf):
  # when React table is selected
  if len(selected_etf) > len(st.session_state.pre_select):
    for i in selected_etf:
      if i[0] not in st.session_state.main_menu:
        st.session_state.main_menu.insert(1, i[0]) 
        st.session_state.msg = ('success', i[1] + ' is opened in the top menu')
      else:
        st.session_state.msg = ('warning', i[1] + ' already opened in the top menu')
  else:
    st.session_state.msg = ('none', '')

  st.session_state.pre_select = [i[0] for i in selected_etf]
  return 


def search_etfs():
  # search button
  txt = st.session_state.search.lower()
  search_isin = search(txt, conn)
  get_etfs(search_isin['ISINCode'], conn)

  for asset_class in asset_classes:
    st.session_state.filter_dict[asset_class] = {}

  st.session_state.msg = ('none', '')
  return 



  #------------------------------------------------------------------- Filter  

with st.sidebar:
  

  st.selectbox('Select Country', exchanges, format_func=lambda x: x[2] + ' (' + x[3] + ')',key='exchange', on_change=change_scope, args=(conn,))

  col_filter_header = st.columns([2,3])
  col_filter_header[0].subheader('Filter')
  col_filter_header[1].empty()


  asset_class = st.selectbox('Asset Class', asset_classes,key='asset_class', on_change=change_scope, args=(conn,))


  results = run_filter(st.session_state.asset_class, st.session_state.exchange[3], conn)
  col_filter_header[1].button("Show ETFs: " + str(len(results)) , on_click=get_etfs, args=(results["ISINCode"], conn,))  
    
  filter_group = filter_pd[filter_pd['AssetClass'].isin([st.session_state.asset_class, 'Common'])]

  filter_labels = filter_group[['Column','Label']].drop_duplicates().to_records(index=False)
    
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
      
    if label == 'Dividend Treatment':
      st.write('---')
    
    st.multiselect(label, filter_stats, default = default_value,format_func = lambda x:x+' ('+str(filter_stats[x]) + ')', key=filter_key ,on_change=add_filter, args=(asset_class, column, label, filter_key, ))
    


#------------------------------------------------------
def display_filter():
  col_search, _, col_delete = st.columns([3,1,1])
  search = col_search.text_input('', placeholder='Search ETFs', key='search', on_change=search_etfs)
    
  
  
  col_header1, _, col_header3= st.columns([3,1,1])
  with col_header1:
    st.write('')
    st.write('')
    st.radio('',['Overview', 'Performance', 'Fund Flow', 'Income'], key='display')
  with col_header3:
    if st.session_state.display == 'Performance':
      st.selectbox('Performance', ['Cumulative','Annualised', 'Calendar Year'], key='return_type')
    else: 
      st.selectbox('Fund Currency', ['USD', 'EUR', 'GBP', 'Fund currency'], key='currency')

  data = table_format(st.session_state.display_data, st.session_state.display, \
                 st.session_state.currency if 'currency' in st.session_state else None, \
                st.session_state.return_type if 'return_type' in st.session_state else None)


  rows = selectable_data_table(data.to_dict(orient='records'), st.session_state.display, \
                               st.session_state.currency if 'currency' in st.session_state else None, \
                                st.session_state.return_type if 'return_type' in st.session_state else None)


  if len(rows)>0:
    update_selections(data.loc[data['id'].isin(rows), ['ExchangeTicker', 'FundName']].to_records(index=False))


  #----------------------------------------------------------------------------------- Overview
  return

def display_etf(etf_ticker, display_data, conn):
  etf_info, navs = get_etf_overview(etf_ticker, display_data, conn)
  indices = get_equity_indices_names(conn)
  etfs = get_etfs_lst(conn, st.session_state.exchange[3])


  display_header(etf_info)

  with st.expander('Overview', expanded=True):
    display_overview(etf_info, navs)
   

  with st.expander('Fund Details', expanded=True):
    display_detail(etf_info['ISINCode'], etf_info, conn)


  with st.expander('Listing', expanded=False):
    display_listing(etf_info['ISINCode'], conn)


  with st.expander('Performance', expanded=False):
    display_performance(etf_info['ISINCode'], etf_info['Name'], indices, etfs, conn)
    
  #-------------------------------------------------------------------------------------- Holding
  with st.expander('Holdings'):
    display_holding(etf_info['ISINCode'], conn)
   
  #-------------------------------------------------------------------------------------- Fund Flow
  with st.expander('Fund Flow', expanded=False):
    display_fundflow(etf_info['ISINCode'], etf_info['Name'], etfs, conn)

  #-------------------------------------------------------------------- Dividend
  if etf_info['Distribution'] == 'Distributing':
    with st.expander('Dividend', expanded=False):

      display_dividend(etf_info['ISINCode'], etf_info['Name'], conn)

    #-------------------------------------------------------------------- Similar ETFs
  with st.expander('Similar ETFs', expanded=False):

    for k, v in etf_info['Similar_ETFs'].items():
      st.write('**'+k+'**')
      similar_etfs_data, similar_etfs_header = get_similar_etfs_detail(v, conn)

      similar_etfs_table(similar_etfs_data, similar_etfs_header)
      st.write('')

  return 
     
#---------------------------------------------------------------- Main Menu
icons = ['fullscreen-exit'] + ['bounding-box'] * (len(st.session_state.main_menu)-1)
menu = option_menu(None, st.session_state.main_menu, 
    icons=icons, 
    menu_icon="cast", default_index=st.session_state.default_page , orientation="horizontal")
#main_menu

script = 'screening'
if menu == 'ETF Screening':
  st.session_state.default_page = 0
  display_filter()
  
else:
  display_etf(menu, st.session_state.display_data, conn)


if st.session_state.msg[0] == 'success':
  st.success(st.session_state.msg[1])
if st.session_state.msg[0] == 'warning': 
  st.warning(st.session_state.msg[1])



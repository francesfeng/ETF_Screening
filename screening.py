import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

from streamlit_option_menu import option_menu

from src import style
from react_table import selectable_data_table
from src.data import init_connection
from src.data import get_filters
from src.data import initialise_data
from src.data import get_etfs_data
from src.data import table_format
from src.data import get_exchanges
from src.data import search
from src.data import get_etf_port_overview


from src.components.display import display_etf
from src.components.com_components import metric


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

def add_filter(asset_class, column, label, label_key, conn):
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


def update_selections(selected_etf, menu_placeholder):
  # when React table is selected
  if len(selected_etf) > len(st.session_state.pre_select):
    for i in selected_etf:
      if i[0] not in st.session_state.main_menu:
        st.session_state.main_menu.insert(1, i[0]) 
        st.session_state.msg = ('success', i[0] + ': ' + i[1] + ' is opened in the top menu')
        with menu_placeholder:
          st.session_state.selected_menu = display_menu_selection()
      else:
        st.session_state.msg = ('warning', i[0] + ': ' + i[1] + ' already opened in the top menu')

      
  else:
    st.session_state.msg = ('none', '')

  st.session_state.pre_select = [i[0] for i in selected_etf]


  return 



def search_etfs(conn, asset_classes):
  # search button
  txt = st.session_state.search.lower()
  search_isin = search(txt, conn)
  get_etfs(search_isin['ISINCode'], conn)

  for asset_class in asset_classes:
    st.session_state.filter_dict[asset_class] = {}

  st.session_state.msg = ('none', '')
  return 


def add_compare(tickers):
  for i in tickers:
    if i not in st.session_state.compare:
      st.session_state.compare.append(i)
  st.session_state.default_home_page  = 1

  return  

def add_port(tickers, conn):
  for i in tickers:
    if i not in st.session_state.port:
      st.session_state.port.append(i)
  #get_portfolio_etfs(conn)
  st.session_state.default_home_page = 2
  return


#------------------------------------------------------
def display_filter(conn, asset_classes ,menu_placeholder):
  col_search, _, col_delete = st.columns([3,1,1])
  search = col_search.text_input('', placeholder='Search ETFs', key='search', on_change=search_etfs, args=(conn,asset_classes))
    
  col_header1, _, col_header3= st.columns([2,2,1])
  with col_header1:
    st.write('')
    st.write('')
    st.radio('',['Overview', 'Performance', 'Fund Flow', 'Income'], key='display')
  with col_header3:
    if st.session_state.display == 'Performance':
      st.selectbox('Performance', ['Cumulative','Annualised', 'Calendar Year'], key='return_type')
    else: 
      st.selectbox('Fund Currency', ['USD', 'EUR', 'GBP', 'Fund currency'], key='currency')

  msg_placeholder = st.empty()

  data = table_format(st.session_state.display_data, st.session_state.display, \
                 st.session_state.currency if 'currency' in st.session_state else None, \
                st.session_state.return_type if 'return_type' in st.session_state else None)


  rows = selectable_data_table(data.to_dict(orient='records'), st.session_state.display, \
                               st.session_state.currency if 'currency' in st.session_state else None, \
                                st.session_state.return_type if 'return_type' in st.session_state else None)


  if len(rows)>0:
    update_selections(data.loc[data['id'].isin(rows), ['ExchangeTicker', 'FundName']].to_records(index=False), menu_placeholder)

  with msg_placeholder:
    if st.session_state.msg[0] == 'success':
      st.success(st.session_state.msg[1])
    if st.session_state.msg[0] == 'warning': 
      st.warning(st.session_state.msg[1])


  col_footer = st.columns(2)
  st.markdown('<style>div.row-widget.stButton > button{float:left; width: 200px; height: 50px;}</style>', unsafe_allow_html=True)
   
  with col_footer[0]:
    st.button('Add to Comparison (' + str(len(rows)) + ')', on_click = add_compare, args=(data.loc[data['id'].isin(rows), 'ExchangeTicker'].to_list(),))

  with col_footer[1]:
    st.button('Add to portfolio  (' + str(len(rows)) + ')', on_click = add_port, args=(data.loc[data['id'].isin(rows), 'ExchangeTicker'].to_list(), conn))

  return


def display_menu_selection():
  icons = ['fullscreen-exit'] + ['bounding-box'] * (len(st.session_state.main_menu)-1)
  menu = option_menu('Selected ETFs', st.session_state.main_menu, \
    icons=icons, \
    menu_icon="cast", default_index=st.session_state.default_page , orientation="horizontal",\
    styles={"nav-link": {"--hover-color": "#eee"},"menu-title": {"font-size": "medium"},} )
  return menu


  #------------------------------------------------------------------- Filter  

def app(conn):
  filter_pd = get_filters()
  asset_classes = filter_pd['AssetClass'].unique()
  exchanges = get_exchanges(conn)


  if 'default_page' not in st.session_state:
    st.session_state.default_page = 0


  if 'main_menu' not in st.session_state:
    st.session_state.main_menu = ['View all']

  if 'selected_menu' not in st.session_state:
    st.session_state.selected_menu = 'View all'

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

  #---------------------------------------------------------------- Side bar
  
  

  with st.sidebar:
    
    st.selectbox('Select Country', exchanges, format_func=lambda x: x[2] + ' (' + x[3] + ')',key='exchange', on_change=change_scope, args=(conn,))

    col_filter_header = st.columns([1,3])
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
      
      st.multiselect(label, filter_stats, default = default_value,format_func = lambda x:x+' ('+str(filter_stats[x]) + ')', key=filter_key ,on_change=add_filter, args=(asset_class, column, label, filter_key, conn, ))
      

  

  #main_menu
  placeholder = st.empty()
  with placeholder:
    st.session_state.selected_menu = display_menu_selection()

  if st.session_state.selected_menu == 'View all':
    st.session_state.default_page = 0
    display_filter(conn, asset_classes, placeholder)
    
  else:
    display_etf(st.session_state.selected_menu, st.session_state.display_data, conn)

  return




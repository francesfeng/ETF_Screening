import streamlit as st
import pandas as pd
import numpy as np
import datetime

from src.data import get_exchanges
from src.data import get_equity_indices_names
from src.data import get_etfs_lst
from src.data import get_etf_compare_overview
from src.data import get_fundflow
from src.data import get_dividend
from src.data import get_div_latest


from src.components.com_components import compare_table
from src.components.com_components import perf_compare_table
from src.components.com_components import holding_table
from src.components.com_components import performance_table


from src.components.compare_element import compare_holding
from src.components.compare_element import compare_div



from src.viz import fundflow_graph
from src.viz import performance_graph


from src.components.display import display_performance
from src.components.display import display_fundflow


def remove(idx, tickers):
	st.session_state.compare_detail.pop(tickers[idx])
	st.session_state.compare.pop(idx)

	return

def add_etf():
	if len(st.session_state.search_etf) == 0:
		return

	else:
		if len(st.session_state.search_etf) > len(st.session_state.compare_pre):
			for i in st.session_state.search_etf:
				if i[2] not in st.session_state.compare:
					st.session_state.compare.append(i[2])

		st.session_state.compare_pre = st.session_state.search_etf

	return

def get_etf_details(conn):
	tickers = []
	for i in st.session_state.compare:
		if i not in st.session_state.compare_detail:
			tickers += [i]

	if len(tickers) > 0:
		overview = get_etf_compare_overview(tickers, st.session_state.display_data, conn)

	for i in tickers:
		st.session_state.compare_detail[i] = overview[i]
	return

def app(conn):
	
	st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
	st.write('<style>div.row-widget.stButton {display:flex;justify-content: center;}</style>', unsafe_allow_html=True)
	
	if 'compare_detail' not in st.session_state:
		st.session_state.compare_detail = {}

	if 'compare_pre' not in st.session_state:
		st.session_state.compare_pre = []

	

	#--------------------------------------------

	exchanges = get_exchanges(conn, include_all=True)


	col_search = st.columns([1,4])
	with col_search[0]:
		exchange_select = st.selectbox('Select Country/Exchange', exchanges, format_func=lambda x: x[2] + ' (' + x[3] + ')' if x[2] != 'All' else x[2])

	with col_search[1]:
		etfs = get_etfs_lst(conn, exchange_select[3] if exchange_select[3] != ' ' else None, include_class=True)
		st.multiselect("Search ETFs to add to portfolio",etfs,format_func=lambda x: """{} | {} | {}""".format(x[2], x[3],x[4]) if len(x) > 1 else '', key='search_etf', on_change = add_etf)
			
	get_etf_details(conn)
	if len(st.session_state.compare_detail) == 0:
		st.warning('No ETFs are addded to Compare')
	
	
	#--------------------------------------------
	else: 
		st.write("***")
		indices = get_equity_indices_names(conn)
		isins = [v['ISINCode'] for k, v in st.session_state.compare_detail.items()]
		names = [v['Name'] for k, v in st.session_state.compare_detail.items()]
		tickers = [k for k in st.session_state.compare_detail]

		for i, col in enumerate(st.columns([1] + [2]*len(st.session_state.compare_detail))):
			if i!= 0:
				col.button('❌', key=tickers[i-1], on_click=remove, args=(i-1, tickers,))
				
		compare_table(st.session_state.compare_detail)

		st.write("----")
		st.write('**Returns (Cumulative)**')

		perf_dict = {'1M': '1M (%)', '3M': '3M (%)', '6M': '6M (%)', 'YTD': 'YTD (%)', '1Y': '1Y (%)', '3Y': '3Y (%)', '5Y': '5Y (%)', '10Y': '10Y (%)'}
		perf_compare_table({k: v['Return'] for k, v in st.session_state.compare_detail.items()}, perf_dict)
		
		st.write("----")
		col_flow_header = st.columns(4)
		col_flow_header[0].write('**Fund Flow**')
		flow_currency = col_flow_header[3].selectbox('', ['USD','GBP','EUR'])
		flow_header = 'flow_USD' if flow_currency == 'USD' else 'flow_EUR' if flow_currency == 'EUR' else 'flow_GBP'
		currency_symbole = '$' if flow_currency == 'USD' else '€' if flow_currency == 'EUR' else '£'


		flow_dict = {'1M': '1M', '3M': '3M', '6M':'6M', 'YTD':'YTD', '1Y': '1Y', '3Y':'3Y', '5Y':'5Y'}
		perf_compare_table({k: v['Flow'][flow_header] for k, v in st.session_state.compare_detail.items()}, flow_dict, currency_symbole)

		st.write("----")

		holdings = {k: v['Top10'] for k, v in st.session_state.compare_detail.items()}
		holding_table(holdings)

		#-------------------------------------------- Performance
		st.write('')
		with st.expander('Performance'):
			display_performance(isins, names, indices, etfs, conn)
	
			
		#----------------------------------------------- Holding
		with st.expander('Holdings'):
			compare_holding(isins, names, conn)

		#----------------------------------------------- Fund Flow
		with st.expander('Fund Flow'):
			display_fundflow(isins, names, etfs, conn, tickers)

		#----------------------------------------------- Dividend
		div_isins = [v['ISINCode']for k, v in st.session_state.compare_detail.items() if v['Distribution'] == 'Distributing']

		if len(div_isins) > 0 :
			with st.expander('Dividend'):
				compare_div(div_isins, isins, names, conn)




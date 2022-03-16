import streamlit as st
import pandas as pd
import numpy as np
import datetime
from src.data import get_etf_overview
from src.data import get_tr
from src.data import normalise
from src.data import get_perf_period
from src.data import volatility
from src.data import get_vol_period
from src.data import drawdown
from src.data import get_drawdown_period
from src.data import get_holding
from src.data import get_fundflow
from src.data import get_dividend
from src.data import get_div_latest
from src.data import init_data

from src.components import compare_table
from src.components import perf_compare_table
from src.components import holding_table
from src.components import performance_table
from src.components import div_table

from src.viz import performance_graph
from src.viz import legend_graph
from src.viz import performance_grouped_bar_graph
from src.viz import draw_holding_type
from src.viz import draw_full_holding_graph
from src.viz import fundflow_graph


def remove(idx, tickers):
	st.session_state.compare_detail.pop(tickers[idx])
	st.session_state.compare.pop(idx)

	return

def add_etf():
	if len(st.session_state.search_etf) == 0:
		return

	etf_exist = []
	etf_add = []
	etf_isins = []
	etf_remove = []

	return

def app(conn):
	
	st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
	st.write('<style>div.row-widget.stButton {display:flex;justify-content: center;}</style>', unsafe_allow_html=True)
	if 'compare_detail' not in st.session_state:
		st.session_state.compare_detail = {}

	for i in st.session_state.compare:
		if i not in st.session_state.compare_detail:
			if i in st.session_state.clicked:
				st.session_state.compare_detail[i] = st.session_state.clicked[i]
			else:
				st.session_state.compare_detail[i], _ = get_etf_overview(i, st.session_state.display_data, conn)

	#--------------------------------------------

	etfs_lst = init_data(conn)
	etf_add = st.multiselect("Search ETFs to add to portfolio",etfs_lst,format_func=lambda x: """{} | {} | {}""".format(x[1], x[2],x[3]) if len(x) > 1 else '', key='search_etf', on_change = add_etf)
			
	st.write(etfs_lst[:10])
	if len(st.session_state.compare_detail) == 0:
		st.warning('No ETFs are addded to Compare')
	
	
	#--------------------------------------------
	else: 
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
			isins = [v['ISINCode'] for k, v in st.session_state.compare_detail.items()]
			# launch_dates = pd.to_datetime([v['Details']['LaunchDate'] for k, v in st.session_state.compare_detail.items()])
			# latest = min(launch_dates)

			names_pd = pd.DataFrame(zip(isins,[v['Name'] for k, v in st.session_state.compare_detail.items()]), columns=['ISINCode','FundName'])
			perf_cum = get_tr(isins, None,conn)
			cumulative, cum_col, annualised, ann_col, calendar, cal_col = get_perf_period(isins, st.session_state.display_data['performance'] ,names_pd, conn)
			legend = legend_graph(names_pd, 'FundName',names_pd['FundName'].to_list() )
		 
			perf_select = st.radio('', ['Cumulative', 'Annualised', 'Calendar'])
			col_perf_header1, col_perf_header2 = st.columns(2)
			col_perf_header1.empty()
			col_perf_header1.empty()

			#-------------------------------------------- Cumulative
			if perf_select == 'Cumulative':
				with col_perf_header1:
					st.write("**Cumulative Return**")
					st.caption('Total returns are shown on a Net Asset Value (NAV) basis, with gross income reinvested for dividend paying ETFs.')
				with col_perf_header2:
					st.write('')
					select_label = names_pd.to_records(index=False)
					cum_selected = st.multiselect('', names_pd['FundName'], default=names_pd['FundName'])
					isin_selected = names_pd.loc[names_pd['FundName'].isin(cum_selected), 'ISINCode']


				perf_cum_norm = normalise(perf_cum[perf_cum['ISINCode'].isin(isin_selected)], names_pd)
				alt_cum = performance_graph(perf_cum_norm, 'TotalReturn',cum_selected , names_pd, '.1f', 'FundName')
				st.altair_chart(alt_cum, use_container_width=True)

				performance_table(cumulative, cum_col, is_decimal=False)

			#-------------------------------------------- Annualised
			if perf_select == 'Annualised':
				with col_perf_header1:
					st.write("**Annualised Return**")
					st.caption('Total returns are shown on a Net Asset Value (NAV) basis, with gross income reinvested for dividend paying ETFs.')
				
				performance = st.session_state.display_data['performance']

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

			#------------------------------------------ Calender
			if perf_select == 'Calendar':
				calendar_pd = pd.DataFrame(calendar)
				
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

				cal_min = np.nanmin(calendar_pd[cal_col].values)
				cal_max = np.nanmax(calendar_pd[cal_col].values)
				for i, col in enumerate(st.columns(len(cal_col))):
					with col:
						alt_cal = performance_grouped_bar_graph(calendar_pd, 'FundName', cal_col[i], cal_col[i], \
							names_pd['FundName'].to_list(), [cal_min, cal_max], True if i==0 else False)
						st.altair_chart(alt_cal, use_container_width=True)

				calendar = calendar_pd[np.insert(cal_col, 0, 'FundName')].to_dict(orient='records')
				performance_table(calendar, cal_col)

		#----------------------------------------------- Vol

		with st.expander('Risk'):
			risk_select = st.radio('', ['Volatility', 'Max Drawdown'])
			col_risk_header1, col_risk_header2 = st.columns(2)
			col_risk_header1.empty()
			col_risk_header1.empty()

			with col_risk_header2:
				st.write('')
				select_label = names_pd.to_records(index=False)
				risk_selected = st.multiselect('', names_pd['FundName'], default=names_pd['FundName'], key='risk_select')
				isin_selected = names_pd.loc[names_pd['FundName'].isin(risk_selected), 'ISINCode']

			if risk_select == 'Volatility':
				with col_risk_header1:
					st.write("**Volatility**")
					st.caption('Daily Volatility based on Total returns.')

				vol = volatility(perf_cum[perf_cum['ISINCode'].isin(isin_selected)], names_pd)

				alt_vol = performance_graph(vol, 'Volatility',risk_selected , names_pd, '.1%', 'FundName')
				
				st.altair_chart(alt_vol, use_container_width=True)
				vol_period, vol_names = get_vol_period(isins, names_pd, conn)

				performance_table(vol_period, vol_names)

		#----------------------------------------------- Drawdown
			if risk_select == 'Max Drawdown':
				with col_risk_header1:
					st.write("**Max drawdown**")
					st.caption('Daily Drawdown based on Total returns.')

				dd = drawdown(perf_cum[perf_cum['ISINCode'].isin(isin_selected)], names_pd)

				alt_dd = performance_graph(dd, 'Drawdown',risk_selected , names_pd, '.1%', 'FundName')
				st.altair_chart(alt_dd, use_container_width=True)

				dd_period, dd_date_period, dd_names = get_drawdown_period(isins, names_pd, conn)
				performance_table(dd_period, dd_names, dd_date_period)

		#----------------------------------------------- Holding
		with st.expander('Holdings'):
			holding_type, holding_all = get_holding(isins, conn)
			type_dict = {'Country':'Geography', 'Sector':'Sector', 'Exchange': 'Exchange', 'Credit Rating': 'Credit Rating', 
					'Maturity': 'Maturity', 'Currency': 'Currency', 'Asset Class':'AssetClass'}
			types_unique = holding_type['HoldingType'].unique()

			type_dict = {k: type_dict[k] for i, k in enumerate(type_dict) if type_dict[k] in types_unique and type_dict[k]!='Top10'}
			names_dic = names_pd.set_index('ISINCode').to_dict()['FundName']

			col_holding1, col_holding2 = st.columns(2)
			col_holding1.write('')
			label = col_holding1.radio('', [k for k in type_dict] + ['All Holdings'])

			if label != 'All Holdings':
				for i, col in enumerate(st.columns(len(isins))):
					with col:
						holding_data = holding_type[(holding_type['ISINCode'] == isins[i])&(holding_type['HoldingType'] == type_dict[label])]
						title = names_dic[isins[i]]
						label_width = 25 if label == 'Country' else 50 if label == 'Credit Rating' else 80 if label == 'Maturity' else 50 if label == 'Currency' else 120
						col.write('**'+title+'**')
						alt_holding = draw_holding_type(holding_data, 'Flag' if label == 'Country' else 'Holding','', label_width)
						st.altair_chart(alt_holding, use_container_width=True)
			else:
				holding_selected = col_holding2.selectbox('', isins, index=0,format_func=lambda x: names_dic[x])
				full_holding_data = holding_all[holding_all['ISINCode'] == holding_selected]
				alt_full_holding = draw_full_holding_graph(full_holding_data)
				st.altair_chart(alt_full_holding, use_container_width=True)
				#alt_holding = draw_holding_type(holding_type[])

			st.write('')
			st.write('')
		#----------------------------------------------- Fund Flow
		with st.expander('Fund Flow'):
			flow_monthly = get_fundflow(isins, 'flow_USD' ,names_pd, conn)
			date_min = min(flow_monthly['TimeStamp'])
			date_max = max(flow_monthly['TimeStamp'])

			flow_header = st.columns([1,2,1,1])

			flow_currency_select = flow_header[0].radio('Currency', ['USD', 'GBP', 'EUR'], key='flow2')
			flow_from = flow_header[2].date_input('from', value=max(date_max - datetime.timedelta(days=365), date_min) ,min_value=date_min, max_value=date_max )
			flow_to = flow_header[3].date_input('to', value=date_max ,min_value=flow_from, max_value=date_max)
			
			currency_col = 'flow_USD' if flow_currency_select == 'USD' else 'flow_EUR' if flow_currency_select == 'EUR' else 'flow_GBP'
			if flow_currency_select != 'USD':
				flow_monthly = get_fundflow(isins, currency_col, names_pd, conn)

			flow_monthly = flow_monthly[(flow_monthly['TimeStamp'] >= pd.to_datetime(flow_from, utc=True)) &(flow_monthly['TimeStamp'] <= pd.to_datetime(flow_to, utc=True))]
			flow_pivot = flow_monthly.pivot(index='Dates', columns='ISINCode', values=currency_col)
			flow_pivot = flow_pivot[[i for i in isins if i in flow_pivot.columns]]
			flow_pivot = flow_pivot.fillna(0)
			pp_flow = fundflow_graph(flow_pivot, names_pd)
			st.plotly_chart(pp_flow, use_container_width=True)


			flow_period = [{**{'FundName': v['Name']}, **v['Flow'][currency_col]} for k, v in st.session_state.compare_detail.items()]
			flow_period_names = ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y']
			flow_len = max([len(i) for i in flow_period])-1
			flow_period_names = flow_period_names[:flow_len]

			performance_table(flow_period, flow_period_names, header_suffix=' (' + flow_currency_select + ')', is_num=False, is_large=True)

			st.write('')
			st.write('')

		#----------------------------------------------- Dividend
		div_isins = [v['ISINCode']for k, v in st.session_state.compare_detail.items() if v['Distribution'] == 'Distributing']

		if len(div_isins) > 0 :
			with st.expander('Dividend'):
				div = get_dividend(div_isins, conn)
				div = div.merge(names_pd, how='left', on='ISINCode')
				
				date_min = min(div['TimeStamp'])
				date_max = max(div['TimeStamp'])

				col_divs = st.columns([1, 2, 1, 1])
				col_divs[0].write('')
				col_divs[0].write('')
				div_select = col_divs[0].radio('', ['Dividend Yield', 'Dividend'])

				flow_from = col_divs[2].date_input('from', value=date_min ,min_value=date_min, max_value=date_max )
				flow_to = col_divs[3].date_input('to', value=date_max ,min_value=flow_from, max_value=date_max)
				div_data = div[(div['TimeStamp'] >= pd.to_datetime(flow_from, utc=True)) &(div['TimeStamp'] <= pd.to_datetime(flow_to, utc=True))]

				if div_select == 'Dividend Yield':
					alt_div = performance_graph(div_data, 'Yield',[names_dic[i] for i in div_isins] , names_pd, '.1%', 'FundName')
					st.altair_chart(alt_div, use_container_width=True)

				if div_select == 'Dividend':
					st.altair_chart(legend, use_container_width=True)
					div_pivot = div_data.pivot(index='Dates', columns='ISINCode', values='Dividend')

					div_pivot = div_pivot.fillna(0)
					pp_div = fundflow_graph(div_pivot, names_pd)
					st.plotly_chart(pp_div, use_container_width=True)


				div_latest = get_div_latest(div_isins, names_pd, conn)
				div_table(div_latest, ['Ex-Dividend Date', 'Dividend', 'Yield (%)', 'Dividend Growth'])

				st.write('')
				st.write('')


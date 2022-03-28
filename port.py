import streamlit as st
import pandas as pd
import numpy as np
import datetime

import altair as alt

from src.data import get_exchanges
from src.data import get_etfs_lst
from src.data import get_etf_port_overview
from src.data import get_benchmark_port
from src.data import get_sector
from src.data import get_benchmark_weight
from src.data import get_ranks
from src.data import get_price
from src.data import get_usd_prices_div
from src.data import calc_return

from src.data import calc_date_range
from src.data import normalise
from src.data import __get_frequency_index
from src.data import num_format
from src.data import calc_mom_returns




from src.viz import draw_pie_graph
from src.viz import draw_sub_pie_graph
from src.viz import draw_radar_graph

from src.viz import performance_graph
from src.viz import performance_overlay_graph



from src.components.com_components import display_msg
from src.components.com_components import metric

from src.components.port_element import customise_portfolio
from src.components.port_element import display_port_holding
from src.components.port_element import display_performance_contribution
from src.components.port_element import display_port_perf

def get_portfolio_etfs(conn):
	tickers = []
	for i in st.session_state.port:
		if i not in st.session_state.port_detail:
			tickers += [i]

	if len(tickers) > 0:		
		overview = get_etf_port_overview(tickers, conn)

		for i in tickers:
			st.session_state.port_detail[i] = overview[i]

		weight = round(100/len(st.session_state.port_detail),2)

		for k in st.session_state.port_detail:
			st.session_state.port_detail[k]['Weight'] = weight

		last_ticker = [k for k in st.session_state.port_detail][-1]
		st.session_state.port_detail[last_ticker]['Weight'] = 100 - weight * (len(st.session_state.port_detail) - 1)	

		st.session_state.port_weight = 100.00	
	return

def add_etf(conn):
	if len(st.session_state.search_etf) == 0:
		st.session_state.port_pre = [] 
		st.session_state.msg = ['', '']
		return

	else:
		if len(st.session_state.search_etf) > len(st.session_state.port_pre):
			for i in st.session_state.search_etf:
				if i[2] not in st.session_state.port:
					st.session_state.port.append(i[2])
					st.session_state.msg = ['Success', i[2] + ' ' + i[3] + 'added to the portfolio']

				else:
					st.session_state.msg = ['Fail', i[2] + ' ' + i[3] + 'already exists in the portfolio']

		st.session_state.port_pre = st.session_state.search_etf
		
	return

def empty_portfolio():
	for i in st.session_state.port_detail:
		del st.session_state[i]
	st.session_state.port = []
	st.session_state.port_detail = {}
	st.session_state.msg = ['', '']
	return

def get_total_weight(ticker):
	old_weight = st.session_state.port_detail[ticker]['Weight']
	new_weight = st.session_state[ticker]


	if st.session_state.port_weight - old_weight + new_weight > 100:
		st.session_state.msg = ['Fail', "Total portfolio weight cannot exceed 100%. " + ticker + ' is re-weighted']
		revised_weight = 100 - st.session_state.port_weight + old_weight
		st.session_state.port_detail[ticker]["Weight"] = revised_weight
		st.session_state[ticker] = revised_weight
		st.session_state.port_weight = 100
	else:		
		st.session_state.port_detail[ticker]["Weight"] = new_weight
		st.session_state[ticker] = new_weight
		st.session_state.port_weight = round(st.session_state.port_weight - old_weight + new_weight,2)
		st.session_state.msg = ['', '']

	st.session_state.analyse_msg = ['','']
	return


def remove_etf(ticker, name):
	old_weight = st.session_state[ticker]
	del st.session_state.port_detail[ticker]
	
	del st.session_state[ticker]
	if ticker in st.session_state.port:
		st.session_state.port.remove(ticker)
	st.session_state.port_weight -= old_weight

	st.session_state.msg = ['Info', name + ' has been removed from the portfolio']
	return

def reset_weight():
	for k in st.session_state.port_detail:
		st.session_state.port_detail[k]['Weight'] = 0.0
		st.session_state[k] = 0.0
		st.session_state.port_weight = 0.0
	return


def run_portfolio():

	if st.session_state.port_weight < 100:
		weights = [v['Weight'] for k, v in st.session_state.port_detail.items()]

		if 0 in weights:
			st.session_state.analyse_msg = ["Fail", "There are ETFs with 0 allocation, please assign a weight before running the portfolio"]	
			st.session_state.port_run = False
			return 

		else:
			new_weights = [round(i/st.session_state.port_weight*100,2) for i in weights]
			new_weights[-1] = 100.00- sum(new_weights[:-1])

			for i, k in enumerate(st.session_state.port_detail):
				st.session_state.port_detail[k]['Weight'] = new_weights[i]
				st.session_state[k] = new_weights[i]

			st.session_state.port_weight = 100.0
			st.session_state.analyse_msg = ['Info',"Each ETF allocation has been re-weighted to meet total weight of 100%"]

	st.session_state.msg = ['', '']
	st.session_state.port_run = True

	return 

def app(conn):

	if 'port_detail' not in st.session_state:
		st.session_state.port_detail = {}

	if 'port_pre' not in st.session_state:
		st.session_state.port_pre = []

	if 'port_weight' not in st.session_state:
		st.session_state.port_weight = 0

	if 'msg' not in st.session_state:
		st.session_state.msg = ['', '']

	if 'analyse_msg' not in st.session_state:
		st.session_state.analyse_msg = ['', '']

	if 'port_run' not in st.session_state:
		st.session_state.port_run = False

	exchanges = get_exchanges(conn, include_all=True)


	col_search = st.columns([1,4])
	with col_search[0]:
		exchange_select = st.selectbox('Select Country/Exchange', exchanges, format_func=lambda x: x[2] + ' (' + x[3] + ')' if x[2] != 'All' else x[2])

	with col_search[1]:
		etfs = get_etfs_lst(conn, exchange_select[3] if exchange_select[3] != ' ' else None, include_class=True)
		st.multiselect("Search ETFs to add to portfolio",etfs,format_func=lambda x: """{} | {} | {}""".format(x[2], x[3],x[4]) if len(x) > 1 else '', key='search_etf', on_change = add_etf, args = (conn,))
			
	get_portfolio_etfs(conn)


	if len(st.session_state.port_detail) == 0:
		st.warning('No ETFs are addded to Compare')

	#------------------------------------------------------------------- Display Portfolio
	else:

		with st.expander('Portfolio construction', expanded=True):
			col_port0, warning_msg, _, col_port2 = st.columns([2, 10, 2, 2])

			with col_port0:
				st.write('')
				st.write('**Holdings**', "(", len(st.session_state.port_detail), ")" , key="holding_count")

			with warning_msg:
				display_msg(st.session_state.msg)

			with col_port2:
				st.write('')
				st.button("↩️ Reset weight to 0", help = "Reset weight to zero", on_click = reset_weight)
			

			customise_portfolio(st.session_state.port_detail, get_total_weight, remove_etf)

			col_analysis = st.columns([2, 7 ,1])

			with col_analysis[0]:
				st.write('')
				st.button("💹 Analyse my portfolio", on_click = run_portfolio)

			with col_analysis[1]:
				display_msg(st.session_state.analyse_msg)

			with col_analysis[2]:
				st.write('')
				st.button("🗑️ Delete all", help="Remove all EFFs in this portfolio", on_click = empty_portfolio, key='empty')

			st.write('')


		if st.session_state.port_run == True:
			port = [(v['ISINCode'], v['Weight']) for k, v in st.session_state.port_detail.items()]
			port = pd.DataFrame(port, columns=['ISINCode', 'Weight']).set_index('ISINCode')
			
			#-------------------------------------------------------------Constituents
			with st.expander('Constituents'):
				
				col_consti = st.columns([2,2,1])
				with col_consti[0]:
					st.write('')
					st.write('')
					st.write("**My portfolio**")
					
				with col_consti[1]:
					bench_port = get_benchmark_port(conn)
					st.selectbox("Benchmark portfolio:", bench_port, key='bench_port')
				
				bench = get_benchmark_weight(st.session_state.bench_port, conn)

				port_sector = get_sector(port, conn)
				bench_sector = get_sector(bench, conn)

				port_sector['Type'] = 'My Portfolio'
				bench_sector['Type'] = 'Benchmark'
				sectors = pd.concat([port_sector, bench_sector], axis=0, ignore_index=True)
				sectors = sectors.pivot_table(index='Sector', columns='Type', values = 'Weight')

				fig_pie = draw_sub_pie_graph(sectors['My Portfolio'].sort_values(ascending=False),\
										sectors['Benchmark'].sort_values(ascending=False)
					)
				st.plotly_chart(fig_pie, use_container_width=True)

			#-------------------------------------------------------------Ranking
			with st.expander('Ranking'):
				port_ranks = get_ranks(port, conn)
				bench_ranks = get_ranks(bench, conn)

				col_rank = st.columns([6,1,2,1,1,2])

				col_rank[2].write("")
				col_rank[2].write("**Average Rank**")	
				col_rank[3].metric(label="My portfolio", value=port_ranks['Rank'], delta='')
				col_rank[4].metric(label="", value='vs.', delta='')
				col_rank[5].metric(label="Benchmark", value=bench_ranks['Rank'], delta='')

				col_rank2 = st.columns([6,1,5,1])
				with col_rank2[0]:
					fig_radar = draw_radar_graph(
								[v for k, v in port_ranks.items()][:-1], 
								[v for k, v in bench_ranks.items()][:-1], 
								[k for k in port_ranks][:-1], 
								'My Portfolio',
								'Benchmark')	
					st.plotly_chart(fig_radar, use_container_width=True)

				with col_rank2[2]:
					st.selectbox('Explanations', ['Cost Rank', 'Return Rank'])
	
					st.write("""Size is measured by assets under management (AUM). The larger ETF’s size is, 
								the more profitable to operate and maintain the ETF.
							Large size ETFs also tend to have high liquidity and low tracking errors. """)

			#-------------------------------------------------------------Holding
			with st.expander('Holdings'):
				display_port_holding(port, bench, conn)


			with st.expander('Portfolio Strategies'):
				port_prices, port_div = get_usd_prices_div(port.index, conn)
				bench_prices, bench_div = get_usd_prices_div(bench.index, conn)

				perf_date_min = max(port_prices.groupby('ISINCode')['Dates'].min())

				col_perf = st.columns([2, 2, 2])
				col_perf[0].number_input("Start to invest", min_value=0, value = 10000,step=100, key='start_value')
				col_perf[1].selectbox(".", ["GBP", "EUR", "USD"], key='currency')
				col_perf[2].date_input("Inception date", value=perf_date_min, min_value=perf_date_min, key='start_date')

				col_perf1 = st.columns([2,2,2])
				col_perf1[0].number_input("Contribution/Withdraw", min_value=0 ,value=500, key='contribution_amount')
				col_perf1[1].selectbox(".", ['Contribution', 'Withdraw'], key='contribution')
				col_perf1[2].selectbox('Contribution/Withdraw Frequency', ['Monthly', 'Quarterly', 'Annual'], key='contribution_frequency')

				col_perf2 = st.columns([2,2,2])
				col_perf2[0].selectbox("Rebalance frequency", ["Monthly", "Quarterly", "Annual"], key='rebalance')
				col_perf2[1].selectbox("Dividend treatment", ["Distribute","Reinvest"], key='dividend')


				st.write("***")
				
				port_returns, _, _ = calc_return(port_prices, port_div, port, st.session_state.start_value, st.session_state.start_date, \
							st.session_state.currency, (1 if st.session_state.contribution == 'Contribution' else -1) * st.session_state.contribution_amount, st.session_state.contribution_frequency, \
							st.session_state.rebalance, st.session_state.dividend, 'My Portfolio'
							)
				perf_date_max = port_returns.iloc[-1]['Dates']

				col_perf = st.columns([2, 4, 2, 2])

				col_perf[0].write('')
				col_perf[0].write('**Performance backtesting**')
				with col_perf[1]:
					period_select = st.radio('Period', ['1Y', '3Y', '5Y', '10Y', 'All', 'Custom period'], index=4, key='perf1')

				if period_select == 'Custom period':
					perf_from = col_perf[2].date_input('From', value=max(perf_date_max - datetime.timedelta(3652), perf_date_min), min_value=perf_date_min, max_value=perf_date_max)
					perf_to = col_perf[3].date_input('To', value=perf_date_max, min_value=perf_from, max_value=perf_date_max)
				else: 
					perf_from = None
					perf_to = None

				if st.session_state.dividend == 'Reinvest':
					perf_col = ['Dates', 'Name', 'TR']
				else:
					perf_col = ['Dates', 'Name', 'PR', 'Income']

				if st.session_state.contribution_amount != 0:
					perf_col += ['Contribution']

				perf = port_returns[perf_col].rename(columns={perf_col[2]: 'Portfolio Valuation'})
				perf_dict = port_returns.iloc[-1].to_dict()
				perf = perf.melt(id_vars=['Dates', 'Name']).rename(columns={'variable':'Type', 'value': 'Price'})
				perf_names = perf['Type'].unique()

				perf_start, perf_end = calc_date_range(period_select, perf_from, perf_to, perf_date_min, perf_date_max)
				perf = perf[(perf['Dates'] >= perf_start) & (perf['Dates'] <= perf_end)]

				col_graph = st.columns([5,1])
				with col_graph[0]:
					# reduce to monthly data if rows exceeds 3000
					perf_display = perf if len(perf) < 3000 else perf[np.append(__get_frequency_index(perf['Dates'], 'Monthly'),True)]
					alt_cum = performance_graph(perf_display, 'Price', perf_names, perf_names, '~s','Type', )
					st.altair_chart(alt_cum, use_container_width=True)

				with col_graph[1]:
					st.write('**Currence Valuation**')
					st.caption('as of ' + str(perf_dict['Dates']))
					
					curr_symbol = '$' if st.session_state.currency == 'USD' else '£' if st.session_state.currency == 'GBP' else '€'
					st.metric(label='Total Portfolio Value:', value=num_format(perf_dict[perf_col[2]],curr_symbol), delta=None)

					income_title = 'Dividend received in Cash:' if st.session_state.dividend == 'Distribute' else 'Dividend received and reinvested'
					st.metric(label=income_title, value=num_format(perf_dict['Income'],curr_symbol), delta=None)

					st.metric(label='Total Cash Contribution/Withdraw', value=num_format(perf_dict['Contribution'],curr_symbol), delta=None)

					net_gain = perf_dict[perf_col[2]] - perf_dict['Contribution']
					st.metric(label='Net Gain', value=num_format(net_gain,curr_symbol), delta='{:.2%}'.format(net_gain/perf_dict['Contribution']))

				st.write('')
				st.write('')

			#------------------------------------------------------------------------ Overview
			with st.expander('Risk & Returns'):
				port_names = [('My Portfolio', '', 'My Portfolio')]
				port_names += [(v['ISINCode'], k, v['Name']) for k, v in st.session_state.port_detail.items()]
				port_names = pd.DataFrame(port_names,columns=['ISINCode', 'Ticker', 'Name'])

				bench_returns, _, _ = calc_return(bench_prices, bench_div, bench, 200, st.session_state.start_date, \
							st.session_state.currency, 0, st.session_state.contribution_frequency, \
							st.session_state.rebalance, st.session_state.dividend, 'Benchmark'
							)

				display_port_perf(port_returns, bench_returns, etfs, port_names, perf_date_min, perf_date_max, conn)


			with st.expander("ETFs Performance contributions", expanded=True):
				display_performance_contribution(port_returns[['Dates', 'TR_Net']], port_prices[['ISINCode', 'Dates', 'TR_USD']],port_names[['ISINCode', 'Name']])

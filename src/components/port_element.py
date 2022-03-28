import streamlit as st
import pandas as pd
import numpy as np
import datetime

from src.data import get_combined_holding
from src.data import calc_date_range
from src.data import normalise
from src.data import __get_frequency_index
from src.data import calc_mom_returns
from src.data import get_equity_indices_names
from src.data import get_equity_indices
from src.data import get_price
from src.data import calc_cum_returns
from src.data import volatility
from src.data import calc_vol_period
from src.data import drawdown
from src.data import calc_drawdown_period
from src.data import calc_ann_returns
from src.data import calc_cal_returns


from src.viz import legend_graph
from src.viz import draw_holding_grouped
from src.viz import draw_full_holding_graph
from src.viz import performance_graph
from src.viz import performance_overlay_graph
from src.viz import legend_graph
from src.viz import performance_grouped_bar_graph

from src.components.com_components import performance_table

def customise_portfolio(port, reweight, remove_etf):

	col_header= st.columns([1, 3, 2, 1, 1, 1, 1, 2, 1])
	col_list = ['Ticker','Name', 'Sector', 'NAV', 'AUM','Cost', 'Rank']
	col_names = ['ExchangeTicker','Name', 'Sector', 'NAV', 'AUM', 'Cost', 'Rank5']

	for i, col in enumerate(col_header[:-2]):
		col.write("**" + col_list[i] + "**", key= col_list[i])
	with col_header[-2]:
		st.write("**Weight**", "(", st.session_state.port_weight, "% )")
	
	for i, k in enumerate(port):
		cols = st.columns([1, 3, 2, 1, 1, 1, 1, 2, 1])
		for j, col in enumerate(cols[:-2]):
			if j == 0:
				col.write(k)
			else:
				col.write(port[k][col_names[j]])	

		cols[-2].number_input("", min_value = 0.0, max_value = 100.0, value=port[k]['Weight'], step = 1.0, key=k, on_change=reweight, args=(k, ))
		cols[-1].button("❌", key='delete'+str(i), on_click=remove_etf, args = (k, port[k]['Name'], ))

	return


def display_port_holding(port, bench, conn):
	holding_type, holding_all = get_combined_holding(port, bench, conn)
	types = holding_type['HoldingType'].unique()
	types_dict = {'Top10': 'Top 10 Holdings', 'Sector': 'Sector', 'Geography': 'Country', 'AssetClass': 'Asset Class',\
				'Currency': 'Currency', 'Exchange': 'Exchange (equities only)', 'Maturity': 'Maturity (bond holding only)', \
				'Credit Rating': 'Credit Rating (bold holding only)'}
	main_type = ['Top10', 'Sector', 'Geography']
	other_type = ['AssetClass', 'Currency']
	other_type = list(set(types) - set(main_type))

	col_holding = st.columns(2)
	holding_select = col_holding[0].radio(label = '', options = ['Main','Other Characters','Full Holdings'])
	names_pd = pd.DataFrame(['My Portfolio', 'Benchmark'],columns = ['Portfolio'])
	legend = legend_graph(names_pd, 'Portfolio', None) 
				
	if holding_select == 'Main':
					
		st.altair_chart(legend, use_container_width=True)
		for i, col in enumerate(st.columns([4,1]*len(main_type))):
			if i%2 == 0:
				with col:
					type_idx = int(i/2)
					holding_data = holding_type[holding_type['HoldingType'] == main_type[type_idx]]
					alt_holding = draw_holding_grouped(holding_data, 'Weight', 'Portfolio', 'Holding', ['My Portfolio', 'Benchmark'], types_dict[main_type[type_idx]])
					st.altair_chart(alt_holding, use_container_width=True)

	elif holding_select == 'Other Characters':
					
		st.altair_chart(legend, use_container_width=True)
		for i, col in enumerate(st.columns([4,1]*len(other_type))):
			if i%2 == 0:
				with col:
					type_idx = int(i/2)
					holding_data = holding_type[holding_type['HoldingType'] == other_type[type_idx]]
					alt_holding = draw_holding_grouped(holding_data, 'Weight', 'Portfolio', 'Holding', ['My Portfolio', 'Benchmark'], types_dict[other_type[type_idx]])
					st.altair_chart(alt_holding, use_container_width=True)

	else:

		name_selected = col_holding[1].selectbox('', ['My Portfolio', 'Benchmark'])
		full_holding_data = holding_all[holding_all['Portfolio'] == name_selected]
		if len(full_holding_data) > 0:
			alt_full_holding = draw_full_holding_graph(full_holding_data, num_per_page=15)
			st.altair_chart(alt_full_holding, use_container_width=True)
		else:
			st.warning('Woops... Sorry, missing data. We are working on it')
	st.write('')
	return


def display_performance_contribution(port_data, etfs_data, port_names):


	perf_date_min = port_data.iloc[0]['Dates']
	perf_date_max = port_data.iloc[-1]['Dates']


	prices_contri = port_data.rename(columns={'TR_Net': 'Price'})
	prices_contri['ISINCode'] = 'My Portfolio'

	prices_contri = pd.concat([prices_contri, etfs_data.rename(columns={'TR_USD':'Price'})], axis=0)
	prices_contri = normalise(prices_contri, port_names)

	relative_type = st.radio('', ['Month-on-Month Returns','Relative performance to portfolio' ])
	col_rel= st.columns(2)
	with col_rel[0]:
		st.caption('The relative performance is to compare each ETFs total return to the constituents ETFs total return, \
				without net contribution or withdraw assumptions')
					

	with col_rel[1]:
		if len(port_names) > 6: 
			last = prices_contri.groupby('Name')['Name','Dates', 'Price'].tail(1)
			last = last[~last['Name'].isin(['My Portfolio'])].sort_values('Price', ascending=True)
			top5 = last.tail(5)['Name']
			bottom5 = last.head(5)['Name']

			perf_type = st.radio('',['Top 5 ETFs', 'Bottom 5 ETFs', 'Customise'])
			multi_list = top5 if perf_type == 'Top 5 ETFs' else bottom5 if perf_type == 'Bottom 5 ETFs' else None
			etfs_select = st.multiselect('', port_names['Name'],  multi_list)
		else:
			etfs_select = st.multiselect('', port_names['Name'][1:],  port_names['Name'][1:])

	col_rel1 = st.columns([2,1,1])
	with col_rel1[0]:
		period_select = st.radio('Period', ['1Y', '3Y', '5Y', '10Y', 'All', 'Custom period'], key='period1')

		if period_select == 'Custom period':
			perf_from = col_rel1[1].date_input('From', value=max(perf_date_max - datetime.timedelta(3652), perf_date_min), min_value=perf_date_min, max_value=perf_date_max, key='date_from_contri')
			perf_to = col_rel1[2].date_input('To', value=perf_date_max, min_value=perf_from, max_value=perf_date_max, key='date_to_contri')
		else: 
			perf_from = None
			perf_to = None

	perf_start, perf_end = calc_date_range(period_select, perf_from, perf_to, perf_date_min, perf_date_max)
	price_display = prices_contri[prices_contri['Name'].isin(['My Portfolio']+etfs_select)]

	price_display = price_display[(price_display['Dates'] >= perf_start)&(price_display['Dates'] <= perf_end)]
	price_display = normalise(price_display, port_names)

	if len(price_display) / len(port_names) > 1000:
		dates = price_display['Dates'].unique()
		dates = dates[np.append(__get_frequency_index(dates, 'Monthly'),True)]
		price_display = price_display[price_display['Dates'].isin(dates)]
				
	if relative_type == 'Relative performance to portfolio':
		alt_perf_contri = performance_graph(price_display, 'Price', port_names['Name'].to_list(), port_names, '~s', 'Name')
		st.altair_chart(alt_perf_contri.properties(height=500), use_container_width=True)
	else:

		etfs_mom, port_mom = calc_mom_returns(price_display)

		alt_perf_mom = performance_overlay_graph(etfs_mom, port_mom,'Return',etfs_select, ['My Portfolio'] , '.1%', 'Name')
		st.altair_chart(alt_perf_mom, use_container_width=True)
			
	return

def display_port_perf(port_returns, bench_returns, etfs, port_names, perf_date_min, perf_date_max, conn ):

	perf_port = port_returns[['Dates', 'TR_Net']].rename(columns={'TR_Net': 'Price'})
	perf_port['Name'] = 'My Portfolio'

	perf_bench = bench_returns[['Dates', 'TR_Net']].rename(columns={'TR_Net': 'Price'})
	perf_bench['Name'] = 'Benchmark'

	perf = pd.concat([perf_port, perf_bench], axis=0)	

	perf_select = st.radio('', ['Cumulative', 'Annualised', 'Calendar', 'Volatility', 'Max Drawdown'])
	col_compare_header = st.columns(2)

	#----- add compare widget
	with col_compare_header[1]:
		compare_type = st.radio('Add to chart', ['Equity indices', 'ETFs', 'Portfolio ETFs'])
		indices = get_equity_indices_names(conn)
		compare_lst = indices if compare_type == 'Equity indices' else etfs if compare_type=='ETFs' else port_names[1:].to_records()
					
		compare_select = st.multiselect('', compare_lst, format_func = lambda x: x[2] + ' ' + x[3])

					
	if len(compare_select) > 0:
		compare_tickers = [i[1] for i in compare_select]
		compare_names = [(i[1],i[3]) for i in compare_select]
		compare_names = pd.DataFrame(compare_names, columns = ['ISINCode', 'Name'])

		if compare_type == 'Equity indices':
			compare_prices = get_equity_indices(compare_tickers, conn, min(perf_date_min, perf_date_max-datetime.timedelta(365*10+2)), perf_date_max)
		else:
			compare_prices = get_price(compare_tickers, conn, min(perf_date_min, perf_date_max-datetime.timedelta(365*10+2)), perf_date_max, 'Total Return')

		compare_prices = compare_prices.merge(compare_names, how='left', on='ISINCode')
		perf = pd.concat([perf, compare_prices[['Dates','Price','Name']]], axis=0)

	names_display = ['My Portfolio', 'Benchmark'] + (compare_names['Name'].to_list() if len(compare_select) > 0 else [])
	legend = legend_graph(pd.DataFrame(names_display, columns=['Name']), 'Name', None) 

	cumulative, cum_periods = calc_cum_returns(perf[perf['Name'].isin(names_display)], perf_date_max, pd.DataFrame(names_display, columns=['Name']), 'Name')

	#----- add time period select widget
	if perf_select == 'Cumulative' or perf_select == 'Volatility' or perf_select=='Max Drawdown':
		col_perf_header = st.columns([2,6,2,2])

		with col_perf_header[0]:
			st.write('')
			select_bench = st.checkbox('Add benchmark')
		with col_perf_header[1]:
			period_select = st.radio('Period', ['1Y', '3Y', '5Y', '10Y', 'All', 'Custom period'])

		if period_select == 'Custom period':
			perf_from = col_perf_header[2].date_input('From', value=max(perf_date_max - datetime.timedelta(3652), perf_date_min), min_value=perf_date_min, max_value=perf_date_max, key='date_from1')
			perf_to = col_perf_header[3].date_input('To', value=perf_date_max, min_value=perf_from, max_value=perf_date_max, key='date_to1')
		else: 
			perf_from = None
			perf_to = None

		names_select = ['My Portfolio']
		if select_bench:
			names_select += ['Benchmark']
		else:
			names_select += ['']
		if len(compare_select) > 0:
			names_select += compare_names['Name'].to_list()

		perf_start, perf_end = calc_date_range(period_select, perf_from, perf_to, perf_date_min, perf_date_max)

		perf_display = perf[perf['Name'].isin(names_select)]
		perf_display = perf_display[(perf_display['Dates'] >= perf_start)&(perf_display['Dates'] <= perf_end)]

		perf_display = normalise(perf_display, names_pd=None, col_name='Name', val_name='Price')

		dates = perf_display['Dates'].unique()
		if len(dates) > 1000:
			dates = dates[np.append(__get_frequency_index(dates, 'Monthly'),True)]
			perf_display = perf_display[perf_display['Dates'].isin(dates)]

		if perf_select == 'Cumulative':
			with col_compare_header[0]:
				st.write('')
				st.write("**Cumulative Return**")
				st.caption('Total returns are shown on a Net Asset Value (NAV) basis, with gross income reinvested for dividend paying ETFs.')

			alt_cum = performance_graph(perf_display, 'Price',names_select, names_select, '.1f', 'Name')
			st.altair_chart(alt_cum, use_container_width=True)
					
			performance_table(cumulative, cum_periods)

				
		if perf_select == 'Volatility':
			with col_compare_header[0]:
				st.write("**Volatility**")
				st.caption('Monthly Volatility based on Total returns.')
			
			vol = volatility(perf_display,  names_pd=None, col_name='Name', val_name='Price')
			alt_vol = performance_graph(vol, 'Volatility',names_select , names_select, '.1%', 'Name')
			st.altair_chart(alt_vol, use_container_width=True)

			vol_cum, vol_periods = calc_vol_period(perf, perf_date_max ,pd.DataFrame(names_display, columns=['Name']), 'Name')
			performance_table(vol_cum.to_dict(orient='records'), vol_periods)

		if perf_select == 'Max Drawdown':
			with col_compare_header[0]:
				st.write("**Max drawdown**")
				st.caption('Daily Drawdown based on Total returns.')

			dd = drawdown(perf_display,  names_pd=None, col_name='Name', val_name='Price')
			alt_dd = performance_graph(dd, 'Drawdown',names_select , names_select, '.1%', 'Name')
			st.altair_chart(alt_dd, use_container_width=True)
			dd_period, dd_dates, dd_names = calc_drawdown_period(perf, perf_date_max, pd.DataFrame(names_display, columns=['Name']), 'Name')
			performance_table(dd_period, dd_names, dd_dates)

	if perf_select == 'Annualised':
		annualised, ann_periods = calc_ann_returns(cumulative, cum_periods, col_names=['Name'])

		with col_compare_header[0]:
			st.write("**Annualised Return**")
			st.caption('Total returns are shown on a Net Asset Value (NAV) basis, with gross income reinvested for dividend paying ETFs.')

		if annualised is None:
			st.warning('The fund history is less than one year, there no annualised return is displayed')

		else: 
			ann_min = min(np.nanmin(annualised[ann_periods].values),0)
			ann_max = max(np.nanmax(annualised[ann_periods].values),0)
						
			st.altair_chart(legend, use_container_width=True)
			for i, col in enumerate(st.columns(len(ann_periods))):
				with col:
					alt_ann = performance_grouped_bar_graph(annualised,'Name',ann_periods[i], ann_periods[i], \
                                  names_display, [ann_min, ann_max], True if i == 0 else False)
					st.altair_chart(alt_ann, use_container_width=True)

			performance_table(annualised.to_dict(orient='records'), ann_periods)

	if perf_select == 'Calendar':
		calendar, cal_periods = calc_cal_returns(perf, perf_date_min, perf_date_max, pd.DataFrame(names_display, columns=['Name']), 'Name')
		
		with col_compare_header[0]:
			st.write("**Calendar Return**")
			st.caption('Total returns are shown on a Net Asset Value (NAV) basis, with gross income reinvested for dividend paying ETFs.')
		
		col_perf_header = st.columns(2)
		with col_perf_header[0]:
			if len(cal_periods) >=5:
				cal_selected= st.select_slider('', cal_periods, value=[cal_periods[0], cal_periods[4]])
				left_idx = np.where(np.array(cal_periods) == cal_selected[0])[0][0]

				right_idx = np.where(np.array(cal_periods) == cal_selected[1])[0][0] + 1
				cal_select = cal_periods[left_idx:right_idx]
			else:
				cal_select = cal_periods

		cal_min = min(np.nanmin(calendar[cal_select].values), 0)
		cal_max = max(np.nanmax(calendar[cal_select].values),0)

		st.altair_chart(legend, use_container_width=True)
		for i, col in enumerate(st.columns(len(cal_select))):
			with col:
				alt_cal = performance_grouped_bar_graph(calendar, 'Name', cal_select[i], cal_select[i], \
			                              names_display, [cal_min, cal_max], True if i==0 else False)
				st.altair_chart(alt_cal, use_container_width=True)

		performance_table(calendar[['Name'] + cal_select].to_dict(orient='records'), cal_select)

	return
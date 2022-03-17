
import streamlit as st 
import pandas as pd
import numpy as np
import datetime

from src.data import get_listing
from src.data import get_etf_details
from src.data import format_flow
from src.data import calc_date_range
from src.data import get_price
from src.data import add_compare_prices
from src.data import calc_cum_returns
from src.data import calc_ann_returns
from src.data import calc_cal_returns

from src.data import normalise
from src.data import volatility
from src.data import calc_vol_period
from src.data import drawdown
from src.data import calc_drawdown_period

from src.data import get_holding
from src.data import get_fundflow
from src.data import get_flow_period

from src.data import add_compare_flow
from src.data import get_flow_period
from src.data import get_dividend
from src.data import get_div_latest

from src.viz import performance_line_simple
from src.viz import draw_radar_graph
from src.viz import draw_grouped_bar_vertical
from src.viz import draw_top_holding_graph
from src.viz import performance_graph
from src.viz import performance_grouped_bar_graph
from src.viz import draw_holding_type
from src.viz import draw_full_holding_graph
from src.viz import legend_graph
from src.viz import fundflow_graph
from src.viz import dividend_graph


from src.components.com_components import metric
from src.components.com_components import performance_table
from src.components.com_components import table_num
from src.components.com_components import div_table


def display_header(etf_info):
	st.title(etf_info['Name'])
	st.markdown(etf_info['ExchangeTicker'] + """     -      """ + etf_info['ISINCode'])
	st.markdown("***")
	cols_headers = st.columns([1,1,1,1,1,1,1,1])
	cols_headers[0].metric(label="NAV (1M %)", value=etf_info['NAV'], delta=str(etf_info['NAV_1MChange'])+'%')
	cols_headers[1].metric(label="AUM (1M Chg.)", value=etf_info['AUM'], delta=etf_info['AUM_1MChange'] )
	cols_headers[2].metric(label="Cost per Annum", value=str(etf_info['Cost']) + '%',)
	cols_headers[3].metric(label="Rating", value=etf_info['Rank'],)

	cols_headers[5].metric(label="Exchange Price", value=etf_info['ExchangePrice'],)
	cols_headers[6].metric(label="3M Avg. Volume", value=etf_info['Volume'],)
	cols_headers[7].metric(label="Exchange", value=etf_info['Exchange'],)

	return


def display_overview(etf_info, navs):
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
	col_overview_lines2[3].metric(label='ETF', value=etf_info['Ranks']['Rank'])
	col_overview_lines2[4].markdown(" &nbsp; &nbsp;  |")
	col_overview_lines2[5].metric(label='Sector Average', value=etf_info['Ranks']['sectorRank'])

	col_overview_lines3 = st.columns(3)
	with col_overview_lines3[0]:
		obj = etf_info['Objective']
		st.markdown('<p style="text-align:justify;display: -webkit-box; -webkit-line-clamp: 6;-webkit-box-orient: vertical; overflow: hidden;">' + obj + '</p>', unsafe_allow_html=True)
		st.markdown("[Read more...](#index-description)", unsafe_allow_html=True)
		st.write('**Asset Class:**', etf_info['AssetClass'])
		st.write('**Sector**:', etf_info['Sector'])
		st.write('**Index**:', etf_info['IndexName'])
		st.write('**Dividend Treatment**:', etf_info['Distribution']) 
		

	with col_overview_lines3[1]:
		period_length = np.array([-52, -52*3, -52*5, 0])
		period_start = period_length[periods == period_select].item(0)
		alt_perf = performance_line_simple(navs[period_start:], 'NAV', '.2f')
		st.altair_chart(alt_perf, use_container_width=True)

	with col_overview_lines3[2]:
		rank_categories = ['Cost', 'Return', 'AUM', 'TrackingError', 'Volume']
		data1 = [etf_info['Ranks'].get(k+'Rank') for k in rank_categories]
		data2 = [etf_info['Ranks'].get('sector_' +k+'Rank') for k in rank_categories]

		fig_radar = draw_radar_graph(data1, data2, rank_categories, 'ETF','Sector')  
		st.plotly_chart(fig_radar, use_container_width=True)

		st.write('')
	col_overview_lines4 = st.columns(3)
	with col_overview_lines4[0]:
		similar_etfs = etf_info['Similar_ETFs_top']
		st.write('##### Similar ETFs')  

		etfs_len = 5 if len(similar_etfs)>=5 else len(similar_etfs)
		header = """| Name | 1Year  (%) |
| --- | ----------- |"""
		for i in range(etfs_len):
			header += """
|""" + similar_etfs[i]['FundName']  + """ | """ + table_num(similar_etfs[0]['Return']) + """ | """

		st.markdown(header, unsafe_allow_html=True)

	with col_overview_lines4[1]:
		st.write('##### Fund Flow')

		currency_select = st.radio('', ['USD', 'GBP', 'EUR'])
		flow_period = ['1M', '3M','6M', 'YTD', '1Y']
		flow = format_flow(etf_info['Flow'], currency_select, flow_period)
		
		alt_flow = draw_grouped_bar_vertical(flow, 'Fund Flow', 'Type', 'Period', ['ETF', 'Sector Average'],flow_period )
		st.altair_chart(alt_flow, use_container_width=True)

	with col_overview_lines4[2]:
		st.write('##### Top 5 Holdings')
		top5 = pd.DataFrame(etf_info['Top10'])
		if len(top5) == 0:
			st.warning('Woops, No holding data available, we are working on it')
		else:
			alt_top5 = draw_top_holding_graph(top5)
			st.write(alt_top5, use_container_width=True)

	return


def display_detail(isin, etf_info, conn):
	details = get_etf_details([isin], conn)
	st.write('##### Investment objective')
	st.write(etf_info['Objective'])
	st.write('##### Index Description')
	st.write(details['IndexDescription'])

	st.write('##### Listing')
	col_listings = st.columns(5)
	col_listings[0].markdown(metric('ISIN', etf_info["ISINCode"]), unsafe_allow_html=True)
	col_listings[1].markdown(metric('Exchange Ticker', etf_info["ExchangeTicker"].split(' ')[1]), unsafe_allow_html=True)
	col_listings[2].markdown(metric('Trading / Fund Currency', etf_info["TradingCurrency"] + ' / '+etf_info["FundCurrency"]),unsafe_allow_html=True)
	col_listings[3].markdown(metric('Launch Date', details['LaunchDate']), unsafe_allow_html=True)
	col_listings[4].markdown(metric('Exchange', etf_info["Exchange"]), unsafe_allow_html=True)

	st.write('')
	st.write('')
	st.write('##### Key Facts')
	col_facts = st.columns(5)
	col_facts[0].markdown(metric('TER', str(etf_info["Cost"])+'%'), unsafe_allow_html=True)
	col_facts[1].markdown(metric('NAV', etf_info["NAV"]), unsafe_allow_html=True)
	col_facts[2].markdown(metric('AUM', etf_info["AUM"]),unsafe_allow_html=True)
	col_facts[3].markdown(metric('Shares Outstanding', '{:,}'.format(int(details['Shares']))), unsafe_allow_html=True)
	col_facts[4].markdown(metric('3M Average Volume', etf_info["Volume"]), unsafe_allow_html=True)

	st.write('')
	st.write('')
	st.write('##### Benchmark')
	col_bench = st.columns(5)
	col_bench[0].markdown(metric('Index', etf_info["IndexName"]), unsafe_allow_html=True)
	col_bench[1].markdown(metric('Index Provider', details["IndexProvider"]), unsafe_allow_html=True)
	col_bench[2].markdown(metric('Index Rebalance', details["RebalanceFrequency"]),unsafe_allow_html=True)
	col_bench[3].markdown(metric('Replication Method', details['Replication']), unsafe_allow_html=True)
	col_bench[4].markdown(metric('Currency Hedged', details["CurrencyHedge"]), unsafe_allow_html=True)

	if etf_info['Distribution'] == 'Distributing':
		st.write('')
		st.write('')
		st.write('##### Dividend')
		col_divs = st.columns(5)
		col_divs[0].markdown(metric('Last Dividend', details["Dividend"]), unsafe_allow_html=True)
		col_divs[1].markdown(metric('Dividend Yield', str(round(details["Yield"],2))+'%'), unsafe_allow_html=True)
		col_divs[2].markdown(metric('Frequency', details["CashFlowFrequency"]),unsafe_allow_html=True)
		col_divs[3].markdown(metric('Ex-Dividend Date', details['exDivDate']), unsafe_allow_html=True)

	st.write('')
	st.write('')
	st.write('##### Structure')
	col_str = st.columns(5)
	col_str[0].markdown(metric('Fund Manager', details["FundCompany"]), unsafe_allow_html=True)
	col_str[1].markdown(metric('Castodian', details["Custodian"]), unsafe_allow_html=True)
	col_str[2].markdown(metric('Domicile', details["Domicile"]),unsafe_allow_html=True)
	col_str[3].markdown(metric('Legal Structure', details['LegalStructure']), unsafe_allow_html=True)
	col_str[4].markdown(metric('UCITS', details["UCITS"]), unsafe_allow_html=True)

	st.write('')
	st.write('')
	st.write('##### Statistics')
	col_stats = st.columns(5)
	col_stats[0].markdown(metric('3-Year Return (Cumulative)', etf_info["NAV_3YChange_Cum"]), unsafe_allow_html=True)
	col_stats[1].markdown(metric('3-Year Return (Annualised)', etf_info["NAV_3YChange_Ann"]), unsafe_allow_html=True)
	col_stats[2].markdown(metric('3-Year Volatility', details["Volatility"]),unsafe_allow_html=True)
	col_stats[3].markdown(metric('3-Year Max Drawdown', details['Drawdown']), unsafe_allow_html=True)
	col_stats[4].markdown(metric('Tracking Error', details["TrackingError3Y"]), unsafe_allow_html=True)

	st.write('')
	st.write('')

	return


def display_listing(isin, conn):
	listing = get_listing([isin], conn)
	col_listing_header = st.columns(7)
	col_listing_header[0].write('**Exchange**')
	col_listing_header[1].write('**Exchange Ticker**')
	col_listing_header[2].write('**Trading Currency**')
	col_listing_header[3].write('**Price**')
	col_listing_header[4].write('**3M Avg. Volume**')
	col_listing_header[5].write('**SEDOL**')
	col_listing_header[6].write('**Trading Hour**')

	for i, value in enumerate(listing):
		col_listing = 'col_listing'+str(i)
		col_listing = st.columns(7)
		col_listing[0].write(value['Exchange'])
		col_listing[1].write(value['ExchangeTicker'])
		col_listing[2].write(value['TradingCurrency'])
		col_listing[3].write(value['Price'])
		col_listing[4].write(value['Volume3M'])
		col_listing[5].write(value['SEDOL'])
		col_listing[6].write(value['ExchangeHour'])

	return


def display_performance(isin, fundname, indices, etfs, conn):
	perf_cum = get_price([isin], conn)
	perf_date_min = perf_cum['Dates'].iloc[0]
	perf_date_max = perf_cum['Dates'].iloc[-1]

	perf_default_date = max(perf_date_max - datetime.timedelta(days=3652), perf_date_min)
	perf_select = st.radio('', ['Cumulative', 'Annualised', 'Calendar', 'Volatility', 'Max Drawdown'])

	col_compare_header = st.columns(2)
	with col_compare_header[0]:
		if perf_select == 'Cumulative':
			st.write('')
			st.write("**Cumulative Return**")
			st.caption('Total returns are shown on a Net Asset Value (NAV) basis, with gross income reinvested for dividend paying ETFs.')

	with col_compare_header[1]:
		compare_type = st.radio('Add to chart', ['Equity indices', 'ETFs'])
		compare_lst = indices if compare_type == 'Equity indices' else etfs
		compare_select = st.multiselect('', compare_lst, format_func = lambda x: x[2] + ' ' + x[3])

	prices, names_pd = add_compare_prices(compare_select, compare_type, \
                                          perf_cum, [(isin, fundname)], \
                                          min(perf_date_min, perf_date_max - datetime.timedelta(3652)), None, 'Total Return',\
                                          conn)

	cumulative, cum_periods = calc_cum_returns(prices, perf_date_max, names_pd)
	legend = legend_graph(names_pd, 'Name',names_pd['Name'].to_list() )

	if perf_select == 'Cumulative' or perf_select == 'Volatility' or perf_select=='Max Drawdown':
		col_perf_header = st.columns([3,6,2,2])

		with col_perf_header[0]:
			price_type = st.radio('Display', ['NAV', 'Total Return'])
		with col_perf_header[1]:
			period_select = st.radio('Period', ['1Y', '3Y', '5Y', '10Y', 'All', 'Custom period'])

		if period_select == 'Custom period':
			perf_from = col_perf_header[2].date_input('From', value=perf_default_date, min_value=perf_date_min)
			perf_to = col_perf_header[3].date_input('To', value=perf_date_max, min_value=perf_from)
		else: 
			perf_from = None
			perf_to = None

		perf_start, perf_end = calc_date_range(period_select, perf_from, perf_to, perf_date_min, perf_date_max)

		if price_type == 'NAV':

			prices_nav, _ = add_compare_prices(compare_select, compare_type, \
                                          perf_cum, [(isin, fundname)], \
                                          min(perf_date_min, perf_date_max - datetime.timedelta(3652)), None, 'NAV',\
                                          conn)
			prices_display = prices_nav[(prices_nav['Dates'] >= perf_start) & (prices_nav['Dates'] <= perf_end)]

		else:
			prices_display = prices[(prices['Dates'] >= perf_start) & (prices['Dates'] <= perf_end)]

		if perf_select == 'Cumulative':
			if len(compare_select) > 0:
				prices_display = normalise(prices_display, names_pd)

			alt_cum = performance_graph(prices_display, 'Price',names_pd['Name'].to_list() , names_pd, '.1f', 'Name')
			st.altair_chart(alt_cum, use_container_width=True)

			performance_table(cumulative, cum_periods)
		#-------------------------------------------------------------------------------------- Volatility

		if perf_select == 'Volatility':
			with col_compare_header[0]:
				st.write("**Volatility**")
				st.caption('Daily Volatility based on Total returns.')

			vol = volatility(prices_display, names_pd)
			alt_vol = performance_graph(vol, 'Volatility',names_pd['Name'].to_list() , names_pd, '.1%', 'Name')
			st.altair_chart(alt_vol, use_container_width=True)

			vol_cum, vol_periods = calc_vol_period(prices, perf_date_max ,names_pd)
			performance_table(vol_cum.to_dict(orient='records'), vol_periods)

		#-------------------------------------------------------------------------------------- Max drawdown
		if perf_select == 'Max Drawdown':
			with col_compare_header[0]:
				st.write("**Max drawdown**")
				st.caption('Daily Drawdown based on Total returns.')

			dd = drawdown(prices_display, names_pd)

			alt_dd = performance_graph(dd, 'Drawdown',names_pd['Name'].to_list() , names_pd, '.1%', 'Name')
			st.altair_chart(alt_dd, use_container_width=True)
			dd_period, dd_dates, dd_names = calc_drawdown_period(prices, perf_date_max, names_pd)
			performance_table(dd_period, dd_names, dd_dates)

	#-------------------------------------------------------------------------------------- Annualised
	if perf_select == 'Annualised':
		annualised, ann_periods = calc_ann_returns(cumulative, cum_periods)
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
                                              names_pd['Name'].to_list(), [ann_min, ann_max], True if i == 0 else False)
					st.altair_chart(alt_ann, use_container_width=True)

			performance_table(annualised.to_dict(orient='records'), ann_periods)

	#-------------------------------------------------------------------------------------- Calender
	if perf_select == 'Calendar':
		calendar, cal_periods = calc_cal_returns(prices, perf_date_min, perf_date_max, names_pd)
		st.altair_chart(legend, use_container_width=True)

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

		for i, col in enumerate(st.columns(len(cal_select))):
			with col:
				alt_cal = performance_grouped_bar_graph(calendar, 'Name', cal_select[i], cal_select[i], \
                                          names_pd['Name'].to_list(), [cal_min, cal_max], True if i==0 else False)
				st.altair_chart(alt_cal, use_container_width=True)

		performance_table(calendar[['ISINCode','Name'] + cal_select].to_dict(orient='records'), cal_select)
	st.write('')
	st.write('')

	return


def display_holding(isin, conn):
	holding_type, holding_all = get_holding([isin], conn)

	if len(holding_type) == 0:
		st.warning('Woops, No holding data available, we are working on it')
	else:
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

	return


def display_fundflow(isin, fundname, etfs_list, conn):

	flow_monthly = get_fundflow([isin], 'flow_USD', conn)

	col_flow_header = st.columns(2)

	with col_flow_header[0]:
		st.write("**Fund Flow**")
		st.caption('Total returns are shown on a Net Asset Value (NAV) basis, with gross income reinvested for dividend paying ETFs.')

	with col_flow_header[1]:
		compare_select = st.multiselect('Add to chart', etfs_list, format_func = lambda x: x[2] + ' ' + x[3])

	flow_date_min = min(flow_monthly['Dates'])
	flow_date_max = max(flow_monthly['Dates'])

	col_flow_header1 = st.columns([3,6,2,2])

	with col_flow_header1[0]:
		flow_currency_select = st.radio('Currency', ['USD', 'GBP', 'EUR'], key='flow2')

	with col_flow_header1[1]:
		period_select = st.radio('Period', ['1Y', '3Y', '5Y', '10Y', 'All', 'Custom period'], index=1, key='flow_period')

		if period_select == 'Custom period':
			flow_from = col_flow_header1[2].date_input('From', value=max(flow_date_min, flow_date_max-datetime.timedelta(365)), min_value=flow_date_min, max_value=flow_date_max)
			flow_to = col_flow_header1[3].date_input('To', value=flow_date_max, min_value=perf_from, max_value=flow_date_max)
		else: 
			flow_from = None
			flow_to = None

	flow_start, flow_end = calc_date_range(period_select, flow_from, flow_to, flow_date_min, flow_date_max)

	flows, flow_names_pd = add_compare_flow(compare_select, flow_currency_select, \
                                            flow_monthly, [(isin, fundname)], \
                                            flow_start, flow_end, conn
                                            )

	legend = legend_graph(flow_names_pd, 'Name',flow_names_pd['Name'].to_list() )
	st.altair_chart(legend, use_container_width=True)

	pp_flow = fundflow_graph(flows, flow_names_pd)
	st.plotly_chart(pp_flow, use_container_width=True)

	flow_period, flow_period_names = get_flow_period(flow_names_pd['ISINCode'].to_list(), flow_currency_select, flow_names_pd, conn)

	currency_symbole = '$' if flow_currency_select == 'USD' else  '£' if flow_currency_select == 'GBP' else '€'
	performance_table(flow_period, flow_period_names, header_suffix=' (' + currency_symbole + ')', is_num=False)
	st.write('')
	st.write('')

	return


def display_dividend(isin, fundname, conn):

	div = get_dividend([isin], conn)
	div_date_min = div['Dates'].iloc[0]
	div_date_max = div['Dates'].iloc[-1]
	div_default_date = max(div_date_max - datetime.timedelta(days=3652), div_date_min)

	div_header = st.columns([6,2,2,3])

	with div_header[0]:
		period_select = st.radio('Period', ['1Y', '3Y', '5Y', '10Y', 'All', 'Custom period'], index=1, key='div_period')

		if period_select == 'Custom period':
			div_from = div_header[1].date_input('From', value=max(div_date_min, div_date_max-datetime.timedelta(365)), min_value=div_date_min, max_value=div_date_max)
			div_to = div_header[2].date_input('To', value=div_date_max, min_value=div_from, max_value=div_date_max)
		else: 
			div_from = None
			div_to = None
	div_start, div_end = calc_date_range(period_select, div_from, div_to, div_date_min, div_date_max)

	alt_div = dividend_graph(div[(div['Dates'] >= div_start) & (div['Dates'] <= div_end)])
	st.altair_chart(alt_div, use_container_width=True)

	div_latest = get_div_latest([isin], pd.DataFrame([(isin, fundname)], columns=['ISINCode','Name']), conn)
	div_table(div_latest, ['Ex-Dividend Date', 'Dividend', 'Yield (%)', 'Dividend Growth'])
	st.write('')
	st.write('')

	return








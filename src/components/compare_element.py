import streamlit as st
import pandas as pd
import datetime
from src.data import get_holding
from src.data import get_dividend
from src.data import get_div_latest

from src.components.com_components import div_table

from src.viz import draw_holding_type
from src.viz import draw_full_holding_graph
from src.viz import performance_graph
from src.viz import fundflow_graph
from src.viz import legend_graph


def compare_holding(isins, names, conn):
	holding_type, holding_all = get_holding(isins, conn)
	type_dict = {'Country':'Geography', 'Sector':'Sector', 'Exchange': 'Exchange', 'Credit Rating': 'Credit Rating', 
			'Maturity': 'Maturity', 'Currency': 'Currency', 'Asset Class':'AssetClass'}
	types_unique = holding_type['HoldingType'].unique()

	type_dict = {k: type_dict[k] for i, k in enumerate(type_dict) if type_dict[k] in types_unique and type_dict[k]!='Top10'}
	names_dic = pd.DataFrame(names, index=isins, columns=['Name']).drop_duplicates().to_dict(orient='index')

	col_holding1, col_holding2 = st.columns(2)
	col_holding1.write('')
	label = col_holding1.radio('', [k for k in type_dict] + ['All Holdings'])

	if label != 'All Holdings':
		for i, col in enumerate(st.columns(len(isins))):
			with col:
				holding_data = holding_type[(holding_type['ISINCode'] == isins[i])&(holding_type['HoldingType'] == type_dict[label])]

				title = names_dic[isins[i]]['Name']
				label_width = 25 if label == 'Country' else 50 if label == 'Credit Rating' else 80 if label == 'Maturity' else 50 if label == 'Currency' else 120
				col.write('**'+title+'**')
				if len(holding_data) > 0 :
					alt_holding = draw_holding_type(holding_data, 'Flag' if label == 'Country' else 'Holding','', label_width)
					st.altair_chart(alt_holding, use_container_width=True)
				else:
					st.warning('Woops... Sorry, missing data. We are working on it')
	else:
		holding_selected = col_holding2.selectbox('', isins, index=0,format_func=lambda x: names_dic[x]['Name'])
		full_holding_data = holding_all[holding_all['ISINCode'] == holding_selected]
		if len(full_holding_data) > 0:
			alt_full_holding = draw_full_holding_graph(full_holding_data)
			st.altair_chart(alt_full_holding, use_container_width=True)
		else:
			st.warning('Woops... Sorry, missing data. We are working on it')

	st.write('')
	st.write('')

	return


def compare_div(div_isins, isins ,names, conn):
	div = get_dividend(div_isins, conn)
	names_pd = pd.DataFrame([(isins[i], names[i]) for i in range(len(isins)) if isins[i] in div_isins], columns = ['ISINCode','Name'])
	div = div.merge(names_pd, how='left', on='ISINCode')

	date_min = min(div['Dates'])
	date_max = max(div['Dates'])

	col_divs = st.columns([1, 2, 1, 1])
	col_divs[0].write('')
	col_divs[0].write('')
	div_select = col_divs[0].radio('', ['Dividend Yield', 'Dividend'])

	flow_from = col_divs[2].date_input('from', value=max(date_min, date_max - datetime.timedelta(365*5)) ,min_value=date_min, max_value=date_max )
	flow_to = col_divs[3].date_input('to', value=date_max ,min_value=flow_from, max_value=date_max)
				
	div_data = div[(div['Dates'] >= pd.to_datetime(flow_from, utc=True)) &(div['Dates'] <= pd.to_datetime(flow_to, utc=True))]

	if div_select == 'Dividend Yield':
		alt_div = performance_graph(div_data, 'Yield',names_pd['Name'].to_list() , names_pd, '.1%', 'Name')
		st.altair_chart(alt_div, use_container_width=True)

	if div_select == 'Dividend':
		legend = legend_graph(names_pd, 'Name',names_pd['Name'].to_list() )
		st.altair_chart(legend, use_container_width=True)
		div_pivot = div_data.pivot_table(index='Dates', columns='ISINCode', values='Dividend')

		div_pivot = div_pivot.fillna(0)
		pp_div = fundflow_graph(div_pivot, names_pd)
		st.plotly_chart(pp_div, use_container_width=True)

	div_latest = get_div_latest(div_isins, names_pd, conn)
	div_table(div_latest, ['Ex-Dividend Date', 'Dividend', 'Yield (%)', 'Dividend Growth'])

	st.write('')
	st.write('')


	return


import pandas as pd
import numpy as np
import streamlit as st

from src.style import color_palette
from src.data import num_format


@st.cache
def metric(label, value):
	return '<div data-testid="metric-container"><label r=""><div data-testid="stCaptionContainer" class="css-uyf2hz e16nr0p30"> ' + label + ' </div></label><div data-testid="stMetricValue"><div>' + value + '</div></div></div>'
	#return '<div data-testid="metric-container"><label r=""><div data-testid="stCaptionContainer"> ' + label + ' </div></label><div data-testid="stMetricValue"><div>' + value + '</div></div></div>'

def table_num(num, is_decimal=True):
	num_format = num*100 if is_decimal == True else num
	html_format = ''
	if num > 0:
		html_format = '<p style="color: #0DAA5B; text-align:right">' + str(round(num_format,1)) +'</p>'
	elif num < 0:
		html_format = '<p style="color: #D81414; text-align:right">' + str(round(num_format,1)) +'</p>'
	else:
		html_format = '<p style="text-align:right;"> - </p>'
	return html_format

def table_text(txt):
	html_format = ''
	if txt[0] == '-':
		html_format = '<p style="color: #D81414; text-align:right;">' + txt +'</p>'
	elif txt[0] != '0':
		html_format = '<p style="color: #0DAA5B; text-align:right;">' + txt +'</p>'
	else: 
		html_format = '<p style="text-align:right;"> - </p>'
	return html_format

def cell_color_text(txt, background_color, is_header=False, align='center'):
	html_format = ''

	background = 'background-color: rgb(231, 232, 240, 0.3);' if background_color == True else ''
	header = 'font-weight: bold;' if is_header==True else ''
	text_align = 'text-align:center;' if align=='center' else ''

	html_format = '<p style="' + text_align + background + header + '"> ' + str(txt) + '</p>'

	return html_format

def cell_color_num(num, background_color, is_header=False, currency = None):
	html_format = ''
	background = 'background-color: rgb(231, 232, 240, 0.3);' if background_color == True else ''
	header = 'font-weight: bold;' if is_header==True else ''
	font_color = '' if num == '-' else 'color: #0DAA5B;' if num > 0 else 'color: #D81414;' if num<0 else ''

	num_text = ''
	if currency is None:
		num_text = '-' if num == '-' else str(round(num,1))
	else:
		num_text = num_format(num, currency)

	html_format = '<p style="text-align:center;' + background + header + font_color + '"> ' + num_text + '</p>'

	return html_format
    

def table_num_with_caption(num, text):
	if num < 0:
		html_num = '<p style="color: #D81414; margin-block-end: 0; text-align:right;">' + str(round(num*100,1)) +'</p>'
	else:
		html_num = '<p style="margin-block-end:0; text-align:right;">-</p>'

	html_text = '<div data-testid="stCaptionContainer" class="css-uyf2hz e16nr0p30"><p style="text-align:right;">'+ str(text) + '</p></div>'
	html_format = '<div class="caption_date"><style> .caption_date{gap: 0;} </style><div>' + html_num + '</div>' + html_text + '</div>'
	return html_format


def color_bar(idx, text):
	main_palette, _ = color_palette()
	color = main_palette[idx]
	class_name = 'rectangle' + str(idx)

	html_format = '<div class = "'+ class_name + '"><style> .'+ class_name +' {width: 10px; height: 2rem; background: '+ color + '; display: inline-block;} </style></div>'
	html_format = '<div class="table_header">'+html_format+'<span>' + text + '</span></div>'

	return html_format


def performance_table(data, col_names, data2 = None, header_suffix = ' %', is_num=True, is_decimal=True, is_large=False):
	for i, col in enumerate(st.columns([3] + [1] * len(col_names))):
		if i == 0:
			col.write('Name')
		else:
			col.markdown('<p style="text-align:right">' + col_names[i-1] + header_suffix + '</p>', unsafe_allow_html=True)
	for i, k in enumerate(data):
		for j, col in enumerate(st.columns([3] + [1] * len(col_names))):
			if j == 0:
				col.markdown(color_bar(i, k['Name']), unsafe_allow_html=True)
				#col.write(k['FundName'])
			else:
				if data2 is None:
					item = k[col_names[j-1]]
					item = num_format(item) if is_large == True else item
					
					if is_num == True:
						col.markdown(table_num(item, is_decimal=is_decimal), unsafe_allow_html=True)
					else:
						col.markdown(table_text(item), unsafe_allow_html=True)
				else:
					col.markdown(table_num_with_caption(k[col_names[j-1]],data2[i][col_names[j-1]]), unsafe_allow_html=True)

	return

def div_table(data, col_names):
	for i, col in enumerate(st.columns([3] + [1] * len(col_names))):
		if i == 0:
			col.write('Name')
		else:
			col.write(col_names[i-1])
	for i, k in enumerate(data):
		for j, col in enumerate(st.columns([3] + [1] * len(col_names))):
			if j == 0:
				col.markdown(color_bar(i, k['Name']), unsafe_allow_html=True)
			else:				
				if j == len(col_names):
					col.markdown(table_text(k[col_names[j-1]]), unsafe_allow_html=True)
				else:
					col.write(k[col_names[j-1]])
				

	return


def similar_etfs_table(data, col_names):

	for i, col in enumerate(st.columns([3] + [1] * (len(col_names) - 1 ))):
		col.write(col_names[i])

	for i, k in enumerate(data):
		for j, col in enumerate(st.columns([3] + [1] * (len(col_names)-1))):
			if j == len(col_names)-1:
				col.markdown(table_num(k[col_names[j]]), unsafe_allow_html=True)
			else:
				col.write(k[col_names[j]])
				
	return

def compare_table(data):
	table_dict = {'ExchangeTicker': 'Ticker', 'Name': 'Name', 'ISINCode': 'ISIN','Rank': 'Rank' ,'Distribution': 'Use of Income', 
				'FundCurrency': 'Fund Currency', 'AUM': 'AUM', 'Volume': '3M Avg Volume', 'NAV': 'NAV', 'Cost': 'Cost (%)', 
				'IndexName': 'Index', 'Sector': 'Sector'}
	tickers = [k for k in data]
	for i, k in enumerate(table_dict):		
		for j, col in enumerate(st.columns([1] + [2] * len(data))):
			if j == 0:
				col.markdown(cell_color_text(table_dict[k], i%2+1, is_header=True), unsafe_allow_html=True)
			else:
				col.markdown(cell_color_text(data[tickers[j-1]][k], i%2+1), unsafe_allow_html=True)
	return

def perf_compare_table(data, table_dict, currency=None):
	
	tickers = [k for k in data]
	for i, k in enumerate(table_dict):		
		for j, col in enumerate(st.columns([1] + [2] * len(data))):
			if j == 0:
				col.markdown(cell_color_text(table_dict[k], i%2+1, is_header=True), unsafe_allow_html=True)
			else:
				item = data[tickers[j-1]][k] if k in data[tickers[j-1]] else '-'
				col.markdown(cell_color_num(item, i%2+1, is_header=False, currency=currency), unsafe_allow_html=True)

	return

def holding_table(data):
	tickers = [k for k in data]
	table_header = ['Top 10 Holdings'] + [str(i) for i in range(1,11)]
	for i in range(11):
		for j, col in enumerate(st.columns([2] + [3,1]*len(data))):
			if i == 0:
				if j==0:
					col.markdown(cell_color_text('Top 10 Holdings (%)', i%2+1,is_header=True), unsafe_allow_html=True)
				elif j%2 == 0:
					holdings = data[tickers[int(j/2)-1]]
					weight = round(sum([i['Weight'] for i in holdings]),2)
					col.markdown(cell_color_text(weight, i%2+1, is_header=True), unsafe_allow_html=True)
				else:
					col.markdown(cell_color_text('_', i%2+1), unsafe_allow_html=True)
			else:
				if j == 0:
					col.markdown(cell_color_text(table_header[i], i%2+1 ,is_header=True), unsafe_allow_html=True)
				else:
					ticker = tickers[int(j/2)-1]
					if j%2 == 1: 
						col.markdown(cell_color_text(data[ticker][i-1]['Name'], i%2+1, align='left'), unsafe_allow_html=True)
					else:
						col.markdown(cell_color_text(data[ticker][i-1]['Weight'], i%2+1), unsafe_allow_html=True)

	return








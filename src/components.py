import pandas as pd
import numpy as np
import streamlit as st

from src.style import color_palette


@st.cache
def metric(label, value):
	return '<div data-testid="metric-container"><label r=""><div data-testid="stCaptionContainer" class="css-uyf2hz e16nr0p30"> ' + label + ' </div></label><div data-testid="stMetricValue"><div>' + value + '</div></div></div>'


def table_num(num):
	html_format = ''
	if num > 0:
		html_format = '<p style="color: #0DAA5B;">' + str(round(num*100,1)) +'</p>'
	elif num < 0:
		html_format = '<p style="color: #D81414;">' + str(round(num*100,1)) +'</p>'
	else:
		html_format = '<p> - </p>'
	return html_format

def table_text(txt):
	html_format = ''
	if txt[0] == '-':
		html_format = '<p style="color: #D81414;">' + txt +'</p>'
	elif txt[0] != '0':
		html_format = '<p style="color: #0DAA5B;">' + txt +'</p>'
	else: 
		html_format = '<p> - </p>'
	return html_format



def table_num_with_caption(num, text):
	if num < 0:
		html_num = '<p style="color: #D81414; margin-block-end: 0">' + str(round(num*100,1)) +'</p>'
	else:
		html_num = '<p style="margin-block-end: 0">-</p>'

	html_text = '<div data-testid="stCaptionContainer" class="css-uyf2hz e16nr0p30"><p>'+ text + '</p></div>'
	html_format = '<div class="caption_date"><style> .caption_date{gap: 0;} </style><div>' + html_num + '</div>' + html_text + '</div>'
	return html_format


def color_bar(idx, text):
	main_palette, _ = color_palette()
	color = main_palette[idx]
	class_name = 'rectangle' + str(idx)

	html_format = '<div class = "'+ class_name + '"><style> .'+ class_name +' {width: 10px; height: 2rem; background: '+ color + '; display: inline-block;} </style></div>'
	html_format = '<div class="table_header">'+html_format+'<span>' + text + '</span></div>'

	return html_format


def performance_table(data, col_names, data2 = None, header_suffix = ' %', is_num=True):
	for i, col in enumerate(st.columns([3] + [1] * len(col_names))):
		if i == 0:
			col.write('Name')
		else:
			col.write(col_names[i-1] + header_suffix)
	for i, k in enumerate(data):
		for j, col in enumerate(st.columns([3] + [1] * len(col_names))):
			if j == 0:
				col.markdown(color_bar(i, k['FundName']), unsafe_allow_html=True)
				#col.write(k['FundName'])
			else:
				if data2 is None:
					if is_num == True:
						col.markdown(table_num(k[col_names[j-1]]), unsafe_allow_html=True)
					else:
						col.markdown(table_text(k[col_names[j-1]]), unsafe_allow_html=True)
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
				col.markdown(color_bar(i, k['FundName']), unsafe_allow_html=True)
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
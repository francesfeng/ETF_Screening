import streamlit as st
import pandas as pd
import numpy as np

import altair as alt
import psycopg2
import datetime

import screening
import compare 
import port

from streamlit_option_menu import option_menu

from src import style
from src.data import init_connection

st.set_page_config(layout="wide")
alt.themes.register("lab_theme", style.lab_theme)
alt.themes.enable("lab_theme")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

conn = init_connection()


if 'compare' not in st.session_state:
  st.session_state.compare = []

if 'clicked' not in st.session_state:
  st.session_state.clicked = {}

if 'default_page' not in st.session_state:
  st.session_state.default_page = 0
#-------------------------------------------------------Menu

if 'etfs' not in st.session_state:
  st.session_state.etfs = []

def add_compare():
  st.session_state.etfs.append(1)


st.write('This is home page')
st.button('Add ETF', on_click = add_compare)

script = 'screening'
if main_menu == 'ETF Screening':
  st.session_state.default_page =0
  screening.app(conn)
  
elif main_menu == compare_item:
  st.session_state.default_page  = 1
  compare.app(conn)
else:
  st.session_state.default_page =2
  port.app()
#------------------------------------------------------


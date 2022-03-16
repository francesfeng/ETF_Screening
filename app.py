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

compare_item = "compare" + (' (' + str(len(st.session_state.compare)) + ')' if len(st.session_state.compare) > 0 else '')

main_menu = option_menu(None, ["ETF Screening", compare_item , "Portfolio"], 
    icons=['fullscreen-exit', 'chevron-bar-contract', "bounding-box"], 
    menu_icon="cast", default_index=st.session_state.default_page , orientation="horizontal")
#main_menu

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


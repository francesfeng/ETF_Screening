import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

from streamlit_option_menu import option_menu

from src import style
from react_table import selectable_data_table
from src.data import init_connection


import screening
import compare
import port


#-------------------------------------------------------Initialise

st.set_page_config(layout="wide")
alt.themes.register("lab_theme", style.lab_theme)
alt.themes.enable("lab_theme")

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

conn = init_connection()

#=================================================


if 'compare' not in st.session_state:
  st.session_state.compare = []

if 'port' not in st.session_state:
  st.session_state.port = []

if 'clicked' not in st.session_state:
  st.session_state.clicked = {}

if 'default_home_page' not in st.session_state:
  st.session_state.default_home_page = 0
#-------------------------------------------------------Menu


main_menu = option_menu(None, ['ETF Screening', 'Compare', 'Portfolio'],
                      icons = ['fullscreen-exit', 'bounding-box', 'bounding-box'], 
                      default_index = st.session_state.default_home_page, orientation='horizontal',
                      styles={
        "container": {"padding": "0!important", "background-color": "#1830B7"},
        "icon": {"color": "white", "font-size": "18px"}, 
        "nav-link": {"font-size": "25px", "color": "#E7E8F0","font-family": "sans-serif" ,"font-weight": "lighter", "text-align": "left", "margin":"0px", "--hover-color": "#4659C5"},
        "nav-link-selected": {"background-color": "#1830B7", "color": "white", "font-weight": "bold"},
    })

script = 'screening'
if main_menu == 'ETF Screening':
  st.session_state.default_home_page =0
  screening.app(conn)
  
elif main_menu == 'Compare':
  st.session_state.default_home_page  = 1
  compare.app(conn)
else:
  st.session_state.default_home_page =2
  port.app(conn)
#------------------------------------------------------


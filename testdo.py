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

#st.set_page_config(layout="wide")
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


#-------------------------------------------------------Menu



#st.write('This is home page')
screening
  
st.button('Compare', on_click=compare.app, args=(conn,))

st.button('Portfolio', on_click=port.app)

#------------------------------------------------------


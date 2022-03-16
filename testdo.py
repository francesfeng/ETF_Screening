import pandas as pd
import streamlit as st

from src.data import init_connection


conn = init_connection()

query = """ SELECT * FROM calc_div_yield LIMIT 10 """

st.write(pd.read_sql(query, con=conn))
import os

import streamlit as st
import streamlit.components.v1 as components

parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "frontend/build")

#st.write(build_dir)

_selectable_data_table = components.declare_component(

    #"selectable_data_table", url="http://localhost:3001",

    "selectable_data_table", path=build_dir,
)



def selectable_data_table(data, key=None):
    st.write(build_dir)
    return _selectable_data_table(data=data, default=[], key=key)




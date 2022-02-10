import pandas as pd

import streamlit as st

from react_table import selectable_data_table


data = [
  { "id": 1, "lastName": 'Snow', "firstName": 'Jon', "age": 35 },
  { "id": 2, "lastName": 'Lannister', "firstName": 'Cersei', "age": 42 },
  { "id": 3, "lastName": 'Lannister', "firstName": 'Jaime', "age": 45 },
  { "id": 4, "lastName": 'Snow', "firstName": 'Jon', "age": 35 },
  { "id": 5, "lastName": 'Lannister', "firstName": 'Cersei', "age": 42 },
  { "id": 6, "lastName": 'Lannister', "firstName": 'Jaime', "age": 45 },

];

#df = pd.DataFrame(raw_data, columns=["First Name", "Last Name", "Age"])

#st.write(df)

st.title("The is the beginning of my table")

rows = selectable_data_table(data)
if rows:
    st.write("You have selected", rows)

st.write("here is the end")


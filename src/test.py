from google.cloud.sql.connector import connector
import sqlalchemy
import streamlit as st
import pg8000
import os

import pandas as pd


from google.oauth2 import service_account


credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)


st.write(credentials)


# connector.connect(
#      "project:region:instance",
#      "pg8000",
#      user="service-acc-etfs@green-diagram-337510.iam.gserviceaccount.com",
#      db="postgres",
#      enable_iam_auth=True,
#  )


# @st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None, 'builtins.weakref': lambda _: None})
# def init_connection_engine() -> sqlalchemy.engine.Engine:
#     def getconn() -> pg8000.dbapi.Connection:
#         conn: pg8000.dbapi.Connection = connector.connect(
#         	"project:region:instance",
#         	"pg8000",
#         	enable_iam_auth=True,
#             **st.secrets["gcp_service_account"]
#         )
#         return conn


#     engine = sqlalchemy.create_engine(
#         "postgresql+pg8000://",
#         creator=getconn,
#     )
#     engine.dialect.description_encoding = None
#     return engine

# conn = init_connection_engine
# def getconn():
#     return connector.connect(
#         
#          **st.secrets["gcp_service_account"],
#          enable_iam_auth=True,
#     )
def getconn() -> pg8000.dbapi.Connection:
    conn: pg8000.dbapi.Connection = connector.connect(
        "project:region:instance",
        "pg8000",
        user="service-acc-etfs@green-diagram-337510.iam.gserviceaccount.com",
     	db="postgres",
     	enable_iam_auth=True,
     	#**st.secrets["gcp_service_account"],
    )
    return conn



# pool = sqlalchemy.create_engine(
#     "postgresql+pg8000://",
#     creator=getconn(),
# )

# def run_query(query, conn):
#     with conn.cursor() as cur:
#         cur.execute(query)
#         return cur.fetchall()

pool = sqlalchemy.create_engine(
    "postgresql+pg8000://",
    creator=getconn,
)

st.write('pool is', pool)

query = sqlalchemy.text("SELECT * FROM funds limit 10")

with pool.connect() as db_conn:
	results = db_conn.execute(query).fetchall()
	st.write(results)

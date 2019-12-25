import cx_Oracle
import pandas as pd

# Barra风险数据
user,password='riskdata','riskdata'
dsn=cx_Oracle.makedsn('172.16.100.188','1522','markdb')
def ConnectOracle(user,password,dsn):
    connect=cx_Oracle.connect(user,password,dsn)
    if connect:
        print("链接成功")
    return connect
connect=ConnectOracle(user,password,dsn)

sql_order_get_barra_table_names='SELECT TABLE_NAME FROM USER_TABLES'
table_names=pd.read_sql(sql_order_get_barra_table_names,connect)
for i in table_names.index[3:4]:
    sql_order="select * from "+table_names.iloc[i,0]
    df=0
    df=pd.read_sql(sql_order,connect)
    df.to_pickle('DataSets/'+table_names.iloc[i,0])

# 组合数据
user,password='risk_read','riskRead2019'
dsn=cx_Oracle.makedsn('172.16.100.230','1521','xrisk')
connect=ConnectOracle(user,password,dsn)
sql_order="select * from xrisk.tcrp_hld"
trcp_hld=pd.read_sql(sql_order,connect);df.head()

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
sql_order="select * from BARRA_FACTORRET"
df=pd.read_sql(sql_order,connect);df.head()

# 组合数据
user,password='risk_read','riskRead2019' # risk_read 密码：riskRead2019
dsn=cx_Oracle.makedsn('172.16.100.230','1521','xrisk')
connect=ConnectOracle(user,password,dsn)
sql_order="select * from all_tab_comments"
df=pd.read_sql(sql_order,connect);df.head()
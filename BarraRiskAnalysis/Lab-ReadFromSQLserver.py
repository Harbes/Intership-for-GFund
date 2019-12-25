import pymssql
import pandas as pd

# 链接数据库
server,user,password,database='172.16.100.7','mikuser','mikuser','NWindDB' # 服务器填写IP地址
def ConnectSQLserver(server,user,password,database):
    connect=pymssql.connect(server,user,password,database)
    if connect:
        print('链接成功')
    return connect
connect=ConnectSQLserver(server,user,password,database)

# 设定sql读取命令，转化为pandas.DataFrame
sql_order="SELECT * from ASHAREBALANCESHEET WHERE wind_code = '000421.sz' ORDER BY ANN_DT DESC"
df=pd.read_sql(sql_order,connect);df.head()
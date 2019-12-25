import cx_Oracle
import pymssql
def ConnectOracle(user,password,dsn):
    '''
    用于链接barra、Xrisk等数据库
    '''
    connect=cx_Oracle.connect(user,password,dsn)
    if connect:
        print("链接成功")
    return connect
def ConnectSQLserver(server,user,password,database):
    '''
    用于链接wind数据库
    '''
    connect=pymssql.connect(server,user,password,database)
    if connect:
        print('链接成功')
    return connect


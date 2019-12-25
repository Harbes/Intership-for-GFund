import cx_Oracle
import pymssql

# 链接oracle，例如Barra风险数据库，投资组合数据库
def ConnectOracle(user,password,dsn):
    '''
    范例：
    options_barra_database={'user':'riskdata',
                   'password':'riskdata',
                   'dsn':cx_Oracle.makedsn('172.16.100.188','1522','markdb')}
    options_portfolio_database={'user':'risk_read',
                   'password':'riskRead2019',
                   'dsn':cx_Oracle.makedsn('172.16.100.230','1521','xrisk')}
    connect_barra_risk=ConnectOracle(**options_barra_database)
    connect_portfolio_database=ConnectOracle(**options_portfolio_database)
    '''
    connect=cx_Oracle.connect(user,password,dsn)
    if connect:
        print("链接成功")
    return connect


# 链接SQL server，例如wind数据库
def ConnectSQLserver(server,user,password,database):
    '''
    范例：
    options_WindDatabase={'server':'172.16.100.7',
                      'user':'mikuser',
                      'password':'mikuser',
                      'database':'NWindDB'}
    connect=ConnectSQLserver(**options_WindDatabase)
    '''
    connect=pymssql.connect(server,user,password,database)
    if connect:
        print('链接成功')
    return connect

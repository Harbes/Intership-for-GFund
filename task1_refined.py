# code习惯说明：
#   变量小写，且使用“_”连接
#       sql开头：表示字符串形式的sql命令
#   函数首字母大写，且不使用“_”连接

## todo 任务目标，既能分析单个组合，又能运行所有组合分析

import pandas as pd
import cx_Oracle
import pymssql
import matplotlib.pyplot as plt
import datetime


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
def ConnectSQLserver(server,user,password,database):
    connect=pymssql.connect(server,user,password,database)
    if connect:
        print('链接成功')
    return connect

def GenerateFactorNamesCombination(list_format=True, factors='all'):
    '''
    :param factors: all,style
    :return:
    '''
    factor_name_excluding_industry_country = [
    'CNE5S_BETA', 'CNE5S_BTOP', 'CNE5S_EARNYILD', 'CNE5S_GROWTH', 'CNE5S_LEVERAGE', 'CNE5S_LIQUIDTY',
    'CNE5S_MOMENTUM', 'CNE5S_RESVOL', 'CNE5S_SIZE', 'CNE5S_SIZENL']
    factor_name_all = ['CNE5S_AERODEF', 'CNE5S_AIRLINE',
                       'CNE5S_AUTO', 'CNE5S_BANKS', 'CNE5S_BETA', 'CNE5S_BEV', 'CNE5S_BLDPROD',
                       'CNE5S_BTOP', 'CNE5S_CHEM', 'CNE5S_CNSTENG', 'CNE5S_COMSERV',
                       'CNE5S_CONMAT', 'CNE5S_CONSSERV', 'CNE5S_COUNTRY', 'CNE5S_DVFININS',
                       'CNE5S_EARNYILD', 'CNE5S_ELECEQP', 'CNE5S_ENERGY', 'CNE5S_FOODPROD',
                       'CNE5S_GROWTH', 'CNE5S_HDWRSEMI', 'CNE5S_HEALTH', 'CNE5S_HOUSEDUR',
                       'CNE5S_INDCONG', 'CNE5S_LEISLUX', 'CNE5S_LEVERAGE', 'CNE5S_LIQUIDTY',
                       'CNE5S_MACH', 'CNE5S_MARINE', 'CNE5S_MATERIAL', 'CNE5S_MEDIA',
                       'CNE5S_MOMENTUM', 'CNE5S_MTLMIN', 'CNE5S_PERSPRD', 'CNE5S_RDRLTRAN',
                       'CNE5S_REALEST', 'CNE5S_RESVOL', 'CNE5S_RETAIL', 'CNE5S_SIZE',
                       'CNE5S_SIZENL', 'CNE5S_SOFTWARE', 'CNE5S_TRDDIST', 'CNE5S_UTILITIE']

    if factors=='all':
        factor_list=factor_name_all
    elif factors=='style':
        factor_list=factor_name_excluding_industry_country
    else:
        factor_list=set(factor_name_all).difference(factor_name_excluding_industry_country)
    if not list_format:
        factor_list_for_sql = ''
        for i in factor_list[:-1]:
            factor_list_for_sql = factor_list + i + ','
        factor_list_for_sql += factor_list[-1]
        return factor_list_for_sql
    else:
        return factor_list
def GetFactorReturnsFromSQL(factors='all'):
    factor_list=GenerateFactorNamesCombination(factors=factors)
    sql_get_factor_returns = 'select TRADINGDATE,FACTOR,DLYRETURN from BARRA_FACTORRET where TRADINGDATE BETWEEN ' + \
                             start_date + ' and ' + end_date + ' and Factor in ' + str(tuple(factor_list))
    factor_returns = pd.read_sql(sql_get_factor_returns, connect_barra).set_index(
        ['TRADINGDATE', 'FACTOR']).sort_index().unstack()
    factor_returns.columns = factor_returns.columns.droplevel()
    return factor_returns
def GetAssetExposureFromSQL(factors='all'):
    factor_names = GenerateFactorNamesCombination(list_format=False,factors=factors)
    sql_get_asset_exposure = 'select SECUCODE,TRADINGDATE,' + factor_names + ' from barra_assetexposure where TRADINGDATE BETWEEN ' + start_date + ' and ' + end_date
    asset_exposure = pd.read_sql(sql_get_asset_exposure, connect_barra).set_index(
        ['TRADINGDATE', 'SECUCODE']).sort_index()
    asset_exposure = asset_exposure.loc[~asset_exposure.index.duplicated(keep='last')]  # 删除重复数据
    return asset_exposure

#if __name__ is '__main__':

options_barra_database={'user':'riskdata',
                   'password':'riskdata',
                   'dsn':cx_Oracle.makedsn('172.16.100.188','1522','markdb')}
connect_barra=ConnectOracle(**options_barra_database)
options_xrisk_database={'user':'risk_read',
                   'password':'riskRead2019',
                   'dsn':cx_Oracle.makedsn('172.16.100.230','1521','xrisk')}
connect_xrisk=ConnectOracle(**options_xrisk_database)
options_winddb_datebase={'server':'172.16.100.7',
                     'user':'mikuser',
                     'password':'mikuser',
                     'database':'NWindDB'}
connect_winddb=ConnectSQLserver(**options_winddb_datebase)


start_date,end_date=' 20180701','20180730' # todo 如果没有时间限制？--在组合分析时常用

# 从barra数据库读取factor return
factor_returns=GetFactorReturnsFromSQL()
# 从barra数据库读取asset exposure数据
asset_exposure=GetAssetExposureFromSQL()
tmp_sql='select * from barra_assetexposure where TRADINGDATE BETWEEN  20180701 and 20180705'
tmp_exposure=pd.read_sql(tmp_sql,connect_barra)
tmp_exposure.columns




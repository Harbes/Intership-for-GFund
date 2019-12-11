# code习惯说明：
#   变量小写，且使用“_”连接
#       sql开头：表示字符串形式的sql命令
#   函数首字母大写，且不使用“_”连接

## todo 任务目标，既能分析单个组合，又能运行所有组合分析

import pandas as pd
import numpy as np
import cx_Oracle
import pymssql
import pymysql
import matplotlib.pyplot as plt
import datetime

# 小工具
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
        factor_list=list(set(factor_name_all).difference(factor_name_excluding_industry_country))
    if not list_format:
        factor_list_for_sql = ''
        for i in factor_list[:-1]:
            factor_list_for_sql = factor_list_for_sql + i + ','
        factor_list_for_sql += factor_list[-1]
        return factor_list_for_sql
    else:
        return factor_list
def DatetimeBarraToXrisk(date):
    return datetime.datetime.strptime(str.strip(date),'%Y%m%d').strftime('%Y-%m-%d')
# step1: 准备数据
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
# step2: 读取数据
def GetPrecloseFromWindDB():
    sql_get_preclose='select trade_dt,s_info_windcode,s_dq_preclose from ashareeodprices where trade_dt between ' + start_date + ' and ' + end_date
    preclose=pd.read_sql(sql_get_preclose,connect_winddb)
    preclose['trade_dt']=pd.to_datetime(preclose['trade_dt'],format='%Y%m%d')
    preclose['s_info_windcode'] = preclose['s_info_windcode'].str.slice_replace(6, repl='')
    preclose=preclose.set_index(['trade_dt','s_info_windcode']).sort_index().loc[~preclose.index.duplicated(keep='last')] # 去重
    return preclose['s_dq_preclose']
def GetFactorReturnsFromBarra(factors='all'):
    factor_list=GenerateFactorNamesCombination(factors=factors)
    sql_get_factor_returns = 'select TRADINGDATE,FACTOR,DLYRETURN from BARRA_FACTORRET where TRADINGDATE BETWEEN ' + \
                             start_date + ' and ' + end_date + ' and Factor in ' + str(tuple(factor_list))
    factor_returns = pd.read_sql(sql_get_factor_returns, connect_barra)
    factor_returns['TRADINGDATE']=pd.to_datetime(factor_returns['TRADINGDATE'],format='%Y%m%d')
    factor_returns=factor_returns.set_index(['TRADINGDATE', 'FACTOR']).sort_index()
    #factor_returns.columns = factor_returns.columns.droplevel()
    return factor_returns['DLYRETURN']*100.0
def GetFactorCovarianceFromBarra(factors='all'):
    factor_list = GenerateFactorNamesCombination(factors=factors)
    sql_get_factor_covariance = 'select TRADINGDATE,FACTOR1,FACTOR2,COVARIANCE from BARRA_FACTORCOV where TRADINGDATE BETWEEN ' + \
                             start_date + ' and ' + end_date + ' and Factor1 in ' + str(tuple(factor_list))+' and (Factor2 in '+str(tuple(factor_list))+')'
    factor_covariance = pd.read_sql(sql_get_factor_covariance, connect_barra)
    factor_covariance['TRADINGDATE'] = pd.to_datetime(factor_covariance['TRADINGDATE'], format='%Y%m%d')
    factor_covariance = factor_covariance.set_index(['TRADINGDATE', 'FACTOR1','FACTOR2']).sort_index().unstack().fillna(0.0)
    t_set=set(factor_covariance.index.get_level_values(0))
    for t in t_set:
        factor_covariance.loc[t]=factor_covariance.loc[t].values+factor_covariance.loc[t].values.T-np.diag(np.diag(factor_covariance.loc[t].values))
    return factor_covariance
def GetAssetReturnsFromBarra():
    sql_get_asset_returns = 'select TRADINGDATE,SECUCODE,DLYRETURN from BARRA_ASSETRET where TRADINGDATE BETWEEN ' + \
                             start_date + ' and ' + end_date
    asset_returns = pd.read_sql(sql_get_asset_returns, connect_barra)
    asset_returns['TRADINGDATE']=pd.to_datetime(asset_returns['TRADINGDATE'],format='%Y%m%d')
    asset_returns=asset_returns.set_index(['TRADINGDATE', 'SECUCODE']).sort_index()
    asset_returns=asset_returns.loc[~asset_returns.index.duplicated(keep='last')]
    return asset_returns['DLYRETURN']
def GetSpecificReturnsFramBarra():
    sql_get_specific_returns = 'select TRADINGDATE,SECUCODE,SPECIFICRETURN from BARRA_SPECIFICRET where TRADINGDATE BETWEEN ' + \
                             start_date + ' and ' + end_date
    specific_returns = pd.read_sql(sql_get_specific_returns, connect_barra)
    specific_returns['TRADINGDATE'] = pd.to_datetime(specific_returns['TRADINGDATE'], format='%Y%m%d')
    specific_returns = specific_returns.set_index(['TRADINGDATE', 'SECUCODE']).sort_index()
    return specific_returns.loc[~specific_returns.index.duplicated(keep='last'),'SPECIFICRETURN']
def GetSpecificRiskFromBarra():
    sql_get_specific_risk = 'select TRADINGDATE,SECUCODE,SPECIFICRISK from BARRA_SPECIFICRISK where TRADINGDATE BETWEEN ' + \
                             start_date + ' and ' + end_date
    specific_risk = pd.read_sql(sql_get_specific_risk, connect_barra)
    specific_risk['TRADINGDATE'] = pd.to_datetime(specific_risk['TRADINGDATE'], format='%Y%m%d')
    specific_risk = specific_risk.set_index(['TRADINGDATE', 'SECUCODE']).sort_index()
    return specific_risk.loc[~specific_risk.index.duplicated(keep='last'),'SPECIFICRISK']

def GetAssetExposureFromBarra(factors='all'):
    factor_names = GenerateFactorNamesCombination(list_format=False,factors=factors)
    sql_get_asset_exposure = 'select SECUCODE,TRADINGDATE,' + factor_names + ' from barra_assetexposure where TRADINGDATE BETWEEN ' + start_date + ' and ' + end_date
    asset_exposure = pd.read_sql(sql_get_asset_exposure, connect_barra)
    asset_exposure['TRADINGDATE']=pd.to_datetime(asset_exposure['TRADINGDATE'],format='%Y%m%d')
    asset_exposure=asset_exposure.set_index(['TRADINGDATE', 'SECUCODE']).sort_index()
    asset_exposure =asset_exposure.loc[~asset_exposure.index.duplicated(keep='last')]  # todo 重复
    #asset_exposure=asset_exposure.groupby(level=(0,1)).sum()
    return asset_exposure
def GetBenchmarkWeightsFromWindDB(benchmark='HS300'):
    benchmark_set = {'HS300': ('i_weight', 'aindexhs300closeweight'),
                     'CSI500': ('weight', 'AIndexCSI500Weight'),
                     'SSE50': ('weight', 'AIndexSSE50Weight')  # 上证50只有20180830之前的数据
                     }
    sql_get_benchmark_weights='select trade_dt,s_con_windcode,'+benchmark_set[benchmark][0] +' from '+benchmark_set[benchmark][1] +' where trade_dt BETWEEN ' + start_date + ' and ' + end_date
    weights=pd.read_sql(sql_get_benchmark_weights,connect_winddb)
    weights['trade_dt']=pd.to_datetime(weights['trade_dt'],format='%Y%m%d')
    ## 注意：由于Barra数据库中股票代码没有SH之类的后缀，因此，来自winddb的股票代码需要去除后缀
    weights['s_con_windcode']=weights['s_con_windcode'].str.slice_replace(6,repl='')
    weights=weights.set_index(['trade_dt','s_con_windcode']).sort_index().loc[~weights.index.duplicated(keep='last')] # 去重
    #weights.columns=weights.columns.droplevel()
    return weights
def GetPortfolioWeightsFromXrisk(port_code=None):
    # todo weights=holding*precose/sum() --- 问题：现金分红？
    # todo 添加单个组合数据的读取
    sql_get_from_xrisk = "SELECT PORT_CODE,T_DATE,I_CODE,H_COUNT from xrisk.tcrp_hld WHERE (T_DATE BETWEEN '" + DatetimeBarraToXrisk(
        start_date) + "' AND '" + DatetimeBarraToXrisk(end_date) + "') and (A_TYPE= 'SPT_S')"
    if port_code is not None:
        if type(port_code) is str:
            sql_get_from_xrisk+= " and (PORT_CODE = '"+port_code+"')"
        else:
            sql_get_from_xrisk+=' and (PORT_CODE in '+str(tuple(port_code))+')'
    portfolio_count = pd.read_sql(sql_get_from_xrisk, connect_xrisk)
    portfolio_count['T_DATE']=pd.to_datetime(portfolio_count['T_DATE'],format='%Y-%m-%d')
    portfolio_count =portfolio_count.set_index(['T_DATE', 'I_CODE', 'PORT_CODE']).sort_index().groupby(level=(0, 1, 2)).sum().unstack()  # 使用groupby+sum是因为数据中出现了“分裂”的情况
    portfolio_count.columns = portfolio_count.columns.droplevel()
    preclose=GetPrecloseFromWindDB()
    portfolio_weights=(portfolio_count.reindex(preclose.index).mul(preclose,axis=0)).groupby(level=0).apply(lambda x:x/x.sum())
    return portfolio_weights
# step3：计算
def PortfolioExposure(asset_exposure,portfolio_weights=None):
    # todo
    if portfolio_weights is None:
        return asset_exposure.groupby(level=0).mean().stack()
    elif portfolio_weights.size/len(portfolio_weights)==1.0: #todo 这一层if似乎可以合并到最有一层if中，前提是portfolio_weights是个TN*1的DataFrame，而不是Series
        portfolio_exposure=(asset_exposure.reindex(portfolio_weights.index)*portfolio_weights.values).groupby(level=0).sum()/portfolio_weights.groupby(level=0).sum().values
        return portfolio_exposure.stack()
    else:
        portfolio_exposure=pd.DataFrame(np.nan,index=pd.MultiIndex.from_product((set(asset_exposure.index.get_level_values(0)),asset_exposure.columns)),columns=portfolio_weights.columns).sort_index()
        for p in portfolio_exposure.columns:
            portfolio_exposure[p]=(asset_exposure.mul(portfolio_weights[p].reindex(asset_exposure.index).values,axis=0).groupby(level=0).sum().div(portfolio_weights[p].reindex(asset_exposure.index).groupby(level=0).sum().values,axis=0)).stack()
        return portfolio_exposure
def PortfolioFactorRisk(portfolio_exposure,factor_covariance):
    t_set=set(portfolio_exposure.index.get_level_values(0))
    factor_risk=pd.DataFrame(np.nan,index=t_set,columns=portfolio_exposure.columns).sort_index()
    for t in factor_risk.index:
        for p in factor_risk.columns:
            factor_risk.loc[t,p]=(portfolio_exposure.loc[(t,slice(None)),p].values@factor_covariance.loc[t]).values@portfolio_exposure.loc[(t,slice(None)),p]
    return factor_risk
def PortfolioSpecificRisk(portfolio_weights,specific_risk):
    return (portfolio_weights**2.0).reindex(specific_risk.index).mul(specific_risk,axis=0).groupby(level=0).sum()

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

# todo 如果没有时间限制？--在组合分析时常用
start_date,end_date=' 20180701','20180730' # 全局变量，控制所有输入数据
factors='all' # 'style','others'
benchmark='HS300' #
port_code=None # '76C012'或者['76C012','76C012']



# 从barra数据库读取factor return、specifict return、factor cov
factor_returns=GetFactorReturnsFromBarra(factors='all')
specific_returns=GetSpecificReturnsFramBarra()
factor_covariance=GetFactorCovarianceFromBarra(factors='all')
asset_returns=GetAssetReturnsFromBarra()
# 从barra数据库读取asset exposure数据
asset_exposure=GetAssetExposureFromBarra(factors='all')
# 从wind数据库读取benchmark的权重数据
benchmark_weights = GetBenchmarkWeightsFromWindDB(benchmark='HS300')
# 从xrisk数据库读取并计算现实组合的权重数据
portfolio_weights=GetPortfolioWeightsFromXrisk(port_code=None) # 例如：76C012
#port_76=GetPortfolioWeightsFromXrisk(port_code='76C012')


# 计算组合的因子暴露以及超额暴露、超额收益率
market_exposure=PortfolioExposure(asset_exposure) #等权重的市场组合
benchmark_exposure=PortfolioExposure(asset_exposure, portfolio_weights=benchmark_weights)
portfolio_exposure=PortfolioExposure(asset_exposure,portfolio_weights=portfolio_weights)
excess_market_exposure= market_exposure.sub(benchmark_exposure,axis=0)
excess_portfolio_exposure=portfolio_exposure.sub(benchmark_exposure,axis=0) # DataFrame减去Series，建议使用.sub命令

#计算组合超额收益率
excess_market_return=excess_market_exposure.mul(factor_returns,axis=0)
excess_portfolio_return=excess_portfolio_exposure.mul(factor_returns.values,axis=0)

# 计算组合的因子收益率、特质收益率
portfolio_factor_return=portfolio_exposure.reindex(factor_returns.index).mul(factor_returns,axis=0).groupby(level=0).sum()
portfolio_specific_return=portfolio_weights.reindex(specific_returns.index).mul(specific_returns,axis=0).groupby(level=0).sum()

# todo  组合风险分解[倒推？？？]
# todo 组合收益率出现0？？？
portfolio_weights.loc[('2018-07-11',slice(None)),'002155'].dropna() # 结果： 601390
asset_returns.loc[('2018-07-11','601390')] # 结果0.0


portfolio_returns=portfolio_weights.reindex(asset_returns.index).mul(asset_returns,axis=0).groupby(level=0).sum().replace(0.0,np.nan)
#portfolio_returns-portfolio_factor_return-portfolio_specific_return # todo 存在误差？？？ 已知factor_return*100.0

portfolio_risk=portfolio_returns**2.0 # todo 组合收益率为0？
portfolio_factor_risk=PortfolioFactorRisk(portfolio_exposure,factor_covariance) # todo 量级感觉不对
portfolio_specific_risk=PortfolioSpecificRisk(portfolio_weights,specific_risk) # todo 量级不对


portfolio_risk-(portfolio_factor_risk+portfolio_specific_risk)
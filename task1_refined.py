# code习惯说明：
#   变量小写，且使用“_”连接
#       sql开头：表示字符串形式的sql命令
#   函数首字母大写，且不使用“_”连接
#

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
        # todo 剔除国家因子
        factor_list=list(set(factor_name_all).difference(factor_name_excluding_industry_country).difference(['CNE5S_COUNTRY']))
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
    # preclose数据用于生成组合的权重数据: w = preclose * holdings / sum(...)
    sql_get_preclose='select trade_dt,s_info_windcode,s_dq_preclose from ashareeodprices where trade_dt between ' + start_date + ' and ' + end_date
    preclose=pd.read_sql(sql_get_preclose,connect_winddb)
    preclose['trade_dt']=pd.to_datetime(preclose['trade_dt'],format='%Y%m%d')
    preclose['s_info_windcode'] = preclose['s_info_windcode'].str.slice_replace(6, repl='')
    preclose=preclose.set_index(['trade_dt','s_info_windcode']).sort_index().loc[~preclose.index.duplicated(keep='last')] # 去重
    return preclose['s_dq_preclose']
def GetFactorReturnsFromBarra(factors='all'):
    # 从Barra数据库读取factor return数据；通过factors参数设置不同因子集合
    factor_list=GenerateFactorNamesCombination(factors=factors)
    sql_get_factor_returns = 'select TRADINGDATE,FACTOR,DLYRETURN from BARRA_FACTORRET where TRADINGDATE BETWEEN ' + \
                             start_date + ' and ' + end_date + ' and Factor in ' + str(tuple(factor_list))
    factor_returns = pd.read_sql(sql_get_factor_returns, connect_barra)
    factor_returns['TRADINGDATE']=pd.to_datetime(factor_returns['TRADINGDATE'],format='%Y%m%d')
    factor_returns=factor_returns.set_index(['TRADINGDATE', 'FACTOR']).sort_index()
    return factor_returns['DLYRETURN']*100.0 # todo 量级问题
def GetFactorCovarianceFromBarra(factors='all'):
    # 从Barra数据库读取因子协方差矩阵
    factor_list = GenerateFactorNamesCombination(factors=factors)
    sql_get_factor_covariance = 'select TRADINGDATE,FACTOR1,FACTOR2,COVARIANCE from BARRA_FACTORCOV where TRADINGDATE BETWEEN ' + \
                             start_date + ' and ' + end_date + ' and Factor1 in ' + str(tuple(factor_list))+' and (Factor2 in '+str(tuple(factor_list))+')'
    factor_covariance = pd.read_sql(sql_get_factor_covariance, connect_barra)
    factor_covariance['TRADINGDATE'] = pd.to_datetime(factor_covariance['TRADINGDATE'], format='%Y%m%d')
    t_set = set(factor_covariance['TRADINGDATE'])
    factor_covariance = factor_covariance.set_index(['TRADINGDATE', 'FACTOR1','FACTOR2']).sort_index().unstack().fillna(0.0)
    #t_set = set(factor_covariance.index.get_level_values(0))
    for t in t_set:
        factor_covariance.loc[t]=factor_covariance.loc[t].values+factor_covariance.loc[t].values.T-np.diag(np.diag(factor_covariance.loc[t].values))
    factor_covariance.columns=factor_covariance.columns.droplevel()
    return factor_covariance
def GetAssetReturnsFromBarra(source='barra'):
    # 总Barra数据库读取asset returns数据
    if source.upper()=='BARRA':
        sql_get_asset_returns = 'select TRADINGDATE,SECUCODE,DLYRETURN from BARRA_ASSETRET where TRADINGDATE BETWEEN ' + \
                             start_date + ' and ' + end_date
        asset_returns = pd.read_sql(sql_get_asset_returns, connect_barra)
        asset_returns['TRADINGDATE']=pd.to_datetime(asset_returns['TRADINGDATE'],format='%Y%m%d')
        asset_returns=asset_returns.set_index(['TRADINGDATE', 'SECUCODE']).sort_index()
        asset_returns=asset_returns.loc[~asset_returns.index.duplicated(keep='last')]
        return asset_returns
    else:
        sql_='select trade_dt,s_info_windcode,s_dq_adjpreclose,s_dq_adjclose from ashareeodprices where trade_dt between '+start_date+' and '+end_date
        adj_prc=pd.read_sql(sql_,connect_winddb)
        adj_prc['trade_dt']=pd.to_datetime(adj_prc['trade_dt'],format='%Y%m%d')
        adj_prc['s_info_windcode'] = adj_prc['s_info_windcode'].str.slice_replace(6, repl='')
        asset_returns0=adj_prc.set_index(['trade_dt','s_info_windcode']).T.pct_change().T['s_dq_adjclose'].sort_index()*100.0
        return asset_returns0
    # todo 尝试使用wind数据库
    return asset_returns['DLYRETURN']
def GetAssetExposureFromBarra(factors='all'):
    # 从Barra数据库读取asset exposure
    factor_names = GenerateFactorNamesCombination(list_format=False,factors=factors)
    sql_get_asset_exposure = 'select SECUCODE,TRADINGDATE,' + factor_names + ' from barra_assetexposure where TRADINGDATE BETWEEN ' + start_date + ' and ' + end_date
    asset_exposure = pd.read_sql(sql_get_asset_exposure, connect_barra)
    asset_exposure['TRADINGDATE']=pd.to_datetime(asset_exposure['TRADINGDATE'],format='%Y%m%d')
    asset_exposure=asset_exposure.set_index(['TRADINGDATE', 'SECUCODE']).sort_index()
    asset_exposure =asset_exposure.loc[~asset_exposure.index.duplicated(keep='last')]  # todo 重复
    #asset_exposure=asset_exposure.groupby(level=(0,1)).sum()
    return asset_exposure
def GetSpecificReturnsFramBarra():
    # 从Barra数据库读取specific return
    sql_get_specific_returns = 'select TRADINGDATE,SECUCODE,SPECIFICRETURN from BARRA_SPECIFICRET where TRADINGDATE BETWEEN ' + \
                             start_date + ' and ' + end_date
    specific_returns = pd.read_sql(sql_get_specific_returns, connect_barra)
    specific_returns['TRADINGDATE'] = pd.to_datetime(specific_returns['TRADINGDATE'], format='%Y%m%d')
    specific_returns = specific_returns.set_index(['TRADINGDATE', 'SECUCODE']).sort_index()
    return specific_returns.loc[~specific_returns.index.duplicated(keep='last'),'SPECIFICRETURN']
def GetSpecificRiskFromBarra():
    # 从Barra数据库读取specific risk
    sql_get_specific_risk = 'select TRADINGDATE,SECUCODE,SPECIFICRISK from BARRA_SPECIFICRISK where TRADINGDATE BETWEEN ' + \
                             start_date + ' and ' + end_date
    specific_risk = pd.read_sql(sql_get_specific_risk, connect_barra)
    specific_risk['TRADINGDATE'] = pd.to_datetime(specific_risk['TRADINGDATE'], format='%Y%m%d')
    specific_risk = specific_risk.set_index(['TRADINGDATE', 'SECUCODE']).sort_index()
    return specific_risk.loc[~specific_risk.index.duplicated(keep='last'),'SPECIFICRISK']
def GetBenchmarkWeightsFromWindDB(benchmark='HS300'):
    # 从wind数据库读取benchmark的权重
    # todo 标普低波红利指数可能需要另设
    benchmark_set = {'HS300': ('i_weight', 'aindexhs300closeweight'),
                     'CSI500': ('weight', 'AIndexCSI500Weight'),
                     'SSE50': ('weight', 'AIndexSSE50Weight')  # 上证50只有20180830之前的数据
                     }
    benchmark=benchmark.upper()
    sql_get_benchmark_weights='select trade_dt,s_con_windcode,'+benchmark_set[benchmark][0] +' from '+benchmark_set[benchmark][1] +' where trade_dt BETWEEN ' + start_date + ' and ' + end_date
    weights=pd.read_sql(sql_get_benchmark_weights,connect_winddb)
    weights['trade_dt']=pd.to_datetime(weights['trade_dt'],format='%Y%m%d')
    ## 注意：由于Barra数据库中股票代码没有SH之类的后缀，因此，来自winddb的股票代码需要去除后缀
    weights['s_con_windcode']=weights['s_con_windcode'].str.slice_replace(6,repl='')
    weights=weights.set_index(['trade_dt','s_con_windcode']).sort_index().loc[~weights.index.duplicated(keep='last')] # 去重
    #weights.columns=weights.columns.droplevel()
    return weights*.01
def GetPortfolioWeightsFromXrisk(port_code=None):
    # todo 存在组合hold在某段时期为0的情况：'2019-06-14'，'2019-06-23'，'76H002'---> 先卖出，后买入
    sql_get_from_xrisk = "SELECT PORT_CODE,T_DATE,I_CODE,H_COUNT,H_EVAL_ADDED from xrisk.tcrp_hld WHERE (T_DATE BETWEEN '" + DatetimeBarraToXrisk(
        start_date) + "' AND '" + DatetimeBarraToXrisk(end_date) + "') and (A_TYPE= 'SPT_S')"
    if port_code is not None:
        if type(port_code) is str:
            sql_get_from_xrisk+= " and (PORT_CODE = '"+port_code+"')"
        else:
            sql_get_from_xrisk+=' and (PORT_CODE in '+str(tuple(port_code))+')'
    portfolio_data = pd.read_sql(sql_get_from_xrisk, connect_xrisk)
    portfolio_data.loc[portfolio_data['H_EVAL_ADDED']==0.0,'H_COUNT']=0.0
    portfolio_data['T_DATE']=pd.to_datetime(portfolio_data['T_DATE'],format='%Y-%m-%d')
    portfolio_count =portfolio_data[['T_DATE', 'I_CODE', 'PORT_CODE','H_COUNT']].set_index(['T_DATE', 'I_CODE', 'PORT_CODE']).sort_index().groupby(level=(0, 1, 2)).sum().unstack()  # 使用groupby+sum是因为数据中出现了“分裂”的情况
    portfolio_count.columns = portfolio_count.columns.droplevel()
    preclose = GetPrecloseFromWindDB()
    return portfolio_count.reindex(preclose.index).mul(preclose,axis=0).groupby(level=0).apply(lambda x:x/x.sum())#.groupby(level=0).sum()


# step3：计算
def PortfolioExposure(asset_exposure,weights=None):
    if weights is None:
        return asset_exposure.groupby(level=0).mean().stack()
    elif weights.size/len(weights)==1.0: #todo 这一层if似乎可以合并到最有一层if中，前提是portfolio_weights是个TN*1的DataFrame，而不是Series
        portfolio_exposure=(asset_exposure.reindex(weights.index)*weights).groupby(level=0).sum()
        return portfolio_exposure.stack()
    else:
        portfolio_exposure=pd.DataFrame(np.nan,index=pd.MultiIndex.from_product((set(asset_exposure.index.get_level_values(0)),asset_exposure.columns)),columns=weights.columns).sort_index()
        for p in portfolio_exposure.columns:
            portfolio_exposure[p]=asset_exposure.reindex(weights.index).mul(weights[p],axis=0).groupby(level=0).sum().stack()

        return portfolio_exposure
def PortfolioReturn(asset_returns, weights):
    # todo 组合return计算方式：
    # 个股收益率使用wind数据库计算，而不使用barra数据库中的数据
    return weights.reindex(asset_returns.index).mul(asset_returns, axis=0).groupby(level=0).sum().replace(0.0, np.nan)
def PortfolioFactorRisk(portfolio_exposure,factor_covariance):
    t_set=set(portfolio_exposure.index.get_level_values(0))
    factor_risk=pd.DataFrame(np.nan,index=t_set,columns=portfolio_exposure.columns).sort_index()
    for t in factor_risk.index:
        for p in factor_risk.columns:
            factor_risk.loc[t,p]=factor_covariance.loc[t].values@portfolio_exposure.loc[(t,factor_covariance.columns),p]@portfolio_exposure.loc[(t,factor_covariance.columns),p]
    return factor_risk
def PortfolioSpecificRisk(weights,specific_risk):
    return (weights**2.0).mul(specific_risk.reindex(weights.index),axis=0).groupby(level=0).sum()
def PortfolioFactorReturnDecom(portfolio_exposure,factor_returns):
    portfolio_factor_return_contribution=portfolio_exposure.reindex(factor_returns.index).mul(factor_returns,axis=0)
    style_factor_list=GenerateFactorNamesCombination(factors='style')
    other_factor_list=GenerateFactorNamesCombination(factors='others')#+['CNE5S_COUNTRY']
    portfolio_style_factor_return=portfolio_factor_return_contribution.loc[(slice(None),style_factor_list),slice(None)].groupby(level=0).sum()
    portfolio_other_factor_return=portfolio_factor_return_contribution.loc[(slice(None),other_factor_list),slice(None)].groupby(level=0).sum()
    return portfolio_style_factor_return,portfolio_other_factor_return
def PortfolioSpecificReturn(specific_returns,weights):
    return weights.reindex(specific_returns.index).mul(specific_returns,axis=0).groupby(level=0).sum()

#if __name__ is '__main__':
# 连接数据库
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

# 全局参数设置
start_date,end_date=' 20190603','20190712'
# factors='all' # 'style','others' # todo 添加到字段说明中
benchmark='HS300' #
port_code=None # '76C012'或者['76C012','76C012']


# 从barra数据库读取factor return、specifict return等数据
factor_returns=GetFactorReturnsFromBarra(factors='all')
specific_returns=GetSpecificReturnsFramBarra()
factor_covariance=GetFactorCovarianceFromBarra(factors='all')
specific_risk=GetSpecificRiskFromBarra()
asset_returns=GetAssetReturnsFromBarra()
asset_exposure=GetAssetExposureFromBarra(factors='all')
# 从wind数据库读取benchmark的权重数据
benchmark_weights = GetBenchmarkWeightsFromWindDB(benchmark='HS300')
# 从xrisk数据库读取并计算现实组合的每日权重数据
portfolio_weights=GetPortfolioWeightsFromXrisk(port_code=None) # 例如：76C012; '76H002'出现大量0收益率
#port_76=GetPortfolioWeightsFromXrisk(port_code='76C012')
p76=GetPortfolioWeightsFromXrisk(port_code='76H002') # 权重权重之和出现为0的情况，源于组合在某些日期出现 h_count 没有数据，数据缺失还是已经卖出？

# 计算组合的因子暴露以及超额暴露、超额收益率
market_exposure=PortfolioExposure(asset_exposure) #等权重的市场组合
benchmark_exposure=PortfolioExposure(asset_exposure, portfolio_weights=benchmark_weights)
portfolio_exposure=PortfolioExposure(asset_exposure,portfolio_weights=portfolio_weights)
excess_market_exposure= market_exposure.sub(benchmark_exposure,axis=0)
excess_portfolio_exposure=portfolio_exposure.sub(benchmark_exposure,axis=0) # DataFrame减去Series，建议使用.sub命令

# 计算组合的因子收益率、特质收益率
portfolio_style_factor_return,portfolio_other_factor_return=PortfolioFactorReturnDecom(portfolio_exposure,factor_returns)
portfolio_specific_return=PortfolioSpecificReturn()
# todo  组合风险分解[倒推？？？]
# todo 组合收益率出现0，屏蔽？ ---> 先卖出，后买入
#portfolio_weights.loc[('2018-07-11',slice(None)),'002155'].dropna() # 结果： 601390
#asset_returns.loc[('2018-07-11','601390')] # 结果0.0
portfolio_factor_risk=np.sqrt(PortfolioFactorRisk(portfolio_exposure,factor_covariance)) # todo 量级感觉不对
portfolio_specific_risk=np.sqrt(PortfolioSpecificRisk(portfolio_weights,specific_risk)) # todo 量级不对
np.sqrt((portfolio_factor_risk**2.0).mean())# '005443'
np.sqrt((portfolio_specific_risk**2.0).mean())
portfolio_factor_risk.iloc[-1]

((portfolio_returns['006195']*.01+1.0).cumprod()[-1]-1.0)*100.0


# 对组合收益率验证Barra结构模型
portfolio_returns=PortfolioReturn(asset_returns,portfolio_weights)
portfolio_returns-portfolio_style_factor_return-portfolio_other_factor_return-portfolio_specific_return # todo 存在误差？？？ 已知factor_return*100.0
((portfolio_returns*0.01+1).cumprod().iloc[-1]-1.0)*100.0 # '76H008' ,

benchmark_return=PortfolioReturn(asset_returns,benchmark_weights)
(benchmark_return*.01+1.0).cumprod().iloc[-1]

# todo 数据不匹配是否因为之前的文档没考虑数据重复问题？？？
portfolio_returns['006195'].sum()
portfolio_style_factor_return['006195'].sum()
portfolio_other_factor_return['006195'].sum() # 行业因子
portfolio_specific_return['006195'].sum() # todo 特质收益率是否包括国家因子？？？
(portfolio_returns-portfolio_style_factor_return-portfolio_other_factor_return-portfolio_specific_return).sum() # 国家因子的归属，改变了各成分的符号！！！
# todo 结果写入数据库，首先待确认要写入哪些数据？？？写入什么数据库？



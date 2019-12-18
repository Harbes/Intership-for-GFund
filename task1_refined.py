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
    :param list_format: 输出结果选择以列表或字符串形式呈现，分别用于dataframe的列、sql中读取变量合集
    :param factors: all,style,industry,other 控制不同类型的因子集合，分别对应：所有、风格类因子、行业类因子、国家因子
    :return: 列表形式或字符串形式的因子名称集合
    '''
    # Barra数据库中的因子名称集合
    factor_name_style = [
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

    # 输出不同因子集合
    if factors.upper()=='ALL':
        factor_list=factor_name_all
    elif factors.upper()=='STYLE':
        factor_list=factor_name_style
    elif factors.upper()=='INDUSTRY':
        # todo 剔除国家因子
        factor_list=list(set(factor_name_all).difference(factor_name_style).difference(['CNE5S_COUNTRY']))
    else:
        factor_list=['CNE5S_COUNTRY']

    # 根据需要，输出列表形式的因子名称集合或字符串形式的因子名称集合
    if not list_format:
        factor_list_for_sql = ''
        for i in factor_list[:-1]:
            factor_list_for_sql = factor_list_for_sql + i + ','
        factor_list_for_sql += factor_list[-1]
        return factor_list_for_sql
    else:
        return factor_list
def DatetimeBarraToXrisk(date):
    '''
    由于全局变量start_date、end_date以及Barra数据库中的日期以'20191212'形式的字符串呈现，
    而Xrisk数据库中的日期以'2019-12-12'形式呈现。故在读取Xrisk数据库时，需要对日期格式进行调整
    :param date: 输入'20191212'形式的日期
    :return:  输出'2019-12-12'形式的日期
    '''
    return datetime.datetime.strptime(str.strip(date),'%Y%m%d').strftime('%Y-%m-%d')
# step1: 准备数据
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
# step2: 读取数据
def GetPrecloseFromWindDB():
    '''
    从wind数据库获取股票上个交易日收盘价，用于计算组合权重 w = preclose * holdings / sum(...)
    :return: 生成(T*N)形式的series
    '''
    # 从数据库读取数据
    sql_get_preclose='select trade_dt,s_info_windcode,s_dq_preclose from ashareeodprices where trade_dt between ' + start_date + ' and ' + end_date
    preclose=pd.read_sql(sql_get_preclose,connect_winddb)

    #对字符串形式的日期转化为python的datatime
    preclose['trade_dt']=pd.to_datetime(preclose['trade_dt'],format='%Y%m%d')
    # 去除wind数据库中的股票代码后缀 .SH、.SZ
    preclose['s_info_windcode'] = preclose['s_info_windcode'].str.slice_replace(6, repl='')

    # 以(日期，股票代码)作为index，并去重
    preclose=preclose.set_index(['trade_dt','s_info_windcode']).sort_index().loc[~preclose.index.duplicated(keep='last')]
    return preclose['s_dq_preclose']
def GetFactorReturnsFromBarra(factors='all'):
    '''
     从Barra数据库读取factor return数据；通过factors参数设置不同因子集合
    :return: 生成(T*P)形式的Series
    '''
    # 读取数据
    factor_list=GenerateFactorNamesCombination(factors=factors)
    sql_get_factor_returns = 'select TRADINGDATE,FACTOR,DLYRETURN from BARRA_FACTORRET where TRADINGDATE BETWEEN ' + \
                             start_date + ' and ' + end_date + ' and Factor in ' + str(tuple(factor_list))
    factor_returns = pd.read_sql(sql_get_factor_returns, connect_barra)

    # 转化日期
    factor_returns['TRADINGDATE']=pd.to_datetime(factor_returns['TRADINGDATE'],format='%Y%m%d')

    # 以（日期，因子）为index
    factor_returns=factor_returns.set_index(['TRADINGDATE', 'FACTOR']).sort_index()

    # 输出结果时调整量级（%）
    return factor_returns['DLYRETURN']*100.0
def GetFactorCovarianceFromBarra(factors='all'):
    '''
    从Barra数据库读取因子协方差矩阵
    :param factors:
    :return:
    '''
    # 读取数据
    factor_list = GenerateFactorNamesCombination(factors=factors)
    sql_get_factor_covariance = 'select TRADINGDATE,FACTOR1,FACTOR2,COVARIANCE from BARRA_FACTORCOV where TRADINGDATE BETWEEN ' + \
                             start_date + ' and ' + end_date + ' and Factor1 in ' + str(tuple(factor_list))+' and (Factor2 in '+str(tuple(factor_list))+')'
    factor_covariance = pd.read_sql(sql_get_factor_covariance, connect_barra)

    # 转化日期格式
    factor_covariance['TRADINGDATE'] = pd.to_datetime(factor_covariance['TRADINGDATE'], format='%Y%m%d')

    # 生成(T*P)*P形式的因子协方差矩阵
    t_set = set(factor_covariance['TRADINGDATE'])
    factor_covariance = factor_covariance.set_index(['TRADINGDATE', 'FACTOR1','FACTOR2']).sort_index().unstack().fillna(0.0)
    # 由于cov(a,b)=cov(b,a)且barra数据库中只存有其中一项，因此需要根据对称关系生成完整的协方差矩阵
    for t in t_set:
        factor_covariance.loc[t]=factor_covariance.loc[t].values+factor_covariance.loc[t].values.T-np.diag(np.diag(factor_covariance.loc[t].values))
    # 去除columns索引中多余的双层结构
    factor_covariance.columns=factor_covariance.columns.droplevel()
    return factor_covariance
def GetAssetReturnsFromBarra(source='barra'):
    '''
    从Barra数据库读取asset returns数据，或从wind数据库中生成asset returns数据
    :param source: 控制asset returns数据来源：barra、wind
    :return: (T*N)形式的series
    '''
    if source.upper()=='BARRA':
        # 从barra数据库中读取数据
        sql_get_asset_returns = 'select TRADINGDATE,SECUCODE,DLYRETURN from BARRA_ASSETRET where TRADINGDATE BETWEEN ' + \
                             start_date + ' and ' + end_date
        asset_returns = pd.read_sql(sql_get_asset_returns, connect_barra)
        # 转化日期格式
        asset_returns['TRADINGDATE']=pd.to_datetime(asset_returns['TRADINGDATE'],format='%Y%m%d')
        # 设置（日期，股票代码）为index
        asset_returns=asset_returns.set_index(['TRADINGDATE', 'SECUCODE']).sort_index()
        # 剔除重复数据
        asset_returns=asset_returns.loc[~asset_returns.index.duplicated(keep='last')]
        return asset_returns['DLYRETURN']
    else:
        # 读取数据
        sql_='select trade_dt,s_info_windcode,s_dq_adjpreclose,s_dq_adjclose from ashareeodprices where trade_dt between '+start_date+' and '+end_date
        adj_prc=pd.read_sql(sql_,connect_winddb)
        # 转化日期格式
        adj_prc['trade_dt']=pd.to_datetime(adj_prc['trade_dt'],format='%Y%m%d')
        # 剔除wind股票代码的后缀 SH、SZ
        adj_prc['s_info_windcode'] = adj_prc['s_info_windcode'].str.slice_replace(6, repl='')
        # 计算股票日收益率，并调整量级
        asset_returns=adj_prc.set_index(['trade_dt','s_info_windcode']).T.pct_change().T['s_dq_adjclose'].sort_index()*100.0
        return asset_returns
def GetAssetExposureFromBarra(factors='all'):
    '''
    从Barra数据库读取asset exposure
    :param factors:
    :return: 生成(T*N)*P形式的DataFrame
    '''
    # 生成欲读取的因子集合
    factor_names = GenerateFactorNamesCombination(list_format=False,factors=factors)
    # 读取数据
    sql_get_asset_exposure = 'select SECUCODE,TRADINGDATE,' + factor_names + ' from barra_assetexposure where TRADINGDATE BETWEEN ' + start_date + ' and ' + end_date
    asset_exposure = pd.read_sql(sql_get_asset_exposure, connect_barra)
    # 转化日期格式
    asset_exposure['TRADINGDATE']=pd.to_datetime(asset_exposure['TRADINGDATE'],format='%Y%m%d')
    # 设置（日期，股票代码）为index
    asset_exposure=asset_exposure.set_index(['TRADINGDATE', 'SECUCODE']).sort_index()
    # 去重
    asset_exposure =asset_exposure.loc[~asset_exposure.index.duplicated(keep='last')]
    return asset_exposure
def GetSpecificReturnsFramBarra():
    '''
    从Barra数据库读取specific return
    :return: 生成(T*N)形式的Series
    '''
    # 读取数据
    sql_get_specific_returns = 'select TRADINGDATE,SECUCODE,SPECIFICRETURN from BARRA_SPECIFICRET where TRADINGDATE BETWEEN ' + \
                             start_date + ' and ' + end_date
    specific_returns = pd.read_sql(sql_get_specific_returns, connect_barra)
    # 转化日期格式
    specific_returns['TRADINGDATE'] = pd.to_datetime(specific_returns['TRADINGDATE'], format='%Y%m%d')
    # 设置（日期，股票代码）为index
    specific_returns = specific_returns.set_index(['TRADINGDATE', 'SECUCODE']).sort_index()
    # 去重，并输出series
    return specific_returns.loc[~specific_returns.index.duplicated(keep='last'),'SPECIFICRETURN']
def GetSpecificRiskFromBarra():
    '''
    从Barra数据库读取specific risk
    :return: (T*N)形式的Series
    '''
    # 读取数据
    sql_get_specific_risk = 'select TRADINGDATE,SECUCODE,SPECIFICRISK from BARRA_SPECIFICRISK where TRADINGDATE BETWEEN ' + \
                             start_date + ' and ' + end_date
    specific_risk = pd.read_sql(sql_get_specific_risk, connect_barra)
    # 转换日期格式
    specific_risk['TRADINGDATE'] = pd.to_datetime(specific_risk['TRADINGDATE'], format='%Y%m%d')
    # 设置（日期，股票代码）为index
    specific_risk = specific_risk.set_index(['TRADINGDATE', 'SECUCODE']).sort_index()
    # 去重，并输出Series
    return specific_risk.loc[~specific_risk.index.duplicated(keep='last'),'SPECIFICRISK']
def GetBenchmarkWeightsFromWindDB(benchmark='HS300'):
    '''
    从wind数据库读取benchmark的权重
    :param benchmark: 选择benchmark，具体参见函数内设置的benchmark_set
    :return: 生成(T*N)形式的Series
    '''
    # todo 标普低波红利指数可能需要另设
    # todo 权重设置问题；
    benchmark_set = {'HS300': ('i_weight', 'aindexhs300closeweight'), # 当日收盘权重
                     'CSI500': ('weight', 'AIndexCSI500Weight'),
                     'SSE50': ('weight', 'AIndexSSE50Weight')  # 上证50似乎只有20180830之前的数据
                     }
    benchmark=benchmark.upper()
    if benchmark=='HS300':
        sql_get_benchmark_weights = 'select trade_dt,s_con_windcode,' + benchmark_set[benchmark][0] + ' from ' + \
                                    benchmark_set[benchmark][
                                        1] + ' where trade_dt BETWEEN ' + str(int(start_date)-100) + ' and ' + end_date
        weights = pd.read_sql(sql_get_benchmark_weights, connect_winddb)
        weights['trade_dt'] = pd.to_datetime(weights['trade_dt'], format='%Y%m%d')
        ## 注意：由于Barra数据库中股票代码没有SH之类的后缀，因此，来自winddb的股票代码需要去除后缀
        weights['s_con_windcode'] = weights['s_con_windcode'].str.slice_replace(6, repl='')
        weights = weights.set_index(['trade_dt', 's_con_windcode']).sort_index().loc[
            ~weights.index.duplicated(keep='last')]  # 去重
        return 0.01*weights.unstack().shift(1).loc[start_date:].stack()['i_weight']
    else:
        sql_get_benchmark_weights='select trade_dt,s_con_windcode,'+benchmark_set[benchmark][0] +' from '+benchmark_set[benchmark][1] +' where trade_dt BETWEEN ' + start_date + ' and ' + end_date
        weights=pd.read_sql(sql_get_benchmark_weights,connect_winddb)
        weights['trade_dt']=pd.to_datetime(weights['trade_dt'],format='%Y%m%d')
        ## 注意：由于Barra数据库中股票代码没有SH之类的后缀，因此，来自winddb的股票代码需要去除后缀
        weights['s_con_windcode']=weights['s_con_windcode'].str.slice_replace(6,repl='')
        weights=weights.set_index(['trade_dt','s_con_windcode']).sort_index().loc[~weights.index.duplicated(keep='last')] # 去重
        #weights.columns=weights.columns.droplevel()
        return weights*.01
def GetPortfolioWeightsFromXrisk(port_code=None):
    '''
    从xrisk和wind数据库获取并计算组合权重
    :param port_code: 组合代码；若设置None，则选取所有股票组合
    :return:  生成(T*N)*P形式的DataFrame
    '''
    # 读取组合的holding等数据
    sql_get_from_xrisk = "SELECT PORT_CODE,T_DATE,I_CODE,H_COUNT,H_EVAL_ADDED from xrisk.tcrp_hld WHERE (T_DATE BETWEEN '" + DatetimeBarraToXrisk(
        start_date) + "' AND '" + DatetimeBarraToXrisk(end_date) + "') and (A_TYPE= 'SPT_S')"
    if port_code is not None:
        if type(port_code) is str:
            sql_get_from_xrisk+= " and (PORT_CODE = '"+port_code+"')"
        else:
            sql_get_from_xrisk+=' and (PORT_CODE in '+str(tuple(port_code))+')'
    portfolio_data = pd.read_sql(sql_get_from_xrisk, connect_xrisk)

    # 根据H_EVAL_ADDED是否为0，判断是否当日买入：若是，则剔除当日count数据（不计入当日收益率）
    portfolio_data.loc[portfolio_data['H_EVAL_ADDED']==0.0,'H_COUNT']=0.0

    # 转化日期格式
    portfolio_data['T_DATE']=pd.to_datetime(portfolio_data['T_DATE'],format='%Y-%m-%d')

    # 设置（日期，股票代码，组合代码）为index，并以此进行groupby再加总（使用groupby+sum是因为数据中出现了“分裂”的情况），最后unstack成(T*N)*P形式的DataFrame
    portfolio_count =portfolio_data[['T_DATE', 'I_CODE', 'PORT_CODE','H_COUNT']].set_index(['T_DATE', 'I_CODE', 'PORT_CODE']).sort_index().groupby(level=(0, 1, 2)).sum().unstack()
    # 去除columns中多余的双层索引结构
    portfolio_count.columns = portfolio_count.columns.droplevel()

    # 从wind数据库获取前日收盘价苏剧
    preclose = GetPrecloseFromWindDB()

    # 计算并输出组合权重 w=(count*preclose)/sum(...)
    return portfolio_count.reindex(preclose.index).mul(preclose,axis=0).groupby(level=0).apply(lambda x:x/x.sum())


# step3：计算
def PortfolioExposure(asset_exposure,weights=None):
    '''
    计算组合因子暴露 X_p=W_p'*X
    :param asset_exposure:
    :param weights:
    :return: 生成(T*K)*P的DataFrame
    '''
    # 如果不输入weights数据，默认是等权重的市场组合（市值加权的市场组合似乎更合理）
    if weights is None:
        return asset_exposure.groupby(level=0).mean().stack()

    else:
        # 若输入的是Series，将其转化为DataFrame
        weights=pd.DataFrame(weights)

        # 计算组合暴露：对于任意一个组合p，asset exposure[(T*N)*K]和组合权重[(T*N)*1] 按元素相乘，再依照T分组、加总得到T*K的DataFrame；经stack后，P个组合拼接得到(T*K)*P的DataFrame
        portfolio_exposure=pd.DataFrame(np.nan,index=pd.MultiIndex.from_product((set(asset_exposure.index.get_level_values(0)),asset_exposure.columns)),columns=weights.columns).sort_index()
        for p in portfolio_exposure.columns:
            # 设置(~asset_exposure.isnull()).reindex(weights.index).mul(weights[p],axis=0).groupby(level=0).sum()是为了防止数据有缺失，需要调整权重
            portfolio_exposure[p]=\
                (asset_exposure.reindex(weights.index).mul(weights[p],axis=0).groupby(level=0).sum()/
                 (~asset_exposure.isnull()).reindex(weights.index).mul(weights[p],axis=0).groupby(level=0).sum()).stack()
        return portfolio_exposure
def PortfolioReturn(asset_returns, weights):
    '''
    计算组合简单收益率（日频）
    :return: 生成 T*P 的DataFrame
    '''
    # 由于有些组合中间可能出现“卖出再买入”，这种不连续可能影响后续分析，因此只要出现空仓，就剔除之前的数据
    filter_condition=weights.groupby(level=0).sum().iloc[::-1].cumprod().iloc[::-1]!=0.0
    # 计算组合收益率(日频)
    return (weights.reindex(asset_returns.index).mul(asset_returns, axis=0).groupby(level=0).sum()/
            weights.reindex(asset_returns.index).mul(~asset_returns.isnull(), axis=0).groupby(level=0).sum())[filter_condition]

def PortfolioFactorRisk(portfolio_exposure,factor_covariance):
    '''
    计算组合因子风险：sqrt(X_p'*Sigma_f*X_p)
    :return: 生成 T*P 的DataFrame
    '''
    t_set=set(portfolio_exposure.index.get_level_values(0))
    factor_risk=pd.DataFrame(np.nan,index=t_set,columns=portfolio_exposure.columns).sort_index()
    for t in factor_risk.index:
        for p in factor_risk.columns:
            factor_risk.loc[t,p]=factor_covariance.loc[t].values@portfolio_exposure.loc[(t,factor_covariance.columns),p]@portfolio_exposure.loc[(t,factor_covariance.columns),p]
    return factor_risk**0.5
def PortfolioSpecificRisk(specific_risk,weights):
    '''
    计算组合特质风险：sqrt(W_p'*Cov*W_p)=sqrt(sum(W_p_i**2.0*specific_risk_i**2.0)))
    :return:  生成 T*P 的DataFrame
    '''
    return (weights**2.0).mul(specific_risk.reindex(weights.index)**2.0,axis=0).groupby(level=0).sum()**0.5

def PortfolioFactorReturnDecom(portfolio_exposure,factor_returns,weights):
    '''
    对组合收益率进行分解：风格因子贡献、行业因子贡献、国家因子贡献
    :return: 生成3个 T*P 的DataFrame
    '''
    # 用于剔除空仓组合无效数据
    filter_condition = weights.groupby(level=0).sum().iloc[::-1].cumprod().iloc[::-1] != 0.0

    # 计算不同因子对收益率的贡献：portfolio_exposure[(T*K)*P]与factor_returns[(T*K)]按元素相乘
    portfolio_factor_return_contribution=pd.DataFrame(portfolio_exposure.reindex(factor_returns.index).mul(factor_returns,axis=0))

    # 生成不同类因子集合
    style_factor_list=GenerateFactorNamesCombination(factors='style')
    industry_factor_list=GenerateFactorNamesCombination(factors='industry')#+
    country_factor_list=['CNE5S_COUNTRY']

    # 计算不同类因子收益率贡献：同类因子收益率贡献的加总
    portfolio_style_factor_return=portfolio_factor_return_contribution.loc[(slice(None),style_factor_list),slice(None)].groupby(level=0).sum()[filter_condition]
    portfolio_industry_factor_return=portfolio_factor_return_contribution.loc[(slice(None),industry_factor_list),slice(None)].groupby(level=0).sum()[filter_condition]
    portfolio_country_factor_return=portfolio_factor_return_contribution.loc[(slice(None),country_factor_list),slice(None)].groupby(level=0).sum()[filter_condition]
    return portfolio_style_factor_return,portfolio_industry_factor_return,portfolio_country_factor_return
def PortfolioSpecificReturn(specific_returns,weights):
    '''
    计算组合特质收益率：W_p'*e
    :return: 生成 T*P 的DataFrame
    '''
    return weights.reindex(specific_returns.index).mul(specific_returns,axis=0).groupby(level=0).sum()/\
           weights.reindex(specific_returns.index).mul(~specific_returns.isnull(),axis=0).groupby(level=0).sum()

#if __name__ is '__main__':
# 连接数据库
options_barra_database={'user':'riskdata',
                   'password':'riskdata',
                   'dsn':cx_Oracle.makedsn('172.16.100.188','1522','markdb')}
connect_barra=ConnectOracle(options_barra_database['user'],options_barra_database['password'],options_barra_database['dsn'])
options_xrisk_database={'user':'risk_read',
                   'password':'riskRead2019',
                   'dsn':cx_Oracle.makedsn('172.16.100.230','1521','xrisk')}
connect_xrisk=ConnectOracle(options_xrisk_database['user'],options_xrisk_database['password'],options_xrisk_database['dsn'])
options_winddb_datebase={'server':'172.16.100.7',
                     'user':'mikuser',
                     'password':'mikuser',
                     'database':'NWindDB'}
connect_winddb=ConnectSQLserver(options_winddb_datebase['server'],options_winddb_datebase['user'],options_winddb_datebase['password'],options_winddb_datebase['database'])

# 全局参数设置
start_date,end_date=' 20190604','20190712'
benchmark='HS300' #
port_code=None # '76C012'或者['76C012','76C012']


# 从barra数据库读取factor return、specifict return等数据
factor_returns=GetFactorReturnsFromBarra(factors='all')
specific_returns=GetSpecificReturnsFramBarra()
factor_covariance=GetFactorCovarianceFromBarra(factors='all')
specific_risk=GetSpecificRiskFromBarra()
asset_returns=GetAssetReturnsFromBarra()
asset_exposure=GetAssetExposureFromBarra(factors='all')
# 从wind数据库获取benchmark的权重数据
benchmark_weights = GetBenchmarkWeightsFromWindDB(benchmark='HS300')
# 从xrisk数据库读取并计算现实组合的每日权重数据
portfolio_weights=GetPortfolioWeightsFromXrisk(port_code=None) # 例如：76C012; '76H002'出现大量0收益率
#port_76=GetPortfolioWeightsFromXrisk(port_code='76C012')
#p76=GetPortfolioWeightsFromXrisk(port_code='76H002') # 权重权重之和出现为0的情况，源于组合在某些日期出现 h_count 没有数据，数据缺失还是已经卖出？

# 计算组合的因子暴露以及超额暴露、超额收益率
#market_exposure=PortfolioExposure(asset_exposure) #等权重的市场组合
benchmark_exposure=PortfolioExposure(asset_exposure, weights=benchmark_weights)
portfolio_exposure=PortfolioExposure(asset_exposure,weights=portfolio_weights)
#excess_market_exposure= market_exposure.sub(benchmark_exposure,axis=0)
#excess_portfolio_exposure=portfolio_exposure.sub(benchmark_exposure,axis=0) # DataFrame减去Series，建议使用.sub命令

# 计算组合的因子收益率、特质收益率
portfolio_style_factor_return,portfolio_industry_factor_return,portfolio_country_factor_return=PortfolioFactorReturnDecom(portfolio_exposure,factor_returns,portfolio_weights)
portfolio_specific_return=PortfolioSpecificReturn(specific_returns,portfolio_weights)
benchmark_style_factor_return,benchmark_industry_factor_return,benchmark_country_factor_return=PortfolioFactorReturnDecom(benchmark_exposure,factor_returns,benchmark_weights)
benchmark_specific_return=PortfolioSpecificReturn(specific_returns,benchmark_weights)

portfolio_returns=PortfolioReturn(asset_returns,portfolio_weights)
benchmark_return=PortfolioReturn(asset_returns,benchmark_weights)

# 对组合收益率验证Barra结构模型
# todo 累计收益率不匹配是因为不包括首日？权重问题？
#(portfolio_returns['006195'].iloc[1:]*.01+1.0).cumprod()*100.0-100.0 # todo 样本数据是不包括首日的？
def AggResults(start_date,end_date,port_code=None):
    res=pd.DataFrame(np.nan,index=['cumReturn','cumReturn_barra','SpecificReturn','FactorReturn','StyleFactorReturn','IndustryFactorReturn','CountryFactorReturn','TotalRisk','CommonFactorRisk','ResidualRisk'],
                     columns=list(portfolio_returns.columns)+list([benchmark]))
    res.loc['cumReturn']=\
        np.hstack((portfolio_returns.loc[start_date:end_date].sum(),benchmark_return.loc[start_date:end_date].sum()))# 直接使用port_ret可能会导致分解不成立
    res.loc['SpecificReturn']=np.hstack((portfolio_specific_return.loc[start_date:end_date].sum(),benchmark_specific_return.loc[start_date:end_date].sum()))
    res.loc['StyleFactorReturn']=np.hstack((portfolio_style_factor_return.loc[start_date:end_date].sum(),benchmark_style_factor_return.loc[start_date:end_date].sum()))
    res.loc['IndustryFactorReturn']=np.hstack((portfolio_industry_factor_return.loc[start_date:end_date].sum(),benchmark_industry_factor_return.loc[start_date:end_date].sum()))
    res.loc['CountryFactorReturn']=np.hstack((portfolio_country_factor_return.loc[start_date:end_date].sum(),benchmark_country_factor_return.loc[start_date:end_date].sum()))
    res.loc['FactorReturn']=res.loc[['StyleFactorReturn','IndustryFactorReturn','CountryFactorReturn']].sum()
    res.loc['cumReturn_barra']=res.loc[['FactorReturn','SpecificReturn']].sum()
    res.loc['CommonFactorRisk']=np.hstack((portfolio_factor_risk.loc[start_date:end_date].iloc[-1],np.nan))
    res.loc['ResidualRisk']=np.hstack((portfolio_specific_risk.loc[start_date:end_date].iloc[-1],np.nan))
    res.loc['TotalRisk']=np.sqrt((res.loc[['CommonFactorRisk','ResidualRisk']]**2.0).sum())
    if port_code is None:
        return res
    elif type(port_code)==str:
        return res[list([port_code])+[benchmark]]
    else:
        return res[list(port_code)+[benchmark]]
AggResults(start_date,end_date)

# test Barra decom
(portfolio_returns-portfolio_style_factor_return-portfolio_industry_factor_return-portfolio_country_factor_return-portfolio_specific_return)#.sum()


# todo 结果写入数据库，首先待确认要写入哪些数据？？？写入什么数据库？




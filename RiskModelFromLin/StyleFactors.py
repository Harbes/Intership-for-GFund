# 因子构建方式参见《20190620-渤海证券多因子模型研究系列之九：Barra风险模型(CNE6)之纯因子构建与因子合成》
import cx_Oracle
import pymssql
import pandas as pd
import numpy as np
from pandas.tseries.offsets import Day,DateOffset
from ConnectDatabase import ConnectSQLserver,ConnectOracle


start_date='20191201';end_date='20191212'

def GenerateSqlOrder(cols,table_name,start_date,end_date):
    # todo 注意：cols的顺序是‘交易日期，股票代码，核心变量’
    # 例如 'SELECT trade_dt,s_info_windcode,s_dq_mv from AShareEODDerivativeIndicator where (trade_dt>20191201) and (trade_dt<20191212)'
    return 'SELECT '+cols+' from '+table_name+' where trade_dt between '+start_date+' and '+end_date
def GenerateDataframeFromWinddb(cols,table_name,start_date,end_date):
    #cols='trade_dt,s_info_windcode,s_dq_mv';table_name='AShareEODDerivativeIndicator';
    # 生成读取数据库的命令（字符串）
    sql_order=GenerateSqlOrder(cols,table_name,start_date,end_date)
    # 读取数据，并设置（交易日期，股票代码）为index
    data=pd.read_sql(sql_order,connect_winddb).set_index(cols.split(',')[:2])
    # 去除重复的数据(默认保留最后一项)，然后unstack形成DataFrame
    data_df=data.loc[~data.index.duplicated(keep='last')].sort_index().unstack()
    # 转化时间格式，并删除columns中多余的双层结构
    data_df.index=pd.to_datetime(data_df.index,format='%Y%m%d')
    data_df.columns=data_df.columns.droplevel()
    return data_df
def GenerateWeightsFromHalflife(T,halflife=63):
    lam=0.5**(1.0/halflife)
    w=(1-lam)*lam**(np.arange(T))
    return w[::-1]/w.sum()
def GenerateBetaFromWeightedTimeSeriesRegression(Y, X, halflife=None):
    if halflife is None:
        w=pd.Series(1.0,index=Y.index)
    else:
        w=pd.Series(GenerateWeightsFromHalflife(len(Y),halflife=halflife),index=Y.index)*100.0
    X_=pd.concat([pd.Series(1.0,index=X.index),X],axis=1).mul(w,axis=0)
    Y_=Y.mul(w,axis=0)
    return (np.linalg.inv(X_.T@X_)@X_.T@Y_).iloc[1]
def GenerateStdFromWeightedMovingAverate(X, halflife=None):
    if halflife is None:
        w=pd.Series(1.0,index=X.index)
    else:
        w=pd.Series(GenerateWeightsFromHalflife(len(X),halflife=halflife),index=X.index)
    return ((X-X.mean())**2.0).mul(w,axis=0).sum()
def GenerateMeanFromWeightedMovingAverage(X,halflife=None):
    # todo 以0填充还是以mean填充，当前选择是mean
    w=pd.Series(GenerateWeightsFromHalflife(len(X),halflife=halflife),index=X.index)
    return X.apply(lambda x:x.fillna(x.mean())).mul(w,axis=0).sum()
def GenerateCummulativeRange(X):
    X_=X.fillna(0.0).cumsum()
    return X_.max()-X_.min()

def GetDailyStockReturns(start_date, end_date):
    sql_order=GenerateSqlOrder('trade_dt,s_info_windcode,s_dq_adjpreclose,s_dq_adjclose','ashareeodprices',start_date,end_date)
    ret_stock=GenerateDataframeFromWinddb('trade_dt,s_info_windcode,s_dq_adjclose','ashareeodprices',str(int(start_date)-200),end_date).pct_change()[start_date:]
    return ret_stock.replace(0.0,np.nan)*100.0
def GetBenchmarkReturns(start_date,end_date):
    return GenerateDataframeFromWinddb('trade_dt,s_info_windcode,pct_chg','HS300IEODPrices',start_date,end_date)['000300.SH']

def Generate_LogCap_MidCap(start_date,end_date):
    # todo 非交易日（例如周末）数据也存在
    # 生成 自由流通市值的DataFrame
    free_cap=GenerateDataframeFromWinddb('trade_dt,s_info_windcode,s_dq_mv','AShareEODDerivativeIndicator',start_date,end_date)
    # 剔除<=0.0的流通市值无效数据
    free_cap=free_cap[free_cap>0.0]
    log_cap=np.log(free_cap.replace(np.inf,np.nan))
    mid_cap=log_cap.agg(lambda x: x - x.mean(), axis=1).apply(lambda x: (x ** 3.0) - x * (x ** 4.0).sum() / (x ** 2.0).sum(),
                                                      axis=1)
    return log_cap,mid_cap

def Generate_Beta_HistSigma_DailyStd_CumRange(start_date,end_date):
    ret_bench = GetBenchmarkReturns(str(int(start_date) - 10000), end_date)
    ret_stock = GetDailyStockReturns(str(int(start_date) - 10000), end_date)
    #ret_stock=ret_stock[ret_stock!=0.0]
    # 若 halflife=None，则意味着等权重
    beta=ret_stock.loc[start_date:].apply(lambda x:GenerateBetaFromWeightedTimeSeriesRegression(ret_stock.loc[x.name - DateOffset(months=12):x.name], ret_bench.loc[x.name - DateOffset(months=12):x.name], halflife=63), axis=1)
    sigma_hist=ret_stock.loc[start_date:].apply(lambda x:(ret_stock.loc[x.name-DateOffset(months=12):x.name]-(pd.DataFrame(ret_bench.loc[x.name-DateOffset(months=12):x.name]).values@pd.DataFrame(beta.loc[x.name]).T.values)).std(),axis=1)
    # 若 halflife=None，则意味着等权重
    ret_std=ret_stock.loc[start_date:].apply(lambda x:GenerateStdFromWeightedMovingAverate(ret_stock.loc[x.name - DateOffset(months=12):x.name], halflife=42), axis=1)
    log_ret=np.log(ret_stock*.01+1.0)
    cum_range=log_ret.loc[start_date:].apply(lambda x:GenerateCummulativeRange(log_ret.loc[x.name-DateOffset(months=12):x.name]),axis=1).replace(0.0,np.nan)
    return beta,sigma_hist,ret_std.replace(0.0,np.nan),cum_range

def Generate_STOM_STOQ_STOY_ATVR(start_date,end_date):
    turnover=GenerateDataframeFromWinddb('trade_dt,s_info_windcode,s_dq_freeturnover','AShareEODDerivativeIndicator',str(int(start_date)-10000),end_date)
    STOM=turnover.loc[start_date:].apply(lambda x:turnover.loc[x.name-DateOffset(months=1):x.name].mean(),axis=1)
    STOQ=turnover.loc[start_date:].apply(lambda x:turnover.loc[x.name-DateOffset(months=3):x.name].mean(),axis=1)
    STOY=turnover.loc[start_date:].apply(lambda x:turnover.loc[x.name-DateOffset(months=12):x.name].mean(),axis=1)
    ATVR=turnover.loc[start_date:].apply(lambda x:GenerateMeanFromWeightedMovingAverage(turnover.loc[x.name-DateOffset(months=12):x.name],halflife=63),axis=1)
    return np.log(STOM[STOM>0.0]),np.log(STOQ[STOQ>0.0]),np.log(STOY[STOY>0.0]),ATVR[ATVR>0.0]

def Generate_STRREV(start_date,end_date):
    ret_stock = GetDailyStockReturns(str(int(start_date) - 10000), end_date)


if __name__ is '__main__':
    # 数据库参数设置
    options_winddb_datebase = {'server': '172.16.100.7',
                               'user': 'mikuser',
                               'password': 'mikuser',
                               'database': 'NWindDB'}
    connect_winddb = ConnectSQLserver(options_winddb_datebase['server'], options_winddb_datebase['user'],
                                      options_winddb_datebase['password'], options_winddb_datebase['database'])

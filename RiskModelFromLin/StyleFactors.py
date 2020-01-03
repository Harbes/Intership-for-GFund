# 因子构建方式参见《20190620-渤海证券多因子模型研究系列之九：Barra风险模型(CNE6)之纯因子构建与因子合成》
# todo 不同指标计算的股票池数量不同，最好能先统一股票池，然后对所有相关DataFrame进行reindex(columns=stock_pool)
# todo 部分指标wind存在，但似乎不全
import cx_Oracle
import pymssql
import pandas as pd
import numpy as np
from pandas.tseries.offsets import Day,DateOffset
from ConnectDatabase import ConnectSQLserver,ConnectOracle
from numba import jit

start_date='20191115';end_date='20191212'

# 小工具
def GenerateSqlOrder(cols,table_name,start_date=None,end_date=None):
    '''
    用于生成sql命令
    # 注意：cols的顺序是‘交易日期，股票代码，核心变量’，下同。
    # 例如 'SELECT trade_dt,s_info_windcode,s_dq_mv from AShareEODDerivativeIndicator where (trade_dt>20191201) and (trade_dt<20191212)'
    '''
    if start_date is None:
        if end_date is None:
            return 'SELECT ' + cols + ' from ' + table_name
        else:
            return 'SELECT '+cols+' from '+table_name+' where '+cols.split(',')[0]+' <= '+end_date
    else:
        if end_date is None:
            return 'SELECT ' + cols + ' from ' + table_name + ' where '+cols.split(',')[0]+' >= ' + start_date
        else:
            return 'SELECT '+cols+' from '+table_name+' where '+cols.split(',')[0]+' between '+start_date+' and '+end_date
def GenerateDataframeFromWinddb(cols,table_name,start_date=None,end_date=None):
    '''
    从wind数据库读取数据，并转化成 T*N 的DataFrame
    '''
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
def GenerateWeightsFromHalflife(T,halflife=None):
    '''
    给定halflife参数，生成权重序列
    '''
    if halflife is None:
        halflife=int(T/2)
    lam=0.5**(1.0/halflife)
    w=(1-lam)*lam**(np.arange(T))
    return w[::-1]/w.sum()
def GenerateAlphaBetaFromWeightedTimeSeriesRegression(Y, X, halflife=None,output='beta'):
    '''
    单变量加权最小二乘，输出结果可以设置成截距项（alpha）或变量系数估计（beta）
    '''
    Y = ret_stock.loc[t - DateOffset(months=12):t]
    X=ret_bench.loc[t - DateOffset(months=12):t]
    if halflife is None:
        w=pd.Series(1.0,index=Y.index)
    else:
        w=pd.Series(GenerateWeightsFromHalflife(len(Y),halflife=halflife),index=Y.index)*100.0
    X_=pd.concat([pd.Series(1.0,index=X.index),X],axis=1).mul(w,axis=0)
    Y_=Y.apply(lambda x:x.fillna(x.mean())).mul(w,axis=0)
    res=(np.linalg.inv(X_.T@X_)@X_.T@Y_)
    if output.upper()=='BETA':
        return res.iloc[1]
    else:
        return res.iloc[0]
def GenerateStdFromWeightedMovingAverate(X, halflife=None):
    '''
    利用WMA估计std
    '''
    if halflife is None:
        w=pd.Series(1.0,index=X.index)
    else:
        w=pd.Series(GenerateWeightsFromHalflife(len(X),halflife=halflife),index=X.index)
    return np.sqrt(((X-X.mean())**2.0).mul(w,axis=0).sum())
def GenerateMeanFromWeightedMovingAverage(X,halflife=None):
    '''
    利用WMA估计均值
    '''
    # todo 以0填充还是以mean填充，当前选择是mean
    w=pd.Series(GenerateWeightsFromHalflife(len(X),halflife=halflife),index=X.index)
    return X.apply(lambda x:x.fillna(x.mean())).mul(w,axis=0).sum()
def GenerateCummulativeRange(X):
    '''
    用于计算volatility类中的cumulative range指标
    '''
    X_=X.fillna(0.0).cumsum()
    return X_.max()-X_.min()
def GenerateSeasonalityReturns(X,date):
    '''
    计算过去5年相同月份的平均收益率
    '''
    return X.loc[(X.index.year<date.year)&(X.index.month==date.month)].mean()
def GetTradingCalendar():
    '''
    从wind数据库获取股市交易日历
    '''
    sql_order='''
    SELECT trade_days from ASHARECALENDAR where S_INFO_EXCHMARKET = 'SSE'
    '''
    tc=pd.read_sql(sql_order,connect_winddb)
    return pd.to_datetime(tc['trade_days'],format='%Y%m%d').sort_values()
def GetIndustryClassCITICS(trading_calendar,start_date,end_date):
    '''
    从wind获取中信一级行业分类
    '''
    ind=pd.read_sql(GenerateSqlOrder('entry_dt,s_info_windcode,citics_ind_code','AShareIndustriesClassCITICS'),connect_winddb)
    # 获取一级行业分类
    ind['1st']=ind['citics_ind_code'].str.slice(2,4)
    ind['entry_dt']=pd.to_datetime(ind['entry_dt'],format='%Y%m%d')
    ind_=ind.set_index(['entry_dt','s_info_windcode'])['1st']
    #ind_=ind_.loc[~ind_.index.duplicated(keep='last')]
    # 使用resample、reindex以及多个ffill是避免：1、公布日期非交易日；2、避免resample只能到样本的最新日期
    return ind_.unstack().resample('D').ffill().ffill().reindex(trading_calendar).ffill().loc[start_date:end_date]
def GenerateIndustryMomentum(X,g,w):
    '''
    计算行业动量
    :param X: 个股相对强度
    :param g: 行业分类
    :param w: 权重，一般以自由流通市值的某种单调函数作为权重
    :return:
    '''
    assert (X.shape==g.shape) & (X.shape==w.shape)
    X_w=X*w
    ind_class = list(set(g.iloc[0].dropna()))
    rsi=pd.DataFrame(np.nan,index=X.index,columns=ind_class)
    for t in rsi.index:
        for c in rsi.columns:
            rsi.loc[t,c]=X_w.loc[t,g.loc[t]==c].sum()
        # 获取个股层面的行业相对强度数据
    RS_I = g.apply(lambda x: x.transform(lambda y: rsi.loc[x.name, ind.loc[x.name, y.index]]), axis=1)
    RS_I.columns = g.columns
    # 行业动量
    INDMOM = RS_I-X_w
    return INDMOM



# 获取数据
def GetDailyStockReturns(start_date, end_date):
    '''
    计算个股的日收益率
    '''
    sql_order=GenerateSqlOrder('trade_dt,s_info_windcode,s_dq_adjpreclose,s_dq_adjclose','ashareeodprices',start_date,end_date)
    ret_stock=GenerateDataframeFromWinddb('trade_dt,s_info_windcode,s_dq_adjclose','ashareeodprices',str(int(start_date)-200),end_date).pct_change()[start_date:]
    return ret_stock.replace(0.0,np.nan)*100.0
def GetBenchmarkReturns(start_date,end_date):
    '''
    沪深300指数日收益率
    '''
    return GenerateDataframeFromWinddb('trade_dt,s_info_windcode,pct_chg','HS300IEODPrices',start_date,end_date)['000300.SH']

# 计算因子
# size
def Generate_LogCap_MidCap(start_date,end_date):
    '''
    生成size类相关指标
    '''
    # 生成 自由流通市值的DataFrame
    free_cap=GenerateDataframeFromWinddb('trade_dt,s_info_windcode,s_dq_mv','AShareEODDerivativeIndicator',start_date,end_date)
    # 剔除<=0.0的流通市值无效数据
    free_cap=free_cap[free_cap>0.0]
    log_cap=np.log(free_cap.replace(np.inf,np.nan))
    mid_cap=log_cap.agg(lambda x: x - x.mean(), axis=1).apply(lambda x: (x ** 3.0) - x * (x ** 4.0).sum() / (x ** 2.0).sum(),
                                                      axis=1)
    return log_cap,mid_cap

# volatility
def Generate_Beta_HistSigma_DailyStd_CumRange(start_date,end_date):
    '''
    生成volatility类相关指标
    '''
    ret_bench = GetBenchmarkReturns(str(int(start_date) - 10000), end_date)
    ret_stock = GetDailyStockReturns(str(int(start_date) - 10000), end_date)
    #ret_stock=ret_stock[ret_stock!=0.0]
    # 若 halflife=None，则意味着等权重
    beta=ret_stock.loc[start_date:].apply(lambda x:GenerateAlphaBetaFromWeightedTimeSeriesRegression(ret_stock.loc[x.name - DateOffset(months=12):x.name], ret_bench.loc[x.name - DateOffset(months=12):x.name], halflife=63), axis=1)
    sigma_hist=ret_stock.loc[start_date:].apply(lambda x:(ret_stock.loc[x.name-DateOffset(months=12):x.name]-(pd.DataFrame(ret_bench.loc[x.name-DateOffset(months=12):x.name]).values@pd.DataFrame(beta.loc[x.name]).T.values)).std(),axis=1)
    # 若 halflife=None，则意味着等权重
    ret_std=ret_stock.loc[start_date:].apply(lambda x:GenerateStdFromWeightedMovingAverate(ret_stock.loc[x.name - DateOffset(months=12):x.name], halflife=42), axis=1)
    log_ret=np.log(ret_stock*.01+1.0)
    cum_range=log_ret.loc[start_date:].apply(lambda x:GenerateCummulativeRange(log_ret.loc[x.name-DateOffset(months=12):x.name]),axis=1).replace(0.0,np.nan)
    return beta,sigma_hist,ret_std.replace(0.0,np.nan),cum_range

# liquidity
def Generate_STOM_STOQ_STOY_ATVR(start_date,end_date):
    '''
    liquidity类相关指标
    '''
    turnover=GenerateDataframeFromWinddb('trade_dt,s_info_windcode,s_dq_freeturnover','AShareEODDerivativeIndicator',str(int(start_date)-10000),end_date)
    STOM=turnover.loc[start_date:].apply(lambda x:turnover.loc[x.name-DateOffset(months=1):x.name].mean(),axis=1)
    STOQ=turnover.loc[start_date:].apply(lambda x:turnover.loc[x.name-DateOffset(months=3):x.name].mean(),axis=1)
    STOY=turnover.loc[start_date:].apply(lambda x:turnover.loc[x.name-DateOffset(months=12):x.name].mean(),axis=1)
    ATVR=turnover.loc[start_date:].apply(lambda x:GenerateMeanFromWeightedMovingAverage(turnover.loc[x.name-DateOffset(months=12):x.name],halflife=63),axis=1)
    return np.log(STOM[STOM>0.0]),np.log(STOQ[STOQ>0.0]),np.log(STOY[STOY>0.0]),ATVR[ATVR>0.0]

# momentum
def Generate_STRREV_Seasonality_IndMom(start_date,end_date):
    '''
    momentum类相关指标
    '''
    ret_stock = GetDailyStockReturns(str(int(start_date) - 50100), end_date)
    ret_stock=ret_stock.replace(np.inf,np.nan)
    log_ret=np.log(ret_stock*0.01+1)
    # 短期反转
    rev=log_ret.loc[start_date:].apply(lambda x:GenerateMeanFromWeightedMovingAverage(log_ret.loc[x.name-DateOffset(months=1):x.name],halflife=5),axis=1)
    # 季节
    seas=log_ret.loc[start_date:].apply(lambda x:GenerateSeasonalityReturns(log_ret.loc[x.name-DateOffset(months=61):x.name],x.name),axis=1)
    # 个股相对强度(窗口期为6个月）
    RS_S_6M=log_ret.loc[start_date:].apply(lambda x:GenerateMeanFromWeightedMovingAverage(log_ret.loc[x.name-DateOffset(months=6):x.name],halflife=30),axis=1)
    RS_S_6M=RS_S_6M.replace(0.0,np.nan)
    # 行业相对强度和行业动量
    trading_calendar=GetTradingCalendar()
    ind=GetIndustryClassCITICS(trading_calendar,start_date,end_date)
    free_cap=GenerateDataframeFromWinddb('trade_dt,s_info_windcode,s_dq_mv','AShareEODDerivativeIndicator',start_date,end_date)
    free_cap=free_cap[free_cap>0.0].reindex(ind.index)
    # 避免出现大量的nan影响运行
    stock_pool=RS_S_6M.columns.intersection(free_cap.columns).intersection(ind.columns)
    INDMOM=GenerateIndustryMomentum(RS_S_6M.reindex(columns=stock_pool),ind.reindex(columns=stock_pool),np.sqrt(free_cap.reindex(columns=stock_pool)))
    # 个股相对强度（窗口期1年）
    RS_S_1Y=log_ret.loc[str(int(start_date)-100):].apply(lambda x:GenerateMeanFromWeightedMovingAverage(log_ret.loc[x.name-DateOffset(months=12):x.name],halflife=120),axis=1)
    RS_S_1Y=RS_S_1Y.replace(0.0,np.nan)
    MOM=RS_S_1Y.apply(lambda x:RS_S_1Y.loc[x.name-DateOffset(months=1):x.name-DateOffset(weeks=2)].mean(),axis=1)
    alpha=ret_stock.loc[start_date:].apply(lambda x:GenerateAlphaBetaFromWeightedTimeSeriesRegression(ret_stock.loc[x.name - DateOffset(months=12):x.name], ret_bench.loc[x.name - DateOffset(months=12):x.name], halflife=63,output='alpha'), axis=1)
    return rev.replace(np.inf,np.nan)*100.0,seas[seas!=0.0],RS_S_6M,INDMOM,MOM,alpha

# leverage
def Generate_MarketLev(start_date,end_date):
    '''
    leverage类相关指标
    '''
    # 获取总市值、优先股、长期负债数据
    ME=GenerateDataframeFromWinddb('trade_dt,s_info_windcode,s_val_mv','AShareEODDerivativeIndicator',start_date,end_date)
    # todo PE没找到
    #PE=
    # 长期负债取资产负债表中的非流动负债数据
    LD=GenerateDataframeFromWinddb('report_period,s_info_windcode,tot_non_cur_liab','AShareBalanceSheet',str(int(start_date)-10000))
    # 总负债
    TL=GenerateDataframeFromWinddb('report_period,s_info_windcode,tot_liab','AShareBalanceSheet',str(int(start_date)-10000))
    # 总资产
    TA=GenerateDataframeFromWinddb('report_period,s_info_windcode,tot_assets','AShareBalanceSheet',str(int(start_date)-10000))
    # 过去5年营业收入
    TOR=GenerateDataframeFromWinddb('report_period,s_info_windcode,tot_oper_rev','AShareIncome',str(int(start_date)-50000))
    # 过去五年净利润
    NI=GenerateDataframeFromWinddb('report_period,s_info_windcode,net_profit_excl_min_int_inc','AShareIncome',str(int(start_date)-50000))
    # 过去五年现金及其等价物净增加额
    Cash=GenerateDataframeFromWinddb('report_period,s_info_windcode,net_incr_cash_cash_equ','AShareCashFlow',str(int(start_date)-50000))
    # 每股收益预测，对于由多个分析是预测的数据，采取简单平均处理
    sql_order=GenerateSqlOrder('reporting_period,s_info_windcode,est_eps_diluted','AShareEarningEst',str(int(start_date)-50000))
    eps=pd.read_sql(sql_order,connect_winddb).set_index(['reporting_period','s_info_windcode']).sort_index().groupby(level=(0,1)).mean()['est_eps_diluted'].unstack()
    eps.index=pd.to_datetime(eps.index,format='%Y%m%d')
    # todo 净经营资产
    # 无息流动负债： 应付票据、应付账款、预收账款、应交税费、应付利息、其他应付款、其他流动负债。
    sql_order=GenerateSqlOrder('report_period,s_info_windcode,notes_payable,acct_payable,adv_from_cust,taxes_surcharges_payable,int_payable,oth_payable,oth_cur_liab','AShareBalanceSheet',str(int(start_date)-50000))
    free_liq_lia=pd.read_sql(sql_order,connect_winddb).set_index(['report_period','s_info_windcode']).sum(axis=1).sort_index()
    # 存在很多重复index，但数据都不同。此处的选择是保留最后一项
    free_liq_lia=free_liq_lia.iloc[~free_liq_lia.index.duplicated(keep='last')].unstack()
    # 折旧与摊销
    DA=GenerateDataframeFromWinddb('report_period,s_info_windcode,s_stm_is','AShareFinancialIndicator',str(int(start_date)-10000))
    # 经营活动现金流量净额
    CFO=GenerateDataframeFromWinddb('report_period,s_info_windcode,net_cash_flows_oper_act','AShareCashFlow',str(int(start_date)-10000))
    # 投资活动现金流量净额
    CFI=GenerateDataframeFromWinddb('report_period,s_info_windcode,net_cash_flows_inv_act','AShareCashFlow',str(int(start_date)-10000))
    # 营业成本
    TOC=GenerateDataframeFromWinddb('report_period,s_info_windcode,tot_oper_cost','AShareIncome',str(int(start_date)-10000))
    # 销售毛利率 todo 金融类企业似乎没有这个指标，所以需要删除所有金融类企业吗
    GPM=GenerateDataframeFromWinddb('report_period,s_info_windcode,s_fa_grossprofitmargin','AShareFinancialIndicator',str(int(start_date)-10000))








if __name__ is '__main__':
    # 数据库参数设置
    options_winddb_datebase = {'server': '172.16.100.7',
                               'user': 'mikuser',
                               'password': 'mikuser',
                               'database': 'NWindDB'}
    connect_winddb = ConnectSQLserver(options_winddb_datebase['server'], options_winddb_datebase['user'],
                                      options_winddb_datebase['password'], options_winddb_datebase['database'])



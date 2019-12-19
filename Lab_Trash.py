# 本地python pickle数据
xrisk=pd.read_pickle('C:/Users/shenzheng/PycharmProjects/Intership-for-GFund/DataSets/trcp_hld')
xrisk['T_DATE']=pd.to_datetime(xrisk['T_DATE'],format='%Y-%m-%d')
portfolio_stock=xrisk[['PORT_CODE','T_DATE','I_CODE','H_EVAL','H_EVAL_ADDED','H_COUNT','H_COST']].loc[xrisk['A_TYPE']=='SPT_S'] # 'PORT_CODE','T_DATE','I_CODE','H_COUNT','POSITION','H_COST','H_EVAL','UPDATE_TIME'
portfolio_code=set(portfolio_stock['PORT_CODE']);portfolio_trade_dt=set(portfolio_stock['T_DATE'])
portfolio_stock=portfolio_stock.set_index(['T_DATE','PORT_CODE','I_CODE']).sort_index().groupby(level=(0,1,2)).sum()
# portfolio_stock.loc[portfolio_stock.index.duplicated(keep=False)]# 组合'762001'有三条数据出现重复--->似乎应该相加
portfolio_weights=(portfolio_stock['H_EVAL']-portfolio_stock['H_EVAL_ADDED']).unstack()
portfolio_stock['H_EVAL']-portfolio_stock['H_EVAL_ADDED']
portfolio_stock=xrisk.loc[xrisk['A_TYPE']=='SPT_S'][['T_DATE','PORT_CODE','I_CODE','H_COUNT']].set_index(['T_DATE','PORT_CODE','I_CODE']).groupby(level=(0,1,2)).sum().unstack()
portfolio_stock.head()
portfolio_stock.columns=portfolio_stock.columns.droplevel()
tmp=xrisk.loc[xrisk['I_CODE']=='100201'].head()







def ConnectMySQL(server,user,password,database):
    '''
    范例：
    options_WindDatabase={'server':'localhost',
                      'user':'root',
                      'password':'123456',
                      'database':'winddb'}
    connect=ConnectMySQL(**options_mysql)
    '''
    connect=pymysql.connect(server,user,password,database)
    if connect:
        print('链接成功')
    return connect
options_barra_mysql={'server':'localhost',
                      'user':'root',
                      'password':'1234567890',
                      'database':'barra_risk'}
connect_barra=ConnectMySQL(**options_barra_mysql)
options_winddb_mysql={'server':'localhost',
                     'user':'root',
                     'password':'1234567890',
                     'database':'winddb'}
connect_winddb=ConnectMySQL(**options_winddb_mysql)
options_xrisk_mysql={'server':'localhost',
                      'user':'root',
                      'password':'1234567890',
                      'database':'xrisk'}
connect_xrisk=ConnectMySQL(**options_xrisk_mysql)

# 用于检验个股因子结构
t_set=set(asset_returns.index.get_level_values(0))
sec_set=set(asset_returns.index.get_level_values(1))
d=pd.Series(np.nan,pd.MultiIndex.from_product((t_set,sec_set))).sort_index()
for i in d.index:
    d.loc[i]= asset_returns.loc[i]-(asset_exposure.loc[i]*factor_returns.loc[i[0]]).sum()*100.0-specific_returns.loc[i]



import sys
#from pandas import DataFrame  #DataFrame通常来装二维的表格
import pandas as pd      #pandas是流行的做数据分析的包
#建立字典，键和值都从文件里读出来。键是nam，age……，值是lili，jim……




options_barra_database={'user':'riskdata',
                   'password':'riskdata',
                   'dsn':cx_Oracle.makedsn('172.16.100.188','1522','markdb')}



sql_get_from_barra="SELECT DISTINCT(TRADINGDATE) from BARRA_FACTORRET WHERE TRADINGDATE BETWEEN " +\
                       start_date + " AND " + end_date
    trading_calendar=pd.to_datetime(pd.read_sql(sql_get_from_barra, connect_barra)['TRADINGDATE'].sort_values(),format='%Y%m%d')
    multi_index=pd.MultiIndex.from_product((trading_calendar,set(portfolio_eval.index.get_level_values(1))))
    portfolio_weights=pd.DataFrame(np.nan,index=multi_index,columns=portfolio_eval.columns).sort_index()
    for p in portfolio_weights.columns:
        portfolio_weights[p]=portfolio_eval[p].unstack().reindex(trading_calendar).shift(1).apply(lambda x:x/x.sum(),axis=1).stack()
    return portfolio_weights

# 算数平均与几何平均的差异
T=100000
ret_simple=pd.Series(np.random.randn(T))+0.1
ret_arith=ret_simple.mean();ret_arith-(ret_simple).var()*0.5/100.0
ret_geo=((ret_simple*.01+1).cumprod().iloc[-1])**(1.0/len(ret_simple))*100.0-100.0;ret_geo


# test Barra decom
# 组合
err=(portfolio_returns-portfolio_style_factor_return-portfolio_industry_factor_return-portfolio_country_factor_return-portfolio_specific_return)#.sum()
(err>=0.0).mean() # 0.988506
(err>0.01).mean() # 0.862069
# 个股
def TestAssetReturn(asset_returns,asset_exposure,factor_returns,specific_returns,method='fast'):
    if method=='fast':
        asset_exposure_tmp = asset_exposure.stack().reset_index().set_index(
            ['TRADINGDATE', 'level_2', 'SECUCODE']).unstack()
        asset_exposure_tmp.columns = asset_exposure_tmp.columns.droplevel()
        asset_factor_returns=asset_exposure_tmp.mul(factor_returns.reindex(asset_exposure_tmp.index), axis=0).groupby(level=0).sum().stack()
        return asset_returns-asset_factor_returns-specific_returns
    else:
        asset_factor_returns=pd.Series(np.nan,asset_exposure.index)
        for i in asset_exposure.index:
            asset_factor_returns.loc[i]=\
                (factor_returns.loc[i[0]]*asset_exposure.loc[i]).sum()
        return asset_returns-asset_factor_returns-specific_returns
err2=TestAssetReturn(asset_returns,asset_exposure,factor_returns,specific_returns)
(err2>=0.0).mean() # 0.8204958314361527 0.812373499482339 #
(err2>=0.01).mean() # 0.5978301486866276 0.5919120504665168
(err2>0.0).groupby(level=0).mean().mean() # 0.812373499482339
(err2>0.01).groupby(level=0).mean().mean() #0.5919120504665172






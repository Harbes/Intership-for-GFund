# code习惯说明：
#   变量小写，且使用“_”连接
#       sql开头：表示字符串形式的sql命令
#   函数首字母大写，且不使用“_”连接
import pandas as pd
import cx_Oracle
import pymysql
import matplotlib.pyplot as plt
import datetime

# todo 链接数据库
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
options_barra_database={'user':'riskdata',
                   'password':'riskdata',
                   'dsn':cx_Oracle.makedsn('172.16.100.188','1522','markdb')}
connect_barra=ConnectOracle(**options_barra_database)
options_xrisk_database={'user':'risk_read',
                   'password':'riskRead2019',
                   'dsn':cx_Oracle.makedsn('172.16.100.230','1521','xrisk')}
connect_xrisk=ConnectOracle(**options_xrisk_database)
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


# todo 从barra数据库读取某段时间内的factor return数据
# 为形成 T*K 维度的DataFrame，要同时保留barra_factorret表中的TRADINGDATE,FACTOR,DLYRETURN
factor_name_excluding_industry_country=('CNE5S_BETA', 'CNE5S_BTOP', 'CNE5S_EARNYILD', 'CNE5S_GROWTH', 'CNE5S_LEVERAGE', 'CNE5S_LIQUIDTY',
                                  'CNE5S_MOMENTUM','CNE5S_RESVOL','CNE5S_SIZE','CNE5S_SIZENL')
start_date,end_date=' 20180701','20180730'
sql_get_factor_returns= 'select TRADINGDATE,FACTOR,DLYRETURN from BARRA_FACTORRET where TRADINGDATE BETWEEN ' + \
                        start_date +' and ' + end_date + ' and Factor in ' + str(factor_name_excluding_industry_country)
factor_return=pd.read_sql(sql_get_factor_returns, connect_barra).set_index(['TRADINGDATE', 'FACTOR']).sort_index().unstack()
factor_return.columns=factor_return.columns.droplevel()


# todo 从barra数据库读取某段时间内的asset exposure数据
## 注意 asset_exposure中，股票代码没有SH、SZ之类的后缀
#以下命令用于生成sql命令中 紧跟select的列名（字符串）
factor_name_sql= ''
for i in factor_name_excluding_industry_country[:-1]:
    factor_name_sql= factor_name_sql + i + ','
factor_name_sql+=factor_name_excluding_industry_country[-1]
sql_get_asset_exposure= 'select SECUCODE,TRADINGDATE,' + factor_name_sql + ' from barra_assetexposure where TRADINGDATE BETWEEN ' + start_date + ' and ' + end_date
asset_exposure=pd.read_sql(sql_get_asset_exposure, connect_barra).set_index(['TRADINGDATE', 'SECUCODE']).sort_index()
asset_exposure=asset_exposure.loc[~asset_exposure.index.duplicated(keep='last')] # 删除重复数据


# todo 从wind数据库读取并计算benchmark相关
## 设置常用的benchmark；注意hs300free里有许多不同的编制，是否需要纳入？
benchmark_set={'HS300':('i_weight','aindexhs300closeweight'),
               'CSI500':('weight','AIndexCSI500Weight'),
               'SSE50':('weight','AIndexSSE50Weight') # 上证50只有20180830之前的数据
               }
def GetBenchmarkWeightsFromWindDB(benchmark='HS300'):
    #benchmark='SSE50'
    sql_get_benchmark_weights='select trade_dt,s_con_windcode,'+benchmark_set[benchmark][0] +' from '+benchmark_set[benchmark][1] +' where trade_dt BETWEEN ' + start_date + ' and ' + end_date
    weights=pd.read_sql(sql_get_benchmark_weights,connect_winddb)
    ## 注意：由于Barra数据库中股票代码没有SH之类的后缀，因此，来自winddb的股票代码需要去除后缀
    weights['s_con_windcode']=weights['s_con_windcode'].str.slice_replace(6,repl='')
    weights=weights.set_index(['trade_dt','s_con_windcode']).sort_index().loc[~weights.index.duplicated(keep='last')]
    #weights.columns=weights.columns.droplevel()
    return weights
benchmark_weights=GetBenchmarkWeightsFromWindDB()
def PortfolioExposure(asset_exposure,portfolio_weights=None):
    if portfolio_weights is None:
        return asset_exposure.groupby(level=0).mean()
    else:
        return (asset_exposure.reindex(portfolio_weights.index)*portfolio_weights.values).groupby(level=0).sum()/portfolio_weights.groupby(level=0).sum().values
benchmark_exposure=PortfolioExposure(asset_exposure, portfolio_weights=benchmark_weights)
market_exposure=PortfolioExposure(asset_exposure)


# todo 计算因子超额收益、累计超额收益、超额暴露等
factor_excess_exposure=market_exposure-benchmark_exposure
factor_excess_return=factor_excess_exposure*factor_return

# todo 从xrisk读取现实组合数据
#
portfolio_type={'FUT_BD',
                'FUT_CMDT',
                'FUT_IDX_S',
                'FWD_REPO',
                'SPT_ABS',
                'SPT_BD',
                'SPT_CB',
                'SPT_DED',
                'SPT_ETF',
                'SPT_MMF',
                'SPT_OEF',
                'SPT_OST_C',
                'SPT_OTHD',
                'SPT_REPO',
                'SPT_S',
                'SPT_TMD'}
# 有些组合出现了衍生工具，例如：76C012
def DatetimeFromBarraToXrisk(date):
    return datetime.datetime.strptime(str.strip(date),'%Y%m%d').strftime('%Y-%m-%d')
sql_get_from_xrisk="SELECT PORT_CODE,T_DATE,I_CODE,H_EVAL from xrisk.tcrp_hld WHERE (T_DATE BETWEEN '"+DatetimeFromBarraToXrisk(start_date)+"' AND '"+DatetimeFromBarraToXrisk(end_date)+"') and (A_TYPE= 'SPT_S')"
portfolio_stock=pd.read_sql(sql_get_from_xrisk, connect_xrisk).set_index(['T_DATE', 'PORT_CODE', 'I_CODE']).sort_index().groupby(level=(0,1,2)).sum() # 使用groupby+sum是因为数据中出现了“分裂”的情况
portfolio_weights=portfolio_stock.unstack().apply(lambda x:x/x.sum(),axis=1)
portfolio_weights.columns=portfolio_weights.columns.droplevel()
portfolio_code=set(portfolio_stock.index.get_level_values(1)) # 数据库中有哪些组合
portfolio_trade_dt=set(portfolio_stock.index.get_level_values(0)) # 交易日期


# todo 收益、风险分解
def DecomReturn(weights,asset_exposure,factor_return,specific_return):
    style_return=weights*asset_exposure*factor_return

portfolio_weights.loc[('2018-07-30', '76C012', slice(None))]
asset_exposure.head()


# todo 绘图
# 图形bug太多
for f in factor_name_excluding_industry_country[5:6]:
    fig = plt.figure(figsize=(12,4));ax = fig.add_subplot(111) #
    plt1=ax.bar(factor_excess_return[f].index,factor_excess_return[f].values,label='超额收益') ## todo 中文乱码
    plt2=ax.plot(factor_excess_return[f].cumsum(), color="red",label='累计超额收益')
    ax2=ax.twinx()
    ax2.set_ylim(factor_excess_exposure[f].min(),factor_excess_exposure[f].max())
    plt3=ax2.plot(factor_excess_exposure[f],color='gray',label='超额暴露（右）')
    plt.xticks(range(factor_excess_return[f].size),factor_excess_return[f].index, rotation=45) # 为什么rotation不起作用
    #plt.tick_params(axis='x', labelsize=1) # 调整x轴标签字体大小，但不知为何，此处依然没有作用
    fig.legend(loc=1, bbox_to_anchor=(0,-0.2), bbox_transform=ax.transAxes)
    plt.show()


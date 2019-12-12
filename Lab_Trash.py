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
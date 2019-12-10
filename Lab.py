# 本地python pickle数据
xrisk=pd.read_pickle('C:/Users/shenzheng/PycharmProjects/Intership-for-GFund/DataSets/trcp_hld')
xrisk['T_DATE']=pd.to_datetime(xrisk['T_DATE'],format='%Y-%m-%d')
portfolio_stock=xrisk[['PORT_CODE','T_DATE','I_CODE','H_EVAL','H_EVAL_ADDED']].loc[xrisk['A_TYPE']=='SPT_S'] # 'PORT_CODE','T_DATE','I_CODE','H_COUNT','POSITION','H_COST','H_EVAL','UPDATE_TIME'
portfolio_code=set(portfolio_stock['PORT_CODE']);portfolio_trade_dt=set(portfolio_stock['T_DATE'])
portfolio_stock=portfolio_stock.set_index(['T_DATE','PORT_CODE','I_CODE']).sort_index().groupby(level=(0,1,2)).sum()
# portfolio_stock.loc[portfolio_stock.index.duplicated(keep=False)]# 组合'762001'有三条数据出现重复--->似乎应该相加
portfolio_weights=(portfolio_stock['H_EVAL']-portfolio_stock['H_EVAL_ADDED']).unstack()
portfolio_weights.columns
from sqlalchemy import create_engine
import pandas as pd
import datetime
import cx_Oracle

# todo 数据源

def ReadSPCALHCP(data_source):
    df=pd.read_csv(data_source,sep='\t',header='infer').iloc[:-1] # 剔除最后一行无效数据
    df['TICKER']=df['TICKER'].astype(int).astype(str).str.zfill(6) # ticker 写入是float，需要转化为str，且填充0
    # todo 除了 TICKER 外， STOCK KEY 和 GICS CODE似乎也要采取类似的操作
    df['STOCK KEY']=df['STOCK KEY'].astype(int).astype(str).str.zfill(7)
    df['GICS CODE']=df['GICS CODE'].astype(int).astype(str).str.zfill(8)
    # 增加一列 'OPDATE'，记录操作时间
    df['OPDATE']=datetime.datetime.now()
    return df
def WriteToOracle(connect_str,data):
    engine=create_engine(connect_str)
    data.to_sql('SPCALHCP_NCS_ADJ', con=engine,index=False,if_exists='append')
    return None

# if __name__ == '__main__':
connect_str='oracle+cx_oracle://riskdata:riskdata@172.16.100.188:1522/markdb'
# todo 数据源
data_source='C:/Users/shenzheng/PycharmProjects/Intership-for-GFund/DataSets/20191108_SPCALHCP_NCS_ADJ.SDC'
data_SPCALHCP=ReadSPCALHCP(data_source)
WriteToOracle(connect_str,data_SPCALHCP)
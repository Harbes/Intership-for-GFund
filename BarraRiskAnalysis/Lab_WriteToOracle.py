from sqlalchemy import create_engine
import pandas as pd
import datetime
import cx_Oracle
import os
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
# 读取标普低波红利数据，假定数据格式固定
def ReadSPCALHCP(data_source):
    df=pd.read_csv(data_source,sep='\t',header='infer').iloc[:-1] # 剔除最后一行无效数据
    df['TICKER']=df['TICKER'].astype(int).astype(str).str.zfill(6) # ticker 写入是float，需要转化为str，且填充0
    # 说明：除了 TICKER 外， STOCK KEY 和 GICS CODE似乎也要采取类似的操作
    df['STOCK KEY']=df['STOCK KEY'].astype(int).astype(str).str.zfill(7)
    df['GICS CODE']=df['GICS CODE'].astype(int).astype(str).str.zfill(8)
    # 说明：修改columns，以下划线替代空格
    df.columns=df.columns.map(lambda x:x.strip().replace(' ','_'))
    # 增加一列 'OPDATE'，记录操作时间
    df['OPDATE']=datetime.datetime.now()
    # 记录数据源文件名
    df['FROM_FILE']=re.split('[/]',data_source)[-1]
    return df
# 写入数据库
def WriteToOracle(data,connect_str):
    engine=create_engine(connect_str)
    # todo 修改写入方式
    data.to_sql(table_name, con=engine,index=False,if_exists='append')
    return None
# 扫描文件夹，并返回未写入数据文件名列表
def ScanDir(dir,existed_set):
    file_list=[]
    for f in os.listdir(dir):
        if f not in existed_set:
            file_list+=f
    return file_list

if __name__ == '__main__':
    # 全局变量设置
    table_name='SPCALHCP_NCS_ADJ'
    dir='' # todo 文件放置路径
    options_xrisk_database = {'user': 'risk_read',
                              'password': 'riskRead2019',
                              'dsn': cx_Oracle.makedsn('172.16.100.230', '1521', 'xrisk')}
    connect_str='oracle+cx_oracle://riskdata:riskdata@172.16.100.188:1522/markdb'

    # 读取数据库，获得已读取过的文件名称集合
    connect_xrisk = ConnectOracle(**options_xrisk_database)
    existed_set=set(pd.read_sql('select FROM_FILE from '+table_name,connect_xrisk))
    # 扫描文件夹，返回未曾写入的数据文件名称列表
    file_list=ScanDir(dir,existed_set)
    # 写入数据
    for f in file_list:
        data_source=dir+'/'+f
        # data_source='C:/Users/shenzheng/PycharmProjects/Intership-for-GFund/DataSets/20191108_SPCALHCP_NCS_ADJ.SDC'
        data_SPCALHCP=ReadSPCALHCP(data_source)
        WriteToOracle(data_SPCALHCP,connect_str)





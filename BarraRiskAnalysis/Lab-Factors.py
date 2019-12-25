import pandas as pd
from Barra_GetConnectedToDatabase import *

options_WindDatabase={'server':'172.16.100.7',
                      'user':'mikuser',
                      'password':'mikuser',
                      'database':'NWindDB'}
connect_wind=ConnectSQLserver(**options_WindDatabase)


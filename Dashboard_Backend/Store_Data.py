#Enter the subfolder to the evaluated runs (keep the r in front of the string)
path='01_Data'

#Enter the path to the Run information file (keep the r in front of the string)
info_path='Masterdata.xlsx'



import os
import dash
import plotly.express as px
import pandas as pd
import numpy as np
import pandas as pd 
from pandas import DataFrame
import numpy as np
import math
import pandas as pd
from datetime import datetime
import os

pd.set_option('float_format', '{:f}'.format)

directories= [d for d in os.listdir('Data/') if os.path.isdir(os.path.join('Data/', d))]
print(len(directories))

Dataframes=[[] for _ in range(len(directories))]
Dataframes_reward=list()
Dataframes_qvalues=[[] for _ in range(len(directories))]
Dataframes_emission=[[] for _ in range(len(directories))]
Dataframes_waiting=[[] for _ in range(len(directories))]
Dataframes_replay=[[] for _ in range(len(directories))]
Dataframes_average=[[] for _ in range(len(directories))]
Dataframes_waiting_car=[[] for _ in range(len(directories))]
Dataframes_actions=[[] for _ in range(len(directories))]
Dataframes_reward_replay=[[] for _ in range(len(directories))]
average_list=[[] for _ in range(len(directories))]
Setups=[]

print(directories)
scenarios=['5-6_(1380)', '8-9_(2600)', '17-18_(3100)', '23-24_(470)']
print(Dataframes)
for d, file in enumerate(directories):
    print('Currently preparing ' + file +  'for the dashboard')
    Dataframes_reward.append(pd.read_csv('Data/'+file+'/reward.csv'))
    for scenario in scenarios:
        
        
        Dataframes_replay[d].append(pd.read_csv('Data/'+file+'/test/'+scenario+'/position.csv'))
        Dataframes_emission[d].append(pd.read_csv('Data/'+file+'/test/'+scenario+'/emission.csv'))
        Dataframes_waiting[d].append(pd.read_csv('Data/'+file+'/test/'+scenario+'/waiting.csv'))
        Dataframes_qvalues[d].append(pd.read_csv('Data/'+file+'/test/'+scenario+'/q_list.csv'))
        Dataframes_waiting_car[d].append(pd.read_csv('Data/'+file+'/test/'+scenario+'/waiting_car.csv'))
        Dataframes_actions[d].append(pd.read_csv('Data/'+file+'/test/'+scenario+'/action.csv'))
        Dataframes_reward_replay[d].append(pd.read_csv('Data/'+file+'/test/'+scenario+'/reward_replay.csv'))
        df_waiting=pd.read_csv('Data/'+file+'/test/'+scenario+'/waiting.csv')
        df_emission=pd.read_csv('Data/'+file+'/test/'+scenario+'/emission.csv')
        df_waiting_car=pd.read_csv('Data/'+file+'/test/'+scenario+'/waiting_car.csv')
        average_list.append([' '+file+' '+scenario,round(df_waiting_car['waiting time car'].sum(),0),round(df_waiting_car['waiting time car'].mean(),0),
                         round(df_waiting_car['waiting time car'].std(),0), round(df_emission.sum()/1000,0)])
    Setups.append(file)

Dataframes_average=pd.DataFrame(average_list, columns=['Setup', 'Sum Waiting Time [s]', 'Mean Waiting Time [s]','Std Waiting Time [s]', 'Sum Emission [g]'])

print(len(Dataframes_reward))
print(Setups)
#print(xxx)
print(len(Dataframes_replay[0]))


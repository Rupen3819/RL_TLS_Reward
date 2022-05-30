def utilization(filename,path,run):
    # importing required packages
    import pandas as pd 
    from pandas import DataFrame
    import numpy as np
    import openpyxl
    import math
    import plotly.express as px
    import pandas as pd
    from datetime import datetime
    import plotly.figure_factory as ff
    import random
    pd.set_option('float_format', '{:f}'.format)
    df_time=pd.read_excel(filename+ '\\'+'worker_transition_history.xlsx')
    global_starttime=int(df_time['arrival timestamp'].iloc[0])

    


    df_instation=pd.read_excel(filename+ '\\' +run+'_station_utilization_instation.xlsx')
    df_instation_working=pd.read_excel(filename+ '\\' +run+'_station_utilization_working.xlsx')



    #stations_available=['station_2 1', 'station_2 2', 'station_3', 'station_4 1','station_4 2','station_5 1','station_5 2', 'station_6', 'station_7', 'station_1_8']
    stations_available=list(df_instation.columns.values)
    #print(stations_available)

    #Define colors for the stations
##    colors = {stations_available[0]: 'rgb(255, 0, 0)',
##          stations_available[1]: 'rgb(255, 128, 200)',
##          stations_available[2]: 'rgb(255, 128, 200)',
##        stations_available[3]: 'rgb(255, 255, 200)',
##              stations_available[4]: 'rgb(128, 255, 200)',
##              stations_available[5]: 'rgb(0, 255, 255)',
##              stations_available[6]: 'rgb(0, 128, 255)',
##              stations_available[7]: 'rgb(0, 0, 255)',
##              stations_available[8]: 'rgb(127, 128, 255)',
##              stations_available[9]: 'rgb(255, 51, 255)',
##              stations_available[10]: 'rgb(255, 128, 127)',
##              stations_available[11]: 'rgb(0, 0, 0)',
##              stations_available[12]: 'rgb(128, 0, 0)',
##              stations_available[13]: 'rgb(0, 0, 128)',
##              #stations_available[14]: 'rgb(0, 128, 128)',
##
##              }
    dict_list=[]
    for s in range(len(stations_available)):
        history=df_instation[stations_available[s]].values.tolist()
        inplace=False
        for i in range(len(history)):
            if history[i]!=0 and inplace==False:
                start_time=i
                inplace=True
                #written=False

            if history[i]==0 and inplace==True:
                #print(stations_available[s])
                dict_list.append(dict(Task=stations_available[s], Start=datetime.fromtimestamp(global_starttime+start_time), Finish=datetime.fromtimestamp(global_starttime+i), Resource=('worker'+str(history[i-1]))))
                inplace=False
                
    for s in range(len(stations_available)):
        history=df_instation_working[stations_available[s]].values.tolist()
        inplace=False
        for i in range(len(history)):
            if history[i]!=0 and inplace==False:
                start_time=i
                inplace=True
                #written=False

            if history[i]==0 and inplace==True:
                #print(stations_available[s])
                dict_list.append(dict(Task=stations_available[s]+'working', Start=datetime.fromtimestamp(global_starttime+start_time), Finish=datetime.fromtimestamp(global_starttime+i), Resource=('worker'+str(history[i-1]))))
                inplace=False

    #print(dict_list)

    dict_list_worker=[]
    for s in range(len(stations_available)):
        history=df_instation[stations_available[s]].values.tolist()
        inplace=False
        for i in range(len(history)):
            if history[i]!=0 and inplace==False:
                start_time=i
                inplace=True
                #written=False

            if history[i]==0 and inplace==True:
                #print(stations_available[s])
                dict_list_worker.append(dict(Task=('worker'+str(history[i-1])), Start=datetime.fromtimestamp(global_starttime+start_time), Finish=datetime.fromtimestamp(global_starttime+i), Resource=stations_available[s]))
                inplace=False
                
    for s in range(len(stations_available)):
        history=df_instation_working[stations_available[s]].values.tolist()
        inplace=False
        for i in range(len(history)):
            if history[i]!=0 and inplace==False:
                start_time=i
                inplace=True
                #written=False

            if history[i]==0 and inplace==True:
                #print(stations_available[s])
                dict_list_worker.append(dict(Task=('worker'+str(history[i-1]))+'working', Start=datetime.fromtimestamp(global_starttime+start_time), Finish=datetime.fromtimestamp(global_starttime+i), Resource=stations_available[s]))
                inplace=False

    dict_list_worker = sorted(dict_list_worker, key=lambda k: k['Task']) 
                
    #print(dict_list_worker)

    dict_list = sorted(dict_list, key=lambda k: k['Task'])


    df = pd.DataFrame(dict_list)
    #print(df)

  #  r = lambda: random.randint(0,255)
    #print('#%02X%02X%02X' % (r(),r(),r()))
   # colors = ['#%02X%02X%02X' % (r(),r(),r())]
#
 #   for i in range(1, 15):              
  #      b_dict = dict(Task="M " + str(i), Start=datetime.datetime.utcfromtimestamp((serial - 25569 + (start_value/(60*24.0))) * 86400.0).strftime("%Y-%m-%d %H:%M:%S"), Finish=datetime.datetime.utcfromtimestamp((serial - 25569 + (start_value +duration)/(60*24.0)) * 86400.0).strftime("%Y-%m-%d %H:%M:%S"), Resource="job " + str(i))            
   #     df.append(b_dict)
    #    start_value = 10*i

     #   colors.append('#%02X%02X%02X' % (r(),r(),r()))

    

    fig = ff.create_gantt(df, index_col='Resource', show_colorbar=True,
                          group_tasks=True,title="Station Utilization")
    #fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True, group_tasks=True)
    
    #fig.show()
   # fig.write_html(filename+'\\'+run+'_station_utilization.html')





    df = pd.DataFrame(dict_list_worker)
    #print(df)

    fig = ff.create_gantt(df, index_col='Resource', show_colorbar=True,
                      group_tasks=True,title="Worker Utilization")

    #fig.show()
    fig.write_html(filename+'\\'+run+'_worker_utilization.html')

    


    

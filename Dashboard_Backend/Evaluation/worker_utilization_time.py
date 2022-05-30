def worker_utilization_time(filename,path,run):
    import pandas as pd 
    from pandas import DataFrame
    import numpy as np
    import openpyxl
    import math
    pd.set_option('float_format', '{:f}'.format)
    import plotly.express as px
    import pandas as pd
    from datetime import datetime
    import plotly.figure_factory as ff
    import pandas as pd
    import openpyxl as pxl
    import os
    #Get the starttime
    df_time=pd.read_excel(filename+ '\\'+'worker_transition_history.xlsx')
    global_starttime=int(df_time['arrival timestamp'].iloc[0])

    #Get the global end time
    ObjRead = open("end.txt", "r")
    end_global = int(ObjRead.read())
    
    #Get the instation andinstation_working df
    df_instation=pd.read_excel(filename+'\\'+run+'_station_utilization_instation.xlsx')
    df_instation_working=pd.read_excel(filename+'\\'+run+'_station_utilization_working.xlsx')

    #Create a subfolder for the plots createdin this script
    if not os.path.exists(filename+'\\'+'worker_utilization_time'):
        os.makedirs(filename+'\\'+'worker_utilization_time')

    #Get the available stations
    stations_available=list(df_instation.columns.values)
    dict_list=[]
    average_list=[]
    worker_percentage_list=[]
    worker_station_list=[]
    for i in range(len(stations_available)):
        average_list.append([])
        worker_percentage_list.append([])
        worker_station_list.append([])
    for l in range(len(worker_percentage_list)):
        worker_percentage_list[l]=[0]*(end_global+1)
        worker_station_list[l]=[0]*(end_global+1)
    end_time_cut=0
    for s in range(len(stations_available)):
        start_time=[]
        end_time=[]
        history=df_instation[stations_available[s]].values.tolist()
        history_working=df_instation_working[stations_available[s]].values.tolist()
        inplace=False
        start_time=0
        for i in range(end_global+1):
            if history[i]!=0 and inplace==False:
                start_time=i
                inplace=True
                worker=history[i]

            if history[i]==0 and inplace==True:
                average=0
                end_time=i
                inplace=False
                counter=0
                for t in range(end_time-start_time):
                    if history_working[start_time+t]!=0:
                        counter+=1
                percentage=counter/(end_time-start_time)
                for w in range(end_time-start_time):
                    worker_percentage_list[worker-1][start_time+w]=percentage
                    worker_station_list[worker-1][start_time+w]=stations_available[s]
                start_time=0


    date_list=list()
    for i in range(end_global+1):
        date_list.append(datetime.fromtimestamp(i+int(global_starttime)))

    
    from pandas import DataFrame
    worker_list=['worker_1','worker_2','worker_3','worker_4','worker_5','worker_6','worker_7','worker_7','worker_7','worker_8','worker_9','worker_10']
    for i in range(len(worker_percentage_list)-1):
        if sum(worker_percentage_list[i])!=0:
                if sum(worker_percentage_list[i])!=0:
                    df = DataFrame (worker_percentage_list[i],columns=[worker_list[i]])
                    df['station']=worker_station_list[i]
                    df['date']=date_list
                    fig = px.scatter(df,x='date', y=worker_list[i], hover_data=['station'])
                    fig.update_layout(
                        xaxis_title='datetime',
                        yaxis_title='utilization '+worker_list[i])
                    fig.write_html(filename+'\\'+'worker_utilization_time'+'\\'+str(worker_list[i])+'.html')


    average_percentage=list()
    for i in range(len(worker_percentage_list[0])):
        counter=0
        counter_percentage=0
        for l in range(len(worker_percentage_list)):
            if worker_percentage_list[l][i]!=0:
                counter+=1
                counter_percentage+=worker_percentage_list[l][i]
        if counter_percentage==0:
            average_percentage.append(0)
        else:
            avg=counter_percentage/counter
            average_percentage.append(avg)

    df = DataFrame (average_percentage,columns=['average'])
    df['date']=date_list

    fig = px.line(x=df['date'], y=df['average'])
    fig.update_layout(
        xaxis_title='time(min)',
        yaxis_title='average utilization worker')
    fig.write_html(filename+'\\'+'worker_utilization_time'+'\\'+'average_utilization.html')

    
                
    

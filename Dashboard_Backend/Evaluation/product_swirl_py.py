def product_swirl(filename,path,run):
    import os
    #Enter the path + filename of the fileyou want to process
    
    if os.path.isfile(filename+ '\\' + run+'product_swirl.xls'):
        start_file=filename +'\\'+'product_swirl.xls'
    else:
        start_file=filename +'\\'+'product_swirl.xlsx'

    #Enter the path + filename wherethe processed file should be saved
    save_file=filename+ '\\' + run+'_analysis_part1.xlsx'

    ## Create new txt_file
    average_file = open(filename+ '\\' + run+"_avg.txt","w+")
    average_file.write(" ")
    average_file.close()

    import pandas as pd

    #Enter the path + filename of the fileyou want to process
    if os.path.isfile(filename +'\\'+'worker_transition_history.xlsx'):
        worker_file=filename +'\\'+'worker_transition_history.xlsx'
    else:
        worker_file=filename +'\\'+'worker_transition_history.xls'

    #Create df for the stations and number of workers
    df = pd.ExcelFile(worker_file)

    #Get the number of workers from the amount of sheets in the exported excel file
    number_workers = len(df.sheet_names)-1

    print('Numer of workers: '+str(number_workers))

    

    
    
    

    # importing pandas as pd 
    import pandas as pd 
    from pandas import DataFrame
    import numpy as np
    import plotly.express as px
    from statistics import mean
    #from txt_writer import *
      
    # into a dataframe object 
    df = pd.DataFrame(pd.read_excel(start_file)) 

    #Drop the INIT Jobs
    for i in range(len(df)):
        string = df['job name'][i]
        #print(string)
        if string[0]=='I':
            df=df.drop([i])

    # Sort in descending order for finished timestamp
    df_sorted = df.sort_values(by="finished timestamp")

    df_sort_start=df.sort_values(by='started timestamp')

    globalstarttime=df_sort_start['started timestamp'].head(1)

    starttime=[]
    starttime[0:len(df_sorted)]=globalstarttime

    simtime = (df_sorted['finished timestamp'] - starttime)/60

    df_sorted['sim time (min)'] =simtime

    ind =[]
    for i in range(len(df)):
        ind.append(i+1)

    df_sorted['finish index'] = ind

    p_rate=df_sorted['finish index']/df_sorted['sim time (min)']

    df_sorted['p-rate (1/min)']=p_rate

    df_sorted['p-rate per worker (1/(min/worker))']=df_sorted['p-rate (1/min)']/number_workers

    writer = pd.ExcelWriter(save_file)

    df_sorted.to_excel(writer, sheet_name='Job', index=False)

    writer.save()


    df_sorted = pd.DataFrame(pd.read_excel(save_file)) 

    simulation_time=df_sorted['sim time (min)'].values.tolist()

    time=0
    p_rate_hour=[]
    for i in range(len(simulation_time)):
        target_value = simulation_time[i]-60
        minimum = float("inf")
        for val in simulation_time:
           if abs(val - target_value) < minimum:
               final_value = val
               minimum = abs(val - target_value)
        index=simulation_time.index(final_value)
        if final_value < 60:
            final_value=0
        index_difference=(i-index)+1
        if simulation_time[i]-final_value==0:
            print('Division by zero in p-ratehour loop')
        else:
            p_rate_hour.append(index_difference/(simulation_time[i]-final_value))

    #print(p_rate_hour)
    df_p_rate_hour=DataFrame(p_rate_hour,columns=['p-rate hour (1/min)'])
    df_sorted['p-rate hour (1/min)']=df_p_rate_hour
    df_sorted.append(df_p_rate_hour, ignore_index=False, sort=False)


    p_rate_hour_average10=[]
    i=0
    while i <len(p_rate_hour):
        average = np.sum(p_rate_hour[i:i+10])/10
        for a in range(10):
            p_rate_hour_average10.append(average)
        
        i=i+10
        
    p_rate_hour_average10=p_rate_hour_average10[0:1241]

    #txt_writer(filenem+'\\','average_200='+str(average_200))
    #average_all=mean(p_rate_hour_average10)
    #txt_writer(filenem+'\\','average_all='+str(average_all))

    df_p_rate_hour_average10=DataFrame(p_rate_hour_average10,columns=['p-rate hour average 10 (1/min'])
    df_sorted['p-rate hour average 10 (1/min)']=df_p_rate_hour_average10

    writer = pd.ExcelWriter(save_file)
    df_sorted.to_excel(writer, sheet_name='Job', index=False)

    df = pd.DataFrame(pd.read_excel(start_file))

    for i in range(len(df)):
        string = df['job name'][i]
        #print(string)
        if string[0]=='I':
            df=df.drop([i])
            
    df = df.reset_index(drop=True)


    saved_old=0
    old="1"
    start_jis_scheduled=[]
    end_jis_time=[]
    end_jis_job=[]
    start_jis_job=[]
    start_jis_started=[]
    jis_nrr=[]
    jis_nrr.append(1)
    start_jis_job.append(df['job name'][0])
    for i in range (len(df)):
        string = df['product name'][i]
        under = string.find('_')
        jis_nr = string[under+1:len(string)]
        #print(jis_nr)
        if old!=jis_nr:
            minscheduled=min(df['scheduled timestamp'][saved_old:i-1])
            minstarted=min(df['started timestamp'][saved_old:i-1])
            start_jis_scheduled.append(min(df['scheduled timestamp'][saved_old:i+1]))
            start_jis_started.append(min(df['started timestamp'][saved_old:i+1]))
            #print(min(df['started timestamp'][saved_old:i+1]))
            start_jis_job.append(df['job name'][i])
            end_jis_time.append(max(df['finished timestamp'][saved_old:i]))
            end_jis_job.append(df['job name'][i-1])
            jis_nrr.append(int(jis_nr))
            saved_old=i
        old=jis_nr
    if saved_old==len(df)-1:
        start_jis_scheduled.append((df['scheduled timestamp'][saved_old]))
        start_jis_started.append((df['started timestamp'][saved_old]))
        end_jis_time.append((df['finished timestamp'][saved_old]))
        end_jis_job.append(df['job name'][len(df)-1])

    else:
        start_jis_scheduled.append(min(df['scheduled timestamp'][saved_old:len(df)-1]))
        start_jis_started.append(min(df['started timestamp'][saved_old:len(df)-1]))
        end_jis_time.append(max(df['finished timestamp'][saved_old:len(df)-1]))
        end_jis_job.append(df['job name'][len(df)-1])


    jis_df= pd.DataFrame()
    pd.set_option('float_format', '{:f}'.format)
    jis_df['Jis-wagon nr']=jis_nrr
    jis_df['start job name']=start_jis_job
    jis_df['end job name']=end_jis_job
    jis_df['Jis-wagon scheduled']=start_jis_scheduled
    jis_df['Jis-wagon opening']=start_jis_started
    jis_df['Jis-wagon completion']=end_jis_time

    difference=jis_df['Jis-wagon completion']-jis_df['Jis-wagon opening']
    #print(difference)
    jis_df['Jis-wagon duration (sec)']=difference

    jis_dur_min=jis_df['Jis-wagon duration (sec)']/60
    jis_df['Jis-wagon duration (min)']=jis_dur_min

    time_intervall=[]
    for i in range (len(jis_df)-1):
        a = jis_df['Jis-wagon completion'][i]
        b = jis_df['Jis-wagon completion'][i+1]
        timediff=b-a
        time_intervall.append(timediff)
        a=0
    time_intervall.append(0)

    jis_df['time intervall (sec)']=time_intervall
    jis_df['time intervall (min)']=jis_df['time intervall (sec)']/60

    time0=min(jis_df['Jis-wagon opening'])
    jis_df['sim time (min)']=(jis_df['Jis-wagon opening']-time0)/60

    intervall_name=[]
    for i in range (len(jis_df)-1):
        smaller = jis_df['Jis-wagon nr'][i]
        bigger = jis_df['Jis-wagon nr'][i+1]
        intervall_name.append("Jis " + str(bigger) + " fin after Jis "+str(smaller))
        time_intervall.append(timediff)
    intervall_name.append("")
    jis_df['intervall name']=intervall_name

    ml2per=[]
    job_list=df['job name'].values.tolist()
    #print(job_list)
    for i in range(len(jis_df)):
        for b in range(len(job_list)):
            if jis_df["start job name"][i] == job_list[b]:

                start_index=b
        
        for c in range(len(job_list)):

            if jis_df["end job name"][i] == job_list[c]:
                end_index=c
           
        per=(end_index-start_index+1)*2/32
        ml2per.append(per)
    jis_df["Jis-wagon ml2 products"]=ml2per

    mean_list_worker=jis_df['Jis-wagon ml2 products'].values.tolist()
    mean_value=mean(mean_list_worker)
    average_file = open(filename+ '\\' + run+"_avg.txt","a")
    average_file.write('\n avg Jis-wagon ml2 products='+str(mean_value))
    average_file.close()


    import plotly.express as px
    from plotly import graph_objects as go

    fig = go.Figure(
        data=[
            #go.Bar(
            #name='SIM Start Time',
            #y=jis_df['Jis-Wagon Nr'],
            #x=jis_df['SIM TIME (min)'],
            #offsetgroup=0,
            #orientation='h'
           # ),
            go.Bar(
            name='Jis-wagon duration (min)',
            y=jis_df['Jis-wagon nr'],
            x=jis_df['Jis-wagon duration (min)'],
            offsetgroup=0,
            base=jis_df["sim time (min)"],
            orientation='h'
            ),
            go.Bar(name='time intervall',
            y=jis_df['Jis-wagon nr'],
            x=jis_df['time intervall (min)'],
            base=(jis_df['Jis-wagon duration (min)']+jis_df['sim time (min)']),
            offsetgroup=1,
            orientation='h'
            )
        ],
        layout=go.Layout(
            title='Jis-wagon development',
            yaxis_title='Jis-wagon',
            xaxis_title='time(min)'))
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.write_html(filename+ '\\' + run+'_wagon_development.html')


    import plotly.express as px
    from plotly import graph_objects as go

    mean_list=jis_df['Jis-wagon duration (min)'].values.tolist()
    mean_value=mean(mean_list)
    

    fig = go.Figure(
        data=[
            go.Bar(
            y=jis_df['Jis-wagon duration (min)'],
            x=jis_df['Jis-wagon nr'],
            offsetgroup=0
            )
        ],
        layout=go.Layout(
            title='Jis-wagon duration',
            yaxis_title='Jis-wagon duration',
            xaxis_title='Jis-wagon')
    )

    fig.add_hline(y=mean_value, line_dash="dash")
        
    fig.write_html(filename+ '\\' +run+'_wagon_duration.html')


    #######

    mean_list=df_sorted['p-rate hour (1/min)'].values.tolist()
    mean_value=mean(mean_list)
    
    fig = px.line(df_sorted,x='finish index',y=['p-rate hour (1/min)'])


    fig.update_layout(
        title="p-rate hour",
        yaxis_title='p-rate',
        xaxis_title="finish index",
        showlegend=False
        
        )

    fig.add_hline(y=mean_value,line_dash="dash")

    fig.write_html(filename+ '\\' +run+'_prate_hour.html')

    average_file = open(filename+ '\\' + run+"_avg.txt","a")
    average_file.write('\n avg p-rate hour='+str(mean_value))
    average_file.close()


    mean_list=df_sorted['p-rate hour (1/min)'].values.tolist()
    mean_value=mean(mean_list[200:-1])
    average_file = open(filename+ '\\' + run+"_avg.txt","a")
    average_file.write('\n avg p-rate hour after ramp-up (200jobs)='+str(mean_value))
    average_file.close()

    ###

    fig = px.line(df_sorted,x='finish index',y=['p-rate hour average 10 (1/min)'])

    fig.update_layout(
        title="p-rate hour average 10",
        yaxis_title='p-rate',
        xaxis_title="finish index",
        showlegend=False
        
        )

    fig.add_hline(y=mean_value,line_dash="dash")

    fig.write_html(filename+ '\\' +run+'_prate_hour_average10.html')

    ####

    mean_list_worker=df_sorted['p-rate per worker (1/(min/worker))'].values.tolist()
    mean_value_worker=mean(mean_list_worker)

    mean_list=df_sorted['p-rate (1/min)'].values.tolist()
    mean_value=mean(mean_list)

    fig = px.line(df_sorted,x='finish index',y=['p-rate per worker (1/(min/worker))', 'p-rate (1/min)'])

    fig.update_layout(
        title="Comparison p-rate and p-rate per worker",
        xaxis_title="finish index"
        
        )

    fig.add_hline(y=mean_value_worker, line_dash="dash")

    fig.add_hline(y=mean_value,line_dash="dash")

    fig.write_html(filename+ '\\' +run+'_prate.html')

    average_file = open(filename+ '\\' + run+"_avg.txt","a")
    average_file.write('\n avg p-rate='+str(mean_value))
    average_file.write('\n avg p-rate per worker='+str(mean_value_worker))
    
    average_file.close()

    


    ###For the number ofJis-wagon open simultaneously
    start_time=int(jis_df['Jis-wagon opening'][0])
    end_time=int(jis_df['Jis-wagon completion'].max())
    simul=np.zeros(end_time-start_time)
    for i in range(len(jis_df)):
        start_jis=int(jis_df['Jis-wagon opening'][i]-start_time)
        end_jis=int(jis_df['Jis-wagon completion'][i]-start_time)
        for j in range(end_jis-start_jis):
            simul[start_jis+j]+=1

    from statistics import mean
    average_simul=mean(simul)

    simul_df = pd.DataFrame(simul,columns=['simul'])
    fig = px.area(simul_df,y='simul')
    fig.add_hline(y=average_simul, line_dash='dash')
    fig.update_layout(
        title='Simultaneous Jis-wagon',
        xaxis_title="time (sec)",
        yaxis_title='Jis-wagon open simultaneously'
        
        )

    fig.write_html(filename+ '\\' +run+'_simul_jis.html')

    average_file = open(filename+ '\\' + run+"_avg.txt","a")
    average_file.write('\n avg Jis-wagon run simultaneously='+str(average_simul))
    average_file.close()
    

    ###Histograms
    fig = px.histogram(jis_df, x="Jis-wagon duration (min)",nbins=45)
    fig.update_layout(
        title='Jis-wagon duration histogram'
        )
    fig.write_html(filename+ '\\' +run+'_Jis_duration_hist.html')

    
    mean_list_worker=jis_df['Jis-wagon duration (min)'].values.tolist()
    mean_value=mean(mean_list_worker)
    average_file = open(filename+ '\\' + run+"_avg.txt","a")
    average_file.write('\n avg Jis-wagon duration='+str(mean_value))
    average_file.close()
    

    fig = px.histogram(jis_df, x="time intervall (min)",nbins=50)
    fig.update_layout(
        title='Time intervall histogram'
        )
    fig.write_html(filename+ '\\' +run+'_time_intervall_hist.html')

    mean_list_worker=jis_df['time intervall (min)'].values.tolist()
    mean_value=mean(mean_list_worker)
    average_file = open(filename+ '\\' + run+"_avg.txt","a")
    average_file.write('\n avg time intervall(min)='+str(mean_value))
    average_file.close()

    



    #Write write the edited files
    jis_df.to_excel(writer, sheet_name='Jis-wagon', index=False)
    writer.save()
    

def product_swirl(filename,path,run):
    print(i)
    #Enter the path + filename of the fileyou want to process
    start_file=filename +'\\'+'product_swirl.xls'

    #Enter the path + filename wherethe processed file should be saved
    save_file=filename+ '\\' + 'analysis_part1.xls'

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

    df_sorted['p-Rate']=p_rate

    df_sorted['P-Rate per Worker']=df_sorted['P-Rate']/df_sorted['average workers']

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
        p_rate_hour.append(index_difference/(simulation_time[i]-final_value))

    #print(p_rate_hour)
    df_p_rate_hour=DataFrame(p_rate_hour,columns=['P Rate hour'])
    df_sorted['P-Rate hour']=df_p_rate_hour
    df_sorted.append(df_p_rate_hour, ignore_index=False, sort=False)


    p_rate_hour_average10=[]
    i=0
    while i <len(p_rate_hour):
        average = np.sum(p_rate_hour[i:i+10])/10
        for a in range(10):
            p_rate_hour_average10.append(average)
        
        i=i+10
        
    p_rate_hour_average10=p_rate_hour_average10[0:1241]

    txt_writer(filenem+'\\','average_200='+str(average_200))
    average_all=mean(p_rate_hour_average10)
    txt_writer(filenem+'\\','average_all='+str(average_all))

    df_p_rate_hour_average10=DataFrame(p_rate_hour_average10,columns=['P Rate hour average 10'])
    df_sorted['P-Rate hour average 10']=df_p_rate_hour_average10

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
    jis_df['Jis-Wagon Nr']=jis_nrr
    jis_df['Start job name']=start_jis_job
    jis_df['End job name']=end_jis_job
    jis_df['Jis-Wagon scheduled']=start_jis_scheduled
    jis_df['Jis-Wagon opening']=start_jis_started
    jis_df['Jis-Wagon completion']=end_jis_time

    difference=jis_df['Jis-Wagon completion']-jis_df['Jis-Wagon opening']
    #print(difference)
    jis_df['Jis-Wagon duration']=difference

    jis_dur_min=jis_df['Jis-Wagon duration']/60
    jis_df['Jis-Wagon duration (min)']=jis_dur_min

    time_intervall=[]
    for i in range (len(jis_df)-1):
        a = jis_df['Jis-Wagon completion'][i]
        b = jis_df['Jis-Wagon completion'][i+1]
        timediff=b-a
        time_intervall.append(timediff)
        a=0
    time_intervall.append(0)

    jis_df['Time Intervall (sec)']=time_intervall
    jis_df['Time Intervall (min)']=jis_df['Time Intervall (sec)']/60

    time0=min(jis_df['Jis-Wagon opening'])
    jis_df['SIM TIME (min)']=(jis_df['Jis-Wagon opening']-time0)/60

    intervall_name=[]
    for i in range (len(jis_df)-1):
        smaller = jis_df['Jis-Wagon Nr'][i]
        bigger = jis_df['Jis-Wagon Nr'][i+1]
        intervall_name.append("Jis " + str(bigger) + " fin after Jis "+str(smaller))
        time_intervall.append(timediff)
    intervall_name.append("")
    jis_df['Intervall name']=intervall_name

    ml2per=[]
    job_list=df['job name'].values.tolist()
    #print(job_list)
    for i in range(len(jis_df)):
        for b in range(len(job_list)):
            if jis_df["Start job name"][i] == job_list[b]:

                start_index=b
        
        for c in range(len(job_list)):

            if jis_df["End job name"][i] == job_list[c]:
                end_index=c
           
        per=(end_index-start_index+1)*2/32
        ml2per.append(per)
    jis_df["JIS-Wagon ML2 products"]=ml2per


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
            name='Jis-Wagon duration (min)',
            y=jis_df['Jis-Wagon Nr'],
            x=jis_df['Jis-Wagon duration (min)'],
            offsetgroup=0,
            base=jis_df["SIM TIME (min)"],
            orientation='h'
            ),
            go.Bar(name='Time Intervall',
            y=jis_df['Jis-Wagon Nr'],
            x=jis_df['Time Intervall (min)'],
            base=(jis_df['Jis-Wagon duration (min)']+jis_df['SIM TIME (min)']),
            offsetgroup=1,
            orientation='h'
            )
        ],
        layout=go.Layout(
            yaxis_title='Time in min',
            xaxis_title='JIS-Wagon'))
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.write_html(filename+ '\\' + '_wagon_development.html')


    import plotly.express as px
    from plotly import graph_objects as go

    fig = go.Figure(
        data=[
            go.Bar(
            y=jis_df['Jis-Wagon duration'],
            x=jis_df['Jis-Wagon Nr'],
            offsetgroup=0
            )
        ],
        layout=go.Layout(
            yaxis_title='Jis-Wagon duration',
            xaxis_title='JIS-Wagon')
    )
        
    fig.write_html(filename+ '\\' +'_wagon_duration.html')


    #######
    fig = px.line(df_sorted,x='finish index',y=['P-Rate hour'])


    fig.update_layout(
        title="P-Rate hour",
        xaxis_title="Finish index",
        showlegend=False
        
        )

    fig.write_html(filename+ '\\' +'prate_hour.html')

    ###

    fig = px.line(df_sorted,x='finish index',y=['P-Rate hour average 10'])

    fig.update_layout(
        title="P-Rate hour average 10",
        xaxis_title="Finish index",
        showlegend=False
        
        )

    fig.write_html(filename+ '\\' +'prate_hour_average10.html')

    ####

    fig = px.line(df_sorted,x='finish index',y=['P-Rate per Worker', 'P-Rate'])

    fig.update_layout(
        title="P-Rate hour",
        xaxis_title="Finish index"
        
        )

    fig.write_html(filename+ '\\' +'prate.html')


    #Write write the edited files
    jis_df.to_excel(writer, sheet_name='JIS-Wagon', index=False)
    writer.save()
    

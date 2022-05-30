def worker_transition(filename,path,run):
    # importing required packages
    import pandas as pd 
    from pandas import DataFrame
    import numpy as np
    import openpyxl
    import math
    from datetime import datetime
    import os
    pd.set_option('float_format', '{:f}'.format)


    #Enter the path + filename of the fileyou want to process
    if os.path.isfile(filename +'\\'+'worker_transition_history.xlsx'):
        start_file=filename +'\\'+'worker_transition_history.xlsx'
    else:
        start_file=filename +'\\'+'worker_transition_history.xls'

    #Enter the path + filename wherethe processed file should be saved
    end_file=filename+ '\\' +run+'_analysis_part2.xlsx'
    save_file_w_u=filename+ '\\' +run+'_station_utilization_instation.xlsx'
    save_file_i_u=filename+ '\\' +run+'_station_utilization_working.xlsx'

    #Create df for the stations and number of workers
    df = pd.ExcelFile(start_file)

    #Get the number of workers from the amount of sheets in the exported excel file
    number_workers = len(df.sheet_names)-1
    station_list=list()

    #Get the station numbers from the station_transition_history
    for i in range(number_workers):
        stations=df.parse(i)['destination station'].values.tolist()
        for station in stations:
            station_list.append(station) if station not in station_list else station_list

    print(station_list)

    #Get the start time
    daf=pd.read_excel(start_file, sheet_name=1)
    arrivaldaf=daf['arrival timestamp'].values.tolist()
    start_time=round(arrivaldaf[0])

    #Get the end time

    if os.path.isfile(filename +'\\'+'product_swirl.xlsx'):
        end_time_file=filename +'\\'+'product_swirl.xlsx'
    else:
        end_time_file=filename +'\\'+'product_swirl.xls'

    end_time_daf=pd.read_excel(end_time_file)
    end_daf=end_time_daf['finished timestamp'].values.tolist()
    end_time=round(end_daf[-1]+10000)

    #Initialize the end time
    #end_time=round(arrivaldaf[-1]+80000)

    #Initilalize the instation_list and working_list with zeros as twice as many fields as in the number of stations to allow double occupation
    instation_list=np.zeros((len(station_list)*2,end_time-start_time))
    working_list=np.zeros((len(station_list)*2,end_time-start_time))


    #Lists for the worker summary
    worker_totalwork_list=[]
    worker_totalwait_list=[]
    worker_totalwalking_list=[]
    worker_average_prod_station_list=[]
    worker_average_prod_stationrotation_list=[]
    worker_portion_working_list=[]
    worker_portion_idle_list=[]
    worker_portion_walking_list=[]
    average_rotation=[]


    #Lists for the station summary
    station_totalinplace_list=np.zeros(20)
    station_totalwork_list=np.zeros(20)
    station_totalfree_list=np.zeros(20)
    station_portioninplace_list=np.zeros(20)
    station_totalwork_list
    station_portionempty_list=np.zeros(20)



    #Initialize global end 
    global_end=[]
    global_end.append(0)

    #Function for finding strings
    def find_all(s, c):
        #print(s):
        idx = s.find(c)
        find_list=list()
        while idx != -1:
            find_list.append(idx)
            idx = s.find(c, idx + 1)
        return find_list

    def convert(df):
        df_list_test=df.values.tolist()
        #print(len(df_list_test[0]))
        if len(df_list_test[0])==5:
            #print(len(df_list_test[0]))
            #print('no conversion')
            return df
        else:
            #print('converting worker_transition')
            #print(len(df_list_test[0]))
            import math
            export_list=df.values.tolist()
            for i in range(len(export_list)):
                strg=''
                for b in range(len(export_list[i])-4):
                    if isinstance(export_list[i][b+4],str):
                        #print(b)
                        strg=strg+'('+export_list[i][b+4]+')'+','
                        #working_list.append('('+export_list[i][b+4]+')')
                export_list[i][4]=strg[0:-1]
            df=pd.DataFrame(export_list)
            df=df.drop(df.columns[5:], axis =1)
            export_list=df.values.tolist()
            #if export_list[-1][4]=='':
             #   export_list=export_list[:-1]
              #  export_list[-2][3]=np.nan
               # print(export_list[-2][3])
                #print(export_list[-2])
            df=pd.DataFrame(export_list, columns=['source station', 'destination station', 'arrival timestamp','departure timestamp', 'working timestamps'])

        return df


    def evaluate(df_worker, worker_number):
        #print(len(df_worker))
        df_worker=convert(df_worker)
        #print(worker_number)
        working=df_worker['working timestamps'].values.tolist()
        #Set an variable tomake sure the last row is not used twice
        last_none=False
        #print(working[-2:])
        #print('last'+working[-2:]+'is')
        if working[-1]=='':
            print('worker in last station not working')
            df_worker.drop(df_worker.tail(1).index, inplace = True)
            last_none=True
        if type(working[-1])!=str:
            print('worker in last station not working')
            df_worker.drop(df_worker.tail(1).index, inplace = True)
            last_none=True
        for i in range(len(df_worker)):
            #print(len(df_worker))
            arrival=df_worker['arrival timestamp']
            working=df_worker['working timestamps'].values.tolist()
            arrival_timestamp=df_worker['arrival timestamp'].values.tolist()
            departure_timestamp=df_worker['departure timestamp'].values.tolist()
            
            #get the current station of the worker
            station_name=str(df_worker['destination station'][i])

            #get the position of the station in the list
            station=station_list.index(station_name)
            #print(station)
            

            #Check if the current row is not the last row
            if math.isnan(departure_timestamp[i])==False:
                #get the start and end time ofthe current instation
                start_instation=round(float(df_worker['arrival timestamp'][i]))-start_time
                end_instation=round(float(df_worker['departure timestamp'][i]))-start_time

                #Check if the there is double station
                if np.sum(instation_list[station][start_instation:end_instation])!=0:
                    #print(station)
                    #if station==[4]:
                       # print('sum station 0')
                        #print(start_instation)
                        #print(np.sum(instation_list[station][start_instation:end_instation]))      
                    station=station+len(station_list)

                #Add the current worker_number to the current station instation_list
                instation_list[station][start_instation:end_instation]=worker_number

                #print('1')
                #print(instation_list[4][98178])
                #print('station 1_8_2')
                #print(instation_list[11][98178])
                #print(instation_list[12][98178])
                
                
                
                #For the summary
                station_totalinplace_list[station]=station_totalinplace_list[station]+end_instation-start_instation

        if last_none==False:
            #For the final row, since there is no departure timestamp    
            start_instation=round(float(df_worker['arrival timestamp'][len(df_worker)-1]))-start_time

            #get the current station of the worker
            station_name=str(df_worker['destination station'].iloc[-1])

            #get the position of the station in the list
            station=station_list.index(station_name)

            #Checkif there is a , in the last row
            if working[-1][-1]==',':
                last=round(float(working[-1][find_all(working[-1],',')[-2]+2:find_all(working[-1],')')[-1]]))-start_time
            else:
                last=round(float(working[-1][find_all(working[-1],',')[-1]+2:find_all(working[-1],')')[-1]]))-start_time
                    
            if np.sum(instation_list[station][start_instation:last])!=0:
                station_d_last=station+len(station_list)
                #print(station_d_last)
                #print(station_name)
                #print(station)
                #print(np.sum(instation_list[station][start_instation:last]))
                #print(start_instation)
                #print(last)
                    
            else:
                station_d_last=station

            instation_list[station_d_last][start_instation:last]=worker_number


        start_list_global=list()
        end_list_global=list()
        duration_global=list()
        total_duration_global=list()
        wait_time_global=list()
        total_wait_time_global=list()
        walk_time=list()
        rotation_time=list()

        #fot the working list
        for work in range(len(working)):
            #get the current station of the worker
            #station_name=str(df_worker['destination station'][i])

            #get the position of the station in the list
            #station=station_list.index(station_name)
                
                
            if isinstance(working[work], str):
                column_open = find_all(working[work],'(')
                column_close = find_all(working[work],')')
                cmma=find_all(working[work],',')

                start_list=list()
                end_list=list()

                ci=0
                cc=0
                for z in range(len(column_open)):
                    start_list.append(float(working[work][column_open[z]+1:cmma[ci]]))
                    end_list.append(float(working[work][cmma[cc]+1:column_close[z]]))
                    
                    start_working=round(float(working[work][column_open[z]+1:cmma[ci]]))-start_time
                    end_working=round(float(working[work][cmma[cc]+1:column_close[z]]))-start_time

                    for i in range(len(instation_list)):
                        if instation_list[i][start_working]==worker_number:
                            station_d=i
                            #if station_d == 11:
                                #print(start_working)
                                #print('Achtung')

                    working_list[station_d][start_working:end_working]=worker_number
                    
                    
                    start_work=float(working[work][column_open[z]+1:cmma[ci]])
                    end_work=float(working[work][cmma[cc]+1:column_close[z]])
                    
                    #Go to the next releavant entries in the list of columns or ,
                    ci=ci+2
                    cc=cc+2
                
                
                

                duration_steps=list()
                for q in range(len(start_list)):
                    duration_steps.append(end_list[q]-start_list[q])

                wait_time=list()
                
                ##Define Global End
                if len(end_list)>0:
                    if global_end[-1]<=end_list[-1]:
                        global_end.append(end_list[-1])


                    for b in range(len(start_list)-1):
                        wait_time.append(start_list[b+1]-end_list[b])

                    total_wait_time=np.sum(wait_time)

                    total_duration=np.sum(duration_steps)


                    if math.isnan(departure_timestamp[work])==False:
                        add_wait_time=(start_list[0]-float(arrival_timestamp[work])+(float(departure_timestamp[work])-float(end_list[len(end_list)-1])))
                        total_wait_time=total_wait_time+add_wait_time
                    else:
                        total_wait_time=total_wait_time


                    start_list_global.append(start_list)
                    end_list_global.append(end_list)
                    duration_global.append(duration_steps)
                    total_duration_global.append(np.sum(duration_steps))
                    wait_time_global.append(wait_time)
                    total_wait_time_global.append(total_wait_time)

                    rotation_time.append(0)
                    for c in range(len(arrival_timestamp)-1):
                        rotation_time.append(arrival_timestamp[c+1]-departure_timestamp[c])
                        
                    station_totalwork_list[station]=total_duration+station_totalwork_list[station]


        
        df_total_duration=DataFrame(total_duration_global,columns=['total_duration'])
        df_total_wait_time=DataFrame(total_wait_time_global,columns=['total_wait_time'])
        df_worker['total worktime (min)']=df_total_duration/60
        df_worker['wait time in station (min)']=df_total_wait_time/60
        df_worker['rotation time (min)']=DataFrame(rotation_time)/60
        df_worker['productivity at station']=df_worker['total worktime (min)']/df_worker['wait time in station (min)']
        df_worker['productivity including rotation']=df_worker['total worktime (min)']/(df_worker['wait time in station (min)']+df_worker['rotation time (min)'])
        
        
        worker_totalwork=df_total_duration.sum()
        worker_totalwork_list.append((float(df_total_duration.sum()[0])))
        worker_totalwait=df_total_wait_time.sum
        worker_totalwait_list.append((float(df_total_wait_time.sum()[0])))
        worker_totalwalking=df_worker['rotation time (min)'].sum()*60
        worker_totalwalking_list.append(float(worker_totalwalking))
        worker_average_prod_station_list.append(df_worker['productivity at station'].mean())
        worker_average_prod_stationrotation_list.append(df_worker['productivity including rotation'].mean())
        worker_portion_working=(float(df_total_duration.sum()[0]))/((float(df_total_duration.sum()[0]))+(float(df_total_wait_time.sum()[0]))+float(worker_totalwalking))
        worker_portion_working_list.append(worker_portion_working)
        worker_portion_idle=(float(df_total_wait_time.sum()[0]))/((float(df_total_wait_time.sum()[0]))+float(df_total_duration.sum()[0])+float(worker_totalwalking))
        worker_portion_idle_list.append(worker_portion_idle)
        worker_portion_working=(float(worker_totalwalking))/((float(df_total_wait_time.sum()[0]))+float(df_total_duration.sum()[0])+float(worker_totalwalking))
        worker_portion_walking_list.append(worker_portion_working)

        average_rotation.append(len(arrival)/((end_list[-1]-start_time)/3600))

        
            
        return df_worker

    #Define the number of workers
    number_sheets=pd.read_excel(start_file, sheet_name=None)
    number_sheets=list(number_sheets)
    num_workers=len(number_sheets)
    df_workers = pd.read_excel(start_file, None);
    worker_list = list(df_workers.keys())

    #print(worker_list)

    for i in range(num_workers-1):
        if i==0:
            if worker_list[i][-1] =='1':
                pd_worker_1=evaluate(pd.read_excel(start_file, sheet_name=i),1)
            if worker_list[i][-1] =='2':
                pd_worker_2=evaluate(pd.read_excel(start_file, sheet_name=i),2)
            if worker_list[i][-1] =='3':
                pd_worker_3=evaluate(pd.read_excel(start_file, sheet_name=i),3)
            if worker_list[i][-1] =='4':
                pd_worker_4=evaluate(pd.read_excel(start_file, sheet_name=i),4)
            if worker_list[i][-1] =='5':
                pd_worker_5=evaluate(pd.read_excel(start_file, sheet_name=i),5)
            if worker_list[i][-1] =='6':
                pd_worker_6=evaluate(pd.read_excel(start_file, sheet_name=i),6)
            if worker_list[i][-1] =='7':
                pd_worker_7=evaluate(pd.read_excel(start_file, sheet_name=i),7)
            if worker_list[i][-1] =='8':
                pd_worker_8=evaluate(pd.read_excel(start_file, sheet_name=i),8)
            if worker_list[i][-1] =='9':
                pd_worker_9=evaluate(pd.read_excel(start_file, sheet_name=i),9)
            if worker_list[i][-2] =='1' and worker_list[i][-1] =='0':
                pd_worker_10=evaluate(pd.read_excel(start_file, sheet_name=i),10)
        if i==1:
            if worker_list[i][-1] =='1':
                pd_worker_1=evaluate(pd.read_excel(start_file, sheet_name=i),1)
            if worker_list[i][-1] =='2':
                pd_worker_2=evaluate(pd.read_excel(start_file, sheet_name=i),2)
            if worker_list[i][-1] =='3':
                pd_worker_3=evaluate(pd.read_excel(start_file, sheet_name=i),3)
            if worker_list[i][-1] =='4':
                pd_worker_4=evaluate(pd.read_excel(start_file, sheet_name=i),4)
            if worker_list[i][-1] =='5':
                pd_worker_5=evaluate(pd.read_excel(start_file, sheet_name=i),5)
            if worker_list[i][-1] =='6':
                pd_worker_6=evaluate(pd.read_excel(start_file, sheet_name=i),6)
            if worker_list[i][-1] =='7':
                pd_worker_7=evaluate(pd.read_excel(start_file, sheet_name=i),7)
            if worker_list[i][-1] =='8':
                pd_worker_8=evaluate(pd.read_excel(start_file, sheet_name=i),8)
            if worker_list[i][-1] =='9':
                pd_worker_9=evaluate(pd.read_excel(start_file, sheet_name=i),9)
            if worker_list[i][-2] =='1' and worker_list[i][-1] =='0':
                pd_worker_10=evaluate(pd.read_excel(start_file, sheet_name=i),10)
        if i==2:
            if worker_list[i][-1] =='1':
                pd_worker_1=evaluate(pd.read_excel(start_file, sheet_name=i),1)
            if worker_list[i][-1] =='2':
                pd_worker_2=evaluate(pd.read_excel(start_file, sheet_name=i),2)
            if worker_list[i][-1] =='3':
                pd_worker_3=evaluate(pd.read_excel(start_file, sheet_name=i),3)
            if worker_list[i][-1] =='4':
                pd_worker_4=evaluate(pd.read_excel(start_file, sheet_name=i),4)
            if worker_list[i][-1] =='5':
                pd_worker_5=evaluate(pd.read_excel(start_file, sheet_name=i),5)
            if worker_list[i][-1] =='6':
                pd_worker_6=evaluate(pd.read_excel(start_file, sheet_name=i),6)
            if worker_list[i][-1] =='7':
                pd_worker_7=evaluate(pd.read_excel(start_file, sheet_name=i),7)
            if worker_list[i][-1] =='8':
                pd_worker_8=evaluate(pd.read_excel(start_file, sheet_name=i),8)
            if worker_list[i][-1] =='9':
                pd_worker_9=evaluate(pd.read_excel(start_file, sheet_name=i),9)
            if worker_list[i][-2] =='1' and worker_list[i][-1] =='0':
                pd_worker_10=evaluate(pd.read_excel(start_file, sheet_name=i),10)
        if i==3:
            if worker_list[i][-1] =='1':
                pd_worker_1=evaluate(pd.read_excel(start_file, sheet_name=i),1)
            if worker_list[i][-1] =='2':
                pd_worker_2=evaluate(pd.read_excel(start_file, sheet_name=i),2)
            if worker_list[i][-1] =='3':
                pd_worker_3=evaluate(pd.read_excel(start_file, sheet_name=i),3)
            if worker_list[i][-1] =='4':
                pd_worker_4=evaluate(pd.read_excel(start_file, sheet_name=i),4)
            if worker_list[i][-1] =='5':
                pd_worker_5=evaluate(pd.read_excel(start_file, sheet_name=i),5)
            if worker_list[i][-1] =='6':
                pd_worker_6=evaluate(pd.read_excel(start_file, sheet_name=i),6)
            if worker_list[i][-1] =='7':
                pd_worker_7=evaluate(pd.read_excel(start_file, sheet_name=i),7)
            if worker_list[i][-1] =='8':
                pd_worker_8=evaluate(pd.read_excel(start_file, sheet_name=i),8)
            if worker_list[i][-1] =='9':
                pd_worker_9=evaluate(pd.read_excel(start_file, sheet_name=i),9)
            if worker_list[i][-2] =='1' and worker_list[i][-1] =='0':
                pd_worker_10=evaluate(pd.read_excel(start_file, sheet_name=i),10)
        if i==4:
            if worker_list[i][-1] =='1':
                pd_worker_1=evaluate(pd.read_excel(start_file, sheet_name=i),1)
            if worker_list[i][-1] =='2':
                pd_worker_2=evaluate(pd.read_excel(start_file, sheet_name=i),2)
            if worker_list[i][-1] =='3':
                pd_worker_3=evaluate(pd.read_excel(start_file, sheet_name=i),3)
            if worker_list[i][-1] =='4':
                pd_worker_4=evaluate(pd.read_excel(start_file, sheet_name=i),4)
            if worker_list[i][-1] =='5':
                pd_worker_5=evaluate(pd.read_excel(start_file, sheet_name=i),5)
            if worker_list[i][-1] =='6':
                pd_worker_6=evaluate(pd.read_excel(start_file, sheet_name=i),6)
            if worker_list[i][-1] =='7':
                pd_worker_7=evaluate(pd.read_excel(start_file, sheet_name=i),7)
            if worker_list[i][-1] =='8':
                pd_worker_8=evaluate(pd.read_excel(start_file, sheet_name=i),8)
            if worker_list[i][-1] =='9':
                pd_worker_9=evaluate(pd.read_excel(start_file, sheet_name=i),9)
            if worker_list[i][-2] =='1' and worker_list[i][-1] =='0':
                pd_worker_10=evaluate(pd.read_excel(start_file, sheet_name=i),10)
                
        if i==5:
            if worker_list[i][-1] =='1':
                pd_worker_1=evaluate(pd.read_excel(start_file, sheet_name=i),1)
            if worker_list[i][-1] =='2':
                pd_worker_2=evaluate(pd.read_excel(start_file, sheet_name=i),2)
            if worker_list[i][-1] =='3':
                pd_worker_3=evaluate(pd.read_excel(start_file, sheet_name=i),3)
            if worker_list[i][-1] =='4':
                pd_worker_4=evaluate(pd.read_excel(start_file, sheet_name=i),4)
            if worker_list[i][-1] =='5':
                pd_worker_5=evaluate(pd.read_excel(start_file, sheet_name=i),5)
            if worker_list[i][-1] =='6':
                pd_worker_6=evaluate(pd.read_excel(start_file, sheet_name=i),6)
            if worker_list[i][-1] =='7':
                pd_worker_7=evaluate(pd.read_excel(start_file, sheet_name=i),7)
            if worker_list[i][-1] =='8':
                pd_worker_8=evaluate(pd.read_excel(start_file, sheet_name=i),8)
            if worker_list[i][-1] =='9':
                pd_worker_9=evaluate(pd.read_excel(start_file, sheet_name=i),9)
            if worker_list[i][-2] =='1' and worker_list[i][-1] =='0':
                pd_worker_10=evaluate(pd.read_excel(start_file, sheet_name=i),10)
                
        if i==6:
            if worker_list[i][-1] =='1':
                pd_worker_1=evaluate(pd.read_excel(start_file, sheet_name=i),1)
            if worker_list[i][-1] =='2':
                pd_worker_2=evaluate(pd.read_excel(start_file, sheet_name=i),2)
            if worker_list[i][-1] =='3':
                pd_worker_3=evaluate(pd.read_excel(start_file, sheet_name=i),3)
            if worker_list[i][-1] =='4':
                pd_worker_4=evaluate(pd.read_excel(start_file, sheet_name=i),4)
            if worker_list[i][-1] =='5':
                pd_worker_5=evaluate(pd.read_excel(start_file, sheet_name=i),5)
            if worker_list[i][-1] =='6':
                pd_worker_6=evaluate(pd.read_excel(start_file, sheet_name=i),6)
            if worker_list[i][-1] =='7':
                pd_worker_7=evaluate(pd.read_excel(start_file, sheet_name=i),7)
            if worker_list[i][-1] =='8':
                pd_worker_8=evaluate(pd.read_excel(start_file, sheet_name=i),8)
            if worker_list[i][-1] =='9':
                pd_worker_9=evaluate(pd.read_excel(start_file, sheet_name=i),9)
            if worker_list[i][-2] =='1' and worker_list[i][-1] =='0':
                pd_worker_10=evaluate(pd.read_excel(start_file, sheet_name=i),10)
        if i==7:
            if worker_list[i][-1] =='1':
                pd_worker_1=evaluate(pd.read_excel(start_file, sheet_name=i),1)
            if worker_list[i][-1] =='2':
                pd_worker_2=evaluate(pd.read_excel(start_file, sheet_name=i),2)
            if worker_list[i][-1] =='3':
                pd_worker_3=evaluate(pd.read_excel(start_file, sheet_name=i),3)
            if worker_list[i][-1] =='4':
                pd_worker_4=evaluate(pd.read_excel(start_file, sheet_name=i),4)
            if worker_list[i][-1] =='5':
                pd_worker_5=evaluate(pd.read_excel(start_file, sheet_name=i),5)
            if worker_list[i][-1] =='6':
                pd_worker_6=evaluate(pd.read_excel(start_file, sheet_name=i),6)
            if worker_list[i][-1] =='7':
                pd_worker_7=evaluate(pd.read_excel(start_file, sheet_name=i),7)
            if worker_list[i][-1] =='8':
                pd_worker_8=evaluate(pd.read_excel(start_file, sheet_name=i),8)
            if worker_list[i][-1] =='9':
                pd_worker_9=evaluate(pd.read_excel(start_file, sheet_name=i),9)
            if worker_list[i][-2] =='1' and worker_list[i][-1] =='0':
                pd_worker_10=evaluate(pd.read_excel(start_file, sheet_name=i),10)
        if i==8:
            if worker_list[i][-1] =='1':
                pd_worker_1=evaluate(pd.read_excel(start_file, sheet_name=i),1)
            if worker_list[i][-1] =='2':
                pd_worker_2=evaluate(pd.read_excel(start_file, sheet_name=i),2)
            if worker_list[i][-1] =='3':
                pd_worker_3=evaluate(pd.read_excel(start_file, sheet_name=i),3)
            if worker_list[i][-1] =='4':
                pd_worker_4=evaluate(pd.read_excel(start_file, sheet_name=i),4)
            if worker_list[i][-1] =='5':
                pd_worker_5=evaluate(pd.read_excel(start_file, sheet_name=i),5)
            if worker_list[i][-1] =='6':
                pd_worker_6=evaluate(pd.read_excel(start_file, sheet_name=i),6)
            if worker_list[i][-1] =='7':
                pd_worker_7=evaluate(pd.read_excel(start_file, sheet_name=i),7)
            if worker_list[i][-1] =='8':
                pd_worker_8=evaluate(pd.read_excel(start_file, sheet_name=i),8)
            if worker_list[i][-1] =='9':
                pd_worker_9=evaluate(pd.read_excel(start_file, sheet_name=i),9)
            if worker_list[i][-2] =='1' and worker_list[i][-1] =='0':
                pd_worker_10=evaluate(pd.read_excel(start_file, sheet_name=i),10)
        if i==9:
            if worker_list[i][-1] =='1':
                pd_worker_1=evaluate(pd.read_excel(start_file, sheet_name=i),1)
            if worker_list[i][-1] =='2':
                pd_worker_2=evaluate(pd.read_excel(start_file, sheet_name=i),2)
            if worker_list[i][-1] =='3':
                pd_worker_3=evaluate(pd.read_excel(start_file, sheet_name=i),3)
            if worker_list[i][-1] =='4':
                pd_worker_4=evaluate(pd.read_excel(start_file, sheet_name=i),4)
            if worker_list[i][-1] =='5':
                pd_worker_5=evaluate(pd.read_excel(start_file, sheet_name=i),5)
            if worker_list[i][-1] =='6':
                pd_worker_6=evaluate(pd.read_excel(start_file, sheet_name=i),6)
            if worker_list[i][-1] =='7':
                pd_worker_7=evaluate(pd.read_excel(start_file, sheet_name=i),7)
            if worker_list[i][-1] =='8':
                pd_worker_8=evaluate(pd.read_excel(start_file, sheet_name=i),8)
            if worker_list[i][-1] =='9':
                pd_worker_9=evaluate(pd.read_excel(start_file, sheet_name=i),9)
            if worker_list[i][-2] =='1' and worker_list[i][-1] =='0':
                pd_worker_10=evaluate(pd.read_excel(start_file, sheet_name=i),10)
        if i==10:
            pd_worker_11=evaluate(pd.read_excel(start_file, sheet_name=i))
        if i==11:
            pd_worker_12=evaluate(pd.read_excel(start_file, sheet_name=i))
        if i==12:
            pd_worker_13=evaluate(pd.read_excel(start_file, sheet_name=i))
        if i==13:
            pd_worker_14=evaluate(pd.read_excel(start_file, sheet_name=i))
        if i==14:
            pd_worker_15=evaluate(pd.read_excel(start_file, sheet_name=i))

    dfObj = pd.DataFrame(instation_list)   
    dfObj2= pd.DataFrame(working_list)
    station_totalinplace_list=station_totalinplace_list[1:8]
    station_totalwork_list=station_totalwork_list[1:8]
    global_end_value=global_end[-1]
    total_duration=global_end_value-start_time
    station_totalfree_list[0:7]=total_duration
    station_totalfree_list=station_totalfree_list[0:7]
    station_totalfree_list=station_totalfree_list-station_totalinplace_list

    #Cut the instation_list to the real end time
    end_global = int(global_end_value-start_time)
    
    file2write=open('end.txt','w')
    file2write.write(str(end_global))
    file2write.close()

    

    
    #print(len(instation_list[0]))
    #for i in range(len(instation_list)):
        #del instation_list[0][int(global_end_value-start_time):]
     #   print(type(instation_list))
      #  print(len(instation_list[i]))
       # instation_list[i]=instation_list[i][0:2]
        #print(len(instation_list[i]))
        
    #for i in range(len(working_list)):
     #   working_list[i]=working_list[i][0:int(global_end_value-start_time)]

    #print(len(working_list[0]))
    #print(len(instation_list[0]))





    #############
    #Write and save the dataframes

    #Add the second workplaces to the station_list
    len_station_list=len(station_list)
    for i in range(len_station_list):
        station_list.append(station_list[i]+' 2')

    #Create df from the instation_list
    df_station_instation = pd.DataFrame(instation_list)   
    df_station_instation = df_station_instation.T
    df_station_instation.columns=station_list

    #Create df from the working_list
    df_station_working= pd.DataFrame(working_list)
    df_station_working = df_station_working.T
    df_station_working.columns=station_list

    #Save the df to excel
    df_station_instation.to_excel(save_file_w_u, index=False)
    df_station_working.to_excel(save_file_i_u, index=False)



    ########
    ###Write the Excel file to the disired path specified as 'end_file'
    # create excel writer
    writer = pd.ExcelWriter(end_file)
    # write dataframe to excel sheet named 'marks'
    for i in range(num_workers-1):
        if i==0:
            if worker_list[i][-1] =='1':
                pd_worker_1.to_excel(writer, 'worker_1',index=False)
            if worker_list[i][-1] =='2':
                pd_worker_2.to_excel(writer, 'worker_2',index=False)
            if worker_list[i][-1] =='3':
                pd_worker_3.to_excel(writer, 'worker_3',index=False)
            if worker_list[i][-1] =='4':
                pd_worker_4.to_excel(writer, 'worker_4',index=False)
            if worker_list[i][-1] =='5':
                pd_worker_5.to_excel(writer, 'worker_5',index=False)
            if worker_list[i][-1] =='6':
                pd_worker_6.to_excel(writer, 'worker_6',index=False)
            if worker_list[i][-1] =='7':
                pd_worker_7.to_excel(writer, 'worker_7',index=False)
            if worker_list[i][-1] =='8':
                pd_worker_8.to_excel(writer, 'worker_8',index=False)
            if worker_list[i][-1] =='9':
                pd_worker_9.to_excel(writer, 'worker_9',index=False)
            if worker_list[i][-1] =='10':
                pd_worker_10.to_excel(writer, 'worker_10',index=False)
        if i==1:
            if worker_list[i][-1] =='1':
                pd_worker_1.to_excel(writer, 'worker_1',index=False)
            if worker_list[i][-1] =='2':
                pd_worker_2.to_excel(writer, 'worker_2',index=False)
            if worker_list[i][-1] =='3':
                pd_worker_3.to_excel(writer, 'worker_3',index=False)
            if worker_list[i][-1] =='4':
                pd_worker_4.to_excel(writer, 'worker_4',index=False)
            if worker_list[i][-1] =='5':
                pd_worker_5.to_excel(writer, 'worker_5',index=False)
            if worker_list[i][-1] =='6':
                pd_worker_6.to_excel(writer, 'worker_6',index=False)
            if worker_list[i][-1] =='7':
                pd_worker_7.to_excel(writer, 'worker_7',index=False)
            if worker_list[i][-1] =='8':
                pd_worker_8.to_excel(writer, 'worker_8',index=False)
            if worker_list[i][-1] =='9':
                pd_worker_9.to_excel(writer, 'worker_9',index=False)
            if worker_list[i][-1] =='10':
                pd_worker_10.to_excel(writer, 'worker_10',index=False)
        if i==2:
            if worker_list[i][-1] =='1':
                pd_worker_1.to_excel(writer, 'worker_1',index=False)
            if worker_list[i][-1] =='2':
                pd_worker_2.to_excel(writer, 'worker_2',index=False)
            if worker_list[i][-1] =='3':
                pd_worker_3.to_excel(writer, 'worker_3',index=False)
            if worker_list[i][-1] =='4':
                pd_worker_4.to_excel(writer, 'worker_4',index=False)
            if worker_list[i][-1] =='5':
                pd_worker_5.to_excel(writer, 'worker_5',index=False)
            if worker_list[i][-1] =='6':
                pd_worker_6.to_excel(writer, 'worker_6',index=False)
            if worker_list[i][-1] =='7':
                pd_worker_7.to_excel(writer, 'worker_7',index=False)
            if worker_list[i][-1] =='8':
                pd_worker_8.to_excel(writer, 'worker_8',index=False)
            if worker_list[i][-1] =='9':
                pd_worker_9.to_excel(writer, 'worker_9',index=False)
            if worker_list[i][-1] =='10':
                pd_worker_10.to_excel(writer, 'worker_10',index=False)
        if i==3:
            if worker_list[i][-1] =='1':
                pd_worker_1.to_excel(writer, 'worker_1',index=False)
            if worker_list[i][-1] =='2':
                pd_worker_2.to_excel(writer, 'worker_2',index=False)
            if worker_list[i][-1] =='3':
                pd_worker_3.to_excel(writer, 'worker_3',index=False)
            if worker_list[i][-1] =='4':
                pd_worker_4.to_excel(writer, 'worker_4',index=False)
            if worker_list[i][-1] =='5':
                pd_worker_5.to_excel(writer, 'worker_5',index=False)
            if worker_list[i][-1] =='6':
                pd_worker_6.to_excel(writer, 'worker_6',index=False)
            if worker_list[i][-1] =='7':
                pd_worker_7.to_excel(writer, 'worker_7',index=False)
            if worker_list[i][-1] =='8':
                pd_worker_8.to_excel(writer, 'worker_8',index=False)
            if worker_list[i][-1] =='9':
                pd_worker_9.to_excel(writer, 'worker_9',index=False)
            if worker_list[i][-1] =='10':
                pd_worker_10.to_excel(writer, 'worker_10',index=False)
        if i==4:
            if worker_list[i][-1] =='1':
                pd_worker_1.to_excel(writer, 'worker_1',index=False)
            if worker_list[i][-1] =='2':
                pd_worker_2.to_excel(writer, 'worker_2',index=False)
            if worker_list[i][-1] =='3':
                pd_worker_3.to_excel(writer, 'worker_3',index=False)
            if worker_list[i][-1] =='4':
                pd_worker_4.to_excel(writer, 'worker_4',index=False)
            if worker_list[i][-1] =='5':
                pd_worker_5.to_excel(writer, 'worker_5',index=False)
            if worker_list[i][-1] =='6':
                pd_worker_6.to_excel(writer, 'worker_6',index=False)
            if worker_list[i][-1] =='7':
                pd_worker_7.to_excel(writer, 'worker_7',index=False)
            if worker_list[i][-1] =='8':
                pd_worker_8.to_excel(writer, 'worker_8',index=False)
            if worker_list[i][-1] =='9':
                pd_worker_9.to_excel(writer, 'worker_9',index=False)
            if worker_list[i][-1] =='10':
                pd_worker_10.to_excel(writer, 'worker_10',index=False)
        if i==5:
            if worker_list[i][-1] =='1':
                pd_worker_1.to_excel(writer, 'worker_1',index=False)
            if worker_list[i][-1] =='2':
                pd_worker_2.to_excel(writer, 'worker_2',index=False)
            if worker_list[i][-1] =='3':
                pd_worker_3.to_excel(writer, 'worker_3',index=False)
            if worker_list[i][-1] =='4':
                pd_worker_4.to_excel(writer, 'worker_4',index=False)
            if worker_list[i][-1] =='5':
                pd_worker_5.to_excel(writer, 'worker_5',index=False)
            if worker_list[i][-1] =='6':
                pd_worker_6.to_excel(writer, 'worker_6',index=False)
            if worker_list[i][-1] =='7':
                pd_worker_7.to_excel(writer, 'worker_7',index=False)
            if worker_list[i][-1] =='8':
                pd_worker_8.to_excel(writer, 'worker_8',index=False)
            if worker_list[i][-1] =='9':
                pd_worker_9.to_excel(writer, 'worker_9',index=False)
            if worker_list[i][-1] =='10':
                pd_worker_10.to_excel(writer, 'worker_10',index=False)
        if i==6:
            if worker_list[i][-1] =='1':
                pd_worker_1.to_excel(writer, 'worker_1',index=False)
            if worker_list[i][-1] =='2':
                pd_worker_2.to_excel(writer, 'worker_2',index=False)
            if worker_list[i][-1] =='3':
                pd_worker_3.to_excel(writer, 'worker_3',index=False)
            if worker_list[i][-1] =='4':
                pd_worker_4.to_excel(writer, 'worker_4',index=False)
            if worker_list[i][-1] =='5':
                pd_worker_5.to_excel(writer, 'worker_5',index=False)
            if worker_list[i][-1] =='6':
                pd_worker_6.to_excel(writer, 'worker_6',index=False)
            if worker_list[i][-1] =='7':
                pd_worker_7.to_excel(writer, 'worker_7',index=False)
            if worker_list[i][-1] =='8':
                pd_worker_8.to_excel(writer, 'worker_8',index=False)
            if worker_list[i][-1] =='9':
                pd_worker_9.to_excel(writer, 'worker_9',index=False)
            if worker_list[i][-1] =='10':
                pd_worker_10.to_excel(writer, 'worker_10',index=False)
        if i==7:
            if worker_list[i][-1] =='1':
                pd_worker_1.to_excel(writer, 'worker_1',index=False)
            if worker_list[i][-1] =='2':
                pd_worker_2.to_excel(writer, 'worker_2',index=False)
            if worker_list[i][-1] =='3':
                pd_worker_3.to_excel(writer, 'worker_3',index=False)
            if worker_list[i][-1] =='4':
                pd_worker_4.to_excel(writer, 'worker_4',index=False)
            if worker_list[i][-1] =='5':
                pd_worker_5.to_excel(writer, 'worker_5',index=False)
            if worker_list[i][-1] =='6':
                pd_worker_6.to_excel(writer, 'worker_6',index=False)
            if worker_list[i][-1] =='7':
                pd_worker_7.to_excel(writer, 'worker_7',index=False)
            if worker_list[i][-1] =='8':
                pd_worker_8.to_excel(writer, 'worker_8',index=False)
            if worker_list[i][-1] =='9':
                pd_worker_9.to_excel(writer, 'worker_9',index=False)
            if worker_list[i][-1] =='10':
                pd_worker_10.to_excel(writer, 'worker_10',index=False)
        if i==8:
            if worker_list[i][-1] =='1':
                pd_worker_1.to_excel(writer, 'worker_1',index=False)
            if worker_list[i][-1] =='2':
                pd_worker_2.to_excel(writer, 'worker_2',index=False)
            if worker_list[i][-1] =='3':
                pd_worker_3.to_excel(writer, 'worker_3',index=False)
            if worker_list[i][-1] =='4':
                pd_worker_4.to_excel(writer, 'worker_4',index=False)
            if worker_list[i][-1] =='5':
                pd_worker_5.to_excel(writer, 'worker_5',index=False)
            if worker_list[i][-1] =='6':
                pd_worker_6.to_excel(writer, 'worker_6',index=False)
            if worker_list[i][-1] =='7':
                pd_worker_7.to_excel(writer, 'worker_7',index=False)
            if worker_list[i][-1] =='8':
                pd_worker_8.to_excel(writer, 'worker_8',index=False)
            if worker_list[i][-1] =='9':
                pd_worker_9.to_excel(writer, 'worker_9',index=False)
            if worker_list[i][-1] =='10':
                pd_worker_10.to_excel(writer, 'worker_10',index=False)
        if i==9:
            if worker_list[i][-1] =='1':
                pd_worker_1.to_excel(writer, 'worker_1',index=False)
            if worker_list[i][-1] =='2':
                pd_worker_2.to_excel(writer, 'worker_2',index=False)
            if worker_list[i][-1] =='3':
                pd_worker_3.to_excel(writer, 'worker_3',index=False)
            if worker_list[i][-1] =='4':
                pd_worker_4.to_excel(writer, 'worker_4',index=False)
            if worker_list[i][-1] =='5':
                pd_worker_5.to_excel(writer, 'worker_5',index=False)
            if worker_list[i][-1] =='6':
                pd_worker_6.to_excel(writer, 'worker_6',index=False)
            if worker_list[i][-1] =='7':
                pd_worker_7.to_excel(writer, 'worker_7',index=False)
            if worker_list[i][-1] =='8':
                pd_worker_8.to_excel(writer, 'worker_8',index=False)
            if worker_list[i][-1] =='9':
                pd_worker_9.to_excel(writer, 'worker_9',index=False)
            if worker_list[i][-1] =='10':
                pd_worker_10.to_excel(writer, 'worker_10',index=False)
        if i==10:
            if worker_list[i][-1] =='1':
                pd_worker_1.to_excel(writer, 'worker_1',index=False)
            if worker_list[i][-1] =='2':
                pd_worker_2.to_excel(writer, 'worker_2',index=False)
            if worker_list[i][-1] =='3':
                pd_worker_3.to_excel(writer, 'worker_3',index=False)
            if worker_list[i][-1] =='4':
                pd_worker_4.to_excel(writer, 'worker_4',index=False)
            if worker_list[i][-1] =='5':
                pd_worker_5.to_excel(writer, 'worker_5',index=False)
            if worker_list[i][-1] =='6':
                pd_worker_6.to_excel(writer, 'worker_6',index=False)
            if worker_list[i][-1] =='7':
                pd_worker_7.to_excel(writer, 'worker_7',index=False)
            if worker_list[i][-1] =='8':
                pd_worker_8.to_excel(writer, 'worker_8',index=False)
            if worker_list[i][-1] =='9':
                pd_worker_9.to_excel(writer, 'worker_9',index=False)
            if worker_list[i][-1] =='10':
                pd_worker_10.to_excel(writer, 'worker_10',index=False)

    #For the worker summary
    pd_worker_summary=pd.read_excel(start_file, sheet_name=len(worker_list)-1) 
    pd_worker_summary=pd_worker_summary.drop(columns=['total working time (s)', 'total idle time (s)',
       'total walking time (s)'])
    pd_worker_summary['total working time (min)']=DataFrame(worker_totalwork_list)/60
    pd_worker_summary['total idle time (min)']=DataFrame(worker_totalwait_list)/60
    pd_worker_summary['total rotation time (min)']=DataFrame(worker_totalwalking_list)/60
    pd_worker_summary['percentage working']=DataFrame(worker_portion_working_list)
    pd_worker_summary['percentage idle']=DataFrame(worker_portion_idle_list)
    pd_worker_summary['percentage rotation']=DataFrame(worker_portion_walking_list)
    pd_worker_summary['average rotations (1/h)']=DataFrame(average_rotation)
    pd_worker_summary.to_excel(writer, 'worker summary', index=False)
    writer.save()

    from statistics import mean

    mean_value=mean(worker_portion_working_list)
    average_file = open(filename+ '\\' + run+"_avg.txt","a")
    average_file.write('\n avg percentage worker working ='+str(mean_value))
    average_file.close()
    mean_percentage_working=mean_value

    mean_value=mean(worker_portion_idle_list)
    average_file = open(filename+ '\\' + run+"_avg.txt","a")
    average_file.write('\n avg percentage worker idle='+str(mean_value))
    average_file.close()
    mean_percentage_idle=mean_value

    mean_value=mean(worker_portion_walking_list)
    average_file = open(filename+ '\\' + run+"_avg.txt","a")
    average_file.write('\n avg percentage worker rotating='+str(mean_value))
    average_file.close()
    mean_percentage_rotating=mean_value

    from plotly import graph_objects as go

    fig = go.Figure(
        data=[
            go.Bar(
            name='mean percentage working',
            y=[mean_percentage_working],
            #x=[' '],
            offsetgroup=0,
            orientation='v'
            ),
            go.Bar(
            name='mean percentage rotating',
            y=[mean_percentage_rotating],
            #x=[' '],
            offsetgroup=0,
            base=mean_percentage_working,
            orientation='v'
            ),
            go.Bar(name='mean percentage idle',
            y=[mean_percentage_idle],
            #x=[' '],
            base=(mean_percentage_working+mean_percentage_rotating),
            offsetgroup=0,
            orientation='v'
            )
        ],
        layout=go.Layout(
            newshape_line_width=400,
            ))
    #fig['layout']['yaxis']['autorange'] = "reversed"
    #fig.show()
    fig.write_html(filename+ '\\' +run+'_mean_percentage_worker.html')
    

    
    
   # print('Product Swirl DataFrames are written successfully to Excel Sheet.')




    

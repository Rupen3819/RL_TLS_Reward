def station_summary(filename,path,run):
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
    import openpyxl as pxl
    pd.set_option('float_format', '{:f}'.format)

    ObjRead = open("end.txt", "r")
    end_global = int(ObjRead.read())

    #Function for searching the index of a run in a list 
    def search(list, run):
        for i in range(len(list)):
            if list[i] == run:
                return list.index(run)
        return 100

    #Load the instation and working dataframes
    df_instation=pd.read_excel(filename+ '\\' +run+'_station_utilization_instation.xlsx')
    df_instation_working=pd.read_excel(filename+ '\\' +run+'_station_utilization_working.xlsx')

    #Transform instation_working 
    instation_working=df_instation_working.values.tolist()
    instation_working=np.array(instation_working).T

    #Transform instation
    instation=df_instation.values.tolist()
    instation=np.array(instation).T


    stations_average_working=[]
    sum_time_working=[]
    sum_time_instation=[]
    for station in instation_working:
        non_zero=0
        for values in station:
            if values!=0:
                non_zero+=1
        stations_average_working.append(non_zero/len(station))
        sum_time_working.append(non_zero)

    stations_average_instation=[]
    for station in instation:
        non_zero=0
        for values in station:
            if values!=0:
                non_zero+=1
        stations_average_instation.append(non_zero/len(station))
        sum_time_instation.append(non_zero)

    def double_station(first,second):
        sum_equal=0
        for i in range(len(instation[0])):
            if instation[first][i]!=0 and instation[second][i]!=0:
                sum_equal+=1
            
        sum_equal_working=0
        for i in range(len(instation_working[0])):
            if instation_working[first][i]!=0 and instation_working[second][i]!=0:
                sum_equal_working+=1
            
        sum_time_instation[first]=sum_time_instation[first]+sum_time_instation[second]-sum_equal
        sum_time_working[first]=sum_time_working[first]+sum_time_working[second]-sum_equal_working

    for i in range(int((len(instation_working)/2))):
        double_station(i,i+int((len(instation_working)/2)))


    sum_time_instation=sum_time_instation[0:int((len(instation_working)/2))]
    sum_time_working=sum_time_working[0:int((len(instation_working)/2))]


    #stations_available=['station_2', 'station_3', 'station_4', 'station_5','station_6','station_7','station_1_8']
    stations_available=list(df_instation.columns.values)[0:int((len(instation_working)/2))]
    average_list=list()
    for i in range(len(stations_available)):
        average_list.append([stations_available[i],sum_time_instation[i]/60,(end_global-sum_time_working[i])/60,sum_time_working[i]/sum_time_instation[i],sum_time_instation[i]/end_global])
        #print(end_global)
        #print(sum_time_instation[i])
        

        
    df = pd.DataFrame(average_list, columns = ['station name', 'sum time active (min)', 'sum time inactive (min)','percentage working active', 'percentage active'])



    import plotly.express as px
    fig = px.bar(df, x="station name", y="percentage active", orientation='v')
    fig.write_html(filename+'\\'+run+'_station_percentage_active.html')

    import plotly.express as px
    fig = px.bar(df, x="station name", y="percentage working active", orientation='v')
    fig.write_html(filename+'\\'+run+'_station_percentage_working_total.html')

    excel_book=pxl.load_workbook(filename+ '\\' +run+'_analysis_part2.xlsx')

    with pd.ExcelWriter(filename+ '\\' +run+'_analysis_part2.xlsx', engine='openpyxl') as writer:
        writer.book = excel_book
        writer.sheets = {
            worksheet.title: worksheet
            for worksheet in excel_book.worksheets
        }
        df.to_excel(writer, 'station summary', index=False)
        writer.save()


    print('DataFrames are written successfully to Excel Sheet.')


    

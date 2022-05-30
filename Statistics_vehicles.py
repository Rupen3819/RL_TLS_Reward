import os.path
import sys
import optparse
import random
import traci
import time
import numpy as np
from xml.dom import minidom
from statistics import mean
from utils import *
import pandas as pd

class Statistic_Vehicles():
    def __init__(self):
        self.current_stats=None
        self.current_stats_name=None

    def create_overview(self,path):
        stops=list()
        decelerations=list()
        waiting=list()
        time_on_site=list()
        CO2=list()
        df_excel = pd.read_excel(os.path.join(path,'test.xlsx'), sheet_name=None)
        for key in df_excel:
            stops.append(df_excel[key]['Number_of_stops'].mean())
            decelerations.append(df_excel[key]['Number_of_decelerations'].mean())
            time_on_site.append(df_excel[key]['Total_time_on_site'].mean())
            CO2.append(df_excel[key]['Total_CO2_emission'].mean())
            waiting.append(df_excel[key]['Accumulated_waiting_time'].mean())


        df_overview=pd.DataFrame(data={'Average_stops': [mean(stops)],'Average_decelerations':[mean(decelerations)], 'Average_time_on_site':[mean(time_on_site)], 'Average_CO2_emission':[mean(CO2)],
                                                 'Average_waiting_time':[mean(waiting)]})

        with pd.ExcelWriter(os.path.join(path, 'test.xlsx'), mode='a', engine="openpyxl", index=False) as writer:
            df_overview.to_excel(writer, sheet_name='Overview', index=False)

    def store_current_stats(self, path):
        for index, row in self.current_stats.iterrows():
            self.current_stats.loc[index,'Departure_of_site']=row['Driving_Sequence'][-1][0][0]
            self.current_stats.loc[index, 'Total_time_on_site'] = len(row['Driving_Sequence'])
        if os.path.exists(os.path.join(path,'test.xlsx')):
            with pd.ExcelWriter(os.path.join(path,'test.xlsx'), mode='a', engine="openpyxl", index=False) as writer:
                self.current_stats.to_excel(writer, sheet_name=str(self.current_stats_name), index=False)

        else:
            with pd.ExcelWriter(os.path.join(path,'test.xlsx'), engine="openpyxl") as writer:
                self.current_stats.to_excel(writer, sheet_name=str(self.current_stats_name), index=False)


    def create_stats(self, name):
        self.current_stats=pd.DataFrame(columns=['ID','Route_ID','Route','Arrival_at_site', 'Departure_of_site', 'Last_step_speed',
                                                 'Number_of_stops', 'Number_of_decelerations', 'Total_CO2_emission',
                                                 'Driving_Sequence', 'Total_time_on_site', 'Accumulated_waiting_time'])
        self.current_stats_name=name

    def add_stats(self):
        cars= traci.vehicle.getIDList()
        #print(self.current_stats)

        for car in cars:
            if car in set(self.current_stats['ID']):
                #print(f'last step speed {self.current_stats.loc[self.current_stats["ID"] == car]["Last_step_speed"].item()}')
                if self.current_stats.loc[self.current_stats['ID'] == car, 'Last_step_speed'].item() >= traci.vehicle.getSpeed(car):
                    self.current_stats.loc[self.current_stats['ID'] == car, 'Number_of_decelerations'] += 1

                if self.current_stats.loc[self.current_stats['ID'] == car, 'Last_step_speed'].item() >= 0.3 and traci.vehicle.getSpeed(car) <= 0.3:
                    #print('stops',self.current_stats.loc[self.current_stats['ID'] == car, 'Last_step_speed'].item(), traci.vehicle.getSpeed(car), car)
                    self.current_stats.loc[self.current_stats['ID'] == car, 'Number_of_stops'] += 1

                if self.current_stats.loc[self.current_stats['ID'] == car, 'Last_step_speed'].item() > traci.vehicle.getSpeed(car):
                    self.current_stats.loc[self.current_stats['ID'] == car, 'Number_of_decelerations'] += 1

                if traci.vehicle.getSpeed(car) <= 0:
                    self.current_stats.loc[self.current_stats['ID'] == car, 'Accumulated_waiting_time'] += 1

                self.current_stats.loc[self.current_stats['ID'] == car,'Total_CO2_emission'] += traci.vehicle.getCO2Emission(car)
                self.current_stats.loc[self.current_stats['ID'] == car, 'Driving_Sequence'].item().append(
                    [[traci.simulation.getTime()], [traci.vehicle.getPosition(car)],
                    [traci.vehicle.getNextTLS(car)]])

                self.current_stats.loc[self.current_stats['ID'] == car,'Last_step_speed'] = traci.vehicle.getSpeed(car)
                #print(self.current_stats.loc[self.current_stats['ID'] == car]['Driving_Sequence'])
                #print(self.current_stats.loc[self.current_stats['ID'] == car]['Driving_Sequence'][0])
                #print(self.current_stats.loc[self.current_stats['ID'] == car]['Driving_Sequence'][0][0])
                #self.current_stats.loc[self.current_stats['ID'] == car]['Driving_Sequence'][0][0].append(traci.simulation.getTime())
                #self.current_stats.loc[self.current_stats['ID'] == car]['Driving_Sequence'].append([[traci.simulation.getTime()], [traci.vehicle.getPosition(car)],
                 #                               [traci.vehicle.getNextTLS(car)]])


                #print(self.current_stats.loc[self.current_stats['ID'] == car]['Daparture_of_site'])

            else:
                #print(car)
                #print(self.current_stats.ID)
                new_car = {'ID': car,'Route_ID':traci.vehicle.getRouteID(car),'Route':traci.vehicle.getRoute(car), 'Arrival_at_site': traci.simulation.getTime(), 'Departure_of_site': 0,
                           'Last_step_speed': traci.vehicle.getSpeed(car), 'Number_of_stops': 0,
                           'Number_of_decelerations': 0, 'Total_CO2_emission': 0,
                           'Driving_Sequence': [[traci.simulation.getTime()], [traci.vehicle.getPosition(car)],
                                                [traci.vehicle.getNextTLS(car)]],
                           'Total_time_on_site': None, 'Accumulated_waiting_time': 0}
                self.current_stats = self.current_stats.append(new_car, ignore_index=True)
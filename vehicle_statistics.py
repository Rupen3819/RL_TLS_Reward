import os.path
from statistics import mean

import pandas as pd
import traci

STATS_FILE = 'test.xlsx'


def generate_summary(path):
    """Generate a summary sheet of the vehicle statistics from the existing worksheets"""
    stops = []
    decelerations = []
    waiting = []
    time_on_site = []
    emissions = []
    worksheets = pd.read_excel(os.path.join(path, STATS_FILE), sheet_name=None)

    for sheet_name in worksheets:
        sheet_df = worksheets[sheet_name]
        stops.append(sheet_df['Number_of_stops'].mean())
        decelerations.append(sheet_df['Number_of_decelerations'].mean())
        time_on_site.append(sheet_df['Total_time_on_site'].mean())
        emissions.append(sheet_df['Total_CO2_emission'].mean())
        waiting.append(sheet_df['Accumulated_waiting_time'].mean())

    df_overview = pd.DataFrame(data={
        'Average_stops': [mean(stops)],
        'Average_decelerations': [mean(decelerations)],
        'Average_time_on_site': [mean(time_on_site)],
        'Average_CO2_emission': [mean(emissions)],
        'Average_waiting_time': [mean(waiting)]
    })

    with pd.ExcelWriter(os.path.join(path, STATS_FILE), mode='a', engine="openpyxl", index=False) as writer:
        df_overview.to_excel(writer, sheet_name='Overview', index=False)


class VehicleStatistics:
    def __init__(self):
        self.stats = None
        self.stats_name = None

    def create(self, name):
        """Initialize/reset the current statistics, to begin collecting new statistics under a specific name"""
        self.stats_name = name
        self.stats = pd.DataFrame(columns=[
            'ID', 'Route_ID', 'Route', 'Arrival_at_site', 'Departure_of_site', 'Last_step_speed',
            'Number_of_stops', 'Number_of_decelerations', 'Total_CO2_emission',
            'Driving_Sequence', 'Total_time_on_site', 'Accumulated_waiting_time'
        ]).set_index('ID')

    def update(self):
        """Update the statistics for all vehicles, based on the current state of the simulation"""
        if self.stats is None:
            return

        car_ids = traci.vehicle.getIDList()
        time = traci.simulation.getTime()

        for car_id in car_ids:
            vehicle_speed = traci.vehicle.getSpeed(car_id)
            vehicle_position = traci.vehicle.getPosition(car_id)
            next_tls = traci.vehicle.getNextTLS(car_id)
            driving_sequence = [[time], [vehicle_position], [next_tls]]

            if car_id in self.stats.index:
                row = self.stats.loc[car_id]

                last_step_speed = row['Last_step_speed'].item()
                vehicle_emission = traci.vehicle.getCO2Emission(car_id)

                if last_step_speed > vehicle_speed:
                    row['Number_of_decelerations'] += 1

                if last_step_speed >= 0.3 and vehicle_speed <= 0.3:
                    row['Number_of_stops'] += 1

                if vehicle_speed <= 0:
                    row['Accumulated_waiting_time'] += 1

                row['Total_CO2_emission'] += vehicle_emission
                row['Driving_Sequence'].item().append(driving_sequence)
                row['Last_step_speed'] = vehicle_speed

            else:
                new_car = {
                    'ID': car_id,
                    'Route_ID': traci.vehicle.getRouteID(car_id),
                    'Route': traci.vehicle.getRoute(car_id),
                    'Arrival_at_site': traci.simulation.getTime(),
                    'Departure_of_site': 0,
                    'Last_step_speed': vehicle_speed,
                    'Number_of_stops': 0,
                    'Number_of_decelerations': 0,
                    'Total_CO2_emission': 0,
                    'Driving_Sequence': driving_sequence,
                    'Total_time_on_site': 0,
                    'Accumulated_waiting_time': 0
                }
                self.stats = pd.concat([self.stats, pd.DataFrame([new_car])], ignore_index=True)

    def save(self, path):
        """Save the current statistics as a new sheet in the spreadsheet"""
        for index, row in self.stats.iterrows():
            self.stats.loc[index, 'Departure_of_site'] = row['Driving_Sequence'][-1][0][0]
            self.stats.loc[index, 'Total_time_on_site'] = len(row['Driving_Sequence'])

        file_path = os.path.join(path, STATS_FILE)
        write_mode = 'a' if os.path.exists(file_path) else 'w'

        with pd.ExcelWriter(file_path, mode=write_mode, engine="openpyxl") as writer:
            self.stats.to_excel(writer, sheet_name=str(self.stats_name))

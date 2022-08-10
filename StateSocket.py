import io
import os

from lxml.etree import XMLParser
from concurrent.futures import ThreadPoolExecutor
import socket
import xml.etree.ElementTree as ET
import time

# Programm for collecting data from a Sumo simulation. The Sumo simulation writes the RawDump in each Simulation
# Step into an XML socket. The class provides a socket for this purpose.
# It is important that the startSocket() method is started before Sumo is started.
# Because Sumo must connect to the socket. The port for the connection can be retrieved after executing startSocket() v
# ia returnSocketNumber().
# Here is the start parameter for SUMO:
# --netstate Host(127.0.0.1):Port
#
# The socket based approach was chosen because it is very performance friendly. The simulation only has to write
# the raw dump as a whole into the socket. The interpretation and aggregation is done in a separate programme.
# This eliminates the need for time-consuming data access via traci

# The Programm reads the socket in each Simulation step, interprets the XML content and aggregates the data over
# the course of the simulation. Currently, the number of stops of each vehicle, the wait/hold time and the travel
# time are determined. All vehicles the whole simulation run are currently collected in a collection(vehicleMap)
# of vehicle objects of the class EvaluationVehicle. These objects contain the aggregated data related to the
# individual vehicle. The data is then evaluated in the evaluate() method at the end of the simulation run. Currently,
# the entire simulated network is evaluated. However, the classer can also be modified in such a way that the evaluation
# is edge/lane specific.
from xml import etree


class StateSocket():

    # Collection stores all Vehicles which were in the Simulation in the Simulation run.
    # Key= vehicleID Value= EvaluationVehicle object.
    vehicleMap = {}
    edgesMap = {}
    lanesMap = {}
    step = 0


    def __init__(self):
        self.Dir = os.path.dirname(os.path.abspath(__file__))


    #Method that evaluates the collected data and calculates a score, e.g. PI. This method executed at the end of the
    # simulation run. Currently, the average of stops, waiting time and travel
    # time is calculated for all vehicles in the vehicleMap.
    def evaluate(self):

        vHalte = 0;
        vWtime = 0;
        vResiezeit = 0;
        numbofVehicles = len(self.vehicleMap)
        numbOflanes = 0

        pi = 0

        for edgeID in self.edgesMap:
            vehicleMap = self.edgesMap[edgeID]
            if len(vehicleMap) > 0:
                vStarke = len(vehicleMap) / (self.step / 60 / 4)
                vHalteTemp = 0
                vWtimeTemp = 0
                vReisezeitTemp = 0
                if vStarke>0:
                    numbOflanes += 1

                for vehicleID in vehicleMap:
                    vehicle = vehicleMap[vehicleID]
                    vHalteTemp += vehicle.numberOfHalts
                    vWtimeTemp += vehicle.waitingSteps
                    vReisezeitTemp += vehicle.travelSteps

                avgHalte = vHalteTemp/len(vehicleMap)
                avgWtime = vWtimeTemp/len(vehicleMap)/4
                avgResiezeit = vReisezeitTemp/len(vehicleMap)/4

            pi += avgHalte * 60 * avgWtime * vStarke


            vHalte += avgHalte
            vWtime += avgWtime
            vResiezeit += avgResiezeit


        avgHalte = vHalte / numbOflanes
        # average waiting time in seconds /4 because Sumo executes 4 simulation steps per seconds
        avgWZeit =  vWtime / numbOflanes
        # average travel Time
        avgRZeit =  vResiezeit / numbOflanes

        return avgHalte, avgWZeit, pi

    # Starts the Socket Thred via a ThreadPoolExecutor
    def return_num_vehicles_lane(self):
        for edgeID in self.edgesMap:
            vehicleMap = self.edgesMap[edgeID]
            #print(vehicleMap)
            #print(len(self.edgesMap[edgeID].num_cars))
            #print(xxx)
            break
        pass

    def startSocket(self):
        print('started_socket')
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(self.socketThread)
        #self.socketThread()

    #Thread that starts the XML socket. The socket blocks, so it must be a separate thread.
    # Starts the port on localhost with a free port. The port number is stored in self.socketnumber
    # and can then be can then be retrieved via returnSocketNumber().

    def socketThread(self):
        HOST = "127.0.0.1" # Localhost Adress
        PORT = 0 # Port 0 pickes a Free Port

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, PORT))


        self.socketnumber = s.getsockname()[1]

        s.listen()
        conn, addr1 = s.accept()
        self.conn = conn


    # Retrieve the port number after startSocket() has been executed.
    def returnSocketNumber(self):
        return self.socketnumber

    def repairXMLString(self, xmlString):
        #print(xmlString)
        #xmlString = "</timestep>" + '\n' + xmlString
        xmlArr = xmlString.split('\n')
        numbOfTimestep = 0
        #xmlArr[0] = "</timestep>"
        for line in xmlArr:

            if line.strip() == "</timestep>":
                numbOfTimestep += 1
        if numbOfTimestep > 1:
            print("XML Fail Repairing XML String")
            writeToNewString = False
            newString = ""
            for line in xmlArr:
                if writeToNewString:
                    newString += line + '\n'
                if line.strip() == "</timestep>":
                    writeToNewString = True
            return newString

        return xmlString

                    #Method for reading the socket. List a data block of a defined size (conn.recv(16777216)) from the socket.
    # Parsed the XML, interprets the data and stores it in the vehicles of the vehicleMap
    def readSocket_old(self):
        xmlstring = self.conn.recv(16777216).decode("utf-8")
        #xmlstring = self.repairXMLString(xmlstring)
        print(xmlstring)
        if self.step > 0:
            try:
                parser = XMLParser(recover=True)
                root = ET.parse(io.StringIO(xmlstring), parser).getroot()

                #tree = ET.ElementTree(ET.fromstring(xmlstring))
                #root = tree.getroot()
                edgesI = root.iter('edge')
                for e in edgesI:
                    edgeID = e.attrib['id']
                    if edgeID in self.edgesMap:
                        edgeVehicleMap = self.edgesMap[edgeID]
                    else:
                        self.edgesMap[edgeID] = {}
                        edgeVehicleMap = self.edgesMap[edgeID]


                    laneI = e.iter('lane')
                    for l in laneI:
                        laneID = l.attrib['id']
                        #if not laneID in laneVehicleMap:
                        vehicleI = l.iter('vehicle')
                        for v in vehicleI:
                            vehicleID = v.attrib['id']
                            speed = float(v.attrib['speed'])
                            if not vehicleID in edgeVehicleMap:
                                edgeVehicleMap[str(vehicleID)] = EvaluationVehicle(str(vehicleID))
                            edgeVehicleMap[str(vehicleID)].travelSteps += 1
                            if speed < 0.1:
                                if edgeVehicleMap[str(vehicleID)].halting:
                                    edgeVehicleMap[str(vehicleID)].waitingSteps += 1
                                else:
                                    edgeVehicleMap[str(vehicleID)].halting = True
                                    edgeVehicleMap[str(vehicleID)].waitingSteps += 1
                                    edgeVehicleMap[str(vehicleID)].numberOfHalts += 1
                            else:
                                edgeVehicleMap[str(vehicleID)].halting = False
                    self.edgesMap[edgeID] = edgeVehicleMap
            except:
                print("XML Error")
                print(xmlstring)
                print("____________________________________________________________________________________________")

        self.step += 1

    def readSocket(self):
        xmlstring = self.conn.recv(16777216).decode("utf-8")
        #print(xmlstring)
        #print(xmlstring)
        #xmlstring = self.repairXMLString(xmlstring)
        if self.step > 0:
            #try:
            parser = XMLParser(recover=True)
            root = ET.parse(io.StringIO(xmlstring), parser).getroot()
            edgesI = root.iter('edge')
            for e in edgesI:
                edgeID = e.attrib['id']
                if not edgeID in self.edgesMap:
                    self.edgesMap[str(edgeID)] = EvaluationEdge(str(edgeID))
                num_cars_edge = 0
                laneI = e.iter('lane')
                for l in laneI:
                    num_cars_lane = 0
                    laneID = l.attrib['id']
                    if not laneID in self.lanesMap:
                        self.lanesMap[str(laneID)] = EvaluationLane(str(laneID))
                    vehicleI = l.iter('vehicle')
                    for v in vehicleI:
                        num_cars_edge += 1
                        num_cars_lane += 1
                        vehicleID = v.attrib['id']
                        if not vehicleID in self.vehicleMap:
                            self.vehicleMap[str(vehicleID)] = EvaluationVehicle(str(vehicleID))
                    self.edgesMap[str(edgeID)].num_cars.append(num_cars_lane)
                self.edgesMap[str(edgeID)].num_cars.append(num_cars_edge)

            #except:
                #pass
                #print("XML Error")
                #print(xmlstring)
                #print("____________________________________________________________________________________________")

        self.step += 1


    # Reset Method
    def reset(self):
        print('reset')
        self.vehicleMap = {}
        self.edgesMap = {}
        self.lanesMap = {}
        print(self.edgesMap)
        self.step = 0



# Data container class for the individual vehicles. Stores all relevant data for the Evlaution vehicle-related.
class EvaluationVehicle:
    id = ""
    halting = False
    numberOfHalts = 0
    waitingSteps = 0
    travelSteps = 0

    def __init__(self, id):
        self.id = id

class EvaluationLane:
    id = ""
    num_cars = []

    def __init__(self,id):
        self.id = id

class EvaluationEdge:
    id = ""
    num_cars = []

    def __init__(self, id):
        self.id = id

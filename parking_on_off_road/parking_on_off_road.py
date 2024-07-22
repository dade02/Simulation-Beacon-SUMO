#!/usr/bin/env python3


import os
import sys
import optparse

#dobbiamo importare alcuni moduli da /tools di sumo
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary # controllo per binary nelle variabili d' ambiente
import traci
import xml.etree.ElementTree as ET
import traci.constants as tc
import sumolib


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options

def is_vehicle_parked(vehicle_id):
    stop_state = traci.vehicle.getStopState(vehicle_id)
    return stop_state & tc.STOP_PARKING != 0

def is_near_parkage(vehicle_id,parkage_id,parking_to_edge):
    if parking_to_edge[parkage_id] == traci.vehicle.getRoadID(vehicle_id).split('_')[0]:  # se sono sulla stessa lane
        vehicle_position = traci.vehicle.getLanePosition(vehicle_id)
        park_start = traci.parkingarea.getStartPos(parkage_id)
        park_end = traci.parkingarea.getEndPos(parkage_id)

        if vehicle_position > park_start - 15 and vehicle_position < park_end - 15:
           return True

    return False

def park_vehicle(vehicle_id, parkage_id, parking_car_parked, parking_capacity, parked_vehicles):
    occupied_count = parking_car_parked[parkage_id]
    capacity = parking_capacity[parkage_id]
    print("-----------------------------")
    print(f"veicolo: {vehicle_id}")
    print(f"capacità parcheggio: {capacity}, numero posti occupati: {occupied_count}")
    print("-----------------------------")

    if occupied_count < capacity:
        # print("C' è spazio")
        traci.vehicle.setParkingAreaStop(vehicle_id, parkage_id, 40)  # 10 è la durata della sosta
        parking_car_parked[parkage_id] += 1
        parked_vehicles[vehicle_id] = parkage_id
        return True

    return False

def is_exit_Parkage(vehicle_id,parking_id,parking_to_edge):
    if parking_to_edge[parking_id] == traci.vehicle.getRoadID(vehicle_id).split('_')[0]:
        vehicle_position = traci.vehicle.getLanePosition(vehicle_id)
        park_end = traci.parkingarea.getEndPos(parking_id)

        if vehicle_position > park_end:
            return True
    return False


def get_route_distance(net, from_edge, to_edge):
    from_edge_obj = net.getEdge(from_edge)
    to_edge_obj = net.getEdge(to_edge)
    route = net.getShortestPath(from_edge_obj, to_edge_obj)
    if route:
        distance = sum(edge.getLength() for edge in route[0])
        return distance, route
    return float('inf'), None


#in futuro considereremo l' heat-map sfruttando Beacon
def find_empty_parkages(parking_capacity,parking_list):
    empty_parkages = []
    for p in parking_list:
        if len(traci.parkingarea.getVehicleIDs(p)) < parking_capacity[p]:
            empty_parkages.append(p)
    return empty_parkages



def find_nearest_parkage(parkage_id,parking_list, parking_to_edge,parking_capacity):
    nearestParkage = None
    nearestDistance = float('inf')
    nearestRoute = None
    current_edge = traci.parkingarea.getLaneID(parkage_id).split('_')[0]
    # Caricare la rete stradale utilizzando sumolib
    net = sumolib.net.readNet("parking_on_off_road.net.xml")

    #itera sui parcheggi liberi
    for parkage in find_empty_parkages(parking_capacity,parking_list):
        if parkage != parkage_id:
            parking_edge = traci.parkingarea.getLaneID(parkage).split('_')[0]
            # Calcolare la distanza lungo la rete stradale
            distance, route = get_route_distance(net, current_edge, parking_edge)

            if distance < nearestDistance:
                nearestDistance = distance
                nearestParkage = parkage
                nearestRoute = route

    print(f"Il parcheggio più vicino a {parkage_id} è {nearestParkage} con una distanza {nearestDistance}")
    return nearestRoute


#contiene il loop di controllo TraCI
def run():
    parking_list = traci.parkingarea.getIDList()  #lista parcheggi

    parked_vehicles = {}                          # veicoli parcheggiati co relativi parcheggi
    recent_parking_vehicles = {}                  # veicoli che hanno lasciato da poco il parcheggio e relativo tempo d' uscita
    parking_to_edge = {}                          #parcheggi e edge associati
    # Popola il dizionario
    for parking_id in parking_list:
        # Ottieni l'ID della corsia associata al parcheggio
        lane_id = traci.parkingarea.getLaneID(parking_id)
        # Estrai l'ID dell'edge dalla corsia
        edge_id = lane_id.split('_')[0]
        # Aggiungi al dizionario
        parking_to_edge[parking_id] = edge_id


    # Percorso al file di configurazione delle aree di parcheggio
    parking_file = "parking_on_off_road.add.xml"
    # Dizionario per mappare i parcheggi alle capacità
    parking_capacity = {}
    # Parse the XML file
    tree = ET.parse(parking_file)
    root = tree.getroot()
    # Trova tutte le aree di parcheggio e ottieni le loro capacità
    for parking_area in root.findall('parkingArea'):
        parking_id = parking_area.get('id')
        capacity = int(parking_area.get('roadsideCapacity'))
        parking_capacity[parking_id] = capacity
    # Stampa il dizionario


    #numero posti occupati per parcheggio, tenendo conto anche dei veicoli che stanno per parcheggiare
    parking_car_parked = {}
    for parking in parking_list:
        parking_car_parked[parking] = 0

    COOLDOWN_PERIOD = 10000 # differenza di tempo tra un parcheggio ed un altro (di fatto non riparcheggia più)
    exitLane = 'E1'






    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        current_time = traci.simulation.getTime()


        for vehicle_id in traci.vehicle.getIDList():
            if not is_vehicle_parked(vehicle_id):
                if vehicle_id in parked_vehicles:
                    if is_exit_Parkage(vehicle_id,parked_vehicles[vehicle_id],parking_to_edge):
                        print(f"veicolo {vehicle_id} uscito dal parcheggio {parked_vehicles[vehicle_id]}")
                        parking_car_parked[parked_vehicles[vehicle_id]] -= 1    #aggiorno il numero di veicoli parcheggiati
                        del parked_vehicles[vehicle_id]                         #rimuovo da veicoli parcheggiati
                        recent_parking_vehicles[vehicle_id] = current_time      #aggiungo a veicoli usciti da poco


                        #una volta che il veicolo è uscito dal parcheggio, lo indirizzo in una strada di uscita
                        net = sumolib.net.readNet("parking_on_off_road.net.xml")
                        from_edge_obj = net.getEdge(traci.vehicle.getLaneID(vehicle_id).split('_')[0])
                        to_edge_obj = net.getEdge(exitLane)
                        route = net.getShortestPath(from_edge_obj, to_edge_obj)
                        path = route[0]
                        edge_ids = [edge.getID() for edge in path]

                        traci.vehicle.setRoute(vehicle_id,edge_ids)
                        traci.vehicle.setColor(vehicle_id, (0, 255, 0, 255))


                if vehicle_id not in recent_parking_vehicles:   #se il veicolo non ha lasciato un parcheggio da poco
                    traci.vehicle.setColor(vehicle_id, (255, 0, 0, 255))
                    for parking_id in parking_list:
                        # se il veicolo è abbastanza vicino al parcheggio
                        if is_near_parkage(vehicle_id,parking_id,parking_to_edge) and vehicle_id not in parked_vehicles:
                            if park_vehicle(vehicle_id, parking_id, parking_car_parked, parking_capacity, parked_vehicles):
                                print("Veicolo parcheggiato")
                            else:
                                print("Veicolo non parcheggiato: non vi è più spazio! Cerco nuovo parcheggio")
                                nearestRoute = find_nearest_parkage(parking_id, parking_list,parking_to_edge,parking_capacity)
                                print(nearestRoute)
                                if nearestRoute != None:
                                    path = nearestRoute[0]
                                    edge_ids = [edge.getID() for edge in path]
                                    print(edge_ids)
                                    traci.vehicle.setRoute(vehicle_id, edge_ids)






                else:
                    if current_time - recent_parking_vehicles[vehicle_id] > COOLDOWN_PERIOD:
                        del recent_parking_vehicles[vehicle_id]



    traci.close()





#main entry point
if __name__ == '__main__':
    options = get_options()

    #controlla binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    #TraCI inizializza Sumo come sottoprocesso e dopo questo script si connette ed esegue
    traci.start([sumoBinary, "-c", "parking_on_off_road.sumocfg", "--tripinfo-output", "tripinfo-xml"])
    run()
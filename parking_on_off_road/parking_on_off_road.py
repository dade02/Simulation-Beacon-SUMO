#!/usr/bin/env python3


import optparse
import os
import sys

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
import numpy as np
from collections import defaultdict
import random
import csv

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
    """print("-----------------------------")
    print(f"veicolo: {vehicle_id}")
    print(f"capacità parcheggio: {capacity}, numero posti occupati: {occupied_count}")
    print("-----------------------------")"""

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



"""def find_nearest_parkage(parkage_id,parking_list, parking_to_edge,parking_capacity):
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
    return nearestRoute"""


def is_vehicle_near_junction(vehID, net, threshold_distance=20.0):
    try:
        # Ottieni la posizione corrente del veicolo
        vehicle_position = traci.vehicle.getPosition(vehID)
        #print(f"Vehicle {vehID} position: {vehicle_position}")

        # Ottieni l'ID dell'edge su cui si trova il veicolo
        current_edge = traci.vehicle.getRoadID(vehID)
        #print(f"Edge {current_edge}")

        # Verifica se l'edge corrente è una junction
        if current_edge.startswith(':'):
            #print(f"{current_edge} è una junction, non un edge.")
            return False, None

        # Ottieni l'ID della junction di destinazione dell'edge corrente
        next_junction_id = net.getEdge(current_edge).getToNode().getID()

        print(f"Next Junction ID: {next_junction_id}")

        if not next_junction_id:
            print(f"Nessuna junction trovata per l'edge {current_edge}")
            return False, None

        # Ottieni la posizione della prossima junction
        junction_position = traci.junction.getPosition(next_junction_id)
        #print(f"Junction {next_junction_id} position: {junction_position}")

    except Exception as e:
        print(f"Error: {e}")
        return False, None

        # Calcola la distanza tra il veicolo e la prossima junction
    distance = np.linalg.norm(np.array(vehicle_position) - np.array(junction_position))
    #print(f"Distance to junction {next_junction_id}: {distance}")

    # Verifica se il veicolo è entro la soglia di distanza dalla prossima junction
    if distance <= threshold_distance:
        print(f"Veicolo {vehID} vicino alla prossima junction {next_junction_id}")
        return True, next_junction_id

    return False, None




def get_reachable_edges_from_lane(laneID):
    reachable_edges = []

    # Ottieni la lista delle connessioni (successive lanes) dalla lane corrente
    connections = traci.lane.getLinks(laneID)

    for conn in connections:
        connected_laneID = conn[0]  # La lane connessa è il primo elemento della tupla
        connected_edgeID = traci.lane.getEdgeID(connected_laneID)

        if connected_edgeID not in reachable_edges:
            reachable_edges.append(connected_edgeID)

        # è l' edge di entrata
        if 'E2' in reachable_edges:
            reachable_edges.remove('E2')

    return reachable_edges

"""def update_car_history_edge(car_history_edge, edges, vehicle_id):
    for edge in edges:
        # Incrementa il conteggio dei passaggi sull'edge per il veicolo
        if edge not in car_history_edge[vehicle_id]:
            car_history_edge[vehicle_id][edge] = 0
        car_history_edge[vehicle_id][edge] += 1"""


def select_next_edge(vehicle_id, lane_history, possible_edges):
    """
    Seleziona la prossima lane per il veicolo, dando priorità alle lane con meno passaggi.
    """
    # Ottieni il numero di passaggi per ciascuna delle possibili lane
    pass_counts = [lane_history[vehicle_id].get(edge, 0) for edge in possible_edges]

    # Calcola le probabilità inversamente proporzionali al numero di passaggi
    max_pass_count = max(pass_counts) if pass_counts else 1
    weights = [max_pass_count - count + 1 for count in pass_counts]  # Più passaggi, meno peso

    # Seleziona una lane basata su queste probabilità
    selected_lane = random.choices(possible_edges, weights=weights, k=1)[0]
    return selected_lane, pass_counts


def set_vehicle_route(vehicle_id, edge_history, reachable_edges):
    current_time = traci.simulation.getTime()

    """
    Imposta il percorso del veicolo verso una delle possibili lane, dando priorità alle lane con meno passaggi.
    """
    selected_edge, pass_count = select_next_edge(vehicle_id, edge_history, reachable_edges)



    # Imposta il percorso del veicolo verso l'edge selezionato
    traci.vehicle.changeTarget(vehicle_id, selected_edge)

    # Aggiorna il conteggio dei passaggi sulla lane selezionata
    if selected_edge not in edge_history[vehicle_id]:
        edge_history[vehicle_id][selected_edge] = 0
    edge_history[vehicle_id][selected_edge] += 1

    return selected_edge,current_time,pass_count

def estrai_numero(veicolo):
    # Splitta la stringa sul punto e prendi la parte dopo il punto
    return int(veicolo.split('.')[1])


def random_point_on_map():
    # Ottenere i limiti della mappa
    edges = traci.edge.getIDList()
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    NET_FILE = "parking_on_off_road.net.xml"
    tree = ET.parse(NET_FILE)
    root = tree.getroot()
    ENTRY_EXIT_LIST = ["E1","-E1","E2","-E2"]
    try:
        # Iterare su tutti gli edge
        for edge in root.findall('edge'):
            # Iterare su tutte le lane all'interno dell'edge
            if edge.get("id") not in ENTRY_EXIT_LIST:
                for lane in edge.findall('lane'):
                    # Ottenere il valore dell'attributo shape
                    shape = lane.get('shape')
                    if shape:
                        # Analizzare i punti dell'attributo shape
                        points = shape.split()
                        for point in points:
                            x, y = map(float, point.split(','))
                            # Aggiornare i limiti
                            min_x = min(min_x, x)
                            min_y = min(min_y, y)
                            max_x = max(max_x, x)
                            max_y = max(max_y, y)



    except Exception as e:
        print(f"Errore durante l'ottenimento dei limiti della mappa dal file di rete: {e}")
        return None, None, None, None

    # Generare un punto casuale all'interno dei limiti della mappa
    random_x = random.uniform(min_x, max_x)
    random_y = random.uniform(min_y, max_y)


    # Generare randomicamente edge_id = None con una probabilità del 10%
    if random.random() < 0.1:  # Probabilità del 10%
        return (random_x, random_y), None


    # Ottenere l'edge più vicino al punto casuale
    edge_id = None
    while edge_id == None or edge_id in ENTRY_EXIT_LIST:
        print(f"edge id: {edge_id}")
        try:
            random_x = random.uniform(min_x, max_x)
            random_y = random.uniform(min_y, max_y)
            edge_id, _, _ = traci.simulation.convertRoad(random_x, random_y, isGeo=False)
        except traci.TraCIException:
            edge_id = None


    connected_edges = []
    # Verificare se l'edge_id è quello di una junction e trovare un edge connesso
    if edge_id and edge_id.startswith(':'):
        for edge in root.findall('edge'):
            print(f"to: {edge.get('to')}, current {(edge_id.split('_')[0])[1:]}")
            if edge.get("to") == (edge_id.split('_')[0])[1:]:
                connected_edges.append(edge.get("id"))

        for e in connected_edges:
            if not e.startswith(':') and e not in ENTRY_EXIT_LIST:
                edge_id = e
                break

    return (random_x, random_y), edge_id


def get_vehicle_ids_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    vehicle_ids = []
    for flow in root.findall('.//flow'):
        flow_id = flow.get('id')
        number = int(flow.get('number', 0))
        for i in range(number):
            vehicle_ids.append(f"{flow_id}.{i}")

    return vehicle_ids


def set_vehicle_destinations():
    # Nome del file CSV
    file_name = 'setDestination.csv'

    # Controlla se il file esiste già
    if os.path.exists(file_name):
        print(f"{file_name} esiste già. Non verrà sovrascritto.")
        return

    # Ottenere la lista dei veicoli dal file XML
    vehicle_ids = get_vehicle_ids_from_xml('parking_on_off_road.rou.xml')

    if not vehicle_ids:
        print("Nessun veicolo presente nel file XML.")
        return



    # Lista per salvare le destinazioni dei veicoli
    destinations = []

    for vehicle_id in vehicle_ids:
        # Ottenere un punto casuale sulla mappa
        (x, y), edge_id = random_point_on_map()
        if edge_id:
            destinations.append([vehicle_id, edge_id, x, y])
        else:
            destinations.append([vehicle_id,"None"])

    # Scrivere le destinazioni nel file CSV
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['VehicleID', 'EdgeID', 'X', 'Y'])
        for destination in destinations:
            writer.writerow(destination)


    print(f"{file_name} è stato creato con le destinazioni dei veicoli.")


def get_vehicle_destinations():
    # Nome del file CSV
    file_name = 'setDestination.csv'

    # Controlla se il file esiste
    if not os.path.exists(file_name):
        print(f"{file_name} non esiste. Assicurati che il file esista e riprova.")
        return

    # Legge le destinazioni dal file CSV
    destinations = {}
    with open(file_name, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            vehicle_id = row['VehicleID']
            edge_id = row['EdgeID']
            destinations[vehicle_id] = edge_id

    return destinations

#contiene il loop di controllo TraCI
def see_heatMap(vehicle_id):
    pass


def run():
    set_vehicle_destinations()
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

    #per ogni veicolo tengo traccia di quante volte ha percorso un edge
    car_history_edge = defaultdict(dict)

    recent_changed_route_cars = {}
    REROUTE_PERIOD = 10000

    exit_lane_list = []
    car_arrived_in_b = [] #lista di veicoli arrivati al loro punto B

    destinations = get_vehicle_destinations() # punti B per ciascun veicolo
    print(destinations)
    net = sumolib.net.readNet("parking_on_off_road.net.xml")


    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        current_time = traci.simulation.getTime()


        for vehicle_id in traci.vehicle.getIDList():
            #se il veicolo non è arrivato al punto b
            if vehicle_id not in car_arrived_in_b and destinations[vehicle_id] == "None":
                car_arrived_in_b.append(vehicle_id)
                see_heatMap(vehicle_id)

            if vehicle_id not in car_arrived_in_b:
                print(f"arrivo: {traci.vehicle.getRoute(vehicle_id)[-1]} newdest: {destinations[vehicle_id]}")
                #se c'è arrivato ora
                if traci.vehicle.getLaneID(vehicle_id).split('_')[0] == destinations[vehicle_id]:
                    car_arrived_in_b.append(vehicle_id)
                elif traci.vehicle.getRoute(vehicle_id)[-1] != destinations[vehicle_id]:
                    traci.vehicle.changeTarget(vehicle_id, destinations[vehicle_id])
                    for edge in traci.vehicle.getRoute(vehicle_id):
                        if edge not in car_history_edge[vehicle_id]:
                           car_history_edge[vehicle_id][edge] = 0
                        car_history_edge[vehicle_id][edge] += 1

                traci.vehicle.setColor(vehicle_id, (255, 0, 0, 255))
            if vehicle_id in car_arrived_in_b:
                if not is_vehicle_parked(vehicle_id):

                    #lista di veicoli che stanno uscendo
                    if traci.vehicle.getRoadID(vehicle_id) == exitLane and vehicle_id not in exit_lane_list:
                        exit_lane_list.append(vehicle_id)

                    if recent_changed_route_cars.get(vehicle_id) != None and current_time - recent_changed_route_cars[vehicle_id] > REROUTE_PERIOD:
                        del recent_changed_route_cars[vehicle_id]
                    # se il veicolo ha appena svoltato, non vi è più il controllo sul doppio risettaggio
                    if recent_changed_route_cars.get(vehicle_id) != None:
                        #print(f"edge veicolo: {vehicle_id} edge attuale: {traci.vehicle.getRoadID(vehicle_id)} meta: {traci.vehicle.getRoute(vehicle_id)[-1]}")
                        #se sono alla destinazione
                        if traci.vehicle.getRoadID(vehicle_id) == traci.vehicle.getRoute(vehicle_id)[-1]:
                            del recent_changed_route_cars[vehicle_id]
                            print(f"Il veicolo {vehicle_id} può di nuovo cambiare rotta")
                            print(f"occorrenze: { car_history_edge[vehicle_id]}")


                    if vehicle_id in parked_vehicles:
                        if is_exit_Parkage(vehicle_id,parked_vehicles[vehicle_id],parking_to_edge):
                            #print(f"veicolo {vehicle_id} uscito dal parcheggio {parked_vehicles[vehicle_id]}")
                            parking_car_parked[parked_vehicles[vehicle_id]] -= 1    #aggiorno il numero di veicoli parcheggiati
                            del parked_vehicles[vehicle_id]                         #rimuovo da veicoli parcheggiati
                            recent_parking_vehicles[vehicle_id] = current_time      #aggiungo a veicoli usciti da poco

                            # cancello ultimo reroute se esiste
                            # se la destinazione non è la lane dove vi è adesso il veicolo
                            print(
                                f"veicolo {vehicle_id} strada {traci.vehicle.getLaneID(vehicle_id).split('_')[0]} dest {traci.vehicle.getRoute(vehicle_id)[-1]}")

                            if traci.vehicle.getRoute(vehicle_id)[-1] != traci.vehicle.getLaneID(vehicle_id).split('_')[0]:
                                if traci.vehicle.getRoute(vehicle_id)[-1] != exitLane:
                                    print(
                                        f"la destinazione {traci.vehicle.getRoute(vehicle_id)[-1]} per il veicolo {vehicle_id} è da eliminare")
                                    car_history_edge[vehicle_id][traci.vehicle.getRoute(vehicle_id)[-1]] -= 1
                                    if car_history_edge[vehicle_id][traci.vehicle.getRoute(vehicle_id)[-1]] == 0:
                                        del car_history_edge[vehicle_id][traci.vehicle.getRoute(vehicle_id)[-1]]




                            #una volta che il veicolo è uscito dal parcheggio, lo indirizzo in una strada di uscita

                            """from_edge_obj = net.getEdge(traci.vehicle.getLaneID(vehicle_id).split('_')[0])
                            to_edge_obj = net.getEdge(exitLane)
                            route = net.getShortestPath(from_edge_obj, to_edge_obj)
                            path = route[0]
                            edge_ids = [edge.getID() for edge in path]
    
                            traci.vehicle.setRoute(vehicle_id,edge_ids)"""
                            traci.vehicle.changeTarget(vehicle_id,exitLane)
                            traci.vehicle.setColor(vehicle_id, (0, 255, 0, 255))



                    if vehicle_id not in recent_parking_vehicles:   #se il veicolo non ha lasciato un parcheggio da poco
                        #traci.vehicle.setColor(vehicle_id, (255, 0, 0, 255))
                        for parking_id in parking_list:
                            # se il veicolo è abbastanza vicino al parcheggio
                            if is_near_parkage(vehicle_id,parking_id,parking_to_edge) and vehicle_id not in parked_vehicles:
                                if park_vehicle(vehicle_id, parking_id, parking_car_parked, parking_capacity, parked_vehicles):
                                    print(f"Veicolo {vehicle_id} parcheggiato")
                                else:
                                    #print("Veicolo non parcheggiato: non vi è più spazio! Cerco nuovo parcheggio")

                                    #trova il parcheggio libero più vicino
                                    """nearestRoute = find_nearest_parkage(parking_id, parking_list,parking_to_edge,parking_capacity)
                                    print(nearestRoute)
                                    if nearestRoute != None:
                                        path = nearestRoute[0]
                                        edge_ids = [edge.getID() for edge in path]
                                        print(edge_ids)
                                        traci.vehicle.setRoute(vehicle_id, edge_ids)"""

                        #logica di rerouting casuale
                        destination_edge = traci.vehicle.getRoute(vehicle_id)[-1]
                        if destination_edge != exitLane:
                            # controllo se il veicolo è vicino ad una junction
                            near_junction, junctionID = is_vehicle_near_junction(vehicle_id,net, 25)
                            if near_junction:
                                all_edges = traci.edge.getIDList()
                                # Filtra per ottenere solo gli edge reali
                                real_edges = [edge for edge in all_edges if not edge.startswith(':')]
                                #print(f"{traci.vehicle.getLaneID(vehicle_id).split('_')[0]} {real_edges}")

                                # se il veicolo non è sulla junction
                                if traci.vehicle.getLaneID(vehicle_id).split('_')[0] in real_edges:

                                    if recent_changed_route_cars.get(vehicle_id) == None:
                                        #print(f"Vehicle {vehicle_id} is near junction {junctionID}")
                                        # calcolo gli edge raggiungibili
                                        laneID = traci.vehicle.getLaneID(vehicle_id)
                                        reachable_edges = get_reachable_edges_from_lane(laneID)
                                        #print(f"Vehicle {vehicle_id} on lane {laneID} can reach edges: {reachable_edges}")
                                        if reachable_edges:
                                            selected_lane, recent_changed_route_cars[
                                                vehicle_id], pass_count = set_vehicle_route(vehicle_id, car_history_edge,
                                                                                            reachable_edges)
                                            #print(f"numero di passaggi: {pass_count}")
                                            #print(f"Vehicle {vehicle_id} routed to lane {selected_lane}")




                    else:
                        if current_time - recent_parking_vehicles[vehicle_id] > COOLDOWN_PERIOD:
                            del recent_parking_vehicles[vehicle_id]




    traci.close()

    print(f"Numero veicoli usciti: {len(exit_lane_list)}")
    print(sorted(exit_lane_list, key=estrai_numero))
    print("Storico passaggi veicoli")
    for v in car_history_edge:
        print(f"veicolo {v} : edges {car_history_edge[v]}")


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
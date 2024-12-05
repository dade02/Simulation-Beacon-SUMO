#!/usr/bin/env python3
import math
import optparse
import os
import sys
from contextlib import nullcontext
from xxlimited_35 import error

# dobbiamo importare alcuni moduli da /tools di sumo
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # controllo per binary nelle variabili d' ambiente
import traci
import xml.etree.ElementTree as ET
import traci.constants as tc
import sumolib
import numpy as np
from collections import defaultdict
import random
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import copy
import imageio.v2 as imageio
import shutil
import concurrent.futures


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options


def get_parking_coordinates(parking_id, additional_file):
    """
    Ottiene le coordinate di un parcheggio.

    Parametri:
    - parking_id: L'ID del parcheggio.
    - additional_file: Il percorso del file aggiuntivo (additional.xml).

    Restituisce:
    - Una tupla con le coordinate del parcheggio (x, y) o None se non trovato.
    """
    import xml.etree.ElementTree as ET  # Importa il modulo per la gestione dell'XML

    try:
        # Analizza il file aggiuntivo per ottenere le informazioni sul parcheggio
        tree_additional = ET.parse(additional_file)
        root_additional = tree_additional.getroot()

        # Trova il parcheggio specificato
        parking_area = root_additional.find(f".//parkingArea[@id='{parking_id}']")
        if parking_area is None:
            # print(f"ParkingArea with ID {parking_id} not found in {additional_file}")
            return None

        lane_id = parking_area.get("lane")
        start_pos = float(parking_area.get("startPos"))
        end_pos = float(parking_area.get("endPos"))

        # Debugging
        # (f"Found parking area: Lane ID: {lane_id}, Start Position: {start_pos}, End Position: {end_pos}")

        # Ottieni le coordinate della corsia
        lane_coords = traci.lane.getShape(lane_id)
        if not lane_coords:
            # print(f"No coordinates found for lane {lane_id}")
            return None

        # print(f"Lane coordinates: {lane_coords}")

        # Calcola le coordinate del parcheggio
        parking_coords = []

        for i in range(len(lane_coords) - 1):
            x1, y1 = lane_coords[i]
            x2, y2 = lane_coords[i + 1]
            segment_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            # Verifica se il parcheggio è all'interno del segmento
            if start_pos < (i + 1) * segment_length and end_pos > i * segment_length:
                if segment_length > 0:
                    proportion_start = (start_pos - i * segment_length) / segment_length
                    proportion_end = (end_pos - i * segment_length) / segment_length

                    x_start = x1 + (x2 - x1) * proportion_start
                    y_start = y1 + (y2 - y1) * proportion_start
                    x_end = x1 + (x2 - x1) * proportion_end
                    y_end = y1 + (y2 - y1) * proportion_end

                    # Aggiungi solo il primo punto calcolato
                    # print(f"aggiunto {x_start}")
                    parking_coords.append((x_start, y_start))
                    break  # Esci dal ciclo dopo aver trovato la prima coordinata

        # Restituisci la prima coordinata trovata
        if parking_coords:
            # print("ritorna coordinate")
            return parking_coords[0]

    except Exception as e:
        print(f"An error occurred: {e}")

    # print(f"ritorna none per il parcheggio {parking_id}")
    return None  # Restituisci None se non trovi il parcheggio


class HeatMap:

    def __init__(self, xml_file, additional_file):
        """
        Inizializza una nuova heatmap leggendo la dimensione dell'area da un file XML
        e calcolando automaticamente i confini della mappa.

        Parametri:
        - xml_file: Il percorso al file XML da cui leggere 'dimensione_area'.
        """
        # Leggi la dimensione dell'area dal file XML
        self.dimensione_area = self._read_dimensione_area_from_xml(xml_file)

        # Ottieni i confini della rete automaticamente
        (self.minX, self.minY), (self.maxX, self.maxY) = traci.simulation.getNetBoundary()
        # print(f"Net Boundaries: minX={self.minX}, minY={self.minY}, maxX={self.maxX}, maxY={self.maxY}")

        # Regola i confini per includere tutti i parcheggi
        self._expand_boundaries_for_parking(additional_file)

        # Calcolo delle dimensioni della matrice
        self.cols = math.ceil((self.maxX - self.minX) / self.dimensione_area)
        self.rows = math.ceil((self.maxY - self.minY) / self.dimensione_area)

        # Inizializzazione della matrice della heatmap come matrice di liste
        self.heat_map = [[[] for _ in range(self.cols)] for _ in range(self.rows)]

    def _expand_boundaries_for_parking(self, additional_file):
        """
        Espande i confini della mappa per includere tutte le posizioni di parcheggio.

        Parametri:
        - additional_file: Il file XML contenente le informazioni sui parcheggi.
        """
        tree_additional = ET.parse(additional_file)
        root_additional = tree_additional.getroot()

        # Trova tutti gli elementi 'parkingArea'
        for parking_area in root_additional.findall(".//parkingArea"):
            parking_id = parking_area.get("id")
            posX, posY = get_parking_coordinates(parking_id, additional_file)

            if posX is not None and posY is not None:
                # Espansione dei confini se necessario
                if posX < self.minX:
                    self.minX = posX
                if posX > self.maxX:
                    self.maxX = posX
                if posY < self.minY:
                    self.minY = posY
                if posY > self.maxY:
                    self.maxY = posY

    def _read_dimensione_area_from_xml(self, xml_file):
        """
        Legge il valore di 'dimensione_area' da un file XML.

        Parametri:
        - xml_file: Il percorso al file XML.

        Restituisce:
        - dimensione_area: Il valore letto dal file XML.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Trova l'elemento 'dimensione_area' e restituisci il suo valore come float
        dimensione_area = float(root.find('dimensione_area').text)
        return dimensione_area

    def update(self, parkage, parked_id=None, real_parkages=False):
        # caso heatmap
        if not real_parkages:

            """
            Aggiorna la heatmap in base alla posizione del parcheggio e allo stato di parkage.
            """

            posX, posY = get_parking_coordinates(parked_id, 'parking_on_off_road.add.xml')

            # Calcola gli indici della matrice per la posizione del veicolo
            col_index = math.floor((posX - self.minX) / self.dimensione_area)
            row_index = math.floor((posY - self.minY) / self.dimensione_area)

            # Verifica che gli indici siano all'interno dei limiti della matrice
            if 0 <= col_index < self.cols and 0 <= row_index < self.rows:
                if not parkage:
                    self.heat_map[row_index][col_index].append(1)
                else:
                    self.heat_map[row_index][col_index].append(-1)
        # caso mappa parcheggi reali
        else:
            """
            Trovo tutti i parcheggi nella rete stradale 
            """
            tree_additional = ET.parse('parking_on_off_road.add.xml')
            root_additional = tree_additional.getroot()

            # Trova tutti gli elementi 'parkingArea'
            for parking_area in root_additional.findall(".//parkingArea"):
                parking_id = parking_area.get("id")
                roadside_capacity = int(parking_area.get("roadsideCapacity"))  # Ottieni la capacità del parcheggio
                posX, posY = get_parking_coordinates(parking_id, 'parking_on_off_road.add.xml')
                # Calcola gli indici della matrice per la posizione del veicolo
                col_index = math.floor((posX - self.minX) / self.dimensione_area)
                row_index = math.floor((posY - self.minY) / self.dimensione_area)

                # Verifica che gli indici siano all'interno dei limiti della matrice
                if 0 <= col_index < self.cols and 0 <= row_index < self.rows:
                    self.heat_map[row_index][col_index].append(roadside_capacity)

    def print_heatmap(self, title="Heatmap", real_parkage=False):
        """
        Stampa la heatmap utilizzando Matplotlib, colorando solo le celle con liste non vuote.

        Parametri:
        - title: Il titolo della heatmap (opzionale).
        """
        # Crea una matrice per la visualizzazione
        display_matrix = np.zeros((self.rows, self.cols))  # Usa 0 per celle vuote

        # Assegna il valore 1 alle celle con una lista non vuota
        for i in range(self.rows):
            for j in range(self.cols):
                if self.heat_map[i][j]:  # Controlla se la lista non è vuota
                    display_matrix[i, j] = 1

        # Definisci la colormap: un colore per celle con dati, un colore per celle senza dati
        cmap = mcolors.ListedColormap(['white', 'blue'])  # 'white' per celle senza dati, 'blue' per celle con dati

        # Definisci i limiti della colormap
        bounds = [-0.5, 0.5, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(10, 8))
        cax = plt.imshow(display_matrix, cmap=cmap, norm=norm, interpolation='nearest', origin='lower')

        # Aggiungi una barra di colore per la heatmap
        cbar = plt.colorbar(cax, ticks=[0, 1])

        if not real_parkage:
            cbar.set_label('Presenza di veicoli')
        else:
            cbar.set_label('Presenza di parcheggi')
        cbar.set_ticks([0, 1])

        if not real_parkage:
            cbar.set_ticklabels(['Nessun parcheggio', 'Parcheggio'])

        # Aggiungi un bordo ai confini della griglia
        for i in range(self.rows + 1):
            plt.axhline(i - 0.5, color='black', linewidth=1)
        for j in range(self.cols + 1):
            plt.axvline(j - 0.5, color='black', linewidth=1)

        plt.title(title)
        plt.xlabel('Colonna della griglia')
        plt.ylabel('Riga della griglia')
        plt.show()

    def get_heatmap(self):
        """
        Restituisce la matrice della heatmap.
        """
        return self.heat_map

    def print_heatmap_values(self):
        """
        Stampa i valori all'interno di ciascuna cella della heatmap.
        """
        print("Heatmap Values:")
        for i in range(self.rows - 1, -1, -1):  # Inizia dall'ultima riga e vai verso la prima
            row_values = []
            for j in range(self.cols):
                cell_value = sum(self.heat_map[i][j])
                row_values.append(f"{cell_value:4d}")
            print(" ".join(row_values))

        """for i in range(self.rows):
            for j in range(self.cols):
                print(f"elemento heatmap[{i}][{j}] : {self.heat_map[i][j]}")
        """

    def save_heatmap_to_image(self, file_path, title="Heatmap", real_parkage=False):
        """
        Salva la heatmap come un file immagine.

        Parametri:
        - file_path: Il percorso del file immagine in cui salvare la heatmap.
        - title: Il titolo della heatmap (opzionale).
        """
        # Crea una matrice per la visualizzazione
        display_matrix = np.zeros((self.rows, self.cols))  # Usa 0 per celle vuote

        # Assegna il valore 1 alle celle con una lista non vuota
        for i in range(self.rows):
            for j in range(self.cols):
                if self.heat_map[i][j]:  # Controlla se la lista non è vuota
                    display_matrix[i, j] = 1

        # Definisci la colormap: un colore per celle con dati, un colore per celle senza dati
        cmap = mcolors.ListedColormap(['white', 'blue'])  # 'white' per celle senza dati, 'blue' per celle con dati

        # Definisci i limiti della colormap
        bounds = [-0.5, 0.5, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(10, 8))
        cax = plt.imshow(display_matrix, cmap=cmap, norm=norm, interpolation='nearest', origin='lower')

        # Aggiungi una barra di colore per la heatmap
        if not real_parkage:
            cbar = plt.colorbar(cax, ticks=[0, 1])
            cbar.set_label('Presenza di veicoli')
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Nessun Dato', 'Dato'])
        else:
            cbar = plt.colorbar(cax, ticks=['No', 'Yes'])
            cbar.set_label('Presenza di parcheggi')
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Nessun Parcheggio', 'Parcheggio'])

        # Aggiungi un bordo ai confini della griglia
        for i in range(self.rows + 1):
            plt.axhline(i - 0.5, color='black', linewidth=1)
        for j in range(self.cols + 1):
            plt.axvline(j - 0.5, color='black', linewidth=1)

        plt.title(title)
        plt.xlabel('Colonna della griglia')
        plt.ylabel('Riga della griglia')

        # Salva il grafico come immagine
        plt.savefig(file_path, bbox_inches='tight')

        # Chiudi la figura per liberare la memoria
        plt.close()

        # print(f"Heatmap salvata come immagine in {file_path}")

    def get_coordinates_from_cell(self, row, col):
        """
        Calcola le coordinate (X, Y) centrali della cella data dalla matrice di calore.
        """
        x = self.minX + col * self.dimensione_area + self.dimensione_area / 2
        y = self.minY + row * self.dimensione_area + self.dimensione_area / 2
        return x, y

    def find_closest_lane(self, posX, posY):
        """
        Trova la corsia (lane) più vicina alle coordinate specificate utilizzando la distanza euclidea.
        """
        lanes = traci.lane.getIDList()
        closest_lane = None
        min_distance = float('inf')

        for lane_id in lanes:
            lane_shape = traci.lane.getShape(lane_id)
            if not lane_shape:
                continue

            for i in range(len(lane_shape) - 1):
                (x1, y1) = lane_shape[i]
                (x2, y2) = lane_shape[i + 1]
                distance = self.distance_point_to_segment(posX, posY, x1, y1, x2, y2)
                if distance < min_distance:
                    min_distance = distance
                    closest_lane = lane_id

        return closest_lane

    def distance_point_to_segment(self, px, py, x1, y1, x2, y2):
        """
        Calcola la distanza del punto (px, py) al segmento definito da (x1, y1) e (x2, y2).
        """
        segment_length_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if segment_length_squared == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / segment_length_squared))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)

        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    """,preference"""

    """,preference"""

    def direct_vehicle_to_best_parking(self, vehicle_id, destinations, parkage_map, net, alfa):
        """
        Parametri:
        - vehicle_id: L'ID del veicolo che deve essere indirizzato verso una zona di parcheggio.
        - destinations: Dizionario che associa vehicle_id a edge_id, che rappresenta la destinazione attuale del veicolo.
        """
        edge_id = destinations.get(vehicle_id)

        if edge_id is None:
            # print(f"Nessuna destinazione trovata per il veicolo con ID {vehicle_id}.")
            return

        """
        bisogna trovare un tradeoff tra distanza di un parcheggio e possibilità che sia effettivamente libero
        Per implementare ciò consideriamo la formula score = alfa * H(i) + (1-alfa) * D(i,d).
        alfa è il coefficiente moltiplicativo (0.5 per dare peso uguale),i è la cella i-esima nella raprresentazione
        della heatmap, H è la norma della possibilità di un parcheggio di essere libero mentre D è la norma della 
        distanza del parcheggio dal punto B, rappresentata da d
        """

        DIS_SUBMAP = 350  # per la norma D consideriamo una sottomappa di raggio 5 km

        max_score = -1000 #provare a settare a -2
        best_lane = None

        print(f'Possibili nuove destinazioni per {vehicle_id}')

        for i in range(self.rows):
            for j in range(self.cols):
                if self.heat_map[i][j]:  # se segnata sulla heatmap
                    num_true_parkage = sum(parkage_map.heat_map[i][j])  # N° parcheggi totali
                    occupied_parkage = sum(self.heat_map[i][j]) * (-1)  # N° parcheggi occupati
                    norm_parkage = 1.00 - (float(occupied_parkage) / num_true_parkage)  # norma H

                    posX, posY = self.get_coordinates_from_cell(i, j)  # coordinate centrali cella
                    nearest_lane = self.find_closest_lane(posX, posY)  # lane più vicina alle coordinate
                    distance_to_B, _ = get_route_distance(net, edge_id, nearest_lane)

                    norm_distance = 1.00 - (float(distance_to_B) / DIS_SUBMAP)  # norma D

                    score = alfa * norm_parkage + (1 - alfa) * norm_distance

                    if score > max_score:
                        max_score = score
                        best_lane = nearest_lane

                    print("--------------------------------------------------------------------------------")
                    print(f"alfa: {alfa}")
                    print(f"Lane {nearest_lane.split('_')[0]} score : {score}")
                    print(f"dettagli - norma H : {norm_parkage} norma D : {norm_distance}")
                    print(f"parcheggi totali: {num_true_parkage} parcheggi occupati: {occupied_parkage} ")
                    print(f"distanza da B: {distance_to_B} ")
                    print(f"{alfa} * {norm_parkage} + (1-{alfa}) * {norm_distance} = {score}")
                    print("--------------------------------------------------------------------------------")

        if best_lane:
            traci.vehicle.changeTarget(vehicle_id, best_lane.split('_')[0])
            destinations[vehicle_id] = best_lane.split('_')[0]
            print(
                f"Il veicolo {vehicle_id} è stato indirizzato verso la corsia : {best_lane.split('_')[0]} con lo score {max_score}")
        else:
            print(f"Nessuna corsia valida trovata per il veicolo {vehicle_id}.")


def is_vehicle_parked(vehicle_id):
    stop_state = traci.vehicle.getStopState(vehicle_id)
    return stop_state & tc.STOP_PARKING != 0


def is_near_parkage(vehicle_id, parkage_id, parking_to_edge):
    if parking_to_edge[parkage_id] == traci.vehicle.getRoadID(vehicle_id).split('_')[0]:  # se sono sulla stessa lane
        vehicle_position = traci.vehicle.getLanePosition(vehicle_id)
        park_start = traci.parkingarea.getStartPos(parkage_id)
        park_end = traci.parkingarea.getEndPos(parkage_id)

        if vehicle_position > park_start - 15 and vehicle_position < park_end - 15:
            # print(f"veicolo {vehicle_id} vicino al parcheggio {parkage_id}")
            return True

    return False


def park_vehicle(vehicle_id, parkage_id, parking_car_parked, parking_capacity, parked_vehicles):
    occupied_count = parking_car_parked[parkage_id]
    capacity = parking_capacity[parkage_id]

    if occupied_count < capacity:
        print("C' è spazio")

        # Frenata graduale prima di fermare il veicolo (new)
        target_speed = 2  # Imposta una velocità bassa prima del parcheggio, es. 2 m/s
        duration = 5  # Tempo per rallentare (in secondi)
        traci.vehicle.slowDown(vehicle_id, target_speed, duration)


        time_stop = random.randint(300, 400)
        traci.vehicle.setParkingAreaStop(vehicle_id, parkage_id, time_stop)  # time_stop è la durata della sosta #? (parametro)
        parking_car_parked[parkage_id] += 1
        parked_vehicles[vehicle_id] = parkage_id
        return True

    return False


def is_exit_Parkage(vehicle_id, parking_id, parking_to_edge):
    if parking_to_edge[parking_id] == traci.vehicle.getRoadID(vehicle_id).split('_')[0]:
        vehicle_position = traci.vehicle.getLanePosition(vehicle_id)
        park_end = traci.parkingarea.getEndPos(parking_id)

        if vehicle_position > park_end:
            return True
    return False


def get_route_distance(net, from_edge, to_edge):
    from_edge_obj = net.getEdge(from_edge.split('_')[0])
    to_edge_obj = net.getEdge(to_edge.split('_')[0])
    route = net.getShortestPath(from_edge_obj, to_edge_obj)
    """if route:
        distance = sum(edge.getLength() for edge in route[0])
        return distance, route"""

    mid_point_B = get_midpoint(net.getEdge(from_edge.split('_')[0]))
    mid_point_parkage = get_midpoint(net.getEdge(to_edge.split('_')[0]))
    pedon_distance = calculate_distance(mid_point_B,
                                        mid_point_parkage)  # calcolo la distanza parcheggio - punto B come

    return pedon_distance, None
    #return float('inf'), None


# in futuro considereremo l' heat-map sfruttando Beacon
def find_empty_parkages(parking_capacity, parking_list):
    empty_parkages = []
    for p in parking_list:
        if len(traci.parkingarea.getVehicleIDs(p)) < parking_capacity[p]:
            empty_parkages.append(p)
    return empty_parkages


def is_vehicle_near_junction(vehID, net, threshold_distance=20.0):
    try:
        # Ottieni la posizione corrente del veicolo
        vehicle_position = traci.vehicle.getPosition(vehID)
        # print(f"Vehicle {vehID} position: {vehicle_position}")

        # Ottieni l'ID dell'edge su cui si trova il veicolo
        current_edge = traci.vehicle.getRoadID(vehID)
        # print(f"Edge {current_edge}")

        # Verifica se l'edge corrente è una junction
        if current_edge.startswith(':'):
            # print(f"{current_edge} è una junction, non un edge.")
            return False, None

        # Ottieni l'ID della junction di destinazione dell'edge corrente
        next_junction_id = net.getEdge(current_edge).getToNode().getID()

        # print(f"veicolo  {vehID} Next Junction ID: {next_junction_id}")

        if not next_junction_id:
            # print(f"Nessuna junction trovata per l'edge {current_edge}")
            return False, None

        # Ottieni la posizione della prossima junction
        junction_position = traci.junction.getPosition(next_junction_id)
        # print(f"Junction {next_junction_id} position: {junction_position}")

    except Exception as e:
        print(f"Error: {e}")
        return False, None

        # Calcola la distanza tra il veicolo e la prossima junction
    distance = np.linalg.norm(np.array(vehicle_position) - np.array(junction_position))
    # print(f"Distance to junction {next_junction_id}: {distance}")

    # Verifica se il veicolo è entro la soglia di distanza dalla prossima junction
    if distance <= threshold_distance:
        # print(f"Veicolo {vehID} vicino alla prossima junction {next_junction_id}")
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

    return selected_edge, current_time, pass_count


def estrai_numero(veicolo):
    # Splitta la stringa sul punto e prendi la parte dopo il punto
    return int(veicolo.split('.')[1])


def random_point_on_map():
    # Ottenere i limiti della mappa

    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    NET_FILE = "parking_on_off_road.net.xml"
    tree = ET.parse(NET_FILE)
    root = tree.getroot()
    ENTRY_EXIT_LIST = ["E85", "-E85", "E86", "-E86"]

    # calcolo lane arrivo
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
    # if random.random() < 0.1:  # Probabilità del 10%
    #   return (random_x, random_y), None

    # Ottenere l'edge più vicino al punto casuale
    edge_id = None
    while edge_id == None or edge_id in ENTRY_EXIT_LIST:
        # print(f"edge id: {edge_id}")
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
            # print(f"to: {edge.get('to')}, current {(edge_id.split('_')[0])[1:]}")
            if edge.get("to") == (edge_id.split('_')[0])[1:]:
                connected_edges.append(edge.get("id"))

        for e in connected_edges:
            if not e.startswith(':') and e not in ENTRY_EXIT_LIST:
                edge_id = e
                break

    # calcolo lane partenza
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
    # if random.random() < 0.1:  # Probabilità del 10%
    #   return (random_x, random_y), None

    # Ottenere l'edge più vicino al punto casuale
    edge_id_p = None
    while edge_id_p == None or edge_id_p in ENTRY_EXIT_LIST:
        # print(f"edge id: {edge_id}")
        try:
            random_x = random.uniform(min_x, max_x)
            random_y = random.uniform(min_y, max_y)
            edge_id_p, _, _ = traci.simulation.convertRoad(random_x, random_y, isGeo=False)
        except traci.TraCIException:
            edge_id_p = None

    connected_edges = []
    # Verificare se l'edge_id è quello di una junction e trovare un edge connesso
    if edge_id_p and edge_id_p.startswith(':'):
        for edge in root.findall('edge'):
            # print(f"to: {edge.get('to')}, current {(edge_id.split('_')[0])[1:]}")
            if edge.get("to") == (edge_id_p.split('_')[0])[1:]:
                connected_edges.append(edge.get("id"))

        for e in connected_edges:
            if not e.startswith(':') and e not in ENTRY_EXIT_LIST:
                edge_id_p = e
                break

    """,preference"""
    return (random_x, random_y), edge_id_p, edge_id


def get_vehicle_number_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    vehicles_number = int(root.find('vehicles_number').text)
    return vehicles_number


def set_vehicle_point_A_B(vehicle_number):
    # Nome del file CSV
    file_name = f'setDestination_{vehicle_number}.csv'

    # Controlla se il file esiste già
    if os.path.exists(file_name):
        print(f"{file_name} esiste già. Non verrà sovrascritto.")
        return

    # Ottenere la lista dei veicoli dal file XML
    # vehicle_number = get_vehicle_number_from_xml('heat_map.xml') #da modificare

    vehicle_ids = []

    for i in range(vehicle_number):
        vehicle_ids.append(f"vehicle_{i}")

    # Lista per salvare le destinazioni dei veicoli
    destinations = []

    for vehicle_id in vehicle_ids:
        # Ottenere un punto casuale sulla mappa
        """, preference"""
        (x, y), edge_id_p, edge_id = random_point_on_map()
        if edge_id:
            """,preference"""
            destinations.append([vehicle_id, edge_id_p, edge_id, x, y])
        else:
            print(f"Errore creazione destinazione veicolo {vehicle_id}")
            destinations.append([vehicle_id, "None"])

    # Scrivere le destinazioni nel file CSV
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        """,'preference'"""
        writer.writerow(['VehicleID', 'EdgeIDp', 'EdgeID', 'X', 'Y'])
        for destination in destinations:
            writer.writerow(destination)

    print(f"{file_name} è stato creato con le destinazioni dei veicoli.")


def get_vehicle_point_A_B(probability_heatmap,vehicle_number):
    # Nome del file CSV
    file_name = f'setDestination_{vehicle_number}.csv'

    # Controlla se il file esiste
    if not os.path.exists(file_name):
        print(f"{file_name} non esiste. Assicurati che il file esista e riprova.")
        return

    # Legge le destinazioni dal file CSV
    starting_lanes = {}
    destinations = {}
    use_heatmap = {}
    # preference = {}
    with open(file_name, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            vehicle_id = row['VehicleID']
            edge_id_p = row['EdgeIDp']
            edge_id = row['EdgeID']
            starting_lanes[vehicle_id] = edge_id_p
            destinations[vehicle_id] = edge_id

            random_float = random.uniform(0, 1)
            if random_float < probability_heatmap:
                use_heatmap[vehicle_id] = 'True'
            else:
                use_heatmap[vehicle_id] = 'False'
            # preference[vehicle_id] = row['preference']

    return starting_lanes, destinations, use_heatmap  # ,preference


def calculate_distance(point1, point2):
    """
    Calcola la distanza euclidea tra due punti.

    Args:
        point1: Una tupla con le coordinate del primo punto (x1, y1).
        point2: Una tupla con le coordinate del secondo punto (x2, y2).

    Returns:
        La distanza euclidea tra i due punti.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_midpoint(edge):
    """
    Calcola il punto medio di un bordo (lane).

    Args:
        edge: Oggetto del bordo, che ha coordinate di inizio e fine.

    Returns:
        Una tupla contenente le coordinate (x, y) del punto medio.
    """
    start = edge.getFromNode().getCoord()  # Coordinate di inizio
    end = edge.getToNode().getCoord()  # Coordinate di fine

    midpoint_x = (start[0] + end[0]) / 2
    midpoint_y = (start[1] + end[1]) / 2

    return (midpoint_x, midpoint_y)


def calcola_analisi_dati(time_A_start, time_B_arrive, parked_in_B, vehicle_use_heatmap, time_parked, edge_parked,
                         origin_destinations, net):
    tot_time_vehicle_heatmap = 0
    tot_time_vehicle_not_heatmap = 0
    num_parcheggio_b_heatmap = 0
    num_parcheggio_b_not_heatmap = 0
    num_use_heatmap = 0
    num_not_use_heatmap = 0
    tot_time_B_to_parkage_heatmap = 0
    tot_time_B_to_parkage_not_heatmap = 0
    tot_distance_B_to_parkage_not_heatmap = 0
    tot_distance_B_to_parkage_heatmap = 0

    # for edge in net.getEdges():
    #   print(edge.getID())

    for v, in_B in parked_in_B.items():
        if vehicle_use_heatmap[v] == 'True':
            num_use_heatmap += 1
        else:
            num_not_use_heatmap += 1

        if in_B == True:
            if vehicle_use_heatmap[v] == 'True':
                tot_time_vehicle_heatmap += time_B_arrive[v] - time_A_start[v]
                num_parcheggio_b_heatmap += 1
            else:
                tot_time_vehicle_not_heatmap += time_B_arrive[v] - time_A_start[v]
                num_parcheggio_b_not_heatmap += 1

        mid_point_B = get_midpoint(net.getEdge(origin_destinations[v]))
        mid_point_parkage = get_midpoint(net.getEdge(edge_parked[v]))
        pedon_distance = calculate_distance(mid_point_B,
                                            mid_point_parkage)  # calcolo la distanza parcheggio - punto B come

        # la distanza euclidea dei punti medi delle due lane

        if vehicle_use_heatmap[v] == 'True':
            tot_time_B_to_parkage_heatmap += time_parked[v] - time_B_arrive[v]
            tot_distance_B_to_parkage_heatmap += pedon_distance

        else:
            tot_time_B_to_parkage_not_heatmap += time_parked[v] - time_B_arrive[v]
            tot_distance_B_to_parkage_not_heatmap += pedon_distance

        # print(f"veicolo {v}, distanza a piedi: {pedon_distance} heatmap?: {vehicle_use_heatmap[v]}, parkage in B?: {parked_in_B[v]}, time for parkage from B:{time_parked[v] - time_B_arrive[v]}")
        # print(f"punto medio B:{mid_point_B}, punto medio parcheggio:{mid_point_parkage},parcheggio: {edge_parked[v]}, destinations:{destinations[v]}, time arrive to B and parkage: {time_B_arrive[v] - time_A_start[v]}")

    t_a_to_b_pb_hm = None
    t_b_to_p_hm = None
    d_b_to_p_hm = None

    t_a_to_b_pb = None
    t_b_to_p = None
    d_b_to_p = None

    print("RISULTATI------------------------")
    if num_use_heatmap != 0:
        if num_parcheggio_b_heatmap != 0:
            t_a_to_b_pb_hm = round(float(tot_time_vehicle_heatmap) / float(num_parcheggio_b_heatmap), 2)
        t_b_to_p_hm = round(float(tot_time_B_to_parkage_heatmap) / float(num_use_heatmap), 2)
        d_b_to_p_hm = round(float(tot_distance_B_to_parkage_heatmap) / float(num_use_heatmap), 2)
        print(f"Tempo medio impiegato per parcheggiare in B per i veicoli che usano la heatmap: {t_a_to_b_pb_hm}s")
        print(f"Tempo medio ricerca posteggio veicoli che usano la heatmap: {t_b_to_p_hm}s")
        print(f"distanza media posteggio - punto B usando la heatmap: {d_b_to_p_hm}m")
    if num_not_use_heatmap != 0:
        if num_parcheggio_b_not_heatmap != 0:
            t_a_to_b_pb = round(float(tot_time_vehicle_not_heatmap) / float(num_parcheggio_b_not_heatmap), 2)
        t_b_to_p = round(float(tot_time_B_to_parkage_not_heatmap) / float(num_not_use_heatmap), 2)
        d_b_to_p = round(float(tot_distance_B_to_parkage_not_heatmap) / float(num_not_use_heatmap), 2)
        print(f"Tempo medio impiegato per parcheggiare in B per i veicoli che non usano la heatmap: {t_a_to_b_pb}s")
        print(f"Tempo medio ricerca posteggio veicoli che non usano la heatmap: {t_b_to_p}s")
        print(f"distanza media posteggio - punto B non usando la heatmap: {d_b_to_p}m")
    print("FINE RISULTATI---------------------")

    return t_a_to_b_pb_hm, t_b_to_p_hm, d_b_to_p_hm, t_a_to_b_pb, t_b_to_p, d_b_to_p


def generate_results(perc_hm, numVeicoli, granularità, alfa, t_A_Bpb_hm, t_B_p_hm, d_B_p_hm, t_A_Bpb, t_B_p, d_B_p):
    # Definizione delle intestazioni delle colonne
    headers = [
        'percentuale_uso_heatmap',
        'veicoli',
        'granularità',
        'alfa',
        'tempo_medio_parcheggio_B_heatmap',
        'tempo_medio_ricerca_posteggio_heatmap',
        'distanza_parcheggio_punto_B_heatmap',
        'tempo_medio_parcheggio_B',
        'tempo_medio_ricerca_posteggio',
        'distanza_parcheggio_punto_B'

    ]

    filename = 'results_data.csv'
    file_exists = os.path.isfile(filename)
    # Scrittura dei dati nel file CSV in modalità append ('a')
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Scrivi le intestazioni solo se il file non esiste
        if not file_exists:
            writer.writerow(headers)

        # Genera e scrivi la righe di dati
        writer.writerow([perc_hm, numVeicoli, granularità, alfa, t_A_Bpb_hm, t_B_p_hm, d_B_p_hm, t_A_Bpb, t_B_p, d_B_p])

    print(f"File '{filename}' creato con successo!")


def save_agent_csv(perc_use_heatmap, numero_test_percentuale, vehicle_id, A, B, p, heatmap, tempo_percorso,
                   tempo_ricerca_parcheggio, pedon_distance, alfa, vehicle_number):
    # Nome del file CSV
    file_name = 'data_agent.csv'

    # Intestazioni del CSV (puoi cambiare questi nomi secondo necessità)
    header = ['perc_use_heatmap', 'numero_test_percentuale', 'vehicle_id', 'A', 'B', 'p', 'heatmap',
              'tempo_percorso', 'tempo_ricerca_parcheggio', 'pedon_distance', 'alfa', 'vehicle_number']

    # Dati da salvare
    data = [perc_use_heatmap, numero_test_percentuale, vehicle_id, A, B, p, heatmap,
            tempo_percorso, tempo_ricerca_parcheggio, pedon_distance, alfa, vehicle_number]

    # Verifica se il file esiste già o meno
    try:
        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Se il file è vuoto, scrivi prima l'intestazione
            file.seek(0, 2)  # Vai alla fine del file
            if file.tell() == 0:  # Se il file è vuoto
                writer.writerow(header)

            # Scrivi i dati
            writer.writerow(data)

    except Exception as e:
        print(f"Errore durante la scrittura nel file CSV: {e}")


# Funzione che avvia una singola simulazione SUMO
"""def run_simulation(p, i):
    # Avvia SUMO con l'opzione --start per iniziare subito la simulazione
    traci.start(
        [sumoBinary, "-c", "parking_on_off_road.sumocfg", "--tripinfo-output", f"tripinfo_{p}_{i}.xml", "--start"])

    # Chiama la funzione run che gestisce la logica della simulazione
    run(p, i)

    # Chiudi la connessione con SUMO alla fine della simulazione
    traci.close()


# Funzione che avvia tutte le simulazioni in parallelo
def start_simulations_in_parallel(percentuali_heatmap, numero_test_percentuale):
    # Crea un ProcessPoolExecutor per gestire i processi paralleli
    with ProcessPoolExecutor() as executor:
        # Lista di futuri per gestire le esecuzioni parallele
        futures = []

        # Loop per avviare le simulazioni
        for p in percentuali_heatmap:
            for i in range(numero_test_percentuale):
                # Submit invia l'esecuzione della funzione run_simulation al pool di processi
                futures.append(executor.submit(run_simulation, p, i))

        # Attendi che tutte le simulazioni siano terminate
        for future in futures:
            future.result()  # Questo metodo blocca l'esecuzione finché il processo non è terminato

"""

def popola_heatmap_parcheggi_percentuale(file_xml, percentuale): #new

    # Carica e analizza il file XML
    tree = ET.parse(file_xml)
    root = tree.getroot()


    parcheggi_id = []

    # Scorre ogni elemento 'parkingArea' nel file XML
    for parking in root.findall('parkingArea'):
        id = parking.get('id')
        parcheggi_id.append(id)

    # Calcola il numero di id da selezionare in base alla percentuale
    num_id_selezionate = max(0, int((percentuale / 100) * len(parcheggi_id)))

    id_casuali = []

    if num_id_selezionate > 0:
        # Seleziona casualmente gli id in base al numero calcolato
        id_casuali = random.sample(parcheggi_id, num_id_selezionate)



    return id_casuali

def set_initial_vehicles_heatmap(vehicles,use_heatmap,id_parkages,heatmap):
    start_vehicles = []
    vehicles_in_parking = {}

    for v in vehicles:
        if use_heatmap[v] == 'True' and len(start_vehicles) < len(id_parkages):
            start_vehicles.append(v)

    for v,id_p in zip(start_vehicles,id_parkages):
        vehicles_in_parking[v] = id_p
        heatmap.update(True,id_p)
        heatmap.update(False, id_p)

    return vehicles_in_parking


def run(percentuali_heatmap, numero_test_percentuale, alfa, vehicle_number):
    parkage_map = HeatMap(xml_file='heat_map.xml', additional_file='parking_on_off_road.add.xml')
    parkage_map.update(True, real_parkages=True)
    # parkage_map.print_heatmap_values()
    # salva mappa parcheggi reali
    parkage_map.save_heatmap_to_image('real_parkages.jpg', 'parcheggi reali', True)

    # lista formata da tutte le transizioni intermedie della heatmap ( posso creare una GIF )
    storic_heatmap = []

    set_vehicle_point_A_B(vehicle_number)
    parking_list = traci.parkingarea.getIDList()  # lista parcheggi

    parked_vehicles = {}  # veicoli parcheggiati co relativi parcheggi
    recent_parking_vehicles = {}  # veicoli che hanno lasciato da poco il parcheggio e relativo tempo d' uscita
    parking_to_edge = {}  # parcheggi e edge associati

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

    # numero posti occupati per parcheggio, tenendo conto anche dei veicoli che stanno per parcheggiare
    parking_car_parked = {}
    for parking in parking_list:
        parking_car_parked[parking] = 0

    COOLDOWN_PERIOD = 10000  # differenza di tempo tra un parcheggio ed un altro (di fatto non riparcheggia più)
    exitLane = 'E86'

    # per ogni veicolo tengo traccia di quante volte ha percorso un edge
    car_history_edge = defaultdict(dict)

    recent_changed_route_cars = {}
    REROUTE_PERIOD = 10000

    # exit_lane_list = []
    car_arrived_in_b = []  # lista di veicoli arrivati al loro punto B

    """,preference """
    starting_lanes, destinations, use_heatmap = get_vehicle_point_A_B(
        percentuali_heatmap,vehicle_number)  # punti B per ciascun veicolo (parametro)
    vehicle_use_heatmap = copy.deepcopy(use_heatmap)  # usato per ricavare i dati delle analisi
    origin_destinations = copy.deepcopy(destinations)  # copia i punti B originali

    print("Veicoli che usano la heatmap: ", end='')
    cont_use_heatmap = 0
    for v, use in use_heatmap.items():
        if use == 'True':
            print(f"{v},", end='')
            cont_use_heatmap += 1

    perc_use_heatmap = round(float(cont_use_heatmap) / float(len(use_heatmap)), 2)  # percentuale effettiva

    # print(destinations)
    net = sumolib.net.readNet("parking_on_off_road.net.xml")

    print("INIZIO HEATMAP")

    heatmap = HeatMap(xml_file='heat_map.xml', additional_file='parking_on_off_road.add.xml')
    print("FINE HEATMAP")

    delay_start = 10  # delay di partenza di un veicolo dall' altro (parametro)
    current_delay_time = 0  # delay attuale
    time_A_start = {}  # momento della sim    ulazione in cui parte il veicolo ( dal punto A )

    time_B_arrive = {}  # momento in cui sono arrivato in B (solo casi in cui ho parcheggiato in B )
    parked_in_B = {}  # veicolo, booleano (True se il veicolo a parcheggiato nel punto B corrispondente )
    time_parked = {}  # momenti nei quali i veicoli si sono posteggiati ( veicolo, tempo)
    edge_parked = {}  # edge dove il veicolo si è posteggiato

    vehicle_index = 0  # indice del veicolo che sta per partire

    vehicle = list(starting_lanes.keys())[vehicle_index]
    st_lane = starting_lanes[vehicle]  # Ottieni la lane di partenza



    #new
    vehicles_in_parking = set_initial_vehicles_heatmap(list(starting_lanes.keys()),use_heatmap,popola_heatmap_parcheggi_percentuale('parking_on_off_road.add.xml',20),heatmap)
    print(vehicles_in_parking)

    STOP_PARKAGE_INIT = 20

    if len(vehicles_in_parking) == 0:
        print(f"Faccio partire il veicolo: {vehicle}")
        # Crea una route per il veicolo basata sull'edge (st_lane)
        traci.route.add(routeID=f"route_{vehicle}", edges=[st_lane])
        traci.vehicle.add(
            vehID=vehicle,
            routeID=f"route_{vehicle}",  # Route creata dinamicamente per ogni veicolo
            departPos="0",  # Posizione iniziale sulla corsia
            departSpeed="0.1"  # Velocità massima alla partenza
        )
        traci.vehicle.setMaxSpeed(vehicle, 5.0)
        vehicle_index += 1
        time_A_start[vehicle] = traci.simulation.getTime()

    else:
        for vehicle in vehicles_in_parking:
            parking_id = vehicles_in_parking[vehicle]
            start_lane = traci.parkingarea.getLaneID(parking_id).split('_')[0]
            start_position = traci.parkingarea.getStartPos(parking_id)


            traci.route.add(routeID=f"route_{vehicle}",edges = [start_lane])
            # Aggiungiamo il veicolo alla simulazione e lo parcheggiamo
            traci.vehicle.add(
                vehID=vehicle,
                routeID=f"route_{vehicle}",
                depart="now",
                departPos=start_position,  # Posizione iniziale sulla corsia
                departSpeed="0.1"
            )
            traci.vehicle.setMaxSpeed(vehicle, 5.0)
            # Parcheggiamo il veicolo nel parcheggio specificato
            traci.vehicle.setParkingAreaStop(vehicle, parking_id,STOP_PARKAGE_INIT)
            time_A_start[vehicle] = traci.simulation.getTime() + STOP_PARKAGE_INIT
            #starting_lanes.pop(vehicle, None)

        """else:
            # Crea una route per il veicolo basata sull'edge (st_lane)
            traci.route.add(routeID=f"route_{vehicle}", edges=[st_lane])
            traci.vehicle.add(
                vehID=vehicle,
                routeID=f"route_{vehicle}",  # Route creata dinamicamente per ogni veicolo
                departPos="0",  # Posizione iniziale sulla corsia
                departSpeed="0.1"  # Velocità massima alla partenza
            )
        vehicle_index += 1"""

    # strutture dati per ricavare i dati delle nostre analisi


    points_A = []  # punti A dei veicoli
    points_parcheggio = []  # punti parcheggi dei veicoli
    # vehicle_use_heatmap per verificare l' uso della heatmap

    while traci.simulation.getMinExpectedNumber() > 0:
        # print(f"numero veicoli attuale: {traci.vehicle.getIDList()}")
        # if len(traci.vehicle.getIDList()) == 0:
        # break

        # print(f"veicoli aspettati:{ traci.simulation.getMinExpectedNumber()}")

        current_time = traci.simulation.getTime()
        # print(f"Tempo attuale: {current_time} secondi")

        #da aggiustare
        #print(f"vehicle_index: {vehicle_index}")
        if current_delay_time % delay_start == 0 and vehicle_index < len(starting_lanes):
            vehicle = list(starting_lanes.keys())[vehicle_index]  # Prendi il veicolo
            #print(list(starting_lanes.keys()))
            #print(vehicles_in_parking)
            print(f"considero il veicolo {vehicle}")
            while vehicle in vehicles_in_parking:
                vehicle_index += 1
                vehicle = list(starting_lanes.keys())[vehicle_index]  # Prendi il veicolo successivo

            #print(f"Faccio partire il veicolo: {vehicle}")

            # new

            #parking_id = vehicles_in_parking[vehicle]
            start_lane = starting_lanes[vehicle]
            #start_position = traci.parkingarea.getStartPos(parking_id)

            traci.route.add(routeID=f"route_{vehicle}", edges=[start_lane])
            
            traci.vehicle.add(
                vehID=vehicle,
                routeID=f"route_{vehicle}",
                depart="now",
                departPos=0,  # Posizione iniziale sulla corsia
                departSpeed="0.1"
            )
            time_A_start[vehicle] = current_time



            traci.vehicle.setMaxSpeed(vehicle, 5.0)

            vehicle_index += 1  # Incrementa solo dopo aver aggiunto il veicolo



        current_delay_time += 1

        traci.simulationStep()

        if len(traci.vehicle.getIDList()) == 0:
            print("Tutti i veicoli sono usciti: fine simulazione!")
            break

        for vehicle_id in traci.vehicle.getIDList():
            if use_heatmap[vehicle_id] == 'True':
                print(f"veicolo {vehicle_id} usa la heatmap")
                """,preference"""
                heatmap.direct_vehicle_to_best_parking(vehicle_id, destinations, parkage_map, net, alfa)
                use_heatmap[vehicle_id] = None

            if vehicle_id not in car_arrived_in_b:
                # print(f"arrivo: {traci.vehicle.getRoute(vehicle_id)[-1]} newdest: {destinations[vehicle_id]}")
                # se c'è arrivato ora
                if traci.vehicle.getLaneID(vehicle_id).split('_')[0] == destinations[vehicle_id]:
                    car_arrived_in_b.append(vehicle_id)
                    print(f"veicolo {vehicle_id} arrivato in B: {destinations[vehicle_id]}")
                    time_B_arrive[vehicle_id] = current_time

                    # print(f"Veicolo {vehicle_id} arrivato a destinazione B" )
                elif traci.vehicle.getRoute(vehicle_id)[-1] != destinations[vehicle_id]:
                    traci.vehicle.changeTarget(vehicle_id, destinations[vehicle_id])
                    for edge in traci.vehicle.getRoute(vehicle_id):
                        if edge not in car_history_edge[vehicle_id]:
                            car_history_edge[vehicle_id][edge] = 0
                        car_history_edge[vehicle_id][edge] += 1

                if not use_heatmap[vehicle_id] == None:
                    traci.vehicle.setColor(vehicle_id, (255, 0, 0, 255))  # non usano heatmap
                else:
                    traci.vehicle.setColor(vehicle_id, (255, 0, 255, 255))  # usano heatmap

            if vehicle_id in car_arrived_in_b:
                if not is_vehicle_parked(vehicle_id):

                    # lista di veicoli che stanno uscendo
                    # if traci.vehicle.getRoadID(vehicle_id) == exitLane and vehicle_id not in exit_lane_list:
                    #    exit_lane_list.append(vehicle_id)

                    if recent_changed_route_cars.get(vehicle_id) != None and current_time - recent_changed_route_cars[
                        vehicle_id] > REROUTE_PERIOD:
                        del recent_changed_route_cars[vehicle_id]
                    # se il veicolo ha appena svoltato, non vi è più il controllo sul doppio risettaggio
                    if recent_changed_route_cars.get(vehicle_id) != None:
                        # print(f"edge veicolo: {vehicle_id} edge attuale: {traci.vehicle.getRoadID(vehicle_id)} meta: {traci.vehicle.getRoute(vehicle_id)[-1]}")
                        # se sono alla destinazione
                        if traci.vehicle.getRoadID(vehicle_id) == traci.vehicle.getRoute(vehicle_id)[-1]:
                            del recent_changed_route_cars[vehicle_id]
                        # print(f"Il veicolo {vehicle_id} può di nuovo cambiare rotta")
                        # print(f"occorrenze: { car_history_edge[vehicle_id]}")

                    if vehicle_id in parked_vehicles:
                        if is_exit_Parkage(vehicle_id, parked_vehicles[vehicle_id], parking_to_edge):
                            # print(f"veicolo {vehicle_id} uscito dal parcheggio {parked_vehicles[vehicle_id]}")
                            parking_car_parked[
                                parked_vehicles[vehicle_id]] -= 1  # aggiorno il numero di veicoli parcheggiati

                            if use_heatmap[
                                vehicle_id] == None:  # se la usa (ho settato a None quelle che lo hanno usato)
                                heatmap.update(False, parked_vehicles[vehicle_id])

                            del parked_vehicles[vehicle_id]  # rimuovo da veicoli parcheggiati
                            recent_parking_vehicles[vehicle_id] = current_time  # aggiungo a veicoli usciti da poco

                            mid_point_B = get_midpoint(net.getEdge(origin_destinations[vehicle_id]))
                            mid_point_parkage = get_midpoint(net.getEdge(edge_parked[vehicle_id]))
                            pedon_distance = calculate_distance(mid_point_B,
                                                                mid_point_parkage)  # calcolo la distanza parcheggio - punto B come

                            save_agent_csv(perc_use_heatmap, numero_test_percentuale, vehicle_id,
                                           starting_lanes[vehicle_id]
                                           , destinations[vehicle_id], edge_parked[vehicle_id],
                                           vehicle_use_heatmap[vehicle_id],
                                           time_B_arrive[vehicle_id] - time_A_start[vehicle_id],
                                           time_parked[vehicle_id] -
                                           time_B_arrive[vehicle_id], round(pedon_distance, 2), alfa, vehicle_number)

                            # cancello ultimo reroute se esiste
                            # se la destinazione non è la lane dove vi è adesso il veicolo
                            # print(
                            # f"veicolo {vehicle_id} strada {traci.vehicle.getLaneID(vehicle_id).split('_')[0]} dest {traci.vehicle.getRoute(vehicle_id)[-1]}")

                            if traci.vehicle.getRoute(vehicle_id)[-1] != traci.vehicle.getLaneID(vehicle_id).split('_')[
                                0]:
                                # if traci.vehicle.getRoute(vehicle_id)[-1] != exitLane:
                                # print(
                                #   f"la destinazione {traci.vehicle.getRoute(vehicle_id)[-1]} per il veicolo {vehicle_id} è da eliminare")
                                car_history_edge[vehicle_id][traci.vehicle.getRoute(vehicle_id)[-1]] -= 1
                                if car_history_edge[vehicle_id][traci.vehicle.getRoute(vehicle_id)[-1]] == 0:
                                    del car_history_edge[vehicle_id][traci.vehicle.getRoute(vehicle_id)[-1]]

                            # una volta che il veicolo è uscito dal parcheggio, lo indirizzo in una strada di uscita

                            """from_edge_obj = net.getEdge(traci.vehicle.getLaneID(vehicle_id).split('_')[0])
                            to_edge_obj = net.getEdge(exitLane)
                            route = net.getShortestPath(from_edge_obj, to_edge_obj)
                            path = route[0]
                            edge_ids = [edge.getID() for edge in path]

                            traci.vehicle.setRoute(vehicle_id,edge_ids)"""
                            traci.vehicle.changeTarget(vehicle_id, exitLane)
                            traci.vehicle.setColor(vehicle_id, (0, 255, 0, 255))

                    if vehicle_id not in recent_parking_vehicles:  # se il veicolo non ha lasciato un parcheggio da poco
                        # traci.vehicle.setColor(vehicle_id, (255, 0, 0, 255))
                        for parking_id in parking_list:
                            # se il veicolo è abbastanza vicino al parcheggio
                            if is_near_parkage(vehicle_id, parking_id,
                                               parking_to_edge) and vehicle_id not in parked_vehicles:
                                if park_vehicle(vehicle_id, parking_id, parking_car_parked, parking_capacity,
                                                parked_vehicles):
                                    print(f"Veicolo {vehicle_id} parcheggiato")
                                    time_parked[vehicle_id] = current_time
                                    edge_parked[vehicle_id] = traci.parkingarea.getLaneID(parking_id).split('_')[0]
                                    if parking_to_edge[parking_id] == destinations[vehicle_id]:
                                        print(f"Veicolo {vehicle_id} parcheggiato nel punto B")
                                        parked_in_B[vehicle_id] = True
                                    else:
                                        print(
                                            f"Veicolo {vehicle_id} parcheggiato in un punto diverso da B (origine: {destinations[vehicle_id]}), parcheggio {parking_to_edge[parking_id]}")
                                        parked_in_B[vehicle_id] = False

                                    if use_heatmap[vehicle_id] == None:
                                        heatmap.update(True, parked_vehicles[vehicle_id])
                                else:
                                    # print("Veicolo non parcheggiato: non vi è più spazio! Cerco nuovo parcheggio")

                                    # trova il parcheggio libero più vicino
                                    """nearestRoute = find_nearest_parkage(parking_id, parking_list,parking_to_edge,parking_capacity)
                                    print(nearestRoute)
                                    if nearestRoute != None:
                                        path = nearestRoute[0]
                                        edge_ids = [edge.getID() for edge in path]
                                        print(edge_ids)
                                        traci.vehicle.setRoute(vehicle_id, edge_ids)"""

                        # logica di rerouting casuale
                        destination_edge = traci.vehicle.getRoute(vehicle_id)[-1]
                        if destination_edge != exitLane:
                            # controllo se il veicolo è vicino ad una junction
                            near_junction, junctionID = is_vehicle_near_junction(vehicle_id, net, 25)
                            if near_junction:
                                all_edges = traci.edge.getIDList()
                                # Filtra per ottenere solo gli edge reali
                                real_edges = [edge for edge in all_edges if not edge.startswith(':')]
                                # print(f"{traci.vehicle.getLaneID(vehicle_id).split('_')[0]} {real_edges}")

                                # se il veicolo non è sulla junction
                                if traci.vehicle.getLaneID(vehicle_id).split('_')[0] in real_edges:

                                    if recent_changed_route_cars.get(vehicle_id) == None:
                                        # print(f"Vehicle {vehicle_id} is near junction {junctionID}")
                                        # calcolo gli edge raggiungibili
                                        laneID = traci.vehicle.getLaneID(vehicle_id)
                                        reachable_edges = get_reachable_edges_from_lane(laneID)
                                        # print(f"Vehicle {vehicle_id} on lane {laneID} can reach edges: {reachable_edges}")
                                        if reachable_edges:
                                            selected_lane, recent_changed_route_cars[
                                                vehicle_id], pass_count = set_vehicle_route(vehicle_id,
                                                                                            car_history_edge,
                                                                                            reachable_edges)
                                            # print(f"numero di passaggi: {pass_count}")
                                            # print(f"Vehicle {vehicle_id} routed to lane {selected_lane}")




                    else:
                        if current_time - recent_parking_vehicles[vehicle_id] > COOLDOWN_PERIOD:
                            del recent_parking_vehicles[vehicle_id]

        deep_copy_heatmap = copy.deepcopy(heatmap)
        storic_heatmap.append(deep_copy_heatmap)

    # calcolo i vari dati
    print("Inizio raccolta analisi dati...")
    print(f"Veicoli totali: {len(use_heatmap)}, di cui {cont_use_heatmap} usano la heatmap")
    print(f"Percentuale {float(cont_use_heatmap) / float(len(use_heatmap)) * 100:.2f}%")

    # print(parked_in_B)
    tempo_A_B_pB_hm, tempo_B_p_hm, distanza_B_p_hm, tempo_A_B_pB, tempo_B_p, distanza_B_p = calcola_analisi_dati(
        time_A_start, time_B_arrive, parked_in_B, vehicle_use_heatmap, time_parked, edge_parked, origin_destinations,
        net)

    generate_results(f"{float(cont_use_heatmap) / float(len(use_heatmap)) * 100:.2f}%", len(use_heatmap),
                     heatmap._read_dimensione_area_from_xml(xml_file='heat_map.xml'), alfa,
                     tempo_A_B_pB_hm,
                     tempo_B_p_hm, distanza_B_p_hm, tempo_A_B_pB, tempo_B_p, distanza_B_p)

    print("Fine raccolta analisi dati")

    # Creo GIF della heatmap-----------------

    # Cartella temporanea per le immagini
    temp_dir = "images_GIF"
    os.makedirs(temp_dir, exist_ok=True)
    image_filenames = []
    N = 70  # Salva un'immagine ogni 10 step

    """print(f"Numero frame heatmap {len(storic_heatmap)}")

    # Salva ogni heatmap come immagine
    for i, heatmap in enumerate(storic_heatmap):
        if i % N == 0:  # Solo ogni N step
            filename = f"{temp_dir}/heatmap_{i}.png"
            storic_heatmap[i].save_heatmap_to_image(filename)
            image_filenames.append(filename)
            print(f"salvata immagine {i}")

    filename = f"{temp_dir}/heatmap_{(len(storic_heatmap)/N)*N+N}.png"
    storic_heatmap[len(storic_heatmap)-1].save_heatmap_to_image(filename)
    image_filenames.append(filename)
    print(f"salvata immagine {(len(storic_heatmap)/N)*N+N}")

    print("fine salva immagini GIF")
    # Crea una GIF a partire dalle immagini
    with imageio.get_writer('heatmap_animation.gif', mode='I', duration=0.5) as writer:
        for filename in image_filenames:    
            image = imageio.imread(filename)
            writer.append_data(image)

    # Rimuovi le immagini temporanee
    for filename in image_filenames:
        os.remove(filename)

    # Elimina la cartella temporanea
    shutil.rmtree(temp_dir)
    print("GIF creata con successo!")

    #fine creazione GIF heatmap-------------------------"""

    traci.close()

    # salva heatmap
    heatmap.save_heatmap_to_image('heatmap.jpg')

    print(f"Destinazioni originali:{origin_destinations}")
    print(f"nuove Destinazioni originali:{destinations}")

    # print(f"Numero veicoli usciti: {len(exit_lane_list)}")
    # print(sorted(exit_lane_list, key=estrai_numero))
    # print("Storico passaggi veicoli")
    # for v in car_history_edge:
    #    print(f"veicolo {v} : edges {car_history_edge[v]}")


def simula_percentuale(p, i):
    # Avvia SUMO con l'opzione --start per iniziare subito la simulazione
    traci.start(
        [sumoBinary, "-c", "parking_on_off_road.sumocfg", "--tripinfo-output", f"tripinfo_{p}_{i}.xml",
         "--start", "--quit-on-end"])

    # Chiama la funzione run che gestisce la logica della simulazione
    run(p, i)


# main entry point
if __name__ == '__main__':
    options = get_options()

    # controlla binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # 0,0.25,0.5,0.75,1
    percentuali_heatmap = [1] #poi mettere anche 0
    numero_test_percentuale =  10 # 10

    # 0.2,0.3,0.4,0.5,0.6,0.7,0.8
    alfa = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]  # coefficiente dei pesi per calcolare lo score (più è piccola più do' peso alla distanza)
    # start_simulations_in_parallel(percentuali_heatmap, numero_test_percentuale)
    vehicle_number = [300]  #,250,300
    for p in percentuali_heatmap:
        for a in alfa:
            for v_num in vehicle_number:
                for i in range(numero_test_percentuale):
                    # Avvia SUMO con l'opzione --start per iniziare subito la simulazione
                    traci.start(
                        [sumoBinary, "-c", "parking_on_off_road.sumocfg", "--tripinfo-output", f"tripinfo_{p}_{i}.xml",
                         "--start", "--quit-on-end"])

                    # Chiama la funzione run che gestisce la logica della simulazione
                    run(p, i, a, v_num)

    # Numero di processi paralleli, puoi modificarlo a seconda del numero di core della tua macchina
    """numero_processi = 3

    # Parallelizza le simulazioni
    with concurrent.futures.ProcessPoolExecutor(max_workers=numero_processi) as executor:
        futures = []
        for p in percentuali_heatmap:
            for i in range(numero_test_percentuale):
                futures.append(executor.submit(simula_percentuale, p, i))

        # Aspetta che tutte le simulazioni siano completate
        concurrent.futures.wait(futures)"""
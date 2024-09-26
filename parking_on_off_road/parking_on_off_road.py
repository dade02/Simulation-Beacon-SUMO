#!/usr/bin/env python3
import math
import optparse
import os
import sys
from contextlib import nullcontext
from xxlimited_35 import error

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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
            #print(f"ParkingArea with ID {parking_id} not found in {additional_file}")
            return None

        lane_id = parking_area.get("lane")
        start_pos = float(parking_area.get("startPos"))
        end_pos = float(parking_area.get("endPos"))

        # Debugging
        #(f"Found parking area: Lane ID: {lane_id}, Start Position: {start_pos}, End Position: {end_pos}")

        # Ottieni le coordinate della corsia
        lane_coords = traci.lane.getShape(lane_id)
        if not lane_coords:
            #print(f"No coordinates found for lane {lane_id}")
            return None

        #print(f"Lane coordinates: {lane_coords}")

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
                    #print(f"aggiunto {x_start}")
                    parking_coords.append((x_start, y_start))
                    break  # Esci dal ciclo dopo aver trovato la prima coordinata

        # Restituisci la prima coordinata trovata
        if parking_coords:
            #print("ritorna coordinate")
            return parking_coords[0]

    except Exception as e:
        print(f"An error occurred: {e}")

    #print(f"ritorna none per il parcheggio {parking_id}")
    return None  # Restituisci None se non trovi il parcheggio


class HeatMap:



    def __init__(self, xml_file,additional_file):
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
        #print(f"Net Boundaries: minX={self.minX}, minY={self.minY}, maxX={self.maxX}, maxY={self.maxY}")

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

    def update(self,parkage,parked_id = None, real_parkages = False):
        #caso heatmap
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
        #caso mappa parcheggi reali
        else:
            """
            Trovo tutti i parcheggi nella rete stradale 
            """
            tree_additional = ET.parse('parking_on_off_road.add.xml')
            root_additional = tree_additional.getroot()

            # Trova tutti gli elementi 'parkingArea'
            for parking_area in root_additional.findall(".//parkingArea"):
                parking_id = parking_area.get("id")
                posX, posY = get_parking_coordinates(parking_id, 'parking_on_off_road.add.xml')
                # Calcola gli indici della matrice per la posizione del veicolo
                col_index = math.floor((posX - self.minX) / self.dimensione_area)
                row_index = math.floor((posY - self.minY) / self.dimensione_area)

                # Verifica che gli indici siano all'interno dei limiti della matrice
                if 0 <= col_index < self.cols and 0 <= row_index < self.rows:
                    self.heat_map[row_index][col_index].append(1)

    def print_heatmap(self, title="Heatmap",real_parkage=False):
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
        cax = plt.imshow(display_matrix, cmap=cmap, norm=norm, interpolation='nearest',origin='lower')

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
    def save_heatmap_to_image(self, file_path, title="Heatmap",real_parkage=False):
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
        cax = plt.imshow(display_matrix, cmap=cmap, norm=norm, interpolation='nearest',origin='lower')

        # Aggiungi una barra di colore per la heatmap
        if not real_parkage:
            cbar = plt.colorbar(cax, ticks=[0, 1])
            cbar.set_label('Presenza di veicoli')
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Nessun Dato', 'Dato'])
        else:
            cbar = plt.colorbar(cax, ticks=['No', 'Yes'])
            cbar.set_label('Presenza di parcheggi')
            cbar.set_ticks([0,1])
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

        #print(f"Heatmap salvata come immagine in {file_path}")




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

    def find_best_parking_zones(self,preference,vehicle_id):
        #filtro in base alle preferenze dei vari veicoli


        # Inizializza tutte le celle a 1
        cell_states = np.ones((self.rows, self.cols))

        # Calcola i valori delle celle in base alla heatmap
        for i in range(self.rows):
            for j in range(self.cols):
                if self.heat_map[i][j]:
                    cell_states[i][j] = sum(self.heat_map[i][j])
                else:
                    cell_states[i][j] = 1 #fuori considerazione


        #print("heatmap intermedia")
        #self.print_heatmap_values()



        best_zones = [] #tutte le possibili aree di parcheggio filtrate


        if preference[vehicle_id] == 'find-possibility':
            """
            Trova le zone della mappa con la maggiore probabilità di parcheggio libero,
            cercando la cella con la somma maggiore ma <= 0.
            """
            max_value = -1e10
            for i in range(self.rows):
                for j in range(self.cols):
                    if cell_states[i][j] > max_value and cell_states[i][j] <= 0:
                        max_value = cell_states[i][j]


            # Cerca le celle con il valore massimo trovato
            for i in range(self.rows):
                for j in range(self.cols):
                    if cell_states[i][j] == max_value:
                        best_zones.append((i, j))

            #print(f"Veicolo {vehicle_id} settaggio destinazione con preferenza find-possibility" )


        elif preference[vehicle_id] == 'distance':
            #inserisce tutti i parcheggi segnati

            for i in range(self.rows):
                for j in range(self.cols):
                    #inserisco tutti i parcheggi segnati sulla heatmap fino ad ora
                    if cell_states[i][j] != 1:
                        best_zones.append((i, j))

            #print(f"Veicolo {vehicle_id} settaggio destinazione con preferenza distance" )
        else:
            #print("Errore settaggio file")
            return None

        #print(f"possibili destinazioni: {best_zones}")
        return best_zones

    def direct_vehicle_to_best_parking(self, vehicle_id, destinations,preference):
        """
        Dirige il veicolo verso la corsia (lane) con la maggiore probabilità di trovare un parcheggio libero,
        prendendo in considerazione la destinazione attuale (edge_id) del veicolo.

        Parametri:
        - vehicle_id: L'ID del veicolo che deve essere indirizzato verso una zona di parcheggio.
        - destinations: Dizionario che associa vehicle_id a edge_id, che rappresenta la destinazione attuale del veicolo.
        """
        edge_id = destinations.get(vehicle_id)

        if edge_id is None:
            #print(f"Nessuna destinazione trovata per il veicolo con ID {vehicle_id}.")
            return

        best_zones = self.find_best_parking_zones(preference,vehicle_id)

        if not best_zones:
            #print("Nessuna area libera identificata. Non è possibile indirizzare il veicolo.")
            return

        best_lane = None
        min_cost = float('inf')

        # Carica la rete stradale utilizzando sumolib
        net = sumolib.net.readNet("parking_on_off_road.net.xml")

        for best_row, best_col in best_zones:
            posX, posY = self.get_coordinates_from_cell(best_row, best_col) #coordinate centrali cella
            nearest_lane = self.find_closest_lane(posX, posY) #lane più vicina alle coordinate

            #distanza aerea
            if nearest_lane:
                lane_edge_id = traci.lane.getEdgeID(nearest_lane)

                try:
                    # Calcola la distanza aerea tra i due edge
                    x1, y1 = net.getEdge(edge_id).getFromNode().getCoord()
                    x2, y2 = net.getEdge(lane_edge_id).getFromNode().getCoord()
                    air_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                    #print(f"air distance: {air_distance} to {nearest_lane} da {edge_id}")

                    if air_distance < min_cost:
                        min_cost = air_distance
                        best_lane = nearest_lane

                except Exception as e:
                    #print(f"Errore nel calcolo della distanza aerea: {e}")
                    continue

                #print(f"best_lane: {best_lane}")

        if best_lane:
            traci.vehicle.changeTarget(vehicle_id, best_lane.split('_')[0])
            destinations[vehicle_id] = best_lane.split('_')[0]
            print(f"Il veicolo {vehicle_id} è stato indirizzato verso la corsia più vicina: {best_lane}")
        else:
            print(f"Nessuna corsia valida trovata per il veicolo {vehicle_id}.")




def is_vehicle_parked(vehicle_id):
    stop_state = traci.vehicle.getStopState(vehicle_id)
    return stop_state & tc.STOP_PARKING != 0

def is_near_parkage(vehicle_id,parkage_id,parking_to_edge):
    if parking_to_edge[parkage_id] == traci.vehicle.getRoadID(vehicle_id).split('_')[0]:  # se sono sulla stessa lane
        vehicle_position = traci.vehicle.getLanePosition(vehicle_id)
        park_start = traci.parkingarea.getStartPos(parkage_id)
        park_end = traci.parkingarea.getEndPos(parkage_id)

        if vehicle_position > park_start - 15 and vehicle_position < park_end - 15:
           #print(f"veicolo {vehicle_id} vicino al parcheggio {parkage_id}")
           return True

    return False

def park_vehicle(vehicle_id, parkage_id, parking_car_parked, parking_capacity, parked_vehicles):
    occupied_count = parking_car_parked[parkage_id]
    capacity = parking_capacity[parkage_id]


    if occupied_count < capacity:
        print("C' è spazio")
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

        #print(f"veicolo  {vehID} Next Junction ID: {next_junction_id}")

        if not next_junction_id:
            #print(f"Nessuna junction trovata per l'edge {current_edge}")
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
        #print(f"Veicolo {vehID} vicino alla prossima junction {next_junction_id}")
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
    ENTRY_EXIT_LIST = ["E85","-E85","E86","-E86"]
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
    #if random.random() < 0.1:  # Probabilità del 10%
     #   return (random_x, random_y), None

    # Generare randomicamente uso heatmap 40%
    if random.random() < 0.7:  # Probabilità del 40%
        heatmap = True
    else:
        heatmap = False


    # Ottenere l'edge più vicino al punto casuale
    edge_id = None
    while edge_id == None or edge_id in ENTRY_EXIT_LIST:
        #print(f"edge id: {edge_id}")
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
            #print(f"to: {edge.get('to')}, current {(edge_id.split('_')[0])[1:]}")
            if edge.get("to") == (edge_id.split('_')[0])[1:]:
                connected_edges.append(edge.get("id"))

        for e in connected_edges:
            if not e.startswith(':') and e not in ENTRY_EXIT_LIST:
                edge_id = e
                break


    if heatmap == True:
        if random.random() < 0.5:
            preference = "distance"
        else:
            preference = "find-possibility"
    else:
        preference = None


    return (random_x, random_y), edge_id,heatmap,preference


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
        (x, y), edge_id, heat_map, preference = random_point_on_map()
        if edge_id:
            destinations.append([vehicle_id, edge_id, x, y,heat_map,preference])
        else:
            print(f"Errore creazione destinazione veicolo {vehicle_id}")
            destinations.append([vehicle_id,"None"])

    # Scrivere le destinazioni nel file CSV
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['VehicleID', 'EdgeID', 'X', 'Y','heatmap','preference'])
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
    use_heatmap = {}
    preference = {}
    with open(file_name, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            vehicle_id = row['VehicleID']
            edge_id = row['EdgeID']
            heat_map = row['heatmap']
            destinations[vehicle_id] = edge_id
            use_heatmap[vehicle_id] = heat_map
            preference[vehicle_id] = row['preference']

    return destinations,use_heatmap,preference




def run():
    parkage_map = HeatMap(xml_file='heat_map.xml', additional_file='parking_on_off_road.add.xml')
    parkage_map.update(True, real_parkages=True)
    parkage_map.print_heatmap_values()
    # salva mappa parcheggi reali
    parkage_map.save_heatmap_to_image('real_parkages.jpg', 'parcheggi reali', True)



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
    exitLane = 'E86'

    #per ogni veicolo tengo traccia di quante volte ha percorso un edge
    car_history_edge = defaultdict(dict)

    recent_changed_route_cars = {}
    REROUTE_PERIOD = 10000

    #exit_lane_list = []
    car_arrived_in_b = [] #lista di veicoli arrivati al loro punto B

    destinations,use_heatmap,preference = get_vehicle_destinations() # punti B per ciascun veicolo
    #print(destinations)
    net = sumolib.net.readNet("parking_on_off_road.net.xml")

    print("INIZIO HEATMAP")

    heatmap = HeatMap(xml_file='heat_map.xml',additional_file='parking_on_off_road.add.xml')
    print("FINE HEATMAP")


    print('Mappa parcheggi')

    print('Fine Mappa parcheggi')


    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        current_time = traci.simulation.getTime()


        for vehicle_id in traci.vehicle.getIDList():
            if use_heatmap[vehicle_id] == 'True':
                print(f"veicolo {vehicle_id} usa la heatmap")
                heatmap.direct_vehicle_to_best_parking(vehicle_id,destinations,preference)
                use_heatmap[vehicle_id] = None

            if vehicle_id not in car_arrived_in_b:
                #print(f"arrivo: {traci.vehicle.getRoute(vehicle_id)[-1]} newdest: {destinations[vehicle_id]}")
                #se c'è arrivato ora
                if traci.vehicle.getLaneID(vehicle_id).split('_')[0] == destinations[vehicle_id]:
                    car_arrived_in_b.append(vehicle_id)
                    #print(f"Veicolo {vehicle_id} arrivato a destinazione B" )
                elif traci.vehicle.getRoute(vehicle_id)[-1] != destinations[vehicle_id]:
                    traci.vehicle.changeTarget(vehicle_id, destinations[vehicle_id])
                    for edge in traci.vehicle.getRoute(vehicle_id):
                        if edge not in car_history_edge[vehicle_id]:
                           car_history_edge[vehicle_id][edge] = 0
                        car_history_edge[vehicle_id][edge] += 1

                if not use_heatmap[vehicle_id] == None:
                    traci.vehicle.setColor(vehicle_id, (255, 0, 0, 255)) #non usano heatmap
                else:
                    traci.vehicle.setColor(vehicle_id, (255, 0, 255, 255))#usano heatmap

            if vehicle_id in car_arrived_in_b:
                if not is_vehicle_parked(vehicle_id):

                    #lista di veicoli che stanno uscendo
                    #if traci.vehicle.getRoadID(vehicle_id) == exitLane and vehicle_id not in exit_lane_list:
                    #    exit_lane_list.append(vehicle_id)

                    if recent_changed_route_cars.get(vehicle_id) != None and current_time - recent_changed_route_cars[vehicle_id] > REROUTE_PERIOD:
                        del recent_changed_route_cars[vehicle_id]
                    # se il veicolo ha appena svoltato, non vi è più il controllo sul doppio risettaggio
                    if recent_changed_route_cars.get(vehicle_id) != None:
                        #print(f"edge veicolo: {vehicle_id} edge attuale: {traci.vehicle.getRoadID(vehicle_id)} meta: {traci.vehicle.getRoute(vehicle_id)[-1]}")
                        #se sono alla destinazione
                        if traci.vehicle.getRoadID(vehicle_id) == traci.vehicle.getRoute(vehicle_id)[-1]:
                            del recent_changed_route_cars[vehicle_id]
                           #print(f"Il veicolo {vehicle_id} può di nuovo cambiare rotta")
                            #print(f"occorrenze: { car_history_edge[vehicle_id]}")


                    if vehicle_id in parked_vehicles:
                        if is_exit_Parkage(vehicle_id,parked_vehicles[vehicle_id],parking_to_edge):
                            #print(f"veicolo {vehicle_id} uscito dal parcheggio {parked_vehicles[vehicle_id]}")
                            parking_car_parked[parked_vehicles[vehicle_id]] -= 1    #aggiorno il numero di veicoli parcheggiati

                            if use_heatmap[vehicle_id] == None: #se la usa (ho settato a None quelle che lo hanno usato)
                                heatmap.update(False,parked_vehicles[vehicle_id] )

                            del parked_vehicles[vehicle_id]                         #rimuovo da veicoli parcheggiati
                            recent_parking_vehicles[vehicle_id] = current_time      #aggiungo a veicoli usciti da poco



                            # cancello ultimo reroute se esiste
                            # se la destinazione non è la lane dove vi è adesso il veicolo
                            #print(
                                #f"veicolo {vehicle_id} strada {traci.vehicle.getLaneID(vehicle_id).split('_')[0]} dest {traci.vehicle.getRoute(vehicle_id)[-1]}")

                            if traci.vehicle.getRoute(vehicle_id)[-1] != traci.vehicle.getLaneID(vehicle_id).split('_')[0]:
                                #if traci.vehicle.getRoute(vehicle_id)[-1] != exitLane:
                                    #print(
                                     #   f"la destinazione {traci.vehicle.getRoute(vehicle_id)[-1]} per il veicolo {vehicle_id} è da eliminare")
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
                                    if use_heatmap[vehicle_id] == None:
                                        heatmap.update(True, parked_vehicles[vehicle_id] )
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

    #salva heatmap
    heatmap.save_heatmap_to_image('heatmap.jpg')



    #print(f"Numero veicoli usciti: {len(exit_lane_list)}")
    #print(sorted(exit_lane_list, key=estrai_numero))
    #print("Storico passaggi veicoli")
    #for v in car_history_edge:
    #    print(f"veicolo {v} : edges {car_history_edge[v]}")



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
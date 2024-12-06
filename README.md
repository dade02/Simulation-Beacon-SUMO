# Parking On/Off Road Simulation

Questo progetto utilizza **SUMO** (Simulation of Urban Mobility) e **TraCI** per simulare la gestione dei parcheggi lungo una rete stradale. Consente di analizzare le dinamiche dei parcheggi con e senza il supporto di una heatmap, calcolare statistiche e produrre visualizzazioni utili per lo studio del traffico urbano.

## **Struttura del Progetto**

Nella directory principale sono presenti i seguenti file:

- **`parking_on_off_road.py`**: Script Python principale che controlla la simulazione, interagendo dinamicamente con la rete tramite il modulo TraCI. Gestisce veicoli e parcheggi e raccoglie i dati della simulazione.
- **`parking_on_off_road.sumocfg`**: File di configurazione di SUMO, che definisce i file richiesti per la simulazione (rete stradale, parcheggi, ecc.).
- **`parking_on_off_road.net.xml`**: Descrive la rete stradale, inclusi incroci e collegamenti.
- **`parking_on_off_road.add.xml`**: Specifica i parcheggi lungo la rete, con parametri come:
  - `id`: Identificativo univoco del parcheggio.
  - `lane`: Corsia su cui il parcheggio si trova.
  - `startPos` e `endPos`: Posizione del parcheggio sulla corsia (in metri).
  - `roadsideCapacity`: Numero di posti disponibili.
- **`manage_data.py`**: Script Python per manipolare ed analizzare i risultati della simulazione. Calcola le medie dei dati raccolti e genera grafici.
- **`heat_map.xml`**: File che definisce la granularità della heatmap nella simulazione.

## **Dipendenze**
Versione Python: **3.11.2** 
Il progetto richiede le seguenti librerie Python:
- `numpy`: Per operazioni matematiche avanzate.
- `pandas`: Per la manipolazione e l'analisi di dati strutturati.
- `seaborn`: Per la creazione di grafici statistici.
- `imageio`: Per la gestione di immagini e video.

## **Installazione**

1. Clona il repository:
    ```bash
    git clone https://github.com/dade02/tirocinio.git
    cd tirocinio
    ```
2. Crea un ambiente virtuale:
    ```bash
    pipenv shell
    ```
3. Installa le dipendenze:
    ```bash
    pipenv install
    ```

## **Avvio della Simulazione**

Per avviare la simulazione, utilizza il comando:
```bash
python3 parking_on_off_road.py

Se desideri disabilitare l'interfaccia grafica di SUMO:
```bash
python3 parking_on_off_road.py --nogui
```
  

---

## **Come Utilizzare `manage_data.py`**  
Lo script `manage_data.py` è progettato per elaborare i dati generati dalla simulazione dei parcheggi, calcolando medie statistiche e producendo grafici per analizzare l'andamento dei parametri simulati.
**Calcolo delle medie dei dati e rappresentazione grafica**  
Lo script prende in input il file `results_data.csv`, che contiene i dati raccolti durante la simulazione, e calcola le medie per ciascun gruppo di simulazioni con parametri simili.  
Per eseguire questa operazione e salvare i risultati in un nuovo file CSV:  
  ```bash
   python3 manage_data.py --output csv NOME_FILE.csv
  ```
Se si dispone già di un file contenente le medie e si vuole solo creare i grafici:
   ```bash
   python3 manage_data.py --skip-media --output grafici
   ```


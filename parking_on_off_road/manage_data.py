import pandas as pd
import numpy as np
import csv
import re
import matplotlib.pyplot as plt

def calcola_media_percentuali(input_csv, output_csv):
    # Leggi il file CSV
    df = pd.read_csv(input_csv)

    # Definisci le colonne che devono essere considerate per il calcolo della media
    colonne_da_mediare = [
        'tempo_medio_parcheggio_B_heatmap',
        'tempo_medio_ricerca_posteggio_heatmap',
        'distanza_parcheggio_punto_B_heatmap',
        'tempo_medio_parcheggio_B',
        'tempo_medio_ricerca_posteggio',
        'distanza_parcheggio_punto_B'
    ]

    # Arrotonda la percentuale di uso alle percentuali desiderate (0, 25, 50, 75, 100)
    percentuali_target = [0, 25, 50, 75, 100]
    df['percentuale_uso_rounded'] = df['percentuale_uso_heatmap'].str.rstrip('%').astype(float)
    df['percentuale_uso_rounded'] = df['percentuale_uso_rounded'].apply(
        lambda x: min(percentuali_target, key=lambda p: abs(p - x)))

    # Raggruppa i dati per la percentuale arrotondata e calcola la media per le colonne specificate ignorando i valori NaN
    df_media = df.groupby('percentuale_uso_rounded')[colonne_da_mediare].mean().reset_index()

    # Salva il risultato in un nuovo CSV
    df_media.to_csv(output_csv, index=False)


# Funzione per estrarre la parte numerica da un vehicle_id, ad esempio "vehicle_10" -> 10
def extract_vehicle_number(vehicle_id):
    match = re.search(r'(\d+)', vehicle_id)
    return int(match.group(1)) if match else 0

# Funzione per ordinare i dati per perc_use_heatmap, numero_test_percentuale e poi vehicle_id
def sort_data(file_path):
    # Legge i dati dal file CSV
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Mantiene separato l'header
    header, rows = data[0], data[1:]

    # Filtra e ordina solo le righe che contengono valori numerici validi nelle colonne 2 e 3
    valid_rows = []
    for row in rows:
        try:
            # Prova a convertire in float le colonne 3 e 2
            perc_use_heatmap = float(row[0])
            numero_test_percentuale = float(row[1])
            valid_rows.append(row)
        except ValueError:
            # Se c'è un errore di conversione, la riga viene ignorata
            print(f"Riga ignorata per errore di conversione: {row}")

    # Ordina i dati validi: prima per perc_use_heatmap, poi per numero_test_percentuale, e infine per vehicle_id (parte numerica)
    rows_sorted = sorted(valid_rows, key=lambda x: (
        float(x[0]),  # perc_use_heatmap
        float(x[1]),  # numero_test_percentuale
        extract_vehicle_number(x[2])  # Estrae il numero dal vehicle_id
    ))

    # Scrive i dati ordinati nuovamente nel file CSV, sovrascrivendo i vecchi
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Scrive l'header
        writer.writerows(rows_sorted)  # Scrive le righe ordinate


def genera_grafici_da_csv(file_csv):
    # Leggi il CSV in un DataFrame pandas
    df = pd.read_csv(file_csv)

    # Ottieni la colonna delle percentuali
    percentuali = df['percentuale_uso_rounded']

    # Lista delle metriche da tracciare
    metriche = ['tempo_medio_parcheggio_B_heatmap', 'tempo_medio_ricerca_posteggio_heatmap',
                'distanza_parcheggio_punto_B_heatmap', 'tempo_medio_parcheggio_B',
                'tempo_medio_ricerca_posteggio', 'distanza_parcheggio_punto_B']

    # Per ogni metrica, genera un grafico
    for metrica in metriche:
        # Verifica se la colonna contiene dati validi
        if df[metrica].notna().any():
            plt.figure()  # Crea una nuova figura per ciascun grafico
            plt.plot(percentuali, df[metrica], marker='o', label=metrica)

            # Aggiungi titolo e label degli assi
            plt.title(f"Andamento di {metrica} rispetto alle percentuali di uso")
            plt.xlabel('Percentuale di uso')
            plt.ylabel(metrica)

            # Mostra la griglia e la legenda
            plt.grid(True)
            plt.legend()

            # Salva il grafico in un file (opzionale)
            plt.savefig(f"grafico_{metrica}.png")

            # Mostra il grafico
            plt.show()





if __name__ == '__main__':
    input_csv = 'results_data.csv'  # Sostituisci con il percorso del tuo file
    output_csv = 'media_percentuali.csv'
    calcola_media_percentuali(input_csv, output_csv)

    # Specifica il percorso del file CSV
    file_path = 'data_agent.csv'
    # Eseguiamo la funzione per ordinare e sovrascrivere il file CSV
    sort_data(file_path)

    genera_grafici_da_csv(output_csv)
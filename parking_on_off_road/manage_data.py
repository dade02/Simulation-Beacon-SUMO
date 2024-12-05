import pandas as pd
import numpy as np
import csv
import re
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

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
    df_media = df.groupby(['percentuale_uso_rounded','alfa','veicoli'])[colonne_da_mediare].mean().reset_index()

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

    # Ottieni le colonne delle percentuali e di alfa
    #percentuali = df['percentuale_uso_rounded']
    #alfas = df['alfa']

    # Lista delle metriche da tracciare
    metriche_heatmap = ['tempo_medio_parcheggio_B_heatmap', 'tempo_medio_ricerca_posteggio_heatmap',
                        'distanza_parcheggio_punto_B_heatmap']
    metriche_no_heatmap = ['tempo_medio_parcheggio_B', 'tempo_medio_ricerca_posteggio',
                           'distanza_parcheggio_punto_B']

    # Raggruppa i dati per 'percentuale_uso_rounded'
    grouped_by_percentuale = df.groupby('percentuale_uso_rounded')

    # Genera un grafico per ciascuna metrica da misurare (con e senza heatmap)
    for metrica_heatmap, metrica_no_heatmap in zip(metriche_heatmap, metriche_no_heatmap):
        # Per ogni gruppo di percentuali (0, 25, 50, 75, 100)
        for percentuale, group in grouped_by_percentuale:
            plt.figure()  # Crea una nuova figura per ciascun grafico

            # Verifica se entrambe le colonne contengono dati validi nel gruppo corrente
            if group[metrica_heatmap].notna().any() and group[metrica_no_heatmap].notna().any():
                # Grafico per la metrica con heatmap (linea blu)
                plt.plot(group['alfa'], group[metrica_heatmap], marker='o', color='blue',
                         label=f"{metrica_heatmap}")

                # Grafico per la metrica senza heatmap (linea arancione)
                plt.plot(group['alfa'], group[metrica_no_heatmap], marker='o', color='orange',
                         label=f"{metrica_no_heatmap}")

                # Aggiungi titolo e label degli assi
                plt.title(f"Andamento di {metrica_heatmap} e {metrica_no_heatmap} (percentuale: {percentuale}%)")
                plt.xlabel('Valore di alfa')
                plt.ylabel('Valore delle metriche')

                # Mostra la griglia e la legenda
                plt.grid(True)
                plt.legend()

                # Salva il grafico in un file (opzionale)
                plt.savefig(f"grafico_{metrica_heatmap}_{metrica_no_heatmap}_percentuale_{percentuale}.png")

                # Mostra il grafico
                plt.show()


def crea_bar_plot(file_csv):
    # Leggi il CSV in un DataFrame pandas
    df = pd.read_csv(file_csv)

    # Lista delle metriche da tracciare
    metriche_heatmap = ['tempo_medio_parcheggio_B_heatmap', 'tempo_medio_ricerca_posteggio_heatmap',
                        'distanza_parcheggio_punto_B_heatmap']
    metriche_no_heatmap = ['tempo_medio_parcheggio_B', 'tempo_medio_ricerca_posteggio',
                           'distanza_parcheggio_punto_B']


    # Raggruppa i dati per 'percentuale_uso_rounded' e 'veicoli'
    grouped_by_percentuale_veicoli = df.groupby(['percentuale_uso_rounded', 'veicoli'])

    # Genera un bar plot per ciascuna coppia (percentuale, veicoli)
    for (percentuale, veicoli), group in grouped_by_percentuale_veicoli:
        plt.figure(figsize=(12, 6))  # Crea una nuova figura per ciascun grafico

        # Combina percentuale e veicoli per la legenda
        legenda_label = f"Percentuale: {percentuale}% - Veicoli: {veicoli}"
        # Tracciamo ogni metrica
        for metrica_heatmap, metrica_no_heatmap in zip(metriche_heatmap, metriche_no_heatmap):

            # Riempie i valori mancanti con 0 per evitare valori NaN nelle colonne delle metriche
            group[metrica_heatmap] = group[metrica_heatmap].fillna(0)
            group[metrica_no_heatmap] = group[metrica_no_heatmap].fillna(0)

            # Se il gruppo ha dati validi per entrambe le metriche
            if group[metrica_heatmap].notna().any() and group[metrica_no_heatmap].notna().any():
                # Creiamo un nuovo DataFrame per visualizzare le colonne affiancate per ogni 'alfa'
                data = pd.DataFrame({
                    'alfa': group['alfa'].astype(str),
                    'heatmap': group[metrica_heatmap],
                    'no_heatmap': group[metrica_no_heatmap]
                })

                # Riorganizza i dati nel formato lungo per Seaborn
                data_melted = data.melt(id_vars='alfa', value_vars=['heatmap', 'no_heatmap'],
                                        var_name='tipo', value_name='valore')

                # Crea il barplot
                sns.barplot(x='alfa', y='valore', hue='tipo', data=data_melted,
                            palette={'heatmap': 'lightblue', 'no_heatmap': 'blue'},
                            dodge=True)

                # Aggiungi titolo e label degli assi
                plt.title(f"Confronto delle metriche per percentuale {percentuale}% e veicoli {veicoli}")
                plt.xlabel('Valore di alfa')
                plt.ylabel(f'Valore di {metrica_heatmap.split("_heatmap")[0]}')  # Mostra cosa stiamo calcolando

                # Mostra la griglia e la legenda
                plt.grid(True)
                plt.legend(title='Tipo di metrica')

                # Salva il grafico in un file (opzionale)
                plt.savefig(f"barplot_percentuale_{percentuale}_veicoli_{veicoli}_{metrica_heatmap.split('_heatmap')[0]}.png")

                # Mostra il grafico
                plt.show()


def crea_grafici_percentuali(file_csv):
    # Leggi il CSV in un DataFrame pandas
    df = pd.read_csv(file_csv)

    # Lista delle metriche da tracciare
    metriche_heatmap = ['tempo_medio_parcheggio_B_heatmap', 'tempo_medio_ricerca_posteggio_heatmap',
                        'distanza_parcheggio_punto_B_heatmap']
    metriche_no_heatmap = ['tempo_medio_parcheggio_B', 'tempo_medio_ricerca_posteggio',
                           'distanza_parcheggio_punto_B']

    # Genera un grafico per ciascuna metrica
    for metrica_heatmap, metrica_no_heatmap in zip(metriche_heatmap, metriche_no_heatmap):
        plt.figure(figsize=(12, 6))  # Crea una nuova figura per ciascun grafico

        # Traccia la metrica con heatmap (linea blu)
        sns.lineplot(data=df, x='percentuale_uso_rounded', y=metrica_heatmap, marker='o',
                     label=f"Heatmap: {metrica_heatmap}", color='blue')

        # Traccia la metrica senza heatmap (linea arancione)
        sns.lineplot(data=df, x='percentuale_uso_rounded', y=metrica_no_heatmap, marker='o',
                     label=f"No Heatmap: {metrica_no_heatmap}", color='orange')

        # Aggiungi titolo, label degli assi e legenda
        plt.title(f"Andamento di {metrica_heatmap.split('_heatmap')[0]} rispetto alla percentuale di veicoli con heatmap")
        plt.xlabel('Percentuale di veicoli che usano la heatmap (%)')
        plt.ylabel('Valore della metrica')
        plt.xticks([0, 25, 50, 75, 100])  # Imposta le etichette sull'asse X
        plt.grid(True)
        plt.legend()

        # Salva il grafico in un file
        filename = f"grafico_{metrica_heatmap.split('_heatmap')[0]}.png"
        plt.savefig(filename)
        print(f"Grafico salvato come {filename}")

        # Mostra il grafico
        plt.show()


def crea_grafici(file_csv):
    """
    Legge i dati dal file CSV e genera grafici heatmap e lineari per le metriche fornite.

    Parametri:
        file_csv (str): Percorso al file CSV contenente i dati.
    """
    # Legge il file CSV in un DataFrame
    df = pd.read_csv(file_csv)

    # Liste delle metriche da tracciare
    metriche_heatmap = ['tempo_medio_parcheggio_B_heatmap', 'tempo_medio_ricerca_posteggio_heatmap','distanza_parcheggio_punto_B_heatmap']
    metriche_no_heatmap = ['tempo_medio_parcheggio_B', 'tempo_medio_ricerca_posteggio','distanza_parcheggio_punto_B']

    # Genera un grafico per ciascuna coppia di metriche
    for metrica_heatmap, metrica_no_heatmap in zip(metriche_heatmap, metriche_no_heatmap):
        plt.figure(figsize=(12, 6))  # Nuova figura per ciascun grafico

        # Traccia la metrica con heatmap (linea blu)
        sns.lineplot(data=df, x='perc_init_heatmap', y=metrica_heatmap, marker='o',
                     label=f"Heatmap: {metrica_heatmap}", color='blue')

        # Traccia la metrica senza heatmap (linea arancione)
        sns.lineplot(data=df, x='perc_init_heatmap', y=metrica_no_heatmap, marker='o',
                     label=f"No Heatmap: {metrica_no_heatmap}", color='orange')

        # Aggiungi titolo, assi e legenda
        plt.title(
            f"Confronto di {metrica_heatmap.split('_heatmap')[0]} rispetto alla percentuale di inizializzazione heatmap")
        plt.xlabel('Percentuale di inizializzazione heatmap (%)')
        plt.ylabel('Valore della metrica')
        plt.xticks([0, 20, 40, 60, 80, 100])  # Imposta le etichette sull'asse X
        plt.grid(True)
        plt.legend()

        # Salva il grafico in un file
        filename = f"grafico_percInitHeatmap_{metrica_heatmap.split('heatmap')[0]}.png"
        plt.savefig(filename)
        print(f"Grafico salvato come {filename}")

        # Mostra il grafico
        plt.show()



if __name__ == '__main__':
    """input_csv = 'results_data.csv'  # Sostituisci con il percorso del tuo file
    output_csv = 'media_percentuali.csv'
    calcola_media_percentuali(input_csv, output_csv)

    # Specifica il percorso del file CSV
    file_path = 'data_agent.csv'
    # Eseguiamo la funzione per ordinare e sovrascrivere il file CSV
    sort_data(file_path)

    crea_bar_plot(output_csv)"""

    parser = argparse.ArgumentParser(description="Genera grafici basati su un CSV di input")
    parser.add_argument('--input_csv', type=str, default='results_data.csv',
                        help="Percorso del file CSV di input")
    parser.add_argument('--output_csv', type=str, default=False,
                        help="Percorso del file CSV di output")
    parser.add_argument('--skip_media', action='store_true',
                        help="Salta il calcolo della media e usa il file CSV esistente")
    #parser.add_argument('--percInitHeatmap',default='percInitHeatmap.csv',
    #                    help="Salta il calcolo della media e usa il file CSV esistente")
    args = parser.parse_args()

    # Verifica se `--input_csv` è stato passato

    #if args.percInitHeatmap:
    #    crea_grafici(args.percInitHeatmap)
    #el
    if args.skip_media:
        crea_bar_plot(args.output_csv)
        crea_grafici_percentuali(args.output_csv)
    else:
        calcola_media_percentuali(args.input_csv, args.output_csv)
        crea_bar_plot(args.output_csv)
        crea_grafici_percentuali(args.output_csv)

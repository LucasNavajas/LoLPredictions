import pandas as pd
import numpy as np
import json

# Carga y preparación de los datos
df = pd.read_csv('data/datasheetv2.csv')

def glicko_update(R_winner, R_loser, RD_winner, RD_loser, K=64, RD_reduction_factor=0.97):
    q = np.log(10) / 400
    g_RD = lambda RD: 1 / np.sqrt(1 + 3 * q**2 * RD**2 / np.pi**2)
    E_winner = 1 / (1 + 10 ** (g_RD(RD_loser) * (R_loser - R_winner) / 400))
    E_loser = 1 / (1 + 10 ** (g_RD(RD_winner) * (R_winner - R_loser) / 400))
    R_winner_updated = R_winner + K * (1 - E_winner)
    R_loser_updated = R_loser - K * E_loser
    # Aplicar un simple factor de reducción al RD para ambos jugadores
    RD_winner_updated = max(RD_winner * RD_reduction_factor, 30)  # Asumir 30 como mínimo RD sugerido por Glicko
    RD_loser_updated = max(RD_loser * RD_reduction_factor, 30)
    return R_winner_updated, R_loser_updated, RD_winner_updated, RD_loser_updated


def calculate_player_glicko_ratings(df):
    unique_player_ids = set(df[['Top1ID', 'Jg1ID', 'Mid1ID', 'Adc1ID', 'Supp1ID', 'Top2ID', 'Jg2ID', 'Mid2ID', 'Adc2ID', 'Supp2ID']].melt()['value'].dropna().unique())
    player_glicko = {int(player_id): 1500 for player_id in unique_player_ids}
    player_RD = {int(player_id): 350 for player_id in unique_player_ids}

    for index, row in df.iterrows():
        equipo_ganador = str(row['TeamWinner'])
        equipo_perdedor = '2' if equipo_ganador == '1' else '1'

        for pos in ['Top', 'Jg', 'Mid', 'Adc', 'Supp']:
            id_ganador = row[pos + equipo_ganador + 'ID']
            id_perdedor = row[pos + equipo_perdedor + 'ID']

            if pd.notnull(id_ganador) and pd.notnull(id_perdedor):
                id_ganador = int(id_ganador)
                id_perdedor = int(id_perdedor)

                # Obtener las clasificaciones y RD actuales
                R_winner, RD_winner = player_glicko[id_ganador], player_RD[id_ganador]
                R_loser, RD_loser = player_glicko[id_perdedor], player_RD[id_perdedor]

                # Actualizar clasificaciones y RD utilizando glicko_update
                R_winner_updated, R_loser_updated, RD_winner_updated, RD_loser_updated = glicko_update(
                    R_winner, R_loser, RD_winner, RD_loser)

                # Actualizar las clasificaciones Glicko y RD en los diccionarios
                player_glicko[id_ganador], player_glicko[id_perdedor] = R_winner_updated, R_loser_updated
                player_RD[id_ganador], player_RD[id_perdedor] = RD_winner_updated, RD_loser_updated

    # Guardar los resultados
    with open('player_glicko_ratings.json', 'w') as f:
        json.dump({"player_glicko": player_glicko, "player_RD": player_RD}, f, indent=4)

    return player_glicko, player_RD

# Ejemplo de uso de la función
player_glicko_ratings, player_RD = calculate_player_glicko_ratings(df)

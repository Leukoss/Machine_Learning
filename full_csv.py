import pandas as pd

# Création du fichier complet
years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020',
         '2021', '2022']
files = ['usagers_', 'vehicules_', 'caracteristiques_', 'lieux_', ]
data_path = 'Data/TRAIN/BAAC-Annee-'

full_data = pd.DataFrame()

for year in years:
    path_to_BAAC = data_path + year + '/'
    year_file = pd.DataFrame()

    for file_name in files:
        path_to_file = path_to_BAAC + file_name + year + '_.csv'
        data = pd.read_csv(path_to_file, encoding="latin1", sep=";",
                           low_memory=False)
        data = data.drop(data.columns[0], axis=1)
        if file_name == 'usagers_':
            year_file = pd.concat([year_file, data], axis=1)
        else:
            if file_name == 'vehicules_':
                year_file = pd.merge(year_file, data, on=['Num_Acc', 'num_veh'],
                                     how='left')
            else:
                year_file = pd.merge(year_file, data, on=['Num_Acc'],
                                     how='left')

    full_data = pd.concat([full_data, year_file], axis=0)

var_cat = []
var_num = []
var_ord = []

# ------------------------------------------------------------------------------
# ----------------------------> USAGERS <---------------------------------------
# ------------------------------------------------------------------------------

col_to_drop = ['Num_Acc', 'id_vehicule_x', 'id_vehicule_y', 'num_veh']

for col in col_to_drop:
    if col in full_data.columns:
        full_data.drop(labels=col_to_drop, axis=1, inplace=True)


def convert_float_to_cat(data_frame, col_name):
    data_frame[col_name] = data_frame[col_name].astype(int)
    data_frame[col_name] = data_frame[col_name].astype('category')


def convert_float_to_int(data_frame, col_name):
    data_frame[col_name] = data_frame[col_name].astype(int)


# -> grav

# Supprime les valeurs "-1"
full_data = full_data[full_data['grav'] != -1]
full_data = full_data.dropna(subset=['grav'])

# Binarise les valeurs pour les accidents graves et non graves
full_data['grav'] = full_data['grav'].replace({1: 0, 4: 0, 2: 1, 3: 1})

# Converti en variable catégorielle
var_cat.append('grav')

# -> sexe

# Binarisation [0, 1] des valeurs de sexe
full_data = full_data[full_data['sexe'] != -1]
full_data['sexe'] = full_data['sexe'].replace({2: 0})

# Converti en variable catégorielle
var_cat.append('sexe')

# -> an_nais

# Supprime les valeurs manquantes
full_data = full_data.dropna(subset=['an_nais'])

# Converti en variable numérique
var_num.append('an_nais')

# -> trajet

# Harmonise les données [1, 2, 3, 4, 5, 9]
full_data['trajet'] = full_data['trajet'].replace({-1: 9, 0: 9})
full_data['trajet'] = full_data['trajet'].fillna(9)

# Converti en variable catégorielle
var_cat.append('trajet')

# -> locp

# Harmonise les données [1, 2, 3, 4, 5, 6, 7, 8, 9]
full_data['locp'] = full_data['locp'].replace({-1: 9, 0: 9})
full_data['locp'] = full_data['locp'].fillna(9)

# Converti en variable catégorielle
var_cat.append('locp')

# -> etatp

# Harmonise les données
full_data['etatp'] = full_data['etatp'].replace({0: -1}).fillna(-1)
full_data['etatp'] = full_data['etatp'].fillna(-1)

# Converti en variable catégorielle
var_cat.append('etatp')

# -> catu

# Pour les années antérieures à 2019 : catu = 4 -> catv = 99 & catu = 3
full_data.loc[full_data['catu'] == 4, 'catv'] = 99
full_data['catu'] = full_data['catu'].replace({4: 3})

# Converti en variable catégorielle
var_cat.append('catu')

# -> place

# Supprime seulement si 'place' est Nan ET que ce n'est pas un piéton. Si la
# ligne n'est pas supprimée, alors le NAN est transformé en '10' pour 'piéton'
full_data = full_data[~((full_data['catu'] != 3) & full_data['place'].isna())]
full_data['place'] = full_data['place'].fillna(10)

# Converti en variable catégorielle
var_cat.append('place')


# -> secu

# Fonction pour extraire le premier chiffre renvoyant à l'équipement utilisé
def extract_first_digit(value):
    if pd.notna(value):
        return int(str(value)[0])
    else:
        return 8


# Récupère les lignes où secu1/secu2/secu3 sont NAN (rare cas après 2019,
# tous les cas avant 2019)
all_nans = full_data[['secu1', 'secu2', 'secu3']].isna().all(axis=1)
filtered_data = full_data[all_nans]

# Conserve seulement le cas où il y a un second digit = 1, sinon applique la
# valeur 8 (non déterminable)
filtered_data.loc[
    (filtered_data['secu'] <= 10) | (filtered_data['secu'] % 10 == 2) | (
            filtered_data['secu'] % 10 == 3), 'secu'] = 8
filtered_data['secu'] = filtered_data['secu'].apply(extract_first_digit)

# Défini dans secu1 le seul équipement noté (1 seul avant 2019) et 8 pour les
# deux autres
filtered_data['secu1'] = filtered_data['secu']
filtered_data['secu2'] = 8
filtered_data['secu3'] = 8
# Met à jour les colonnes secu/secu1/secu2/secu3
full_data.loc[all_nans, :] = filtered_data

# On peut drop la colonne secu qui ne vaut plus rien
full_data.drop(labels=['secu'], axis=1, inplace=True)

# Dans les cas survenus après 2019
list_secu = ['secu1', 'secu2', 'secu3']

for secu_i in list_secu:
    full_data[secu_i] = full_data[secu_i].replace({-1: 8, 9: 8})

# Converti en variable catégorielle
var_cat.append('secu1')
var_cat.append('secu2')
var_cat.append('secu3')

# -> actp

# On définit un dictionnaire permettant de remplacer les str par des int afin
# d'harmoniser les types pour ensuite catégoriser les valeurs
replace_dict = {' -1.0': -1, '0.0': 0, '1.0': 1, '2.0': 2, '3.0': 3, '4.0': 4,
                '5.0': 5, '6.0': 6, '7.0': 7, '8.0': 8, '9.0': 9, 'A': 10,
                'B': 11}
full_data['actp'] = full_data['actp'].replace(replace_dict)

# Harmonise les données (-1, 0, B) → 9
full_data['actp'] = full_data['actp'].replace({-1: 9, 0: 9, 11: 9}).fillna(9)

# Converti en variable catégorielle
var_cat.append('actp')

# SAUVEGARDE UNE VERSION INTERMEDIAIRE
full_data.to_csv('Data/TRAIN_FULL/test_usagers.csv', index=False)


# ------------------------------------------------------------------------------
# ------------------------> CARACTERISTIQUES <----------------------------------
# ------------------------------------------------------------------------------

def convert_str_to_cat(data_frame, col_name):
    data_frame[col_name] = data_frame[col_name].astype('category')


# -> an/mois/jour/hrmn

# Traitement sur les années pour harmoniser la forme des étiquettes [2012 et
# non 12 par exemple]
full_data.loc[(full_data['an'] >= 12) & (full_data['an'] <= 18), 'an'] += 2000


# Harmonise hr:mn de type string en int
def convert_to_int(time_str):
    if isinstance(time_str, str):
        if ':' in time_str:
            hours, minutes = map(int, time_str.split(':'))
            return hours * 100 + minutes
        else:
            return int(time_str)
    else:
        return time_str


full_data['hrmn'] = full_data['hrmn'].apply(convert_to_int)


# Traitement des minutes
def generate_intervals(start, end, step):
    intervals = []
    for i in range(start, end, step):
        intervals.append((i, i + step))
    return intervals


# Generate intervals automatically for each 100 units
intervals = generate_intervals(0, 2400, 100)
simple_values = list(range(len(intervals)))


# Function to map values to simple values based on intervals
def map_to_simple_value(value):
    for i, interval in enumerate(intervals):
        if interval[0] <= value < interval[1]:
            return simple_values[i]
    return None  # Return None if value is outside all intervals


# Apply the mapping function to the column and create a new column with the
# simple values
full_data['hrmn'] = full_data['hrmn'].apply(map_to_simple_value)

# Converti en variable catégorielle
var_cat.append('jour')
var_cat.append('mois')
var_cat.append('an')
var_cat.append('hrmn')

# -> lum

# Suppression des valeurs -1
full_data = full_data[full_data['lum'] != -1]
full_data = full_data.dropna(subset=['lum'])

# Converti en variable catégorielle
var_cat.append('lum')

# -> dep/com/agg

# Suppression de la colonne com
full_data.drop(labels=['com'], axis=1, inplace=True)

# Binarisation de la valeur d'agglomération
full_data['agg'] = full_data['agg'].replace({2: 0})

# Harmonisation des départements
# Supprime les 0 superflus à gauche
full_data['dep'] = full_data['dep'].apply(lambda x: x.lstrip('0'))

# Cas des départements inexistants (100, 970...)
full_data['dep'] = full_data['dep'].apply(lambda x: x[:-1] if len(x) == 3 and x.endswith('0') else x)

# Remplace les valeurs pour la Corse
full_data['dep'] = full_data['dep'].replace({'201': '2A', '202': '2B'})

# En cas d'autres valeurs de département, supprime les lignes
full_data = full_data[full_data['dep'].isin(map(str, range(1, 96))) |
                      full_data['dep'].isin(['971', '972', '974', '976'])]

# Converti en variable catégorielle
var_cat.append('dep')
var_cat.append('agg')

# -> int

# Harmonise les données
full_data['int'] = full_data['int'].replace({-1: 9, 0: 9})

# Converti en variable catégorielle
var_cat.append('int')

# -> atm

# Harmonise les données
full_data['atm'] = full_data['atm'].replace({-1: 9, 0: 9}).fillna(9)

# Converti en variable catégorielle
var_cat.append('atm')

# -> col

# Harmonise les données
full_data['col'] = full_data['col'].replace({-1: 8, 0: 8}).fillna(8)

# Converti en variable catégorielle
var_cat.append('col')

# -> adr, gps, lat, long

# Suppression des colonnes adr, gps, lat, long
full_data.drop(labels=['adr', 'gps', 'lat', 'long'], axis=1, inplace=True)

# SAUVEGARDE UNE VERSION INTERMEDIAIRE
full_data.to_csv('Data/TRAIN_FULL/test_usagers_caractéristiques.csv',
                 index=False)

# ------------------------------------------------------------------------------
# ------------------------> CARACTERISTIQUES <----------------------------------
# ------------------------------------------------------------------------------


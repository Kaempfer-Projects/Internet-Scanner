import glob
import os
import numpy as np
import pandas as pd

path1 = 'D:/Ergebnis-droprows/'

path2 = 'D:/FinalKodierung/'
name1 = 'histstrg_'
df_all_rows = pd.DataFrame()

df1 = pd.read_csv('D:/headerliste.csv')
headerlist = list(df1['Spaltennamen'].values.tolist())

#Replace Histstrg Character
Zuordnunghiststrg = {
    '-': '0',
    's': '1',
    'h': '2',
    'a': '3',
    'd': '4',
    'f': '5',
    'r': '6',
    'c': '7',
    'g': '8',
    't': '9',
    'w': '10',
    'i': '11',
    'q': '12',
    'S': '13',
    'H': '14',
    'A': '15',
    'D': '16',
    'F': '17',
    'R': '18',
    'C': '19',
    'G': '20',
    'T': '21',
    'W': '22',
    'I': '23',
    'Q': '24',
    '^': '25',
}

#Port-Frequency-PortOrig
Zuordnungportorig = {
    '-': '0',
    's': '1',
    'h': '2',
    'a': '3',
    'd': '4',
    'f': '5',
    'r': '6',
    'c': '7',
    'g': '8',
    't': '9',
    'w': '10',
    'i': '11',
    'q': '12',
    'S': '13',
    'H': '14',
    'A': '15',
    'D': '16',
    'F': '17',
    'R': '18',
    'C': '19',
    'G': '20',
    'T': '21',
    'W': '22',
    'I': '23',
    'Q': '24',
    '^': '25',
}

#Port-Frequency-PortOrig
Zuordnungportresp = {
    '-': '0',
    's': '1',
    'h': '2',
    'a': '3',
    'd': '4',
    'f': '5',
    'r': '6',
    'c': '7',
    'g': '8',
    't': '9',
    'w': '10',
    'i': '11',
    'q': '12',
    'S': '13',
    'H': '14',
    'A': '15',
    'D': '16',
    'F': '17',
    'R': '18',
    'C': '19',
    'G': '20',
    'T': '21',
    'W': '22',
    'I': '23',
    'Q': '24',
    '^': '25',
}


#######

# Load data
files = glob.glob(path1 + "/*.csv")
for counter, file in enumerate(files):
    df2 = pd.read_csv(
        file,
        sep=";",
        header=0,
        usecols=[2,4, 5, 6, 7, 8, 9, 10, 11],
        dtype=str,
        engine='python',
    )

    #Treffer-column - binary encoding 
    df2['treffer'] = df2['treffer'].str.replace('True', '1', regex=False)
    df2['treffer'] = df2['treffer'].str.replace('False', '0', regex=False)

    # take care of "-"
    df2.replace(to_replace='-', value='0', inplace=True)

    #Drop histstrg Rows with max length
    df2.drop(df2.loc[df2['histstrg'] == 'RhRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR'].index, inplace=True)
    df2.drop(df2.loc[df2['histstrg'] == 'ShAdtDFafRRRRRRRRRRRRRRRRRR'].index, inplace=True)


    # build dummy columns
    df2 = pd.get_dummies(df2, columns=['proto','connstate'])

    # Insert columns for Histstrg 
    df2['histstrg_0'] = '0'
    df2['histstrg_1'] = '0'
    df2['histstrg_2'] = '0'
    df2['histstrg_3'] = '0'
    df2['histstrg_4'] = '0'
    df2['histstrg_5'] = '0'
    df2['histstrg_6'] = '0'
    df2['histstrg_7'] = '0'
    df2['histstrg_8'] = '0'
    df2['histstrg_9'] = '0'
    df2['histstrg_10'] = '0'
    df2['histstrg_11'] = '0'
    df2['histstrg_12'] = '0'
    df2['histstrg_13'] = '0'
    df2['histstrg_14'] = '0'
    df2['histstrg_15'] = '0'
    df2['histstrg_16'] = '0'
    df2['histstrg_17'] = '0'
    df2['histstrg_18'] = '0'
    df2['histstrg_19'] = '0'
    df2['histstrg_20'] = '0'


    for x, row in df2.iterrows():
        for a, i in enumerate(list(row['histstrg'])):
             df2.at[x,'histstrg_' + str(a)] = df2.at[x,'histstrg_' + str(a)].replace('0',i)

    #Replace Histrg Characters
    for old, new in Zuordnunghiststrg.items():
        df2['histstrg_0'] = df2['histstrg_0'].str.replace(old, new, regex=False)
        df2['histstrg_1'] = df2['histstrg_1'].str.replace(old, new, regex=False)
        df2['histstrg_2'] = df2['histstrg_2'].str.replace(old, new, regex=False)
        df2['histstrg_3'] = df2['histstrg_3'].str.replace(old, new, regex=False)
        df2['histstrg_4'] = df2['histstrg_4'].str.replace(old, new, regex=False)
        df2['histstrg_5'] = df2['histstrg_5'].str.replace(old, new, regex=False)
        df2['histstrg_6'] = df2['histstrg_6'].str.replace(old, new, regex=False)
        df2['histstrg_7'] = df2['histstrg_7'].str.replace(old, new, regex=False)
        df2['histstrg_8'] = df2['histstrg_8'].str.replace(old, new, regex=False)
        df2['histstrg_9'] = df2['histstrg_9'].str.replace(old, new, regex=False)
        df2['histstrg_10'] = df2['histstrg_10'].str.replace(old, new, regex=False)
        df2['histstrg_11'] = df2['histstrg_11'].str.replace(old, new, regex=False)
        df2['histstrg_12'] = df2['histstrg_12'].str.replace(old, new, regex=False)
        df2['histstrg_13'] = df2['histstrg_13'].str.replace(old, new, regex=False)
        df2['histstrg_14'] = df2['histstrg_14'].str.replace(old, new, regex=False)
        df2['histstrg_15'] = df2['histstrg_15'].str.replace(old, new, regex=False)
        df2['histstrg_16'] = df2['histstrg_16'].str.replace(old, new, regex=False)
        df2['histstrg_17'] = df2['histstrg_17'].str.replace(old, new, regex=False)
        df2['histstrg_18'] = df2['histstrg_18'].str.replace(old, new, regex=False)
        df2['histstrg_19'] = df2['histstrg_19'].str.replace(old, new, regex=False)
        df2['histstrg_20'] = df2['histstrg_20'].str.replace(old, new, regex=False)

    #Del Histstrg Column
    del df2["histstrg"]

    #add new table
    df2 = df2.reindex(columns=headerlist, fill_value=0)

    #Output single files
    df2.to_csv(path2 + str(counter) + ".csv", sep=";", index=False)

    # Merge to one big file
    df_all_rows = pd.concat([df2, df_all_rows], ignore_index=True)

    # set df2 free
    df2 = ''
    del df2

#Output Merge
df_all_rows.to_csv("D:/Final/allrows.csv", sep=";", index=False)
df_all_rows.to_csv("D:/Final/allrows.txt", sep=";", index=False)

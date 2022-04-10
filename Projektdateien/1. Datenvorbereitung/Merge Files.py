import glob
import os
import numpy as np
import pandas as pd
path1 = 'D:/Ergebnis-droprows/'
df_all_rows = pd.DataFrame()


#Daten Laden
files = glob.glob(path1 + "/*.csv")
for counter, file in enumerate(files):
      df2 = pd.read_csv(
         file,
         sep=";",
         header=0,
         usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
         dtype=str,
         engine = 'python',
         )

# Merge zu einem Dataframe
      df_all_rows = pd.concat([df2, df_all_rows], ignore_index=True)


#df2 freigeben
      df2 = ''
      del df2

df_all_rows = df_all_rows.reset_index(drop=True)

print(len(df_all_rows.index))

count_index = len(df_all_rows.index)

f = open('D:/AllRows.csv', "w")
f.write(str(count_index))
f.close()
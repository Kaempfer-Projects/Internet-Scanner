import glob
import os
import numpy as np
import pandas as pd
UniqueListe1 = []
UniqueListe2 = []
UniqueListe3 = []
UniqueListe4 = []
UniqueListe5 = []
path1 ='D:/Frequency/'
path2 ='D:/Ergebnis-mitTreffer/Ergebnis/'

#Daten Laden
files = glob.glob(path2 + "/*.csv")
for counter, file in enumerate(files):
      df2 = pd.read_csv(
         file,
         header=0,
         dtype=str,
         sep=";",
         engine = 'python',
         )
      #Frequency
      #print(df2.portorig.value_counts())

      #DropRows Abspeichern
      df2.portorig.value_counts().to_csv(path1 + str(counter) + ".csv", sep=";", index=False)

      #Unique Liste von Spalten aufbauen
      UniqueListe1.extend(df2['ipresp'].drop_duplicates().tolist())
      UniqueListe2.extend(df2['portresp'].drop_duplicates().tolist())
      UniqueListe3.extend(df2['proto'].drop_duplicates().tolist())
      UniqueListe4.extend(df2['portorig'].drop_duplicates().tolist())
      UniqueListe5.extend(df2['iporig'].drop_duplicates().tolist())


      #df2 freigeben
      df2 = ''
      del df2

#In Uniquen Listen doppelte Werte entfernen
UniqueListe1 = list(set(UniqueListe1))
UniqueListe2 = list(set(UniqueListe2))
UniqueListe3 = list(set(UniqueListe3))
UniqueListe4 = list(set(UniqueListe4))
UniqueListe5 = list(set(UniqueListe5))


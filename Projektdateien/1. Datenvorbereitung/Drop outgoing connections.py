import glob
import os
import numpy as np
import pandas as pd
path1 = 'D:/Richtig/input/'
path2 = 'D:/Richtig/output/'

#Load data
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

      #Droprows where Ipresp not (130.185.105.90 oder 130.185.105.2
      df2.drop(df2.loc[(df2['ipresp'] != '130.185.105.90') & (df2['ipresp'] != '130.185.105.2')].index, inplace=True)


      #Output
      df2.to_csv(path2 + str(counter) + ".csv", sep=";", index=False)

      #set df2 free
      df2 = ''
      del df2

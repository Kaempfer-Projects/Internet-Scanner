import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob as glob
from mlxtend.preprocessing import minmax_scaling
import random


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

path2 = 'D:/PartsvonKodierung12000/'
df = pd.DataFrame()
df_minority = pd.DataFrame()
df_majority = pd.DataFrame()
df_new = pd.DataFrame()

#Input
# Daten Laden
files = glob.glob(path2 + "/*.csv")
for counter, file in enumerate(files):
    df2 = pd.read_csv(
        file,
        header=0,
        dtype=float,
        sep=";",
        engine='python',
    )
    # Merge zu einem Dataframe
    df = pd.concat([df2, df], ignore_index=True)

    # df2 freigeben
    df2 = ''
    del df2

df = df.reset_index(drop=True)

col = df.pop("treffer")
df.insert(41, "treffer", col)

df.portorig = minmax_scaling(df, columns=['portorig'])
df.portresp = minmax_scaling(df, columns=['portresp'])
df.duration = minmax_scaling(df, columns=['duration'])
df.origbytes = minmax_scaling(df, columns=['origbytes'])
df.respbytes = minmax_scaling(df, columns=['respbytes'])

df.histstrg_0 = minmax_scaling(df, columns=['histstrg_0'])
df.histstrg_1 = minmax_scaling(df, columns=['histstrg_1'])
df.histstrg_2 = minmax_scaling(df, columns=['histstrg_2'])
df.histstrg_3 = minmax_scaling(df, columns=['histstrg_3'])
df.histstrg_4 = minmax_scaling(df, columns=['histstrg_4'])
df.histstrg_5 = minmax_scaling(df, columns=['histstrg_5'])
df.histstrg_6 = minmax_scaling(df, columns=['histstrg_6'])
df.histstrg_7 = minmax_scaling(df, columns=['histstrg_7'])
df.histstrg_8 = minmax_scaling(df, columns=['histstrg_8'])
df.histstrg_9 = minmax_scaling(df, columns=['histstrg_9'])
df.histstrg_10 = minmax_scaling(df, columns=['histstrg_10'])
df.histstrg_11 = minmax_scaling(df, columns=['histstrg_11'])
df.histstrg_12 = minmax_scaling(df, columns=['histstrg_12'])
df.histstrg_13 = minmax_scaling(df, columns=['histstrg_13'])
df.histstrg_14 = minmax_scaling(df, columns=['histstrg_14'])
df.histstrg_15 = minmax_scaling(df, columns=['histstrg_15'])
df.histstrg_16 = minmax_scaling(df, columns=['histstrg_16'])
df.histstrg_17 = minmax_scaling(df, columns=['histstrg_17'])
df.histstrg_18 = minmax_scaling(df, columns=['histstrg_18'])
df.histstrg_19 = minmax_scaling(df, columns=['histstrg_19'])
df.histstrg_20 = minmax_scaling(df, columns=['histstrg_20'])

df.fillna(0, inplace = True)


df_majority = df[df['treffer'] == 0]
df_minority = df[df['treffer'] == 1]

#df1  speichern
df_majority.to_csv("D:/Undersampling12000/NichtInternetscanner.csv", sep=";", index=False)

#df2  speichern
df_minority.to_csv("D:/Undersampling12000/Internetscanner.csv", sep=";", index=False)


##sample function

### Now, downsamples majority labels equal to the number of samples in the minority class
df_majority = df_majority.sample(len(df_minority), random_state=0)

### concat the majority and minority dataframes
df_new = pd.concat([df_majority,df_minority])

## Shuffle the dataset to prevent the model from getting biased by similar samples
df_new = df_new.sample(frac=1, random_state=0)

#output
df_new.to_csv("D:/Undersampling12000/balanced.csv", sep=";", index=False)


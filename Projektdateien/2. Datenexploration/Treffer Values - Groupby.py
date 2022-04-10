import glob
import os
from typing import List, Union

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

path = 'D:/Final/merge.csv'

WertePortOrig = []
WertePortResp = []
WerteTreffer = []
WerteProto = []
WerteConnstate = []
WerteIpResp = []
WerteOrigBytes = []
WerteRespBytes = []

total_count = pd.Series([], dtype=int)

df2 = pd.read_csv(
    path,
    header=0,
    dtype=str,
    sep=";",
    engine='python',
    )

count_series1 = df2.groupby('treffer')['portorig'].value_counts(ascending = True).unstack(level=0).fillna(0)
count_series2 = df2.groupby('treffer')['portresp'].value_counts(ascending = True).unstack(level=0).fillna(0)
count_series3 = df2['treffer'].value_counts(ascending=True)
count_series4 = df2.groupby('treffer')['proto'].value_counts(ascending = True)
count_series5 = df2.groupby('treffer')['connstate'].value_counts(ascending = True).unstack(level=0).fillna(0)
count_series6 = df2.groupby('treffer')['ipresp'].value_counts(ascending = True).unstack(level=0).fillna(0)
count_series7 = df2.groupby('treffer')['origbytes'].value_counts(ascending = True).unstack(level=0).fillna(0)
count_series8 = df2.groupby('treffer')['respbytes'].value_counts(ascending = True).unstack(level=0).fillna(0)


#Output
count_series1.to_csv('D:/Results/WertePortOrig.csv', sep=";")
count_series2.to_csv('D:/Results/WertePortResp.csv', sep=";")
count_series3.to_csv('D:/Results/WerteTreffer.csv', sep=";")
count_series4.to_csv('D:/Results/WerteProto.csv', sep=";")
count_series5.to_csv('D:/Results/WerteConnstate.csv', sep=";")
count_series6.to_csv('D:/Results/WerteIpResp.csv', sep=";")
count_series7.to_csv('D:/Results/WerteOrigBytes.csv', sep=";")
count_series8.to_csv('D:/Results/WerteRespBytes.csv', sep=";")



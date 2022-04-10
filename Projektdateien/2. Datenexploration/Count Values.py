import glob
import os
import numpy as np
import pandas as pd

path2 = 'D:/Ergebnis-droprows/'

WertePortOrig = []
WertePortResp = []
WerteTreffer = []
WerteProto = []
WerteConnstate = []
WerteIpResp = []
WerteOrigBytes = []
WerteRespBytes = []
AnzahlZeilen = []

total_counts1 = pd.Series([], dtype=int)
total_counts2 = pd.Series([], dtype=int)
total_counts3 = pd.Series([], dtype=int)
total_counts4 = pd.Series([], dtype=int)
total_counts5 = pd.Series([], dtype=int)
total_counts6 = pd.Series([], dtype=int)
total_counts7 = pd.Series([], dtype=int)
total_counts8 = pd.Series([], dtype=int)


# Daten Laden
files = glob.glob(path2 + "/*.csv")
for counter, file in enumerate(files):
    df2 = pd.read_csv(
        file,
        header=0,
        dtype=str,
        sep=";",
        engine='python',
    )

    count_series1 = df2['portorig'].value_counts(ascending = True)
    WertePortOrig.append(count_series1)

    count_series2 = df2['portresp'].value_counts(ascending=True)
    WertePortResp.append(count_series2)

    count_series3 = df2['treffer'].value_counts(ascending=True)
    WerteTreffer.append(count_series3)

    count_series4 = df2['proto'].value_counts(ascending=True)
    WerteProto.append(count_series4)

    count_series5 = df2['connstate'].value_counts(ascending=True)
    WerteConnstate.append(count_series5)

    count_series6 = df2['ipresp'].value_counts(ascending=True)
    WerteIpResp.append(count_series6)

    count_series7 = df2['origbytes'].value_counts(ascending=True)
    WerteOrigBytes.append(count_series7)

    count_series8 = df2['respbytes'].value_counts(ascending=True)
    WerteRespBytes.append(count_series8)

    count_index = len(df2.index)
    AnzahlZeilen.append(count_index)

    # df2 freigeben
    df2 = ''
    del df2

for ser in WertePortOrig:
    total_counts1 : int = total_counts1.add(ser, fill_value=0)

for ser in WertePortResp:
    total_counts2 = total_counts2.add(ser, fill_value=0)

for ser in WerteTreffer:
    total_counts3 = total_counts3.add(ser, fill_value=0)

for ser in WerteProto:
    total_counts4 = total_counts4.add(ser, fill_value=0)

for ser in WerteConnstate:
    total_counts5 = total_counts5.add(ser, fill_value=0)

for ser in WerteIpResp:
    total_counts6 = total_counts6.add(ser, fill_value=0)

for ser in WerteOrigBytes:
    total_counts7 = total_counts7.add(ser, fill_value=0)

for ser in WerteRespBytes:
    total_counts8 = total_counts8.add(ser, fill_value=0)

totalrows = sum(AnzahlZeilen)

#Output
total_counts1.to_csv('D:/Results/WertePortOrig.csv', sep=";")
total_counts2.to_csv('D:/Results/WertePortResp.csv', sep=";")
total_counts3.to_csv('D:/Results/WerteTreffer.csv', sep=";")
total_counts4.to_csv('D:/Results/WerteProto.csv', sep=";")
total_counts5.to_csv('D:/Results/WerteConnstate.csv', sep=";")
total_counts6.to_csv('D:/Results/WerteIpResp.csv', sep=";")
total_counts7.to_csv('D:/Results/WerteOrigBytes.csv', sep=";")
total_counts8.to_csv('D:/Results/WerteRespBytes.csv', sep=";")

f = open('D:/Results/totalrows.csv', "w")
f.write(str(totalrows))
f.close()


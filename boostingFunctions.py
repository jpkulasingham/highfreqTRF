from eelbrain import *
import mne
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import scipy.io
import pdb
from scipy import signal
import time
import socket
import warnings

mne.set_log_level('ERROR')
configure(n_workers = 8)

time1 = time.time()


def permutePred(ds: Dataset, predstr: str, nperm: int =2):

    for i in range(0,nperm):
        xnd = ds[predstr].copy()
        if ds[predstr].has_case:
            for j in range(0,len(ds[predstr])):
                aa = np.roll(ds[predstr][j].x, int((i+1) * len(xnd[j]) / (nperm+1)))
                xnd[j] = NDVar(aa,dims=xnd[j].dims,name=predstr+'_p'+str(i))
        else:
            xnd.x = np.roll(xnd.x, int((i + 1) * len(xnd) / (nperm + 1)))
        xnd.name = predstr+'_p'+str(i)
        ds[xnd.name] = xnd
    return ds


def boostWrap(ds: Dataset, predstr: str, outputFolder: str, subjectF: str, outstr: str, bFrac: list, basislen: float, partitions: int, rstr: str = 'source', permflag: bool = True):
    print(outputFolder)

    resM = boosting(ds[rstr], ds[predstr], bFrac[0], bFrac[1], basis=basislen, partitions=partitions)
    print(f'{subjectF} {predstr} {outstr} Model r max = {resM.r.max():.4g}\n')
    print(f'{subjectF} {predstr} {outstr} Model max = {resM.h.max():.4g}\n')

    with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
        f.write(f'{subjectF} {predstr} {outstr} Model r max = {resM.r.max():.4g}\n')
        f.write(f'{subjectF} {predstr} {outstr} Model max = {resM.h.max():.4g}\n')

    if permflag:
        resN = boosting(ds[rstr], ds[predstr + '_p0'], bFrac[0], bFrac[1], basis=basislen, partitions=partitions)
        print(f'{subjectF} {predstr} {outstr} Noise r max = {resN.r.max():.4g}\n')
        print(f'{subjectF} {predstr} {outstr} Noise max = {resN.h.max():.4g}\n')

        with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
            f.write(f'{subjectF} {predstr} {outstr} Noise r max = {resN.r.max():.4g}\n')
            f.write(f'{subjectF} {predstr} {outstr} Noise max = {resN.h.max():.4g}\n')

        save.pickle([resM,resN],f'{outputFolder}/Source/Pickle/{subjectF}_{predstr}{outstr}.pkl')
    else:
        save.pickle(resM,f'{outputFolder}/Source/Pickle/{subjectF}_{predstr}{outstr}.pkl')

    return resM, resN


def boostWrap_multperm(ds: Dataset, predstr: str, outputFolder: str, subjectF: str, outstr: str, bFrac: list, basislen: float, partitions: int, nperm: int = 1, rstr: str = 'source'):
    print(outputFolder)

    resM = boosting(ds[rstr], ds[predstr], bFrac[0], bFrac[1], basis=basislen, partitions=partitions)
    print(f'{subjectF} {predstr} {outstr} Model r max = {resM.r.max():.4g}\n')
    print(f'{subjectF} {predstr} {outstr} Model max = {resM.h.max():.4g}\n')

    with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
        f.write(f'{subjectF} {predstr} {outstr} Model r max = {resM.r.max():.4g}\n')
        f.write(f'{subjectF} {predstr} {outstr} Model max = {resM.h.max():.4g}\n')

    permds = []
    for i in range(0,nperm):
        resN = boosting(ds[rstr], ds[predstr + '_p' + str(i)], bFrac[0], bFrac[1], basis=basislen, partitions=partitions)
        print(f'{subjectF} {predstr} {outstr} Noise {i} r max = {resN.r.max():.4g}\n')
        print(f'{subjectF} {predstr} {outstr} Noise {i} max = {resN.h.max():.4g}\n')

        with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
            f.write(f'{subjectF} {predstr} {outstr} {i} Noise r max = {resN.r.max():.4g}\n')
            f.write(f'{subjectF} {predstr} {outstr} {i} Noise max = {resN.h.max():.4g}\n')
        permds.append(resN)

    save.pickle([resM,permds],f'{outputFolder}/Source/Pickle/{subjectF}_{predstr}{outstr}.pkl')
    return resM, permds


def boostWrap_multpred_multperm(ds: Dataset, predlist: list, outputFolder: str, subjectF: str, outstr: str, bFrac: list, basislen: float, partitions: int, nperm: int = 1, rstr: str = 'source', selstop = False, delta = 0.005):
    print(outputFolder)


    if selstop:
        resM = boosting(ds[rstr], [ds[pred] for pred in predlist], bFrac[0], bFrac[1], basis=basislen, partitions=partitions, selective_stopping=selstop, delta=delta)
    else:
        resM = boosting(ds[rstr], [ds[pred] for pred in predlist], bFrac[0], bFrac[1], basis=basislen, partitions=partitions, delta = delta)

    print(f'{subjectF} {"-".join(predlist)} {outstr} Model r max = {resM.r.max():.4g}\n')
    with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
        f.write(f'{subjectF} {"-".join(predlist)} {outstr} Model r max = {resM.r.max():.4g}\n')

    for i in range(0,len(predlist)):
        print(f'{subjectF} {predlist[i]} {outstr} Model max = {resM.h[i].max():.4g}\n')
        with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
            f.write(f'{subjectF} {predlist[i]} {outstr} Model max = {resM.h[i].max():.4g}\n')

    permds = []
    for p in range(0,nperm):
        if selstop:
            resN = boosting(ds[rstr], [ds[pred+'_p'+str(p)] for pred in predlist], bFrac[0], bFrac[1], basis=basislen,
                            partitions=partitions, selective_stopping=selstop,delta = delta)
        else:
            resN = boosting(ds[rstr], [ds[pred+'_p'+str(p)] for pred in predlist], bFrac[0], bFrac[1], basis=basislen,
                            partitions=partitions,delta=delta)

        print(f'{subjectF} {"-".join(predlist)} {outstr} Noise {p} r max = {resN.r.max():.4g}\n')
        with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
            f.write(f'{subjectF} {"-".join(predlist)} {outstr} Noise {p} r max = {resN.r.max():.4g}\n')

        for i in range(0, len(predlist)):
            print(f'{subjectF} {predlist[i]} {outstr} Noise {p} max = {resN.h[i].max():.4g}\n')
            with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
                f.write(f'{subjectF} {predlist[i]} {outstr} Noise {p} max = {resN.h[i].max():.4g}\n')

        permds.append(resN)

    save.pickle([resM,permds],f'{outputFolder}/Source/Pickle/{subjectF}_{"-".join(predlist)}{outstr}.pkl')
    return resM, permds
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

    # for i in range(0, len(pred)):
    #     for j=1:nperm
    #     xP = pred[i]
    #     xPn = xP.copy()
    #     xPn2 = xPn.copy()
    #     nn = int(len(xPn2.x) / 2)
    #     nn2 = len(xPn2.x)
    #     xPn2.x[0:nn] = xPn.x[nn:nn2]
    #     xPn2.x[nn:nn2] = xPn.x[0:nn]
    #     xPn = xPn2
    #     ppred[i] = xPn.x

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



def boostWrap_mix(ds: Dataset, predstr: str, outputFolder: str, subjectF: str, outstr: str, bFrac: list, basislen: float, partitions: int, rstr: str = 'source', permflag: bool = True):
    print(outputFolder)

    resM = boosting(ds[rstr], [ds[predstr + '_fg'], ds[predstr + '_bg']], bFrac[0], bFrac[1], basis=basislen, partitions=partitions)
    print(f'{subjectF} {predstr} {outstr} Model r max = {resM.r.max():.4g}\n')
    print(f'{subjectF} {predstr} {outstr} Model fg max = {resM.h[0].max():.4g}\n')
    print(f'{subjectF} {predstr} {outstr} Model bg max = {resM.h[1].max():.4g}\n')

    with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
        f.write(f'{subjectF} {predstr} {outstr} Model r max = {resM.r.max():.4g}\n')
        f.write(f'{subjectF} {predstr} {outstr} Model fg max = {resM.h[0].max():.4g}\n')
        f.write(f'{subjectF} {predstr} {outstr} Model bg max = {resM.h[1].max():.4g}\n')

    if permflag:
        resN = boosting(ds[rstr], [ds[predstr + '_fg_p'], ds[predstr + '_bg_p']], bFrac[0], bFrac[1], basis=basislen, partitions=partitions)
        print(f'{subjectF} {predstr} {outstr} Noise r max = {resN.r.max():.4g}\n')
        print(f'{subjectF} {predstr} {outstr} Noise fg max = {resN.h[0].max():.4g}\n')
        print(f'{subjectF} {predstr} {outstr} Noise bg max = {resN.h[1].max():.4g}\n')

        with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
            f.write(f'{subjectF} {predstr} {outstr} Noise r max = {resN.r.max():.4g}\n')
            f.write(f'{subjectF} {predstr} {outstr} Noise fg max = {resN.h[0].max():.4g}\n')
            f.write(f'{subjectF} {predstr} {outstr} Noise bg max = {resN.h[1].max():.4g}\n')

        save.pickle([resM,resN],f'{outputFolder}/Source/Pickle/{subjectF}_{predstr}_{outstr}.pkl')
    else:
        save.pickle(resM,f'{outputFolder}/Source/Pickle/{subjectF}_{predstr}_{outstr}.pkl')

    return resM, resN




def boostWrap_mix_multperm(ds: Dataset, predstr: str, outputFolder: str, subjectF: str, outstr: str, bFrac: list, basislen: float, partitions: int, nperm: int = 1, rstr: str = 'source'):
    print(outputFolder)

    resM = boosting(ds[rstr], [ds[predstr + '_fg'], ds[predstr + '_bg']], bFrac[0], bFrac[1], basis=basislen, partitions=partitions)
    print(f'{subjectF} {predstr} {outstr} Model r max = {resM.r.max():.4g}\n')
    print(f'{subjectF} {predstr} {outstr} Model fg max = {resM.h[0].max():.4g}\n')
    print(f'{subjectF} {predstr} {outstr} Model bg max = {resM.h[1].max():.4g}\n')

    with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
        f.write(f'{subjectF} {predstr} {outstr} Model r max = {resM.r.max():.4g}\n')
        f.write(f'{subjectF} {predstr} {outstr} Model fg max = {resM.h[0].max():.4g}\n')
        f.write(f'{subjectF} {predstr} {outstr} Model bg max = {resM.h[1].max():.4g}\n')

    permds = []
    for i in range(0, nperm):
        resN = boosting(ds[rstr], [ds[predstr + '_fg_p' + str(i)], ds[predstr + '_bg_p' + str(i)]], bFrac[0], bFrac[1], basis=basislen, partitions=partitions)
        print(f'{subjectF} {predstr} {outstr} Noise {i} r max = {resN.r.max():.4g}\n')
        print(f'{subjectF} {predstr} {outstr} Noise {i} fg max = {resN.h[0].max():.4g}\n')
        print(f'{subjectF} {predstr} {outstr} Noise {i} bg max = {resN.h[1].max():.4g}\n')

        with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
            f.write(f'{subjectF} {predstr} {outstr} Noise {i} r max = {resN.r.max():.4g}\n')
            f.write(f'{subjectF} {predstr} {outstr} Noise {i} fg max = {resN.h[0].max():.4g}\n')
            f.write(f'{subjectF} {predstr} {outstr} Noise {i} bg max = {resN.h[1].max():.4g}\n')
        permds.append(resN)

    save.pickle([resM,permds],f'{outputFolder}/Source/Pickle/{subjectF}_{predstr}{outstr}.pkl')
    return resM, permds


def boostWrap_competeAB_multperm(ds: Dataset, predlist: list, outputFolder: str, subjectF: str, outstr: str, bFrac: list, basislen: float, partitions: int, nperm: int = 1, rstr: str = 'source'):
    print(outputFolder)

    resM = boosting(ds[rstr], [ds[predlist[0]],ds[predlist[1]]], bFrac[0], bFrac[1], basis=basislen, partitions=partitions)
    print(f'{subjectF} {predlist[0]} {predlist[1]} {outstr} Model r max = {resM.r.max():.4g}\n')
    print(f'{subjectF} {predlist[0]} {outstr} Model max = {resM.h[0].max():.4g}\n')
    print(f'{subjectF} {predlist[1]} {outstr} Model max = {resM.h[1].max():.4g}\n')

    with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
        f.write(f'{subjectF} {predlist[0]} {predlist[1]} {outstr} Model r max = {resM.r.max():.4g}\n')
        f.write(f'{subjectF} {predlist[0]} {outstr} Model max = {resM.h[0].max():.4g}\n')
        f.write(f'{subjectF} {predlist[1]} {outstr} Model max = {resM.h[1].max():.4g}\n')

    permdsAs = []
    permdsBs = []
    for i in range(0, nperm):

        resN = boosting(ds[rstr], [ds[predlist[0] + '_p' + str(i)], ds[predlist[1]]], bFrac[0], bFrac[1],
                        basis=basislen, partitions=partitions)
        print(f'{subjectF} {predlist[0]}_p{i} {predlist[1]} {outstr} Noise {i} r max = {resN.r.max():.4g}\n')
        print(f'{subjectF} {predlist[0]}_p{i} {outstr} Model max = {resN.h[0].max():.4g}\n')
        print(f'{subjectF} {predlist[1]} {outstr} Noise {i} max = {resN.h[1].max():.4g}\n')

        with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
            f.write(f'{subjectF} {predlist[0]}_p{i} {predlist[1]}_p{i} {outstr} Noise {i} r max = {resN.r.max():.4g}\n')
            f.write(f'{subjectF} {predlist[0]}_p{i} {outstr} Model max = {resN.h[0].max():.4g}\n')
            f.write(f'{subjectF} {predlist[1]} {outstr} Noise {i} max = {resN.h[1].max():.4g}\n')

        permdsAs.append(resN)

        resN = boosting(ds[rstr], [ds[predlist[0]], ds[predlist[1] + '_p' + str(i)]], bFrac[0], bFrac[1],
                        basis=basislen, partitions=partitions)
        print(f'{subjectF} {predlist[0]} {predlist[1]}_p{i} {outstr} Noise {i} r max = {resN.r.max():.4g}\n')
        print(f'{subjectF} {predlist[0]} {outstr} Model max = {resN.h[0].max():.4g}\n')
        print(f'{subjectF} {predlist[1]}_p{i} {outstr} Noise {i} max = {resN.h[1].max():.4g}\n')

        with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
            f.write(f'{subjectF} {predlist[0]} {predlist[1]}_p{i} {outstr} Noise {i} r max = {resN.r.max():.4g}\n')
            f.write(f'{subjectF} {predlist[0]} {outstr} Model max = {resN.h[0].max():.4g}\n')
            f.write(f'{subjectF} {predlist[1]}_p{i} {outstr} Noise {i} max = {resN.h[1].max():.4g}\n')

        permdsBs.append(resN)



    save.pickle([resM,permdsAs,permdsBs],f'{outputFolder}/Source/Pickle/{subjectF}_{predlist[0]}{predlist[1]}{outstr}.pkl')
    return resM, permdsAs, permdsBs

# def boostWrap_single(ds: Dataset, pred: str, outputFolder: str, subjectF: str, outstr: str, bFrac: list, basislen: float, partitions: int, nperm: int = 1, rstr: str = 'source'):
#
#     resM = boosting(ds[rstr], ds[pred], bFrac[0], bFrac[1], basis=basislen, partitions=partitions)
#     print(f'{subjectF} {pred} {outstr} Model r max = {resM.r.max():.4g}\n')
#     print(f'{subjectF} {pred} {outstr} Model max = {resM.h.max():.4g}\n')
#
#     with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
#         f.write(f'{subjectF} {pred} {outstr} Model r max = {resM.r.max():.4g}\n')
#         f.write(f'{subjectF} {pred} {outstr} Model max = {resM.h.max():.4g}\n')
#
#     permds = []
#     for i in range(0, nperm):
#         resN = boosting(ds[rstr], [ds[predlist[0]], ds[predlist[1] + '_p' + str(i)]], bFrac[0], bFrac[1],
#                         basis=basislen, partitions=partitions)
#         print(f'{subjectF} {predlist[0]} {predlist[1]}_p{i} {outstr} Noise {i} r max = {resN.r.max():.4g}\n')
#         print(f'{subjectF} {predlist[0]} {outstr} Model max = {resN.h[0].max():.4g}\n')
#         print(f'{subjectF} {predlist[1]}_p{i} {outstr} Noise {i} max = {resN.h[1].max():.4g}\n')
#
#         with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
#             f.write(f'{subjectF} {predlist[0]} {predlist[1]}_p{i} {outstr} Noise {i} r max = {resN.r.max():.4g}\n')
#             f.write(f'{subjectF} {predlist[0]} {outstr} Model max = {resN.h[0].max():.4g}\n')
#             f.write(f'{subjectF} {predlist[1]}_p{i} {outstr} Noise {i} max = {resN.h[1].max():.4g}\n')
#
#         permds.append(resN)
#
#     save.pickle([resM,permds],f'{outputFolder}/Source/Pickle/{subjectF}_{predlist[0]}{predlist[1]}{outstr}.pkl')
#     return resM, permds
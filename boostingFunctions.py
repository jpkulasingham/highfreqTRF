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


def permutePred(ds: Dataset, predstr: str, nperm: int = 2):
	"""Permutes the given predictor""" 
    for np in range(0,nperm):
        xnd = ds[predstr].copy()
        if ds[predstr].has_case:
            for j in range(0,len(ds[predstr])):
                aa = np.roll(ds[predstr][j].x, int((np+1) * len(xnd[j]) / (nperm+1)))
                xnd[j] = NDVar(aa,dims=xnd[j].dims,name=predstr+'_p'+str(np))
        else:
            xnd.x = np.roll(xnd.x, int((np + 1) * len(xnd) / (nperm + 1)))
        xnd.name = predstr+'_p'+str(np)
        ds[xnd.name] = xnd
    return ds


def boostWrap(ds: Dataset, predstr: str, output_folder: str, subject: str, outstr: str, bFrac: list, basislen: float, partitions: int, rstr: str = 'source', permflag: bool = True):
    """Boosts the given predictor. If permflag == True, boosts permuted model also"""
    print(output_folder)
    resM = boosting(ds[rstr], ds[predstr], bFrac[0], bFrac[1], basis=basislen, partitions=partitions)
    _write_res(resM, output_folder, subject, predstr, outstr, modelstr='Model')
    if permflag:
        resN = boosting(ds[rstr], ds[predstr + '_p0'], bFrac[0], bFrac[1], basis=basislen, partitions=partitions)
    	_write_res(resN, output_folder, subject, predstr, outstr, modelstr='Noise')
        save.pickle([resM,resN],f'{output_folder}/Source/Pickle/{subject}_{predstr}{outstr}.pkl')
    else:
        save.pickle(resM,f'{output_folder}/Source/Pickle/{subject}_{predstr}{outstr}.pkl')
    return resM, resN


def boostWrap_multperm(ds: Dataset, predstr: str, output_folder: str, subject: str, outstr: str, bFrac: list, basislen: float, partitions: int, nperm: int = 1, rstr: str = 'source'):
    """Boosts the given predictor, and boosts nperm permuted models"""
    print(output_folder)
    resM = boosting(ds[rstr], ds[predstr], bFrac[0], bFrac[1], basis=basislen, partitions=partitions)
    _write_res(resM, output_folder, subject, predstr, outstr, modelstr='Model')
    permds = []
    for i in range(0,nperm):
        resN = boosting(ds[rstr], ds[predstr + '_p' + str(i)], bFrac[0], bFrac[1], basis=basislen, partitions=partitions)
        _write_res(resN, output_folder, subject, predstr, outstr, modelstr=f'Noise{i}')
        permds.append(resN)
    save.pickle([resM,permds],f'{output_folder}/Source/Pickle/{subject}_{predstr}{outstr}.pkl')
    return resM, permds


def boostWrap_multpred_multperm(ds: Dataset, predlist: list, output_folder: str, subject: str, outstr: str, bFrac: list, basislen: float, partitions: int, nperm: int = 1, rstr: str = 'source', selstop = False, delta = 0.005):
    """Boosts the given predictors, and boosts nperm permuted models"""
    print(output_folder)
    resM = boosting(ds[rstr], [ds[pred] for pred in predlist], bFrac[0], bFrac[1], basis=basislen, partitions=partitions, selective_stopping=selstop, delta=delta)
	_write_res_multpred(resM, output_folder, subject, predlist, outstr, modelstr='Model'):
    permds = []
    for p in range(0,nperm):
        resN = boosting(ds[rstr], [ds[pred+'_p'+str(p)] for pred in predlist], bFrac[0], bFrac[1], basis=basislen,
                            partitions=partitions, selective_stopping=selstop,delta = delta)
		_write_res_multpred(resN, output_folder, subject, predlist, outstr, modelstr=f'Noise{i}'):
        print(f'{subject} {"-".join(predlist)} {outstr} Noise {p} r max = {resN.r.max():.4g}\n')
        permds.append(resN)
    save.pickle([resM,permds],f'{output_folder}/Source/Pickle/{subject}_{"-".join(predlist)}{outstr}.pkl')
    return resM, permds


def _write_res(res: BoostingResult, output_folder: str, subject: str, predstr: str, outstr: str, modelstr: str):
	print(f'{subject} {predstr} {outstr} {modelstr} r max = {res.r.max():.4g}\n')
    print(f'{subject} {predstr} {outstr} {modelstr} max = {res.h.max():.4g}\n')
    with open(f'{output_folder}/Source/boosting.txt', 'a+') as f:
        f.write(f'{subject} {predstr} {outstr} {modelstr} r max = {res.r.max():.4g}\n')
        f.write(f'{subject} {predstr} {outstr} {modelstr} max = {res.h.max():.4g}\n')


def _write_res_multpred(res: BoostingResult, output_folder: str, subject: str, predlist: list, outstr: str, modelstr: str):
    print(f'{subject} {"-".join(predlist)} {outstr} {modelstr} r max = {res.r.max():.4g}\n')
    with open(f'{output_folder}/Source/boosting.txt', 'a+') as f:
        f.write(f'{subject} {"-".join(predlist)} {outstr} {modelstr} r max = {res.r.max():.4g}\n')
    for i in range(0,len(predlist)):
        print(f'{subject} {predlist[i]} {outstr} {modelstr} max = {res.h[i].max():.4g}\n')
        with open(f'{output_folder}/Source/boosting.txt', 'a+') as f:
            f.write(f'{subject} {predlist[i]} {outstr} {modelstr} max = {res.h[i].max():.4g}\n')
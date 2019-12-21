from eelbrain import *
import mne
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import scipy.io
import pdb
from scipy import signal
import scipy
import scipy.integrate
import scipy.stats
import math

data = np.random.rand(5, 6)
time = UTS(0, 6, 6)
ppp = NDVar(data, dims=('case', time))
p = plot.UTS(ppp[0])
p.close()
del data, ppp, time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
from sklearn.decomposition import FastICA, PCA
import pdb
import itertools
from matplotlib import cm
from matplotlib.colors import ListedColormap


configure(n_workers=4)
fs = 1000
NPerm = 10000


niceplotfolder = '/Users/pranjeevan/Documents/HighFreq/niceplots_paper/full_FIR'
sqdFHT = '/Users/pranjeevan/Documents/HAYO/denoiseTones'
sqdFH = '/Users/pranjeevan/Documents/HAYO/DenoiseQuiet'
subjects_dir = '/Users/pranjeevan/Documents/HAYO/sourcespace/mri'

## PAPER
pklFc = '/Users/pranjeevan/Documents/HighFreq/ResultsNew/FIR_Compete_4ms'
pklFv = '/Users/pranjeevan/Documents/HighFreq/ResultsNew/FIR_Volume_pkl_compete_4ms_AST'
pklFl = '/Users/pranjeevan/Documents/HighFreq/ResultsNew/FIR_Compete_10ms_LF_logAVW'
##

pklFv = '/Users/pranjeevan/Documents/HighFreq/ResultsNew2/HFVolFIR_70_300_full'
symfile = '/Users/pranjeevan/Documents/BitBucket/personal/highFreq/sym-vol-7-cortex_brainstem_full.pkl'
symflag = True

mv_ext = '.mp4'

if not os.path.exists(niceplotfolder):
    os.makedirs(niceplotfolder)


ctx_labels = ('bankssts', 'inferiortemporal', 'middletemporal',
              'superiortemporal' , 'transversetemporal')
voi1 = ['Brain-Stem', '3rd-Ventricle']
voi_lat1 = ('Thalamus-Proper', 'VentralDC')
voi1.extend('%s-%s' % fmt for fmt in itertools.product(('Left', 'Right'), voi_lat1))
voiL = [f'ctx-lh-{s}' for s in ctx_labels]
voiR = [f'ctx-rh-{s}' for s in ctx_labels]
cortsub = voiL + voiR
stemsub = voi1

thal = ['Left-Thalamus-Proper','Right-Thalamus-Proper']

aparcs = ['superior', 'transverse', 'inferior', 'middle']
aparcs = [x + 'temporal' for x in aparcs]
aparcs += ['bankssts']
aparcR = [x + '-rh' for x in aparcs]
aparcL = [x + '-lh' for x in aparcs]

magma = cm.get_cmap('YlOrRd', 12)
cutoff = 135
magma2 = magma(np.linspace(0, 1, 256-cutoff))
magmavals = np.zeros((256,4))
magmavals[:cutoff, :] = [0, 0, 0, 0]
magmavals[cutoff:256,:] = magma2
newcmpGB = ListedColormap(magmavals)
im_ext = '.png'
glassbrain_h = 6
brain_h = 1000
brain_w = 2500


HAYO_all = [ 'R0840', 'R1801', 'R2079', 'R2083', 'R2084', 'R2085', 'R2086', 'R2092', 'R2093', 'R2094', 'R2107', 'R2130',
             'R2135', 'R2145', 'R2148', 'R2153', 'R2154', 'R2185', 'R2196', 'R2197', 'R2201', 'R2210', 'R2217', 'R2223',
             'R2230', 'R2244', 'R2246', 'R2247', 'R2254', 'R2256', 'R2263', 'R2281', 'R2338', 'R2342', 'R2354', 'R2363',
             'R2396', 'R2408', 'R2409', 'R2428']
HAYO = [ 'R0840', 'R1801', 'R2079', 'R2083', 'R2084', 'R2085', 'R2086', 'R2092', 'R2093', 'R2094', 'R2107', 'R2130',
         'R2135', 'R2145', 'R2148', 'R2153', 'R2154', 'R2185', 'R2196', 'R2197', 'R2201', 'R2210', 'R2217', 'R2223',
         'R2230', 'R2244', 'R2246', 'R2247', 'R2254', 'R2256', 'R2263', 'R2281']
NAA = [ 'R2338', 'R2342', 'R2354', 'R2363', 'R2396', 'R2408', 'R2409', 'R2428']
subjectshandedness = {'R0840': 1, 'R1801': 0.67, 'R2079': 1, 'R2083': 1, 'R2084': 0.78, 'R2085': 0.625, 'R2086': 0.6,
                      'R2092': 1, 'R2093': 1, 'R2094': 0.91, 'R2107': 0.16, 'R2130': -0.18, 'R2135': 0.89, 'R2145': 1,
                      'R2148': 1, 'R2153': 0.68, 'R2154': 0.6, 'R2185': 0.67, 'R2196': 1, 'R2197': 1, 'R2201': -1,
                      'R2210': 1, 'R2217': 1, 'R2223': 0.58, 'R2230': 1, 'R2244': 0.9, 'R2246': 0.48, 'R2247': -1,
                      'R2254': 0.86, 'R2256': 1, 'R2263': 0.8, 'R2281': 0.3, 'R2338': 0.25, 'R2342': -0.5, 'R2354': 0.55,
                      'R2363': 1, 'R2396': 0.8, 'R2408': -0.9, 'R2409': 0.85, 'R2428': 1 }
subjects = HAYO_all


def get_right_subj(ds, thr=0.5):
    return ds.sub([i for i in range(len(ds['subject'])) if subjectshandedness[ds['subject'][i]] > thr])


def change_subjects_dir(ds,subjects_dir=subjects_dir):
    for k in ds.keys():
        if type(ds[k])==NDVar:
            if 'source' in ds[k].dimnames:
                ds[k].source.subjects_dir = subjects_dir
    return ds

###########################################################################################


def load_data_HAYO(pklfolder, lstr, sqdfolder=sqdFH):
    filenames = [f for f in os.listdir(f'{pklfolder}/Source/Pickle') if f.endswith(lstr)]
    filenames.sort()
    mat_old = scipy.io.loadmat('HAYOold.mat')
    mat_young_names = mat_old['HAYOnames'].tolist()
    mat_young_flags = mat_old['HAYOolder']
    first_file = load.unpickle(f'{pklfolder}/Source/Pickle/{filenames[0]}')
    npred = len(first_file[0].h) if type(first_file[0].h) is tuple else 1
    nperm = len(first_file[1]) if type(first_file[1]) is list else 1
    
    subjects = [f[:5] for f in filenames]
    young_flags = [int(mat_young_flags[mat_young_names.index(s)]) for s in subjects]
    model_trf = []
    noise_trf = [list() for i in range(3)]    
    for filename in filenames:
        nd = load.unpickle(f'{pklfolder}/Source/Pickle/{filename}')
        model_trf.append(nd[0])
        if nperm>1:
            for j in range(nperm):
                noise_trf[j].append(nd[1][j])
        else:
            noise_trf[0].append(nd[1])
            
    ds = Dataset()
    ds['modeltrf'] = combine(model_trf)
    for j in range(0,nperm):
        ds[f'noisetrf{j}'] = combine(noise_trf[j])
    ds['subject'] = Factor(subjects, name='subject', random=True)
    ds['youngflag'] = Factor(young_flags, name='youngflag')
    ds.info = {'nperm': nperm, 
                'npred': npred, 
                'sqdfolder': sqdfolder,
                'lstr': lstr,
                'pklfolder' : pklfolder
                }
    return ds

def makeNDVarMvN(ds:Dataset, attrstr:str, fsaverage=True):
    nperm = ds.info['nperm']
    npred = 1 if attrstr == 'r' else ds.info['npred']
    attr = getattr(ds['modeltrf'][0], attrstr)
    attrname = [attr.name] if npred == 1 else [a.name for a in attr]
    modelattr = [[] for _ in range(npred)]
    noiseattr = [[[] for _ in range(nperm)] for _ in range(npred)]
 
    noiseformean_abs = [[[] for _ in range(ds.n_cases)] for _ in range(npred)]
    noiseformean_noabs = [[[] for _ in range(ds.n_cases)] for _ in range(npred)]
    for case in range(ds.n_cases):
        if npred > 1:
            for i in range(npred):
                nd = getattr(ds['modeltrf'][case],attrstr)[i]
                nd.source.subjects_dir = subjects_dir
                if fsaverage: nd.source.subject = 'fsaverage'
                modelattr[i].append(nd)
                for j in range(nperm):
                    nd = getattr(ds[f'noisetrf{j}'][case], attrstr)[i]
                    nd.source.subjects_dir = subjects_dir
                    if fsaverage: nd.source.subject = 'fsaverage'
                    noiseattr[i][j].append(nd)
                    noiseformean_noabs[i][case].append(nd)
                    noiseformean_abs[i][case].append(abs(nd.copy()))
        else:
            nd = getattr(ds['modeltrf'][case],attrstr)
            nd.source.subjects_dir = subjects_dir
            if fsaverage: nd.source.subject = 'fsaverage'
            modelattr[0].append(nd)
            for j in range(nperm):
                nd = getattr(ds[f'noisetrf{j}'][case], attrstr)
                nd.source.subjects_dir = subjects_dir
                if fsaverage: nd.source.subject = 'fsaverage'
                noiseattr[0][j].append(nd)
                noiseformean_noabs[0][case].append(nd)
                noiseformean_abs[0][case].append(abs(nd.copy()))

    noisemean_abs = [list() for i in range(npred)]
    noisemean_noabs = [list() for i in range(npred)]
    for i in range(npred):
        ds[f'{attrstr}_{attrname[i]}_model'] = combine(modelattr[i])
        ds[f'{attrstr}_{attrname[i]}_model'].source.subjects_dir = subjects_dir
        for j in range(nperm):
            ds[f'{attrstr}_{attrname[i]}_noise{j}'] = combine(noiseattr[i][j])
            ds[f'{attrstr}_{attrname[i]}_noise{j}'].source.subjects_dir = subjects_dir
        for nm_abs, nm_noabs in zip(noiseformean_abs[i], noiseformean_noabs[i]):
            noisemean_abs[i].append(combine(nm_abs).mean('case'))  # mean across nperm ('case')
            noisemean_noabs[i].append(combine(nm_noabs).mean('case'))
        ds[f'{attrstr}_{attrname[i]}_noise_mean'] = combine(noisemean_abs[i])
        ds[f'{attrstr}_{attrname[i]}_noise_mean'].source.subjects_dir = subjects_dir
        ds[f'{attrstr}_{attrname[i]}_noise_mean_noabs'] = combine(noisemean_noabs[i])
        ds[f'{attrstr}_{attrname[i]}_noise_mean_noabs'].source.subjects_dir = subjects_dir
    ds.info['attrname_'+attrstr] = attrname

    return ds


def nd_smooth(ds, smooth_s=0.005, attrstr='h'):
    npred = ds.info['npred'] if attrstr == 'h' else 1
    for i_m, m in enumerate(['model','noise_mean']):
        for i_a, attr in enumerate(ds.info[f'attrname_{attrstr}']):
            smooth_hh = []
            print(f'{m}:{attr}')
            for i_c in range(ds.n_cases):
                hh = ds[f'{attrstr}_{attr}_{m}'][i_c].copy()
                hh.source.subject = 'fsaverage'
                if 'space' in hh.dimnames:
                    hh = hh.norm('space')
                smooth_hh.append(hh.abs().smooth('source', smooth_s, window='gaussian'))
                print(f'{m}:{attr} {(i_m * npred * ds.n_cases + i_a * ds.n_cases + i_c)/(ds.n_cases * npred * 2):.5f}%',end='\r')
            ds[f'{attrstr}_smooth_{attr}_{m}'] = combine(smooth_hh)
    print('')
    return ds



def xhemi_surf(ds,attrstr,hemi='rh'):
    for attr in ds.info[f'attrname_{attrstr}']:
        attr2 = attrstr+'_smooth_'+attr+'_model'
        print(f'{attrstr} lateralization: {attr2} {hemi}')

        [ds[f'L_{attr2}_{hemi}'],ds[f'R_{attr2}_{hemi}']] = xhemi(ds[attr2],hemi=hemi)
        ds[f'L_{attr2}_{hemi}'] = ds[f'L_{attr2}_{hemi}'].sub(source=aparcR)
        ds[f'R_{attr2}_{hemi}'] = ds[f'R_{attr2}_{hemi}'].sub(source=aparcR)

        attr2 = attrstr+'_smooth_'+attr+'_noise_mean'
        print(f'r_lateralization: {attr2} {hemi}')

        [ds[f'L_{attr2}_{hemi}'],ds[f'R_{attr2}_{hemi}']] = xhemi(ds[attr2],hemi=hemi)
        ds[f'L_{attr2}_{hemi}'] = ds[f'L_{attr2}_{hemi}'].sub(source=aparcR)
        ds[f'R_{attr2}_{hemi}'] = ds[f'R_{attr2}_{hemi}'].sub(source=aparcR)
    return ds


def splitRL_vol(ds_in,hemi='rh'):
    print('splitRL')
    dsR = Dataset()
    dsL = Dataset()
    ds = Dataset()
    dsR.info = ds_in.info
    dsL.info = ds_in.info
    ds.info = ds_in.info
    sssflag = False
    for k in ds_in.keys():
        print('splitRL_vol '+k)
        if type(ds_in[k])==NDVar:
             if ds_in[k].has_dim('source'):
                ds_in[k].source.subjects_dir = subjects_dir
                if not sssflag:
                    lhverts = []
                    rhverts = []
                    for i in range(len(ds_in[k].source.vertices[0])):
                        if ds_in[k].source.coordinates[i][0]<0:
                            lhverts.append(i)
                        elif ds_in[k].source.coordinates[i][0]>0:
                            rhverts.append(i)
                dsR[k] = ds_in[k].sub(source=rhverts)
                dsL[k] = ds_in[k].sub(source=lhverts)
                ds[f'R_{k}_{hemi}'] = ds_in[k].sub(source=rhverts)
                ds[f'L_{k}_{hemi}'] = NDVar(ds_in[k].sub(source=lhverts).x,ds_in[k].sub(source=rhverts).dims)
                ds[k] = ds_in[k].copy()
        else:
            dsR[k] = ds_in[k].copy()
            dsL[k] = ds_in[k].copy()
            ds[k] = ds_in[k].copy()
    return ds, dsR, dsL


def splitRL_surf(ds_in,hemi='rh'):
    print('splitRL')
    dsR = Dataset()
    dsL = Dataset()
    ds = Dataset()
    dsR.info = ds_in.info
    dsL.info = ds_in.info
    ds.info = ds_in.info
    for k in ds_in.keys():
        print('splitRL_surf' + k)
        if type(ds_in[k])==NDVar:
             if ds_in[k].has_dim('source'):
                dsR[k] = ds_in[k].sub(source=aparcR)
                dsL[k] = ds_in[k].sub(source=aparcL)
                # ds[f'R_{k}_{hemi}'], ds[f'L_{k}_{hemi}'] = xhemi(ds_in[k])
        else:
            dsR[k] = ds[k].copy()
            dsL[k] = ds[k].copy()
    return ds, dsR, dsL


def h_latency(ds,attrstr,timewin=(-0.04,0.22)):
    for i in range(ds.info['npred']):
        attr = ds.info['attrname_'+attrstr][i]
        attr2 = attrstr+'_smooth_'+attr+'_model'
        print(f'h_latency: {attr2}')
        tadd = int(ds[attr2].sub(time=timewin).time.tmin*fs)
        if len(timewin)>1:
            xx = ds[attr2].sub(time=timewin)
            xx2 = np.argmax(xx,axis=2)
        else:
            xx2 = np.argmax(ds[attr2], axis=2)
        xx2 = xx2 + tadd
        ds[attrstr + '_latency_' + str(int(1000*timewin[0])) +'_'+str(int(1000*timewin[1]))+'_smooth_'+attr+'_model'] = NDVar(xx2, dims=(Case,ds[attr2].source))
    return ds


def load_smooth(pklfolder, sqdfolder=sqdFH, lstr='', savestr = '', force_make=False,timewin_lat = [0.02,0.07]):
    if lstr == '':
        if pklfolder == pklFc:
            lstr = '_yangh-carrier.pkl'
        elif pklfolder == pklFv:
            lstr = '_yangh-carriervol-7-cortex_brainstem.pkl'
        elif pklfolder == pklFl:
            lstr = '_env.pkl'
        else:
            raise IOError('Incorrect pickle folder')
    if pklfolder == pklFv:
        symflag1 = symflag
    else:
        symflag1 = False
    if symflag1:
        savestr = savestr+'sym'
    if force_make or not os.path.exists(f'{pklfolder}/Source/ds{savestr}{lstr}'):
        ds = load_data_HAYO(pklfolder,lstr)
        ds = makeNDVarMvN(ds, 'h')
        ds = makeNDVarMvN(ds, 'r')
        del ds['modeltrf']
        for i in range(ds.info['nperm']):
            del ds['noisetrf' + str(i)]
        if symflag1:
            ds = make_sym(ds,symfile)
        ds = nd_smooth(ds, smooth_s=0.005, attrstr='h')
        ds = nd_smooth(ds, smooth_s=0.005, attrstr='r')
        ds = h_latency(ds, 'h',timewin=timewin_lat)
        ds = xhemi_surf(ds,'r')
        ds = change_subjects_dir(ds)
        savestr = f'{pklfolder}/Source/ds{savestr}{lstr}'
        print(f'saving {savestr}')

        save.pickle(ds,savestr)
    else:
        savestr = f'{pklfolder}/Source/ds{savestr}{lstr}'
        print(f'loading {savestr}')
        ds = load.unpickle(savestr)
    return ds


def make_sym(ds,symfile=''):

    if symfile=='' or not os.path.exists(symfile):
        for k in ds.keys():
            if type(ds[k]) == NDVar:
                if 'source' in ds[k].dimnames:
                    coords = ds[k].source.coordinates
                    subsrc = []
                    for i,(x,y,z) in enumerate(coords):
                        if (-x,y,z) in coords:
                            subsrc.append(i)
                    symfile = f'symfile_{ds[k].source.name}.pkl'
                    save.pickle(subsrc,symfile)
                    break

    subsrc = load.unpickle(symfile)
    for k in ds.keys():
        if type(ds[k]) == NDVar:
            if 'source' in ds[k].dimnames:
                ds[k] = ds[k].sub(source=subsrc)
    return ds


########################################################################################################

def myttest(ds,test,attrstr,filename,pklfolder,NPerm=NPerm,roi=None, timewin=None):

    tests = []
    ds2 = Dataset()
    
    if test[:3] == 'MvN':
        for attr in ds.info[f'attrname_{attrstr}']:
            if test == 'MvN':
                attr1 = f'{attrstr}_smooth_{attr}_model'
                attr2 = f'{attrstr}_smooth_{attr}_noise_mean'
                ds2[attrstr] = combine([ds[attr1],ds[attr2]])
                mytest = testnd.ttest_rel
            elif test == 'MvNvector':
                attr1 = f'{attrstr}_{attr}_model'
                attr2 = f'{attrstr}_{attr}_noise_mean'
                ds2[attrstr] = combine([ds[attr1],ds[attr2]])
                mytest = testnd.VectorDifferenceRelated                
            if roi is not None:
                if type(roi) is list:
                    roi = [l for l in roi if l in ds2[attrstr].source.parc.as_labels()]
                ds2[attrstr] = ds2[attrstr].sub(source=roi)
            ds2['condition'] = Factor([1 for i in range(ds.n_cases)] + [0 for i in range(ds.n_cases)], name='condition')
            ds2['subject'] = Factor(ds['subject'].as_labels() + ds['subject'].as_labels(), name='subject', random=True)
            if timewin is not None:
                ds2[attrstr].sub(time=timewin)
            testh = mytest(attrstr, 'condition', '1', '0', match='subject', ds=ds2, samples=NPerm, tfce=True, tail=1)
            tests.append(testh)

    if test[:3] == 'roi':
        for attr in ds.info[f'attrname_{attrstr}']:
            attr1 = f'{attrstr}_smooth_{attr}_model'
            attr2 = f'{attrstr}_smooth_{attr}_noise_mean'
            if type(roi[0]) is list:
                roi[0] = [l for l in roi[0] if l in ds[attr1].source.parc.as_labels()]
            if type(roi[1]) is list:
                roi[1] = [l for l in roi[1] if l in ds[attr1].source.parc.as_labels()]
            if test == 'roiavg':
                ds2[attrstr] = combine([(ds[attr1] - ds[attr2]).sub(source = roi[0]).mean('source'),
                                        (ds[attr1] - ds[attr2]).sub(source = roi[1]).mean('source')])
                mytest = testnd.VectorDifferenceRelated
            elif test == 'roinorm':
                ds2[attrstr] = combine([(ds[attr1] - ds[attr2]).sub(source = roi[0]).norm('space').mean('source'),
                                        (ds[attr1] - ds[attr2]).sub(source = roi[1]).norm('space').mean('source')])
                mytest = testnd.ttest_rel
            ds2['condition'] = Factor([1 for i in range(ds.n_cases)] + [0 for i in range(ds.n_cases)], name='condition')
            ds2['subject'] = Factor(ds['subject'].as_labels() + ds['subject'].as_labels(), name='subject', random=True)
            if timewin is not None:
                ds2[attrstr].sub(time=timewin)
            testh = mytest(attrstr, 'condition', '1', '0', match='subject', ds=ds2, samples=NPerm, tfce=True, tail=0)
            tests.append(testh)


    if test[:3] == 'OvY':
        for attr in ds.info[f'attrname_{attrstr}']:
            if test == 'OvY_MmN':
                attr1 = f'{attrstr}_smooth_{attr}_model'
                attr2 = f'{attrstr}_smooth_{attr}_noise_mean'
                ds2[attrstr] = ds[attr1] - ds[attr2]
                if timewin is not None:
                    ds2[attrstr].sub(time=timewin)
            if test[:7] == 'OvY_RmL':
                hemi = test[-2:]
                attr1 = f'R_{attrstr}_smooth_{attr}_model_{hemi}' 
                attr2 = f'L_{attrstr}_smooth_{attr}_model_{hemi}'
                if timewin is not None:
                    ds2[attrstr].sub(time=timewin)
                ds2[attrstr] = ds[attr1] - ds[attr2]
            if test == 'OvY_latency':
                attr1 = f'{attrstr}_latency_{str(int(1000 * timewin[0]))}_{str(int(1000 * timewin[1]))}_smooth_{attr}_model' 
                ds2[attrstr] = ds[attr1]
            if roi is not None:
                if type(roi) is list:
                    roi = [l for l in roi if l in ds2[attrstr].source.parc.as_labels()]
                ds2[attrstr] = ds2[attrstr].sub(source=roi)
            ds2['condition'] = ds['youngflag']
            ds2['subject'] = ds['subject']

            testh = testnd.ttest_ind(attrstr, 'condition', '1', '0', ds=ds2, samples=NPerm, tfce=True, tail=0)
            tests.append(testh)

    if test[:3] == 'RvL':
        for attr in ds.info[f'attrname_{attrstr}']:
            hemi = test[-3:-1]
            tail = int(test[-1])
            attr1 = f'{attrstr}_smooth_{attr}_model_{hemi}' 
            attr2 = f'{attrstr}_smooth_{attr}_noise_mean_{hemi}'
            ds2[attrstr] = combine([ds[f'R_{attr1}']-ds[f'R_{attr2}'],
                                    ds[f'L_{attr1}']-ds[f'L_{attr2}']])             
            if roi is not None:
                if type(roi) is list:
                    roi = [l for l in roi if l in ds2[attrstr].source.parc.as_labels()]
                ds2[attrstr] = ds2[attrstr].sub(source=roi)
            ds2['condition'] = Factor([1 for i in range(ds.n_cases)] + [0 for i in range(ds.n_cases)], name='condition')
            ds2['subject'] = Factor(ds['subject'].as_labels() + ds['subject'].as_labels(), name='subject', random=True)
            if timewin is not None:
                ds2[attrstr].sub(time=timewin)
            testh = testnd.ttest_rel(attrstr, 'condition', '1', '0', match='subject', ds=ds2, samples=NPerm, tfce=True, tail=tail) 
            tests.append(testh)

    if test[:4] == 'pred':
        attr1 = ds.info[f'attrname_h'][0]
        attr2 = ds.info[f'attrname_h'][1]

        pred1 = ds[f'h_smooth_{attr1}_model'] - ds[f'h_smooth_{attr1}_noise_mean']
        pred2 = ds[f'h_smooth_{attr2}_model'] - ds[f'h_smooth_{attr2}_noise_mean']

        if roi is not None:
            if type(roi) is list:
                roi = [l for l in roi if l in pred1.source.parc.as_labels()]
            pred1 = pred1.sub(source=roi)
            pred2 = pred2.sub(source=roi)

        subjects = []
        conditions = []
        ndvar = []
        if 'space' in pred1.dimnames:
            pred1 = pred1.norm('space')
            pred2 = pred2.norm('space')
        pred1 = pred1.sub(time=(0.02,0.07))
        pred2 = pred2.sub(time=(0.02,0.07))
        if test[-2:] == '3d':
            mytest = testnd.ttest_rel
            kwargs = {'samples': NPerm, 'tfce': True}
        if test[-2:] == '2d':
            pred1 = pred1.norm('time')
            pred2 = pred2.norm('time')
            mytest = testnd.ttest_rel
            kwargs = {'samples':NPerm, 'tfce':True}
        for i in range(len(pred1)):
            ndvar.append(pred1[i])
            conditions.append('1')
            subjects.append(i)
            ndvar.append(pred2[i])
            conditions.append('0')
            subjects.append(i)

        ds2 = Dataset()
        ds2['y'] = combine(ndvar)
        ds2['subject'] = Factor(subjects,random=True)
        ds2['condition'] = Factor(conditions)
        testh = mytest('y', 'condition', '1', '0', match='subject', ds=ds2, **kwargs)
        tests.append(testh)


    print(f'saving: {filename}')
    save.pickle(tests,filename)
    return tests



def setmaxtime(p,maskFlag=True):
    if(maskFlag):
        if len(p[0].figure.get_axes())==2:
            ax1 = p[0].figure.get_axes()[0]
            ax2 = p[0].figure.get_axes()[1]
            m11 = [np.max(ax1.lines[i].get_ydata().__abs__()) for i in range(0, int(len(ax1.lines) / 2))]
            m1 = []
            for i in range(0, len(m11)):
                if m11[i] is np.ma.masked:
                    m1.append(0)
                else:
                    m1.append(float(m11[i]))
            m1t = [np.argmax(ax1.lines[i].get_ydata().__abs__()) for i in range(0, int(len(ax1.lines) / 2))]
            m22 = [np.max(ax2.lines[i].get_ydata().__abs__()) for i in range(0, int(len(ax2.lines) / 2))]
            m2 = []
            for i in range(0, len(m22)):
                if m22[i] is np.ma.masked:
                    m2.append(0)
                else:
                    m2.append(float(m22[i]))
            m2t = [np.argmax(ax2.lines[i].get_ydata().__abs__()) for i in range(0, int(len(ax2.lines) / 2))]
            if (np.max(m1) > np.max(m2)):
                maxtime = ax1.lines[0].get_xdata()[m1t[np.argmax(m1)]]
            else:
                maxtime = ax2.lines[0].get_xdata()[m2t[np.argmax(m2)]]
            p[1].set_time(maxtime)
        else:
            ax1 = p[0].figure.get_axes()[0]
            m11 = [np.max(ax1.lines[i].get_ydata().__abs__()) for i in range(0, int(len(ax1.lines) / 2))]
            m1 = []
            for i in range(0, len(m11)):
                if m11[i] is np.ma.masked:
                    m1.append(0)
                else:
                    m1.append(float(m11[i]))
            m1t = [np.argmax(ax1.lines[i].get_ydata().__abs__()) for i in range(0, int(len(ax1.lines) / 2))]
            maxtime = ax1.lines[0].get_xdata()[m1t[np.argmax(m1)]]
            p[1].set_time(maxtime)


def get_stats(ds,attr,niceplotfolder,tt=None,pp=None,attrstr=None,savestr='',teststr='',roi=None,timewin=None,meanstr='source',normstr='space'):
    niceplotfolder = f'{niceplotfolder}/{teststr}'
    if not os.path.exists(niceplotfolder):
        os.makedirs(niceplotfolder)
    savestr = f'{savestr}_{teststr}'
    print(f'get_stats_{attr}_{savestr}')
    filepath = f'{niceplotfolder}/get_stats_{attr}_{savestr}.txt'
    if attrstr is not None:
        nd = ds[attrstr]
    elif attr == 'r':
        nd = ds['r_smooth_correlation_model']
    else:
        nd = ds['h_yangh_model']
    if roi is not None:
        if type(roi) is list:
            roi = [l for l in roi if l in nd.source.parc.as_labels()]
        nd = nd.sub(source=roi)
    if timewin is not None:
        nd = nd.sub(time=timewin)
    if normstr in nd.dimnames:
        nd = nd.norm(normstr)
    mn = nd.min()
    mx = nd.max()
    mean_mn = nd.mean(meanstr).min()
    mean_mx = nd.mean(meanstr).max()
    mean = nd.mean()
    std = nd.std()
    mean_std = nd.mean(meanstr).std()

    file = open(filepath,'w+')
    file.write(f'min = {mn:.6f}\n')
    file.write(f'max = {mx:.6f}\n')
    file.write(f'mean min = {mean_mn:.6f}\n')
    file.write(f'mean max = {mean_mx:.6f}\n')
    file.write(f'std = {std:.6f}\n')
    file.write(f'mean std = {mean_std:.6f}\n')
    file.write(f'mean = {mean:.6f}\n')

    if tt is not None:
        if 'time' in pp.dimnames:
            pp_src = pp.min('time')
        else:
            pp_src = pp
        sig_src = pp_src < 0.05
        if sum(sig_src) > 0:
            nd2 = nd.sub(source=sig_src)
            mn = nd2.min()
            mx = nd2.max()
            mean_mn = nd2.mean(meanstr).min()
            mean_mx = nd2.mean(meanstr).max()
            mean = nd2.mean()
            std = nd2.std()
            mean_std = nd2.mean(meanstr).std()

            ttmn = pp.min()
            ttmx = pp.max()
            ttN1 = sum(pp_src<0.05)
            ttN2 = sum(pp_src<0.01)
            ttN3 = sum(pp_src<0.001)
            ttN4 = sum(pp_src<0.0001)
            NN = len(pp_src)

            tvalx = tt.max()
            tvaln = tt.min()

            file.write(f'\n\ntt min = {ttmn}\n')
            file.write(f'tt max = {ttmx}\n')
            file.write(f'tt 0.05 {ttN1} of {NN} : {ttN1/NN:.6f}\n')
            file.write(f'tt 0.05 {ttN1} of {NN} : {ttN1/NN:.6f}\n')
            file.write(f'tt 0.01 {ttN2} of {NN} : {ttN2/NN:.6f}\n')
            file.write(f'tt 0.001 {ttN3} of {NN} : {ttN3/NN:.6f}\n')
            file.write(f'tt 0.0001 {ttN4} of {NN} : {ttN4/NN:.6f}\n')
            file.write(f'p val min = {ttmn:.6f}, t val max = {tvalx:.6f}, t val min = {tvaln:.6f}\n')

            file.write(f'min = {mn:.6f}\n')
            file.write(f'max = {mx:.6f}\n')
            file.write(f'mean min = {mean_mn:.6f}\n')
            file.write(f'mean max = {mean_mx:.6f}\n')
            file.write(f'std = {std:.6f}\n')
            file.write(f'mean std = {mean_std:.6f}\n')
            file.write(f'mean = {mean:.6f}\n')

            if attr == 'h' and 'time' in pp.dimnames:
                file.write('\n\n****************TRF analysis*************\n\n')

                sig_time = sum(pp < 0.05) > 0
                for i in range(len(sig_time)):
                    if sig_time.x[i] > 0:
                        t1 = pp.time.times[i]
                        break
                for i in range(len(sig_time)):
                    if sig_time.x[-i] > 0:
                        t2 = pp.time.times[-i]
                        break

                subflag = pp < 0.05
                vals = []
                ndmax = nd.max('case')
                for i in range(pp.x.shape[0]):
                    for j in range(pp.x.shape[1]):
                        if subflag.x[i][j]:
                            vals.append(ndmax.x[i][j])

                file.write(f'vals max = {np.max(vals)}\n')
                file.write(f'vals mean = {np.mean(vals)}\n')
                file.write(f'vals std = {np.std(vals)}\n')

                file.write(f't1 = {t1:.4f}\n')
                file.write(f't2 = {t2:.4f}\n')

                ndforfft = nd.sub(source=sig_src)
                fft = ndforfft.fft()

                argmx = np.argmax(fft.mean('source').x,axis=1)
                mxfft = np.array([fft.frequency[i] for i in argmx])

                mn = mxfft.min()
                mx = mxfft.max()
                mean = mxfft.mean()
                std = mxfft.std()

                file.write(f'fft min = {mn:.6f}\n')
                file.write(f'fft max = {mx:.6f}\n')
                file.write(f'fft std = {std:.6f}\n')
                file.write(f'fft mean of argmax = {mean:.6f}\n')

                argmx = np.argmax(fft.mean("source").mean("case").x)
                file.write(f'fft argmax of mean = {fft.frequency[argmx]:.6f}\n')

                if t1!=t2:
                    nd2 = nd.sub(source=sig_src).sub(time=(t1,t2))

                    freqs = []
                    latencies = []
                    thr = nd2 > nd2.max()/10
                    for i in range(nd2.x.shape[0]):
                        ll = []
                        ff = []
                        for j in range(nd2.x.shape[1]):
                            if thr.x[i][j].any():
                                ll.append(nd2.time.times[np.argmax(nd2.x[i][j][:])])
                                f1 = scipy.signal.find_peaks(nd2.x[i][j])
                                ff.append(1000/np.mean(np.diff(f1[0])))
                        latencies.append(ll)
                        freqs.append(ff)

                    ll2 = []
                    for i in range(len(latencies)):
                        ll2.append(np.mean(latencies[i]))

                    ll3 = [x for x in ll2 if str(x)!='nan']

                    lat_subjN = len(ll3)
                    lat_mean = np.mean(ll3)
                    lat_max = np.max(ll3)
                    lat_min = np.min(ll3)
                    lat_std = np.std(ll3)

                    file.write(f'latencies subj {lat_subjN} of {nd.x.shape[0]} : {100*lat_subjN/nd.x.shape[0]}\n')
                    file.write(f'latencies mean = {lat_mean}\n')
                    file.write(f'latencies min = {lat_min}\n')
                    file.write(f'latencies max = {lat_max}\n')
                    file.write(f'latencies std = {lat_std}\n')

                    ff2 = []
                    for i in range(len(freqs)):
                        ff2.append(np.mean(freqs[i]))

                    ff3 = [x for x in ff2 if str(x)!='nan']

                    if len(ff3)>0:
                        freq_subjN = len(ff3)
                        freq_mean = np.mean(ff3)/2
                        freq_max = np.max(ff3)/2
                        freq_min = np.min(ff3)/2
                        freq_std = np.std(ff3)/2

                        file.write(f'frequencies subj {freq_subjN} of {nd.x.shape[0]} : {100*freq_subjN/nd.x.shape[0]}\n')
                        file.write(f'frequencies mean = {freq_mean}\n')
                        file.write(f'frequencies min = {freq_min}\n')
                        file.write(f'frequencies max = {freq_max}\n')
                        file.write(f'frequencies std = {freq_std}\n')

        else:
            file.write('\n\ntt not signif')
            ttmn = pp.min()
            ttmx = pp.max()
            tvalx = tt.max()
            tvaln = tt.min()
            file.write(f'p val min = {ttmn:.6f}, t val max = {tvalx:.6f}, t val min = {tvaln:.6f}\n')
    file.close()


def combine_vol_src(nd_cb,rrc,rrb):
    hastime = 'time' in nd_cb.dimnames
    if hasattr(rrc.x,'mask'):
        rrc1 = rrc.x.data
        rrb1 = rrb.x.data
        rrc2 = rrc.x.mask
        rrb2 = rrb.x.mask
        xx1 = []
        xx2 = []
        ib = 0
        ic = 0
        for p in src_cb.parc:
            if p == 'Brain-Stem':
                if hastime:
                    xx1.append(rrb1[ib,:])
                    xx2.append(rrb2[ib,:])
                else:
                    xx1.append(rrb1[ib])
                    xx2.append(rrb2[ib])
                ib +=1
            else:
                if hastime:
                    xx1.append(rrc1[ic,:])
                    xx2.append(rrc2[ic,:])
                else:
                    xx1.append(rrc1[ic])
                    xx2.append(rrc2[ic])
                ic +=1
        rr = NDVar(xx1,nd_cb.dims)
        mask = NDVar(xx2,nd_cb.dims)
        return rr.mask(mask)
    else:
        xx = []
        ib = 0
        ic = 0
        for p in src_cb.parc:
            if p == 'Brain-Stem':
                if hastime:
                    xx.append(rrb[ib,:])
                else:
                    xx.append(rrb[ib])
                ib +=1
            else:
                if hastime:
                    xx.append(rrc[ic,:])
                else:
                    xx.append(rrc[ic])
                ic +=1
        return NDVar(xx,nd_cb.dims)



def time_compensate(ds,lag=0.01):
    print(f'time_compensate lag = {lag}')
    for k in ds.keys():
        if type(ds[k])==NDVar:
             if ds[k].has_dim('time'):
                ds[k].time = UTS(ds[k].time.tmin-lag,ds[k].time.tstep,ds[k].time.times.shape[0])
    return ds

def time_sub(ds,timewin):
    print(f'time sub {timewin[0]} {timewin[1]}')
    for k in ds.keys():
        if type(ds[k])==NDVar:
             if ds[k].has_dim('time'):
                ds[k] = ds[k].sub(time=timewin)
    return ds

def run_tests_vol(dsIn,outputfolder=pklFv,plotflag=True,savestr='',niceplotfolder=niceplotfolder,force_make=False,lat_flag=True,timewin=(-1000,1000),timewin_lat=(0.020,0.070),lq=None):
    ds = dsIn.copy()
    ds = time_compensate(ds,lag=0.01)
    vol_space = True
    niceplotfolder = niceplotfolder + '_volume'
    if not os.path.exists(niceplotfolder):
        os.makedirs(niceplotfolder)

    if timewin[0]>-1000:
        savestr = f'{savestr}_timewin_{int(1000*timewin[0])}_{int(1000*timewin[1])}'
        ds = time_sub(ds,timewin)

    if lq != None:
        savestr = f'{savestr}_lq{lq}'
        idx = []
        for s in ds['subject']:
            if subjectshandedness[s]>lq:
                print(s)
                aa = ds['subject'].index(s)
                if len(aa)>0:
                    print(aa[0])
                    idx.append(aa[0])
        ds = ds.sub(idx)

    ds, dsR, dsL = splitRL_vol(ds)

    dsY = ds.sub('youngflag=="1"')
    dsO = ds.sub('youngflag=="0"')
  
    tlists = [
              # ['All','r','MvN',None,None],
              #['Old','r','MvN',None,None],
              #['Young','r','MvN',None,None],
              # ['All','r','MvN','cortex',None],
              # ['Old','r','MvN','cortex',None],
              # ['Young','r','MvN','cortex',None],
              # ['All','r','MvN','brainstem',None],
              # ['Old','r','MvN','brainstem',None],
              # ['Young','r','MvN','brainstem',None],
              # ['All','r','MvN','thalamus',None],
              # ['Old','r','MvN','thalamus',None],
              # ['Young','r','MvN','thalamus',None],
              #
              # ['All','r','RvL_rh0',None,None],
              # ['Old','r','RvL_rh0',None,None],
              # ['Young','r','RvL_rh0',None,None],
              # ['All','r','RvL_rh0','cortex',None],
              # ['Old','r','RvL_rh0','cortex',None],
              # ['Young','r','RvL_rh0','cortex',None],
              # ['All','r','RvL_rh0','brainstem',None],
              # ['Old','r','RvL_rh0','brainstem',None],
              # ['Young','r','RvL_rh0','brainstem',None],
              # ['All','r','RvL_rh0','thalamus',None],
              # ['Old','r','RvL_rh0','thalamus',None],
              # ['Young','r','RvL_rh0','thalamus',None],
              #
              #
              # ['All','r','OvY_MmN',None,None],
              # ['All','r','OvY_MmN','cortex',None],
              # ['All','r','OvY_MmN','brainstem',None],
              # ['All','r','OvY_MmN','thalamus',None],
              # #
              # ['All','r','OvY_RmL_rh',None,None],
              # ['All','r','OvY_RmL_rh','cortex',None],
              # ['All','r','OvY_RmL_rh','brainstem',None],
              # ['All','r','OvY_RmL_rh','thalamus',None],
              #
              # ['All','h','MvN',None,None],
              # ['Old','h','MvN',None,None],
              # ['Young','h','MvN',None,None],
              #['All','h','MvN','cortex',None],
              #['Old','h','MvN','cortex',None],
              #['Young','h','MvN','cortex',None],
              # ['All','h','MvN','brainstem',None],
              #['Old','h','MvN','brainstem',None],
              #['Young','h','MvN','brainstem',None],
              # ['All','h','MvN','thalamus',None],
              # ['Old','h','MvN','thalamus',None],
              # ['Young','h','MvN','thalamus',None],
              # #
              # ['All','h','OvY_MmN',None,None],
              # ['All','h','OvY_MmN','cortex',None],
              # ['All','h','OvY_MmN','brainstem',None],
              # ['All','h','OvY_MmN','thalamus',None],
              # #
              # ['All','h','OvY_latency',None,f'{timewin_lat[0]}_{timewin_lat[1]}'],
              # ['All','h','OvY_latency','cortex',f'{timewin_lat[0]}_{timewin_lat[1]}'],
              # ['All','h','OvY_latency','brainstem',f'{timewin_lat[0]}_{timewin_lat[1]}'],
              # ['All','h','OvY_latency','thalamus',f'{timewin_lat[0]}_{timewin_lat[1]}'],

              # ['All','h','pred',None,None],
              # ['Old','h','pred',None,None],
              # ['Young','h','pred',None,None],
              # ['All','h','pred','cortex',None],
              # ['Old','h','pred2d','cortex',None],
              # ['Young','h','pred2d','cortex',None],
              # # ['All','h','pred','brainstem',None],
              # ['Old','h','pred2d','brainstem',None],
              # ['Young','h','pred2d','brainstem',None],
              # ['All','h','pred','thalamus',None],
              # ['Old','h','pred','thalamus',None],
              # ['Young','h','pred','thalamus',None],
             ]

    roi_from_str = {'cortex':cortsub, 'brainstem':'Brain-Stem', 'thalamus':thal,None:None}
    tests = {}
    for tlist in tlists:
        tlist1 = [t for t in tlist if t is not None]
        teststr = '_'.join(tlist1)
        print(teststr)
        group = tlist[0]
        attrstr = tlist[1]
        tstr = tlist[2]
        roi = roi_from_str[tlist[3]]
        if tlist[4] is not None:
            twin = [float(t) for t in tlist[4].split('_')]
        else:
            twin = None
        filename = f'{outputfolder}/Source/{savestr}_{teststr}.pkl'
        if group == 'All':
            ds_t = ds.copy()
        elif group == 'Old':
            ds_t = dsO.copy()
        elif group == 'Young':
            ds_t = dsY.copy()
        if force_make or not os.path.exists(filename):
            tests[teststr] = myttest(ds_t,attrstr=attrstr,test=tstr,pklfolder=pklFv,filename=filename,roi=roi, timewin=twin)
        else:
            tests[teststr] = load.unpickle(filename)

        for it, ttt in enumerate(tests[teststr]):
            print(f'{teststr} ' + ds.info[f'attrname_{attrstr}'][it] +f' p min = {ttt.p.min():.4f}')

        if plotflag:
            if attrstr == 'r':
                get_stats(ds_t,attrstr,niceplotfolder,tt=tests[teststr][0].t.copy(),pp=tests[teststr][0].p.copy(),savestr=savestr,teststr=teststr,roi=roi,timewin=twin)
            else:
                for ia, attrstr in enumerate(ds.info['attrname_h']):
                    attrstr1 = f'h_smooth_{attrstr}_model'
                    if tstr[:4] == 'pred':
                        it = 0
                    else:
                        it = ia
                    get_stats(ds_t,attr='h',niceplotfolder=niceplotfolder,attrstr=attrstr1,tt=tests[teststr][it].t.copy(),pp=tests[teststr][it].p.copy(),savestr=f'{savestr}_{attrstr}',teststr=teststr,roi=roi,timewin=twin)


    plotnorms = [
                # [None,'r','MvN',None,None],
                #[None,'r','RvL_rh0',None,None],
                #[None,'h','MvN',None,None],
                [None,'h','pred2d',None,None]
                ]

    plotattr = {
                 'rMvN':['r_smooth_correlation_model'],
                 'rRvL_rh0':['R_r_smooth_correlation_model_rh'],
                 'hMvN':[f'h_{aa}_model' for aa in ds.info['attrname_h']],
                 'hpred2d':[f'h_{aa}_model' for aa in ds.info['attrname_h']]
                }

    for plotnorm in plotnorms:
        mx = []
        mn = []
        for tlist in tlists:
            tlist1 = [t for t in tlist if t is not None]
            teststr = '_'.join(tlist1)
            group = tlist[0]
            attrstr = tlist[1]
            tstr = tlist[2]
            roi = roi_from_str[tlist[3]]
            if plotnorm[0] is not None:
                if group is not plotnorm[0]:
                    continue
            if plotnorm[1] is not None:
                if attrstr is not plotnorm[1]:
                    continue
            if plotnorm[2] is not None:
                if tstr is not plotnorm[2]:
                    continue
            if plotnorm[3] is not None:
                if roi is not plotnorm[3]:
                    continue
            print('max '+teststr)
            if group == 'All':
                ds_t = ds.copy()
            elif group == 'Old':
                ds_t = dsO.copy()
            elif group == 'Young':
                ds_t = dsY.copy()
            for ia, attr in enumerate(plotattr[plotnorm[1]+plotnorm[2]]):
                xnd = ds_t[attr].mean('case')
                if roi is not None:
                    if type(roi) is list:
                        roi = [l for l in roi if l in xnd.source.parc.as_labels()]
                    xnd = xnd.sub(source=roi)
                mx.append(xnd.abs().max())
                mn.append(xnd.abs().min())
        mx = 1.2*np.max(mx)
        mn = np.min(mn)
        for tlist in tlists:
            tlist1 = [t for t in tlist if t is not None]
            teststr = '_'.join(tlist1)
            group = tlist[0]
            attrstr = tlist[1]
            tstr = tlist[2]
            roi = roi_from_str[tlist[3]]
            if plotnorm[0] is not None:
                if group is not plotnorm[0]:
                    continue
            if plotnorm[1] is not None:
                if attrstr is not plotnorm[1]:
                    continue
            if plotnorm[2] is not None:
                if tstr is not plotnorm[2]:
                    continue
            if plotnorm[3] is not None:
                if roi is not plotnorm[3]:
                    continue
            if group == 'All':
                ds_t = ds.copy()
            elif group == 'Old':
                ds_t = dsO.copy()
            elif group == 'Young':
                ds_t = dsY.copy()
            print('plot '+teststr)
            if tstr == 'pred2d':
                niceplot_pred(ds_t,niceplotfolder,savestr=savestr+teststr,roi=roi)
            else:
                for ia, attr in enumerate(plotattr[plotnorm[1]+plotnorm[2]]):
                    # if attr == 'difference':
                    #     tt = tests[teststr][0].t
                    if tstr == 'pred2d':
                        it = 0
                    else:
                        it = ia
                    tt = tests[teststr][it].t
                    pp = tests[teststr][it].p
                    masknd = tests[teststr][it].masked_difference()
                    if 'time' in masknd.dimnames:
                        ylim = [0,mx]
                    else:
                        ylim = [mn,mx]
                    ylim = [0,mx]
                    if tlist[3] == 'cortex':
                        peaktimes=(0.035,0.04)
                    elif tlist[3] == 'brainstem':
                        peaktimes = (0.02,0.06)
                    else:
                        peaktimes = (0.02,0.07)
                    niceplot(ds_t, attrstr, niceplotfolder, teststr, tt, pp, masknd, attr = attr, ylim=ylim, roi=roi, savestr=f'{savestr}_{attr}',peaktimes=peaktimes)

    #niceplot_fft(ds,niceplotfolder=niceplotfolder)
    #BF_rOvY_MmN(ds,roi=cortsub+['Brain-Stem'])

    return tests


def run_tests_surf(dsIn, outputfolder=pklFc, plotflag=True, savestr='', niceplotfolder=niceplotfolder, force_make=False,
                  lat_flag=True, timewin=(-1000, 1000), timewin_lat=(0.020, 0.070), lq=None):
    ds = dsIn.copy()
    ds = time_compensate(ds, lag=0.01)
    vol_space = True
    niceplotfolder = niceplotfolder + '_surf'
    if not os.path.exists(niceplotfolder):
        os.makedirs(niceplotfolder)

    if timewin[0] > -1000:
        savestr = f'{savestr}_timewin_{int(1000*timewin[0])}_{int(1000*timewin[1])}'
        ds = time_sub(ds, timewin)

    if lq != None:
        savestr = f'{savestr}_lq{lq}'
        idx = []
        for s in ds['subject']:
            if subjectshandedness[s] > lq:
                print(s)
                aa = ds['subject'].index(s)
                if len(aa) > 0:
                    print(aa[0])
                    idx.append(aa[0])
        ds = ds.sub(idx)

    # ds = xhemi_surf(ds,'h')
    # ds = xhemi_surf(ds,'r')


    dsY = ds.sub('youngflag=="1"')
    dsO = ds.sub('youngflag=="0"')

    tlists = [
        # ['All','r','MvN',None,None],
        #['Old','r','MvN',None,None],
        #['Young','r','MvN',None,None],
        # ['All','r','MvN','cortex',None],
        # ['Old','r','MvN','cortex',None],
        # ['Young','r','MvN','cortex',None],
        # ['All','r','MvN','brainstem',None],
        # ['Old','r','MvN','brainstem',None],
        # ['Young','r','MvN','brainstem',None],
        # ['All','r','MvN','thalamus',None],
        # ['Old','r','MvN','thalamus',None],
        # ['Young','r','MvN','thalamus',None],
        #
        # ['All','r','RvL_rh0',None,None],
        ['Old','r','RvL_rh0',None,None],
        ['Young','r','RvL_rh0',None,None],
        # ['All','r','RvL_rh0','cortex',None],
        # ['Old','r','RvL_rh0','cortex',None],
        # ['Young','r','RvL_rh0','cortex',None],
        # ['All','r','RvL_rh0','brainstem',None],
        # ['Old','r','RvL_rh0','brainstem',None],
        # ['Young','r','RvL_rh0','brainstem',None],
        # ['All','r','RvL_rh0','thalamus',None],
        # ['Old','r','RvL_rh0','thalamus',None],
        # ['Young','r','RvL_rh0','thalamus',None],
        #
        #
        # ['All','r','OvY_MmN',None,None],
        # ['All','r','OvY_MmN','cortex',None],
        # ['All','r','OvY_MmN','brainstem',None],
        # ['All','r','OvY_MmN','thalamus',None],
        # #
        # ['All','r','OvY_RmL_rh',None,None],
        # ['All','r','OvY_RmL_rh','cortex',None],
        # ['All','r','OvY_RmL_rh','brainstem',None],
        # ['All','r','OvY_RmL_rh','thalamus',None],
        #
        # ['All','h','MvN',None,None],
        ['Old','h','MvN',None,None],
        ['Young','h','MvN',None,None],
        # ['All','h','MvN','cortex',None],
        # ['Old','h','MvN','cortex',None],
        # ['Young','h','MvN','cortex',None],
        # ['All','h','MvN','brainstem',None],
        # ['Old','h','MvN','brainstem',None],
        # ['Young','h','MvN','brainstem',None],
        # ['All','h','MvN','thalamus',None],
        # ['Old','h','MvN','thalamus',None],
        # ['Young','h','MvN','thalamus',None],
        # #
        # ['All','h','OvY_MmN',None,None],
        # ['All','h','OvY_MmN','cortex',None],
        # ['All','h','OvY_MmN','brainstem',None],
        # ['All','h','OvY_MmN','thalamus',None],
        # #
        # ['All','h','OvY_latency',None,f'{timewin_lat[0]}_{timewin_lat[1]}'],
        # ['All','h','OvY_latency','cortex',f'{timewin_lat[0]}_{timewin_lat[1]}'],
        # ['All','h','OvY_latency','brainstem',f'{timewin_lat[0]}_{timewin_lat[1]}'],
        # ['All','h','OvY_latency','thalamus',f'{timewin_lat[0]}_{timewin_lat[1]}'],

        # ['All','h','pred',None,None],
        # ['Old','h','pred',None,None],
        # ['Young','h','pred',None,None],
        # ['All','h','pred','cortex',None],
        # ['Old','h','pred2d','cortex',None],
        # ['Young','h','pred2d','cortex',None],
        # # ['All','h','pred','brainstem',None],
        # ['Old','h','pred2d','brainstem',None],
        # ['Young','h','pred2d','brainstem',None],
        # ['All','h','pred','thalamus',None],
        # ['Old','h','pred','thalamus',None],
        # ['Young','h','pred','thalamus',None],
    ]

    roi_from_str = {'cortex': cortsub, 'brainstem': 'Brain-Stem', 'thalamus': thal, None: None}
    tests = {}
    for tlist in tlists:
        tlist1 = [t for t in tlist if t is not None]
        teststr = '_'.join(tlist1)
        print(teststr)
        group = tlist[0]
        attrstr = tlist[1]
        tstr = tlist[2]
        roi = roi_from_str[tlist[3]]
        if tlist[4] is not None:
            twin = [float(t) for t in tlist[4].split('_')]
        else:
            twin = None
        filename = f'{outputfolder}/Source/{savestr}_{teststr}.pkl'
        if group == 'All':
            ds_t = ds.copy()
        elif group == 'Old':
            ds_t = dsO.copy()
        elif group == 'Young':
            ds_t = dsY.copy()
        if force_make or not os.path.exists(filename):
            tests[teststr] = myttest(ds_t, attrstr=attrstr, test=tstr, pklfolder=pklFc, filename=filename, roi=roi,
                                     timewin=twin)
        else:
            tests[teststr] = load.unpickle(filename)

        for it, ttt in enumerate(tests[teststr]):
            print(f'{teststr} ' + ds.info[f'attrname_{attrstr}'][it] + f' p min = {ttt.p.min():.4f}')

        # if plotflag:
        #     if attrstr == 'r':
        #         get_stats(ds_t, attrstr, niceplotfolder, tt=tests[teststr][0].t.copy(), pp=tests[teststr][0].p.copy(),
        #                   savestr=savestr, teststr=teststr, roi=roi, timewin=twin)
        #     else:
        #         for ia, attrstr in enumerate(ds.info['attrname_h']):
        #             attrstr1 = f'h_smooth_{attrstr}_model'
        #             if tstr[:4] == 'pred':
        #                 it = 0
        #             else:
        #                 it = ia
        #             get_stats(ds_t, attr='h', niceplotfolder=niceplotfolder, attrstr=attrstr1,
        #                       tt=tests[teststr][it].t.copy(), pp=tests[teststr][it].p.copy(),
        #                       savestr=f'{savestr}_{attrstr}', teststr=teststr, roi=roi, timewin=twin)

    plotnorms = [
         #[None,'r','MvN',None,None],
        [None,'r','RvL_rh0',None,None],
         [None,'h','MvN',None,None],
        #[None, 'h', 'pred2d', None, None]
    ]

    plotattr = {
        'rMvN': ['r_smooth_correlation_model'],
        'rRvL_rh0': ['R_r_smooth_correlation_model_rh'],
        'hMvN': [f'h_{aa}_model' for aa in ds.info['attrname_h']],
        'hpred2d': [f'h_{aa}_model' for aa in ds.info['attrname_h']]
    }

    peaktimes = (0.035,0.04)
    for plotnorm in plotnorms:
        mx = []
        mn = []
        mx2 = []
        for tlist in tlists:
            tlist1 = [t for t in tlist if t is not None]
            teststr = '_'.join(tlist1)
            group = tlist[0]
            attrstr = tlist[1]
            tstr = tlist[2]
            roi = roi_from_str[tlist[3]]
            if plotnorm[0] is not None:
                if group is not plotnorm[0]:
                    continue
            if plotnorm[1] is not None:
                if attrstr is not plotnorm[1]:
                    continue
            if plotnorm[2] is not None:
                if tstr is not plotnorm[2]:
                    continue
            if plotnorm[3] is not None:
                if roi is not plotnorm[3]:
                    continue
            print('max ' + teststr)
            if group == 'All':
                ds_t = ds.copy()
            elif group == 'Old':
                ds_t = dsO.copy()
            elif group == 'Young':
                ds_t = dsY.copy()
            for ia, attr in enumerate(plotattr[plotnorm[1] + plotnorm[2]]):
                xnd = ds_t[attr].mean('case')
                if roi is not None:
                    if type(roi) is list:
                        roi = [l for l in roi if l in xnd.source.parc.as_labels()]
                    xnd = xnd.sub(source=roi)
                mx.append(xnd.abs().max())
                if 'time' in xnd.dimnames:
                    iadd = np.argmin(np.abs(xnd.time.times - peaktimes[0]))
                    ii = np.argmax(xnd.norm('source').sub(time=peaktimes).x)
                    mx2.append(xnd.sub(time=xnd.time.times[ii + iadd]).abs().max())
        mx = np.max(mx)
        mn = -mx
        if 'time' in xnd.dimnames:
            mx2 = np.max(mx2)
        else:
            mx2 = None
        for tlist in tlists:
            tlist1 = [t for t in tlist if t is not None]
            teststr = '_'.join(tlist1)
            group = tlist[0]
            attrstr = tlist[1]
            tstr = tlist[2]
            roi = roi_from_str[tlist[3]]
            if plotnorm[0] is not None:
                if group is not plotnorm[0]:
                    continue
            if plotnorm[1] is not None:
                if attrstr is not plotnorm[1]:
                    continue
            if plotnorm[2] is not None:
                if tstr is not plotnorm[2]:
                    continue
            if plotnorm[3] is not None:
                if roi is not plotnorm[3]:
                    continue
            if group == 'All':
                ds_t = ds.copy()
            elif group == 'Old':
                ds_t = dsO.copy()
            elif group == 'Young':
                ds_t = dsY.copy()
            print('plot ' + teststr)
            if tstr == 'pred2d':
                niceplot_pred(ds_t, niceplotfolder, savestr=savestr + teststr, roi=roi)
            else:
                for ia, attr in enumerate(plotattr[plotnorm[1] + plotnorm[2]]):
                    # if attr == 'difference':
                    #     tt = tests[teststr][0].t
                    it = ia
                    tt = tests[teststr][it].t
                    pp = tests[teststr][it].p
                    masknd = tests[teststr][it].masked_difference()
                    if 'time' in masknd.dimnames:
                        ylim = [mn, mx]
                    else:
                        ylim = [0, mx]
                    niceplot(ds_t, attrstr, niceplotfolder, teststr, tt, pp, masknd, attr=attr, ylim=ylim, roi=roi,
                             savestr=f'{savestr}_{attr}', peaktimes=peaktimes, mx2=mx2)

    # niceplot_fft(ds,niceplotfolder=niceplotfolder)
    # BF_rOvY_MmN(ds,roi=cortsub+['Brain-Stem'])

    return tests


def BF_rOvY_MmN(ds,roi=None):
    ds2 = Dataset()
    ds2['y'] = ds['r_smooth_correlation_model'] - ds['r_smooth_correlation_noise_mean']
    if roi is not None:
        if type(roi) is list:
            roi = [l for l in roi if l in ds2['y'].source.parc.as_labels()]
        ds2['y'] = ds2['y'].sub(source=roi)
    ds2['y'] = ds2['y'].mean('source')
    ds2['x'] = ds['youngflag']
    n1 = sum([1 for i in ds['youngflag'] if i == '1'])
    n0 = ds.n_cases - n1

    tst = test.TTestInd('y','x',ds=ds2,tail=0)
    print(f'p = {tst.p}')
    print(f't = {tst.t}')
    N = n0 * n1 / (n0 + n1)
    v = n0 + n1 - 2
    BF = BF_t(tst.t,N,v)
    print(f'BF = {BF}')
    return BF

def BF_t(t, N, v):
    def _integral_f(g, t, N, v):
        aa = (1 + N * g) ** (-0.5)
        bb = (1 + ((t ** 2) / ((1 + N * g) * v))) ** (-0.5 * (v + 1))
        cc = (2 * math.pi) ** (-0.5)
        dd = g ** (-3 / 2)
        ee = math.exp(-0.5 / g)
        return aa * bb * cc * dd * ee

    _int_f = lambda x: _integral_f(x,t,N,v)
    denominator = scipy.integrate.quad(_int_f, 0, math.inf)
    numerator = (1 + ((t ** 2) / v)) ** (-0.5 * (v + 1))
    return numerator / denominator[0]

def niceplot_pred(ds,niceplotfolder,savestr=None,roi=None):
    attr1 = ds.info[f'attrname_h'][0]
    attr2 = ds.info[f'attrname_h'][1]

    pred1 = ds[f'h_smooth_{attr1}_model'] - ds[f'h_smooth_{attr1}_noise_mean']
    pred2 = ds[f'h_smooth_{attr2}_model'] - ds[f'h_smooth_{attr2}_noise_mean']

    if roi is not None:
        if type(roi) is list:
            roi = [l for l in roi if l in pred1.source.parc.as_labels()]
        pred1 = pred1.sub(source=roi)
        pred2 = pred2.sub(source=roi)

    subjects = []
    conditions = []
    ndvar = []
    if 'space' in pred1.dimnames:
        pred1 = pred1.norm('space')
        pred2 = pred2.norm('space')
    pred1 = pred1.sub(time=(0.02,0.07))
    pred2 = pred2.sub(time=(0.02,0.07))
    pred1 = pred1.norm('time').mean('source')
    pred2 = pred2.norm('time').mean('source')
    for i in range(len(pred1)):
        ndvar.append(pred1[i])
        conditions.append('1')
        subjects.append(i)
        ndvar.append(pred2[i])
        conditions.append('0')
        subjects.append(i)

    ds2 = Dataset()
    ds2['y'] = combine(ndvar)
    ds2['subject'] = Factor(subjects, random=True)
    ds2['condition'] = Factor(conditions)
    import matplotlib as mpl
    mpl.rcParams['font.size'] = 24
    mpl.rcParams['boxplot.flierprops.linewidth'] = 2
    mpl.rcParams['boxplot.boxprops.linewidth'] = 2
    mpl.rcParams['boxplot.whiskerprops.linewidth'] = 2
    mpl.rcParams['boxplot.capprops.linewidth'] = 2
    mpl.rcParams['boxplot.medianprops.linewidth'] = 2
    mpl.rcParams['boxplot.meanprops.linewidth'] = 2

    p = plot.Boxplot('y', 'condition', match='subject', ds=ds2, h=10, w=6, top=0.015)
    p.figure.axes[0].set_xticks([])
    p.figure.axes[0].set_yticks([])
    p.figure.axes[0].set_xlabel('')
    p.figure.axes[0].set_ylabel('')

    p.figure.axes[0].spines['right'].set_visible(False)
    p.figure.axes[0].spines['top'].set_visible(False)
    if 'Old' in savestr:
        p.figure.axes[0].spines['left'].set_visible(False)
    p.draw()
    p.save(f'{niceplotfolder}/pred_boxplot_{savestr}{im_ext}',bbox_inches='tight')
    p.close()




def niceplot(ds, attrstr, niceplotfolder, teststr, tt, pp, masknd, attr = None, savestr='',
             ylim=None, roi=None, avgdir = True, peaktimes=None, mx2 = None):

    savestr = f'{teststr}_{savestr}'
    niceplotfolder = f'{niceplotfolder}/{teststr}'
    if not os.path.exists(niceplotfolder):
        os.makedirs(niceplotfolder)

    if attrstr == 'h':
        if attr == None:
            xnd = ds['h_yangh_model']
        elif attr == 'difference':
            xnd = masknd
        else:
            xnd = ds[attr]
    elif attrstr == 'r':
        if attr == None:
            xnd = ds['r_correlation_model']
        else:
            xnd = ds[attr]
    if roi is not None:
        if type(roi) is list:
            roi = [l for l in roi if l in xnd.source.parc.as_labels()]
        xnd = xnd.sub(source=roi)

    if roi is None and type(xnd.source) == VolumeSourceSpace:
        xnd = xnd.sub(source=cortsub+['Brain-Stem'])
        savestr = savestr + '_2roi'
        tt = tt.sub(source=cortsub+['Brain-Stem'])
        pp = pp.sub(source=cortsub+['Brain-Stem'])

    if 'case' in xnd.dimnames:
        xnd = xnd.mean('case')

    volflag = type(xnd.source) == VolumeSourceSpace
    trfflag = 'time' in xnd.dimnames
    twocols = trfflag and not volflag

    if trfflag:
        # if ylim is None:
        #     mx = xnd.max()
        #     mn = xnd.min()
        # else:
        #     mx = ylim[0]
        #     mn = ylim[1]


        # if volflag:
            # p = plot.GlassBrain.butterfly(xnd,h=glassbrain_h,vmin=mn,vmax=mx)
            # p[0].save_movie(f'{niceplotfolder}/{savestr}_movie_trf{mv_ext}')
            # p[1].save_movie(f'{niceplotfolder}/{savestr}_movie_brain{mv_ext}')
        if 'space' in xnd.dimnames:
            xndplot = xnd.norm('space')
        else:
            # p = plot.brain.butterfly(xnd,h=1000,w=2500,vmin=mn,vmax=mx)
            # p[0].save_movie(f'{niceplotfolder}/{savestr}_movie_trf{mv_ext}')
            # p[1].save_movie(f'{niceplotfolder}/{savestr}_movie_brain{mv_ext}')
            xndplot = xnd.copy()
        # if timewindow is not None:
        t1 = xndplot.time.times[np.argmax(xndplot.mean('source'))]
        # else:
        #     t1 = xndplot.sub(time=timewindow).time.times[np.argmax(xndplot.sub(time=timewindow).mean('source'))]
        if ylim is None:
            mm = xndplot.abs().max()
            if volflag:
                ylim = ( -mm/10, mm )
            else:
                ylim = (-mm, mm)
        legend = ['left hemisphere', 'right hemisphere', 'not significant']
        niceplot_trf(xndplot, masknd, t1, savestr = savestr, volflag = volflag, legend=legend, niceplotfolder=niceplotfolder, ylim=ylim)

        if peaktimes is None:
            ii = np.argmax(xndplot.max('source').x)
            tmax = xnd.time.times[ii]
            xnd = xnd.sub(time=xndplot.time.times[ii])
            xndplot = xndplot.sub(time=xndplot.time.times[ii])
        else:
            iadd = np.argmin(np.abs(xndplot.time.times-peaktimes[0]))
            ii = np.argmax(xndplot.max('source').sub(time=peaktimes).x)
            tmax = xnd.time.times[ii+iadd]
            xnd = xnd.sub(time=xndplot.time.times[ii+iadd])
            xndplot = xndplot.sub(time=xndplot.time.times[ii+iadd])
        savestr = f'{savestr}_{tmax:.4f}'

    if mx2 is None:
        if ylim is None:
            mx = xnd.abs().max()
            mn = xnd.abs().min()
        else:
            mx = ylim[1]
            mn = ylim[0]
    else:
        mx = mx2
        mn = -mx2

    if volflag:
        magma = cm.get_cmap('hot', 12)
        # frac = 0.9*mn/mx
        cutoff = 135#int(255*(0.5 + frac))
        nn = 256 - cutoff
        magma2 = magma(np.linspace(1, 0, 256 - cutoff))
        # magma2 /= np.max(magma2)
        magmavals = np.zeros((256,4))
        magmavals[:cutoff, :] = [0, 0, 0, 0]
        magmavals[cutoff:256,:] = magma2
        newcmp = ListedColormap(magmavals)
        
        if len(pp.dimnames) == 3:
            brainmask = (pp<0.05).x.any(2).any(1)
        elif len(pp.dimnames) == 2:
            brainmask = (pp<0.05).x.any(1)
        else:
            brainmask = pp < 0.05
        if sum(brainmask) != 0:

            xnd = xnd.sub(source=brainmask)
            p = plot.GlassBrain(xnd,cmap=newcmp,h=glassbrain_h,vmax=mx,annotate=False)
            p.save( f'{niceplotfolder}/{savestr}_brain{im_ext}')
            p.close()

            p = plot.GlassBrain(xnd,cmap=newcmp,h=glassbrain_h,vmax=mx,annotate=False,display_mode='l')
            p.save( f'{niceplotfolder}/{savestr}_brain_l{im_ext}')
            p.close()

            p = plot.GlassBrain(xnd,cmap=newcmp,h=glassbrain_h,vmax=mx,annotate=False,display_mode='r')
            p.save( f'{niceplotfolder}/{savestr}_brain_r{im_ext}')
            p.close()

            p = plot.GlassBrain(xnd,cmap=newcmp,h=glassbrain_h,vmax=mx,annotate=False,display_mode='z')
            p.save( f'{niceplotfolder}/{savestr}_brain_z{im_ext}')
            p.close()

            p = plot.GlassBrain(xnd,cmap=newcmp,h=glassbrain_h,vmax=mx,annotate=False,display_mode='y')
            p.save( f'{niceplotfolder}/{savestr}_brain_y{im_ext}')
            p.close()
    
    else:

        if not twocols:
            magma = cm.get_cmap('magma', 12)
            magmavals = magma(np.linspace(0,1,256))
            magmavals[:,3] = np.linspace(0,1,256)
            magmavals[:1,:] = [0,0,0,0.6]
            magmavals[1:3,:] = [0,0,0,0]
            nsval = 2*ylim[1]/256
            if len(pp.dimnames) == 1:
                for i in range(len(xnd.source)):
                    if pp[i] >= 0.05:
                        xnd[i] = nsval
        else:
            magma = cm.get_cmap('PuOr', 12)
            magmavals = magma(np.linspace(0,1,256))
            purple = [150/256, 24/256, 250/256]
            orange = [250/255, 200/256, 8/256]
            for iii in range(3):
                magmavals[:,iii] = np.linspace(purple[iii],orange[iii],256)
            magmavals[:100,3] = 1
            magmavals[100:128,3] = np.linspace(1,0,28)
            magmavals[128:156,3] = np.linspace(0,1,28)
            magmavals[156:,3] = 1
        newcmp = ListedColormap(magmavals)

        if len(pp.dimnames) == 2:
            brainmask = (pp<0.05).x.any(1)
        else:
            brainmask = (pp<0.05)
        # pdb.set_trace()


        vmax = mx
        vmin = mn

        if sum(brainmask) != 0:
            # xnd = xnd.sub(source=brainmask)
            xnd2 = complete_source_space(xnd)
            xnd = xnd2
            p = plot.brain.brain(xnd, cmap=newcmp,vmin=vmin,vmax=vmax, mask=False, h=1000, w=2500)
            p.save_image(f'{niceplotfolder}/{savestr}_brain{im_ext}')
            p.close()

            p = plot.brain.brain(xnd, cmap=newcmp,vmin=vmin,vmax=vmax, mask=False, h=1000, w=1250, hemi='lh')
            p.set_parallel_view(-15, -32, 47)
            p.save_image(f'{niceplotfolder}/{savestr}_brain_temporal_lh{im_ext}')
            p.close()
            p = plot.brain.brain(xnd, cmap=newcmp,vmin=vmin,vmax=vmax, mask=False, h=1000, w=1250, hemi='rh')
            p.set_parallel_view(-15, -32, 47)
            p.save_image(f'{niceplotfolder}/{savestr}_brain_temporal_rh{im_ext}')
            p.close()

    plot_colorbar(mx,mn,savestr,niceplotfolder,newcmp=newcmp)

    if trfflag and volflag and avgdir and 'space' in xnd.dimnames:
        p = plot.GlassBrain(xnd.norm('space'),cmap=newcmp,h=glassbrain_h,vmax=mx,annotate=False)
        p.save(f'{niceplotfolder}/{teststr}_{savestr}_norm_{im_ext}')
        p.close()
        glassbrain_avgdir(xnd,newcmp,mx,savestr,niceplotfolder)


def plot_colorbar(mx,mn,savestr,niceplotfolder,newcmp=None,frac=None):
    # tkmx = float(f'{mx*0.99:.1g}')
    if newcmp is None:
        magma = cm.get_cmap('YlOrRd', 12)
        cutoff = int(255 * frac)
        magma2 = magma(np.linspace(0, 1, 256 - cutoff))
        magmavals = np.zeros((256, 4))
        magmavals[:cutoff, :] = [0, 0, 0, 0]
        magmavals[cutoff:256, :] = magma2
        newcmp = ListedColormap(magmavals)
        mn = mx*frac

    import matplotlib as mpl
    mpl.rcParams['font.size'] = 50
    X, Y = np.mgrid[-2:3, -2:3]
    Z = mn*np.ones(X.shape)
    Z[0] = mx
    FIGSIZE = (10, 10)
    mpb = plt.pcolormesh(X, Y, Z, cmap=newcmp)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    cbar = plt.colorbar(mpb, ax=ax, ticks=np.array([mn, mx]), orientation='horizontal')
    cbar.ax.set_xticklabels([f'{float(f"{mn:.1g}")}', f'{float(f"{mx:.1g}")}'])
    ax.remove()
    plt.savefig(f'{niceplotfolder}/{savestr}_colorbar_h{im_ext}', bbox_inches='tight')
    plt.close()

    import matplotlib as mpl
    mpl.rcParams['font.size'] = 50
    X, Y = np.mgrid[-2:3, -2:3]
    Z = mn*np.ones(X.shape)
    Z[0] = mx
    FIGSIZE = (10, 10)
    mpb = plt.pcolormesh(X, Y, Z, cmap=newcmp)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    cbar = plt.colorbar(mpb, ax=ax, ticks=np.array([mn, mx]), orientation='vertical')
    cbar.ax.set_xticklabels([f'{float(f"{mn:.1g}")}', f'{float(f"{mx:.1g}")}'])
    ax.remove()
    plt.savefig(f'{niceplotfolder}/{savestr}_colorbar_v{im_ext}', bbox_inches='tight')
    plt.close()



def glassbrain_avgdir(xnd,newcmp,mx,savestr,niceplotfolder):
    iR = [i for i in range(len(xnd)) if xnd.source.coordinates[i][0]>0]
    iL = [i for i in range(len(xnd)) if xnd.source.coordinates[i][0]<0]
    iC = [i for i in range(len(xnd)) if xnd.source.coordinates[i][0]==0]

    xR = xnd.sub(source=iR)
    xL = xnd.sub(source=iL)
    xC = xnd.sub(source=iC)

    if xR.x.shape[0] == 0:
        return

    xRdir = xR.mean('source').x
    xLdir = xL.mean('source').x
    xCdir = xC.mean('source').x


    p = plot.GlassBrain(xnd.norm('space'),cmap=newcmp,h=glassbrain_h,vmax=mx,annotate=False,display_mode='lr')
    axL = p.figure.axes[1]
    axR = p.figure.axes[2]

    alen = 30000
    awid = 3
    oC1 = 12
    oC2 = 33

    if len(xL.source)>0:
        ax = axL
        xx = xL
        xd = xLdir
        i1 = 1
        i2 = 2
        ii = np.argmax(xx.norm('space'))
        pos = xx.source.coordinates[ii]
        ax.arrow(pos[i1]*1000,pos[i2]*1000,xd[i1]*alen,xd[i2]*alen,width=awid,color='k')

    if len(xR.source)>0:
        ax = axR
        xx = xR
        xd = xRdir
        i1 = 1
        i2 = 2
        ii = np.argmax(xx.norm('space'))
        pos = xx.source.coordinates[ii]
        ax.arrow(pos[i1]*1000,pos[i2]*1000,xd[i1]*alen,xd[i2]*alen,width=awid,color='k')
        #
    # ax = axC
    # xx = xL
    # xd = xLdir
    # i1 = 0
    # i2 = 1
    # ii = np.argmax(xx.norm('space'))
    # pos = xx.source.coordinates[ii]
    # ax.arrow(-pos[i1]*1000+oC1,pos[i2]*1000+oC2,-xd[i1]*alen,xd[i2]*alen,width=3,color='k')
    #
    # ax = axC
    # xx = xR
    # xd = xRdir
    # i1 = 0
    # i2 = 1
    # ii = np.argmax(xx.norm('space'))
    # pos = xx.source.coordinates[ii]
    # ax.arrow(-pos[i1]*1000+oC1,pos[i2]*1000+oC2,-xd[i1]*alen,xd[i2]*alen,width=3,color='k')

    p.draw()

    p.save(f'{niceplotfolder}/{savestr}_avgdir{im_ext}')
    p.close()


def niceplot_trf(xnd,maskeddiff,t1,savestr = '',volflag = False,times=None,
                 figsize = (20, 10),ylim=[None],
                 lhcolor = (0, 0.5, 1, 1), rhcolor = (1, 0.4, 0.4, 1), nscolor = (0.5,0.5,0.5,1), mlcolor = (0.4,0.7,0.2,1),
                 fontsize = 15, legend=['left hemisphere','right hemisphere','not significant'], niceplotfolder=niceplotfolder):

    if times is None:
        times = (xnd.time.times[0]+0.01, xnd.time.times[-1]-0.01)
    t1 *= 1000
    fontsize = 50
    figsize = (20,10)
    if not os.path.exists(niceplotfolder):
        os.makedirs(niceplotfolder)

    plt.figure(figsize=figsize)
    xnd = xnd.sub(time=times)
    maskeddiff = maskeddiff.sub(time=times)
    tstep = xnd.time.tstep
    xnd = resample(xnd,5000,window='hamming')
    tstep2 = xnd.time.tstep
    if volflag:
        if len(maskeddiff.dimnames) == 3:
            mm = maskeddiff.x.mask.any(axis=1)
        else:
            mm = maskeddiff.x.mask
    else:
        mm = maskeddiff.x.mask
    mm = np.repeat(mm,int(tstep/tstep2),axis=len(mm.shape)-1)
    z_order = len(xnd.source) - xnd.rms('time').x.argsort() - 1
    yy = xnd.x
    xx = xnd.time.times*1000
    yy1 = []
    xx1 = []
    for i in range(yy.shape[0]):
        yy1.append(xnd.sub(source=i))
        xx1.append(xx)
    yy2 = []
    xx2 = []
    for i in range(yy.shape[0]):
        yyy = []
        xxx = []
        for j in range(yy[i].shape[0]):
            if not mm[i][j]:
                yyy.append(yy[i][j])
                xxx.append(xx[j])
            else:
                yyy.append(None)
                xxx.append(None)
        yy2.append(np.array(yyy))
        xx2.append(np.array(xxx))

    for i in z_order:
        plt.plot(xx1[i],yy1[i],color=nscolor, linewidth = 0.3)

    mlFlag = False
    for i in z_order:
        if volflag:
            if xnd.source.coordinates[i][0]<0:
                color = lhcolor
            elif xnd.source.coordinates[i][0]>0:
                color = rhcolor
            else:
                color = mlcolor
                mlFlag = True
        else:
            if xnd.source[i][0] == 'L':
                color = lhcolor
            else:
                color = rhcolor
        plt.plot(xx2[i], yy2[i], color=color, linewidth=1)

    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    plt.xticks([i for i in range(0,int(1000*times[1]),50)])
    if ylim[0] != None:
        ymax = ylim[1]
        ymin = ylim[0]
    else:
        ymax = np.max(np.abs(yy1))
        ymin = -ymax

    ylim = [ymin,ymax]

    plt.ylim(ylim)

    plt.yticks([i for i in np.linspace(ymin,ymax,5)])

    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    ax = plt.axes()

    ax.tick_params(axis='x', which='major', pad=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_position('zero')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax.grid(which='major',axis='x',color=[0.7,0.7,0.7],linestyle='--',linewidth=2)
    ax.tick_params(direction='out',length=6,width=2,axis='both')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axes().set_position([0.1,0.2,0.9,0.9])
    plt.savefig(f'{niceplotfolder}/trf_{savestr}_{ymax}_{ymin}{im_ext}', bbox_inches='tight',
            pad_inches=0)
    plt.close()

    import pylab

    if mlFlag:
        custom_lines = [Line2D([0], [0], color=lhcolor, lw=4),
                        Line2D([0], [0], color=rhcolor, lw=4),
                        Line2D([0], [0], color=mlcolor, lw=4),
                        Line2D([0], [0], color=nscolor, lw=4)]
        legend = [legend[0],legend[1],'mid line',legend[2]]
    else:
        custom_lines = [Line2D([0], [0], color=lhcolor, lw=4),
                        Line2D([0], [0], color=rhcolor, lw=4),
                        Line2D([0], [0], color=nscolor, lw=4)]

    figlegend = pylab.figure(figsize=(12, 8))
    figlegend.legend(custom_lines, legend, 'center',prop={'size':fontsize})
    figlegend.savefig(f'{niceplotfolder}/trf_{savestr}_legend{im_ext}')
    plt.close()


def niceplot_fft(ds_in,savestr='',niceplotfolder=niceplotfolder,roi=None):
    print('niceplot_fft')
    niceplotfolder = f'{niceplotfolder}/FFT_TRF'
    if not os.path.exists(niceplotfolder):
        os.makedirs(niceplotfolder)
    ffts = []
    ffts2 = []
    names = []
    ffts_inds = []
    for i in [0,1]:
        for pred in ds_in.info['attrname_h']:
            ds = ds_in.sub(f'youngflag=="{i}"')
            predstr = 'h_'+pred+'_model'
            xnd_in = ds[predstr].copy()
            xnd_in.source.subjects_dir = subjects_dir
            # xnd = filter_data(xnd_in,125,115,l_trans_bandwidth=5,h_trans_bandwidth=5)
            # xnd = xnd_in.smooth(dim='source',window_size=0.005,window='gaussian')
            savestr = savestr + '_' + predstr
            xnd = xnd_in
            if roi is not None:
                xnd = xnd.sub(source=roi)
            fs2 = 0.5/xnd.time.tstep
            fres = 1
            ntimes = 2*int(fres*fs2)
            nadd = int(ntimes - xnd.shape[-1])
            xnd2 = NDVar(np.concatenate((xnd.x,np.zeros((xnd.x.shape[0],xnd.x.shape[1],xnd.x.shape[2],nadd))),axis=3),dims=(Case,xnd.source,xnd.space,UTS(xnd.time.tmin,xnd.time.tstep,ntimes)))

            ffts_ind = [xx.fft('time').norm('space').mean('source').x for xx in xnd2]
            ffts_inds.append(ffts_ind)
            ffts.append(xnd2.mean('case').fft('time').norm('space').mean('source').x)
            ffts2.append(xnd2.fft('time').norm('space').mean('case').mean('source').x)
            if i==0:
                names.append('O'+pred)
            else:
                names.append('Y'+pred)


        # p = plot.UTS(fftnd.sub(frequency=(0,200))**2,color='b',legend=False)
    # p.save(f'{niceplot_folder}/fft_TRF_{savestr}_{int(fftpeak)}{im_ext}')
    # p.close()

    fftstemp = []
    for ff in ffts:
        if len(ff) > 122:
            ff2 = ff
            ff2[118:122] = np.linspace(ff[118],ff[122],4)
            fftstemp.append(ff2)
            ffts = fftstemp
    import matplotlib.pyplot as plt
    colors = ['xkcd:crimson', 'xkcd:salmon', 'xkcd:cobalt', 'xkcd:sky blue']

    mxval = np.max([np.max(ffts[i]) for i in range(len(ffts))])
    # ffts = [f/mxval for f in ffts]
    mxi = np.argmax(ffts)
    mxi = int(mxi%501)


    plt.gcf().set_size_inches(15, 10)
    fsize = 40
    lwidth = 4

    plt.plot(ffts[1][:200], color=colors[1], linewidth=lwidth)
    plt.plot(ffts[2][:200], color=colors[2], linewidth=lwidth)
    plt.plot(ffts[0][:200], color=colors[0], linewidth=lwidth)
    plt.plot(ffts[3][:200], color=colors[3], linewidth=lwidth)

    # plt.axvline(mxi)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.xlim([0,200])
    plt.axes().spines['top'].set_visible(False)
    plt.axes().spines['right'].set_visible(False)
    plt.yticks([])
    plt.ylim([0,mxval*1.1])
    plt.xticks([i for i in range(0,210,25)])
    plt.grid(which='both',axis='x',linewidth=2)
    # for tick in plt.gca().xaxis.get_major_ticks():
    #     tick.label.set_fontsize(fsize)
    # plt.legend([ 'Younger HFE', 'Older HFE',  'Younger carrier', 'Older carrier'],fontsize=fsize)
    # plt.xlabel('Frequency [Hz]',fontsize=fsize)
    # plt.ylabel('Power',fontsize=fsize)
    # plt.plot([mxi, mxi], [0, mxval], 'k--', linewidth=lwidth)
    # import matplotlib.patches as patches
    # rect = patches.Rectangle((25, 0), 5, 5, linewidth=1, edgecolor='none', facecolor='white')
    # plt.axes().add_patch(rect)

    plt.savefig(f'{niceplotfolder}/FFT_vol{savestr}_{mxi}.png')

    plt.close()



    import pylab

    custom_lines = [Line2D([0], [0], color=colors[2], lw=4),
                    Line2D([0], [0], color=colors[0], lw=4),
                    Line2D([0], [0], color=colors[3], lw=4),
                    Line2D([0], [0], color=colors[1], lw=4)]
    legend = [ 'Younger HFE', 'Older HFE',  'Younger carrier', 'Older carrier']

    figlegend = pylab.figure(figsize=(12, 8))
    figlegend.legend(custom_lines, legend, 'center', prop={'size': fsize})
    figlegend.savefig(f'{niceplotfolder}/FFT_vol{savestr}_legend.png')
    plt.close()
    return ffts

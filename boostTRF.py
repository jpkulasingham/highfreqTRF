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
import boostingFunctions as bF
import itertools
import nibabel

mne.set_log_level('ERROR')
configure(n_workers=8)

time1 = time.time()

outfolder = 'results'
sqdFolder = '/Users/pranjeevan/Documents/highfreqTRF/sqd_folder'
stimuliFolder = '/Users/pranjeevan/Documents/highfreqTRF/stimuli_folder'
outputFolder = '/Users/pranjeevan/Documents/highfreqTRF/output_folder' + outfolder
forwardSolFolder = '/Users/pranjeevan/Documents/highfreqTRF/sourcespace/fwdpkl'
emptyroomFolder = '/Users/pranjeevan/Documents/highfreqTRF/sourcespace/empty room data'
mriFolder = '/Users/pranjeevan/Documents/highfreqTRF/sourcespace/mri'
aparcLabels = ['superiortemporal-lh','superiortemporal-rh','transversetemporal-lh','transversetemporal-rh']
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
temp = f'{outputFolder}/Source/Pickle'
if not os.path.exists(temp):
    os.makedirs(temp)
if os.path.exists(f'{outputFolder}/Source/boosting.txt'):
    os.remove(f'{outputFolder}/Source/boosting.txt')

srcspace_name = 'vol-7-cortex_brainstem_full'
parc_name = f'parc_sym-aparc+aseg-nowhite-{srcspace_name}.pkl'

nowhite = True
ctxll = ['inferiortemporal', 'middletemporal',
                       'superiortemporal','bankssts', 'transversetemporal']
subctxll = ['Brain-Stem', '3rd-Ventricle','Thalamus-Proper', 'VentralDC']
llA = ['ctx-lh-'+l for l in ctxll] + ['ctx-rh-'+l for l in ctxll] + subctxll + ['Left-'+l for l in subctxll] + ['Right-'+l for l in subctxll]
aparcvols = load.unpickle('aparcvol.pkl')

decimation = 1   #when loading data
datafilter = [70,300]
nperm = 3
ltranswidth = 10
htranswidth = 10
datafreq = 1000
boostingFraction = [-0.05,0.21]
basislen = 0.004
datatime = [0,60]

print('loading stimuli')
matstimfolder = stimuliFolder
carrier = {}
yangh = {}

mat7 = scipy.io.loadmat(f'{matstimfolder}/quiet_h-audspec-1000.mat')
mat8 = scipy.io.loadmat(f'{matstimfolder}/quiet_l-audspec-1000.mat')
freqs = Scalar('frequency',mat7['frequencies'][0],'Hz')
spec7 = NDVar(mat7['specgram'],dims=(UTS(0,0.001,60000),freqs))
spec8 = NDVar(mat8['specgram'],dims=(UTS(0,0.001,60000),freqs))
spec7 = spec7.sub(frequency=(300,4000))
spec8 = spec8.sub(frequency=(300,4000))
spec7filt= filter_data(spec7,datafilter[0],datafilter[1],filter_length='auto',method='fir',fir_design='firwin',l_trans_bandwidth = ltranswidth,h_trans_bandwidth = htranswidth)
spec8filt= filter_data(spec8,datafilter[0],datafilter[1],filter_length='auto',method='fir',fir_design='firwin',l_trans_bandwidth = ltranswidth,h_trans_bandwidth = htranswidth)
filt7 = spec7filt.mean('frequency')
filt8 = spec8filt.mean('frequency')
yangh[7]= filter_data(filt7,datafilter[0],datafilter[1],filter_length='auto',method='fir',fir_design='firwin',l_trans_bandwidth = ltranswidth,h_trans_bandwidth = htranswidth)
yangh[8]= filter_data(filt8,datafilter[0],datafilter[1],filter_length='auto',method='fir',fir_design='firwin',l_trans_bandwidth = ltranswidth,h_trans_bandwidth = htranswidth)

wav7 = load.wav('%s/quiet_h.wav' % stimuliFolder)
wav7.x = wav7.x.astype('float64')
wav7 = resample(wav7,1000)
wav8 = load.wav('%s/quiet_l.wav' % stimuliFolder)
wav8.x = wav8.x.astype('float64')
wav8 = resample(wav8,1000)
carrier[7] = filter_data(wav7,datafilter[0],datafilter[1],filter_length='auto',method='fir',fir_design='firwin',l_trans_bandwidth = ltranswidth,h_trans_bandwidth = htranswidth)
carrier[8] = filter_data(wav8,datafilter[0],datafilter[1],filter_length='auto',method='fir',fir_design='firwin',l_trans_bandwidth = ltranswidth,h_trans_bandwidth = htranswidth)

mat = scipy.io.loadmat('matlabOutHAYO.mat')
badchannelsAll = mat['badChannels'][0]
badchannelsNames = mat['subjectsAll'][0]

subjectFolders = [f for f in listdir(sqdFolder) if ~isfile(join(sqdFolder,f))]
for f in reversed(subjectFolders):
    if(f[0]=='.'):              #remove hidden files like .DS_Store
        subjectFolders.remove(f)
subjectFolders.sort()

rawE1 = mne.io.read_raw_kit(join(emptyroomFolder,'EmptyRoom_PreAM-2_08.02.16.sqd'),preload=True)
rawE2 = mne.io.read_raw_kit(join(emptyroomFolder,'ERT_07.25.14_4.14.sqd'),preload=True)

fnum = 1
fflag = 1
for subjectF in subjectFolders:
    t = time.time()
    subjectFolder = '%s/%s' %(sqdFolder,subjectF)
    print('\n\nsubject %d of %d : %s' % (fnum,len(subjectFolders),subjectFolder))

    sqdFileH = join(subjectFolder, '%s_QHmyF.sqd' % subjectF)
    print('loading file : %s' %(sqdFileH))
    rawH = mne.io.read_raw_kit(sqdFileH,stim_code='channel', stim=range(163,176),preload=True)
    
    sqdFileL = join(subjectFolder,  '%s_QLmyF.sqd' % subjectF)
    print('loading file : %s' %(sqdFileL))
    rawL = mne.io.read_raw_kit(sqdFileL,stim_code='channel', stim=range(163,176),preload=True)

    for j in range(0,len(badchannelsNames)):
        if (badchannelsNames[j][0] == subjectF):
            break
    badCH = badchannelsAll[j+8][0]
    badCL = badchannelsAll[j+9][0]

    if(len(badCH)!=0):
        badCH = [e + 1 for e in badCH]
        for i in range(0, len(badCH)):
            rawH.info['bads'].append('MEG %03d' % (badCH[i]))
        print(rawH.info['bads'])

    if(len(badCL)!=0):
        badCL = [e + 1 for e in badCL]
        for i in range(0, len(badCL)):
            rawL.info['bads'].append('MEG %03d' % (badCL[i]))
        print(rawL.info['bads'])

    if(rawH.info['kit_system_id']==rawE1.info['kit_system_id']):
        rawE = mne.io.RawArray(rawE1.get_data(),rawE1.info)
        eflag = 1
    if(rawH.info['kit_system_id']==rawE2.info['kit_system_id']):
        rawE = mne.io.RawArray(rawE1.get_data(),rawE2.info)
        eflag = 2

    rawH = rawH.filter(datafilter[0],datafilter[1],filter_length='auto',method='fir',fir_design='firwin',l_trans_bandwidth = ltranswidth,h_trans_bandwidth = htranswidth)
    rawL = rawL.filter(datafilter[0],datafilter[1],filter_length='auto',method='fir',fir_design='firwin',l_trans_bandwidth = ltranswidth,h_trans_bandwidth = htranswidth)
    rawE = rawE.filter(datafilter[0],datafilter[1],filter_length='auto',method='fir',fir_design='firwin',l_trans_bandwidth = ltranswidth,h_trans_bandwidth = htranswidth)

    forwardSolFile = f'{forwardSolFolder}/{srcspace_name}/{subjectF}-{srcspace_name}-fwd.fif'
    fwd = mne.read_forward_solution(forwardSolFile)
    fwd['info']['working_dir'] = None
    fwd['info']['command_line'] = None
    fwd['info']['meas_file'] = None

    cov = mne.compute_raw_covariance(rawE)

    # Fixed orientation source estimate:
    invsol = mne.minimum_norm.make_inverse_operator(rawH.info, fwd, cov, loose=1)

    dsH = []
    dsH = load.fiff.events(rawH)
    dsH = dsH.sub(dsH['trigger'] != 163)
    
    dsL = []
    dsL = load.fiff.events(rawL)
    dsL = dsL.sub(dsL['trigger'] != 163)

    stims = load.tsv('HAYOstimulitriggers.txt', delimiter=None, ignore_missing = True)

    stims_aH = align1(stims, dsH['trigger'], 'ch')
    stims_aL = align1(stims, dsL['trigger'], 'ch')

    dsH.update(stims_aH['wav', 'condition'])
    dsL.update(stims_aL['wav', 'condition'])


    dsHP = dsH.copy()
    dsLP = dsL.copy()

    dsHP['yangh'] = combine([yangh[name] for name in dsH['wav']])
    dsHP['carrier'] = combine([carrier[name] for name in dsH['wav']])

    dsLP['yangh'] = combine([yangh[name] for name in dsL['wav']])
    dsLP['carrier'] = combine([carrier[name] for name in dsL['wav']])

    print('inverse solution')
    epH = load.fiff.mne_epochs(dsH,datatime[0],datatime[1])
    epL = load.fiff.mne_epochs(dsL,datatime[0],datatime[1])

    stcH = mne.minimum_norm.apply_inverse_epochs(epH[0], invsol, lambda2=1, method='MNE',pick_ori='vector')
    sndH = load.fiff.stc_ndvar(stcH,subjectF,f'{srcspace_name}',subjects_dir=mriFolder)
    pdb.set_trace()

    sndH = load.fiff.stc_ndvar(stcH,subjectF,f'{srcspace_name}',subjects_dir=mriFolder)
    sndH.source.parc = load.unpickle(parc_name)
    voilist = [s for s in llA if s in sndH.source.parc]
    sndH = sndH.sub(source=voilist)
    sndH = sndH.sub(time=(datatime[0],datatime[1]))

    del stcH

    stcL = mne.minimum_norm.apply_inverse_epochs(epL, invsol, lambda2=1, method='MNE',pick_ori='vector')

    sndL = load.fiff.stc_ndvar(stcL,subjectF,f'{srcspace_name}',subjects_dir=mriFolder)
    sndL.source.parc = load.unpickle(parc_name)
    voilist = [s for s in llA if s in sndH.source.parc]
    sndL = sndL.sub(source=voilist)
    sndL = sndL.sub(time=(datatime[0],datatime[1]))

    del stcL

    print('loading MEG epochs')


    dsHP['source'] = sndH
    dsLP['source'] = sndL

    dsHP['source'] = dsHP['source'].sub(time=(datatime[0],datatime[1]))
    dsLP['source'] = dsLP['source'].sub(time=(datatime[0],datatime[1]))

    dsP = combine([dsHP, dsLP])
    del dsHP
    del dsLP

    #downsampling
    if(datafreq!=1000):
        print('downsampling')
        dsP['source'] = resample(dsP['source'], datafreq)
        dsP['yangh'] = resample(dsP['yangh'], datafreq)
        dsP['carrier'] = resample(dsP['carrier'], datafreq)

    nperm = 3
    predstr = 'yangh'
    dsP = bF.permutePred(dsP,predstr,nperm)

    predstr = 'carrier'
    dsP = bF.permutePred(dsP,predstr,nperm)
    del sndH, sndL, rawH, rawL
    predstr = ['yangh','carrier']

    pdb.set_trace()
    bF.boostWrap_multpred_multperm(dsP, predstr, outputFolder, subjectF, f'{srcspace_name}', boostingFraction, basislen, 4, nperm, rstr= 'source')
    del dsP['source']

    elapsed = time.time() - t
    mins, secs = divmod(elapsed, 60)
    hrs, mins = divmod(mins, 60)

    with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
        f.write('elapsed %d:%02d:%02d\n\n'%(hrs,mins,secs))

    print('elapsed %d:%02d:%02d\n\n'%(hrs,mins,secs))

    elapsed = time.time() - time1
    mins, secs = divmod(elapsed, 60)
    hrs, mins = divmod(mins, 60)
    with open(f'{outputFolder}/Source/boosting.txt', 'a+') as f:
        f.write('file %d of %d: total elapsed %d:%02d:%02d\n\n'%(fnum, len(subjectFolders), hrs,mins,secs))
        f.write('------------------------------------------------\n\n')

    del dsP
    fnum = fnum+1

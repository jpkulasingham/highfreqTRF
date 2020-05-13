from eelbrain import *
import numpy as np
p = plot.UTS(NDVar([1,2],UTS(0,1,2)))
p.close()
import matplotlib.pyplot as plt
import scipy
import os


filtc = [70, 200]
specrange = (300, 4000)
datafreq = 500
fsfilt = 1000
pitchms = 10
stimuli_folder = '/Users/pranjeevan/Documents/HAYO/stimuliQuiet'
matstim_folder = stimuli_folder
outstim_folder = '/Users/pranjeevan/Documents/HAYO/predictorsQuiet'
if not os.path.exists(outstim_folder):
    os.makedirs(outstim_folder)


FILT_KWARGS = {
    'filter_length': 'auto',
    'method': 'fir',
    'fir_design': 'firwin',
    'l_trans_bandwidth': 5,
    'h_trans_bandwidth': 5,
}

pitchc = [60, 200]


means = {}
times = {}
for c in ['h', 'l']:
    print(c)
    mat = scipy.io.loadmat(f'{matstim_folder}/quiet_{c}-audspec-1000.mat')
    freqs = Scalar('frequency', mat['frequencies'][0], 'Hz')
    spec = NDVar(mat['specgram'], dims=(UTS(0, 0.001, 60000), freqs))
    spec = spec.sub(frequency=specrange)
    specfilt = filter_data(spec, filtc[0], filtc[1], **FILT_KWARGS)
    filt = specfilt.mean('frequency')
    hfe = filter_data(filt, filtc[0], filtc[1], **FILT_KWARGS)
    hfe -= hfe.mean()
    hfe /= hfe.std()
    save.pickle(hfe, f'{outstim_folder}/quiet_{c}|hfe.pickle')


    wav = load.wav(f'{stimuli_folder}/quiet_{c}.wav')
    wav.x = wav.x.astype('float64')
    wav = resample(wav, fsfilt)
    carrier = filter_data(wav, filtc[0], filtc[1], **FILT_KWARGS)
    carrier -= carrier.mean()
    carrier /= carrier.std()
    save.pickle(carrier, f'{outstim_folder}/quiet_{c}|carrier.pickle')

    infile = f'/Users/pranjeevan/Documents/BitBucket/personal/HAYO/praat_pitch_out_{pitchms}ms_{c}.txt'
    means1 = []
    times1 = []
    with open(infile,'r') as f:
        for ln in f:
            ll = ln.split(' ')
            if ll[0] == '':
                continue
            t1 = float(ll[0])
            t2 = float(ll[1])
            mm = ll[2]
            means1.append(mm)
            times1.append((t1,t2))
    means2 = [float(x) for x in means1 if '.' in x]
    times2 = [t for t,x in zip(times1, means1) if '.' in x]
    means[c] = [x for x in means2 if x < pitchc[1] and x > pitchc[0]]
    times[c] = [t for t,x in zip(times2, means2) if x < pitchc[1] and x > pitchc[0]]

meansall = means['h'] + means['l']
Nq = 2
qthresh = np.quantile(meansall, [i / Nq for i in range(1, Nq)])
qthresh_str = [f'{q:.2f}' for q in qthresh]

ffts = {}
for c in ['h','l']:
    means1 = means[c]
    times1 = times[c]
    qq = []
    for i in range(Nq):
        qq.append(np.zeros(60000))
    for tt,mn in zip(times1,means1):
        t1 = int(tt[0]*1000)
        t2 = int(tt[1]*1000)
        highflag = True
        for i, th in enumerate(qthresh):
            if mn < th:
                qq[i][t1:t2] = 1
                highflag = False
        if highflag:
            qq[-1][t1:t2] = 1
    qqnds = []
    for i, q in enumerate(qq):
        qqnd = NDVar(q,UTS(0,0.001,60000))
        qqnds.append(qqnd)
        save.pickle(qqnd, f'{outstim_folder}/q{c}_p{pitchms}_Nq_q{i}_{"-".join(qthresh_str)}.pickle')

    for pred in ['hfe', 'carrier']:
        stim = load.unpickle(f'{outstim_folder}/quiet_{c}|{pred}.pickle')
        kk = f'FFT_quiet|{pred}.png'
        if kk in ffts.keys():
            ffts[kk] = combine([ffts[kk], stim.fft().sub(frequency=(0,250))]).mean('case')
        else:
            ffts[kk] = stim.fft().sub(frequency=(0,250))
        xx = []
        for i in range(Nq):
            xx.append(np.zeros(60000))
        for i in range(60000):
            for iq, qq in enumerate(qqnds):
                if qq.x[i] == 1:
                    xx[iq][i] = stim.x[i]
        xxnds = []
        for i, x in enumerate(xx):
            xxnd = NDVar(x, UTS(0,0.001,60000))
            save.pickle(xxnd, f'{outstim_folder}/quiet_{c}|{pred}_p{pitchms}_{Nq}_q{i}.pickle')
            kk = f'FFT_quiet|{pred}_p{pitchms}_{Nq}_q{i}.png'
            if kk in ffts.keys():
                ffts[kk] = combine([ffts[kk], xxnd.fft().sub(frequency=(0, 250))]).mean('case')
            else:
                ffts[kk] = xxnd.fft().sub(frequency=(0, 250))
            xxnds.append(xxnd)


xffts = {}
mxvals = []
for k in ffts.keys():
    xnd = ffts[k]
    xff = scipy.signal.savgol_filter(xnd.x, int(xnd.x.size / 40), 3)
    mxvals.append(np.max(xff))
    xffts[k] = xff

ff = np.linspace(0,250, xff.size)
mxval = np.max(mxvals)
colors = ['blue', 'cyan', 'periwinkle', 'green', 'lime green','olive']
colors = ['xkcd:' + c for c in colors]
fsize = 40
lw = 4
plt.gcf().set_size_inches(15, 10)
for i, k in enumerate(xffts.keys()):
    xff = xffts[k]
    plt.plot(ff, xff ** 2, color=colors[i], linewidth=lw, alpha=0.5)

plt.xlim(0, 250)
plt.xticks([i for i in range(0, 260, 25)])
plt.yticks([])
plt.axes().spines['top'].set_visible(False)
plt.axes().spines['right'].set_visible(False)
plt.grid(linewidth=2)
for tick in plt.gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)
plt.savefig(f'{outstim_folder}/stimsFFT.png')
plt.close()
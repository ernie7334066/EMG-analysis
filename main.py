#!/usr/bin/env python3
import os, cv2
import spikeextractors as se
import spikeinterface.widgets as sw
import spikeinterface.toolkit as st
import pandas as pd
import pyopenephys as pyo
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.animation import FFMpegWriter
import numpy as np
from scipy.stats import zscore
from scipy import signal
from scipy.fft import fft
import datetime

import DataRoutine as dr
import DirsInput as di
import helper as h

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

np.random.seed(0)
ProjectName = 'Octopus Project'
today = str(datetime.date.today())
outfd = di.analysisfd(ProjectName)+'/EMG_signal'
if not os.path.isdir(outfd):
    os.mkdir(outfd)
parent_dir = di.datafd(ProjectName)+'/Bath recording'

electricStim_session3 = ['/octo136_2020-10-04_16-04-30',
                        '/octo137_2020-10-04_16-05-36',
                        '/octo138_2020-10-04_16-07-25',
                        '/octo139_2020-10-04_16-08-25',
                        '/octo140_2020-10-04_16-11-23',
                        '/octo142_2020-10-04_16-42-48',
                        '/octo143_2020-10-04_16-46-13',
                        '/octo144_2020-10-04_16-51-19',
                        '/octo145_2020-10-04_16-52-57']

def EMG_near_electstim(session_dirs=electricStim_session,bandpassFreq=[],Yaxis_offset=0,t_win=[-1.5,1.5],
                  channel_ids=[],channel_pos=[],common_reference=False):
        
    if len(channel_pos)>0:
        sortInd = np.argsort(channel_pos)
        channel_ids_sorted = channel_ids[sortInd]
        channel_ids_ref=channel_ids_sorted[np.arange(1,len(channel_ids),2)]
        channel_ids=channel_ids_sorted[np.arange(0,len(channel_ids),2)]        
        
    recordings,sessionIDs,events = ExtractRecording(session_dirs)
    fig_all, axes_all = plt.subplots(1,2,figsize=[10,5],constrained_layout=True,sharey=True)            
    for idx,(recording_orig,sessionID,event) in enumerate(zip(recordings,sessionIDs,events)):
        if len(channel_ids)>0:
            recording = se.SubRecordingExtractor(recording_orig,channel_ids=channel_ids)
        else:
            channel_ids = recording.get_channel_ids()
        if common_reference:
           #recording = st.preprocessing.common_reference(recording,reference='median')
           recording_ref = se.SubRecordingExtractor(recording_orig,channel_ids=channel_ids_ref)
           traces_ref=recording_ref.get_traces()
           traces=recording.get_traces()
           recording = se.NumpyRecordingExtractor(timeseries=traces-traces_ref, 
                                                  sampling_frequency=recording_ref.get_sampling_frequency())
           channel_ids = recording.get_channel_ids()

        n_chs = len(recording.get_channel_ids())
        sampfreq = recording.get_sampling_frequency()
        signal_stim = dr.loadADC(parent_dir+session_dirs[idx], channel=0)
        tIndices, _ = h.signal_timing(signal_stim)
        if len(tIndices)==0:
            #use stimulation artifact from recording channel to find stimulation timing
            signal_stim = recording.get_traces(channel_ids=channel_ids[0])
            onsets, offsets = h.signal_timing(np.squeeze(signal_stim))
            tIndices,_ = h.combineEvents(onsets, offsets, 0.5, sampfreq)

        if len(bandpassFreq)==2:
            recording = st.preprocessing.bandpass_filter(recording, filter_type='butter',order=4,filtfilt=False,
                                                         freq_min=bandpassFreq[0], freq_max=bandpassFreq[1])
        taxis_aligned = np.arange(t_win[0],t_win[1],1/sampfreq)
        nFrame = recording.get_num_frames()
        if abs(Yaxis_offset)>0:
            offset=np.repeat(np.expand_dims(np.linspace(0,Yaxis_offset,num=n_chs),axis=1),
                             len(taxis_aligned),axis=1)
        else:
            offset=0
        data_xcorr_all=[]
        sig_norm_all, amp_all=[],[]
        var_basline, var_sig=[],[]
        for idx,tindex in enumerate(tIndices):
            if tindex+t_win[1]*sampfreq>nFrame:
                continue
            # Extract signal traces near events
            traces=recording.get_traces(channel_ids=channel_ids,
                                        start_frame=tindex+t_win[0]*sampfreq,end_frame=tindex+t_win[1]*sampfreq)            
            # plot traces
            #if (idx == 0) or (sessionID == 'octo127') or (sessionID == 'octo126') \
            #              or (sessionID == 'octo129') or (sessionID == 'octo131'):                
            traces_plot = traces+offset            
            fig, ax = plt.subplots(1,1,figsize=[10,15],constrained_layout=True)
            ax.plot(taxis_aligned,traces_plot.T)
            #ax.axvspan(0,.005,color='k',alpha=0.2)
            ax.set_title('Trial '+str(idx))
            ax.get_yaxis().set_ticks([])
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_xlabel('Time (s)')
            ax.plot([t_win[0],t_win[0]],[0,-50],color='black')
            ax.set_ylim(Yaxis_offset-abs(Yaxis_offset)/10,abs(Yaxis_offset)/10)
            fname = outfd+'/EMG_electStim_'+sessionID+'_Trial'+str(idx)
            plt.savefig(fname+'.pdf',format='pdf')
            plt.close()
            
            # plot heat map
            traces_rmv = np.copy(traces)
            traces_rmv[:,(taxis_aligned>=0) & (taxis_aligned<.06)]=0
            traces_rmv = np.abs(traces_rmv)
            sig_norm = traces_rmv/np.mean(traces_rmv[:,taxis_aligned<0],axis=1,keepdims=True)            
            for ii in range(sig_norm.shape[0]):
                sig_norm[ii,:] = h.smooth(sig_norm[ii,:],window_len=int(.04*sampfreq)+1,window='hamming')
            fig, ax = plt.subplots(1,1,figsize=[5,5],constrained_layout=True)
            # cbar, _ = h.Hist3D_plot(fig, ax, sig_norm, taxis_aligned, cmap='viridis')
            im = ax.imshow(sig_norm, origin='lower', extent=[taxis_aligned[0], taxis_aligned[-1], 1, sig_norm.shape[0]],
                           vmin=np.min(sig_norm),vmax=np.max(sig_norm),interpolation='nearest',cmap='viridis',aspect='auto')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Normalized signal')
            ax.set_xlabel('Time (s)')
            ax.set_title('Trial '+str(idx))
            ax.invert_yaxis()
            ax.set_ylabel('Distal to proximal arm')
            fname2 = outfd+'/EMG_electStim_heatMap_'+sessionID+'_Trial'+str(idx)
            plt.savefig(fname2+'.pdf',format='pdf')
            plt.close()
            sig_norm_all.append(sig_norm)
            
            # Variance level
            var_basline.append(np.var(traces[:,taxis_aligned<0],axis=1))
            var_sig.append(np.var(traces[:,taxis_aligned>.06],axis=1))
            
            # FFT spectrum
            amp,freqAxis = FFT_spect(traces[:,taxis_aligned>.06])
            amp_all.append(amp)
        
        # mean signal plot    
        sig_norm_all = np.array(sig_norm_all)
        fig, ax = plt.subplots(1,1,figsize=[5,5],constrained_layout=True)
        #cbar, _ = h.Hist3D_plot(fig, ax, np.mean(sig_norm_all,axis=0), taxis_aligned, cmap='viridis')
        im = ax.imshow(np.mean(sig_norm_all,axis=0), origin='lower', extent=[taxis_aligned[0], taxis_aligned[-1], 1, sig_norm_all.shape[0]],
               vmin=np.min(np.mean(sig_norm_all,axis=0)),vmax=np.max(np.mean(sig_norm_all,axis=0)),interpolation='nearest', cmap='viridis',aspect='auto')
        cbar = fig.colorbar(im, ax=ax)       
        cbar.set_label('Mean normalize signal')
        ax.set_xlabel('Time (s)')
        ax.set_title(event['messages'][-1])
        ax.invert_yaxis()
        ax.set_ylabel('Distal to proximal arm')
        fname2 = outfd+'/EMG_electStim_heatMap_'+sessionID
        plt.savefig(fname2+'.pdf',format='pdf')
        plt.close()
        
        # FFT frequency spectrum
        fig, ax = plt.subplots(1,1,figsize=[5,5],constrained_layout=True)
        amp_all = np.array(amp_all)
        cbar, _ = h.Hist3D_plot(fig, ax, np.mean(amp_all,axis=0), freqAxis, cmap='viridis')
        cbar.set_label('Amplitude')
        ax.set_xlabel('Frequency (Hz)')
        ax.invert_yaxis()
        ax.set_ylabel('Distal to proximal arm')
        fname2 = outfd+'/EMG_freqSpect_'+sessionID
        plt.savefig(fname2+'.pdf',format='pdf')
        plt.close()
        
        # signal noise plot
        if sessionID == 'octo144':
            var_basline1 = np.mean(np.array(var_basline),axis=0)
            axes_all[0].plot(np.random.randn(var_basline1.size)*0.05,var_basline1,'b*')
            var_sig1 = np.mean(np.array(var_sig),axis=0)
            axes_all[1].plot(np.random.randn(var_sig1.size)*0.05,var_sig1,'b*',label='flexible probe')
        elif sessionID == 'octo136':
            var_basline2 = np.mean(np.array(var_basline),axis=0)
            axes_all[0].plot(np.random.randn(var_basline2.size)*0.05+1,var_basline2,'r*')
            var_sig2 = np.mean(np.array(var_sig),axis=0)
            axes_all[1].plot(np.random.randn(var_sig2.size)*0.05+1,var_sig2,'r*',label='soft probe')

    axes_all[0].set_yscale('log')
    axes_all[0].set_ylabel('Variance')
    axes_all[0].set_title('Baseline')
    axes_all[1].set_title('Stimulation')
    axes_all[0].set_xticks(np.arange(2))
    axes_all[0].set_xticklabels(['flexible probe','soft probe'])
    axes_all[1].set_xticks(np.arange(2))
    axes_all[1].set_xticklabels(['flexible probe','soft probe'])
    fname = outfd+'/Signal_noise_SoftVSFlexible'
    fig_all.savefig(fname+'.pdf',format='pdf')
    plt.close()    

def FFT_spect(signals,fs=30000,freqRange=[1,500]):
    amp=[]
    for idx in range(signals.shape[0]):
        signal = signals[idx,:]
        # Number of sample points
        N = len(signal)
        # sample spacing
        T = 1.0 / fs
        w = np.blackman(N)
        ywf = fft(signal*w)
        amp.append(2.0/N * np.abs(ywf[1:N//2]))
    freq = np.linspace(0.0, 1.0/(2.0*T), N//2)[1:N//2]
    ind = np.where((freq>=freqRange[0]) & (freq<=freqRange[1]))[0]
    amp = np.array(amp)
    return amp[:,ind],freq[ind]

def ExtractRecording(session_dirs):
    recordings,sessionIDs,Events = [],[],[]
    for session_dir in session_dirs:
        str_ind = session_dir.find('2020')
        sessionIDs.append(session_dir[1:str_ind-1])
        datadir = parent_dir+session_dir
        recordings.append(se.OpenEphysRecordingExtractor(datadir))
        Events.append(dr.loadEventMessage(datadir+'/messages.events'))
    return recordings, sessionIDs, Events
   
def GetFiles(datadir,ext):
    files=[]
    for file in os.listdir(datadir):
        if file.endswith(ext):
            files.append(os.path.join(datadir, file))
    if len(files)==1:
        return files[0]
    else:
        return files

if __name__ == "__main__":  
    
    EMG_near_electstim(session_dirs=electricStim_session3,Yaxis_offset=-2000,bandpassFreq=[60,800],common_reference=True,t_win=[-.1,.6],
                       channel_ids = np.arange(32),
                       channel_pos=[30,26,22,18,14,10,6,2,0,4,8,12,16,20,24,28,32,27,23,19,15,11,7,3,1,5,9,13,17,21,25,29])
    
    
    
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:07:58 2020

@author: ernie
"""
import numpy as np
import csv
from typing import List
import scipy.io as si
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
from math import pi
import pandas


def signal_timing(signal:np.array):
    
    signal_bin = ConvertToBinary(signal)
    first = np.array([False])
    onsets_ind = np.append(first, signal_bin[1:] > signal_bin[:-1])
    offsets_ind = np.append(first, signal_bin[:-1] > signal_bin[1:])
    
    onsets = np.where(onsets_ind)[0]
    offsets = np.where(offsets_ind)[0]
    if len(onsets)>len(offsets):
        onsets = onsets[:-1]
    elif len(offsets)>len(onsets):
        offsets = offsets[1:]
    return onsets,offsets

    # # find onset and offset
    # exceed = signal > np.mean(signal)+threshold*np.std(signal)
    # temp = exceed[1:]^exceed[:-1]
    # ind_l = [i for i, val in enumerate(temp) if val]
    # ind = np.array(ind_l)
    # if ind.size % 2 == 1:
    #     ind = ind[:-1]
    # onset_all = ind[::2]
    # offset_all = ind[1::2]
    # return onset_all,offset_all

def select_unit_id(tsvfile:str, groupLabel:str, columnName='group'):
    cluster_id_list = []
    with open(tsvfile) as tsv:
        reader = csv.DictReader(tsv, dialect='excel-tab')
        for row in reader:
            if row[columnName] == groupLabel:
                cluster_id_list.append(int(row['cluster_id']))
    return cluster_id_list


def combineEvents(onset_all:np.array, offset_all:np.array, isi_thr:float, sampfreq:float):
    # combine a train of pulse into one event
    ISI = (onset_all[1:]-offset_all[:-1])/sampfreq
    separate_on = np.append([True],ISI>isi_thr,axis=0)
    separate_off = np.append(ISI>isi_thr,[True],axis=0)
    onsets = onset_all[separate_on]
    offsets = offset_all[separate_off]
    return onsets, offsets

def raster_plot(sorting:object, cluster_id:List, onsets:List, sampfreq:float, t_win:float, axes:List,
                sortOnset_win=[]):
    for unit, ax in zip(cluster_id,axes):
        spktrain = sorting.get_unit_spike_train(unit).astype('int64')
        spkpos = []
        for onset in onsets:
            spkdiff = np.array(spktrain-onset,dtype=float)
            spkdiff /= sampfreq
            spkpos.append(spkdiff[abs(spkdiff)<=t_win])    
        if len(sortOnset_win)>0:
            spkpos = sorted(spkpos, key = lambda x: np.sum((x>=sortOnset_win[0]) & (x<sortOnset_win[1]) )) 
        # make subplot for each unit
        ax.eventplot(spkpos,colors='black')
        ax.set_title('Unit '+str(unit))
    
def Hist3D_plot(fig:object, axs:object, data:np.array, xaxis:np.array, cmap='RdBu_r'):   
    if cmap == 'RdBu_r':
        z_min, z_max = -np.abs(data).max(), np.abs(data).max()
    else:
        z_min, z_max = np.min(data), np.max(data)
    # plot figure   
    pcm = axs.pcolormesh(xaxis, list(range(data.shape[0]+1)), data, cmap=cmap, vmin=z_min, vmax=z_max)
    cbar = fig.colorbar(pcm, ax=axs)
    return cbar, pcm

        
def PSTH_plot(sorting:object, cluster_id:List, onsets:List, sampfreq:float, t_win:float, axes:List,
              bin_width = 0.005):
    for unit, ax in zip(cluster_id,axes):
        spktrain = np.array(sorting.get_unit_spike_train(unit),dtype=int)
        spkpos = []
        for onset in onsets:
            spkdiff = np.array(spktrain-onset,dtype=float)
            spkdiff /= sampfreq
            spkt = spkdiff[abs(spkdiff)<=t_win]
            spkpos = np.append(spkpos,spkt)
        # make subplot for each unit
        taxis = list(np.arange(-t_win,t_win,bin_width))
        count,edge = np.histogram(spkpos, bins=taxis) 
        ax.bar(edge[:-1], count/(bin_width*len(onsets)), width=bin_width,align='edge')
        
def ConvertToBinary(X:np.array):
    X_binary = np.array([False]*len(X))
    X_binary[X>np.max(X)/2] = True
    return X_binary

def ReadRotaryEncoder(A_orig:np.array, B_orig:np.array, ppr=256, radius=8):
    A = ConvertToBinary(A_orig-np.min(A_orig))
    B = ConvertToBinary(B_orig-np.min(B_orig))
    first = np.array([False])
    A_rise = np.append(first, A[1:] > A[:-1])
    A_fall = np.append(first, A[:-1]> A[1:])
    B_rise = np.append(first, B[1:] > B[:-1])
    B_fall = np.append(first, B[:-1]> B[1:])
    cck = A & B
    counter = np.array([0]*len(A))
    counter[A_rise | A_fall| B_rise| B_fall] = 1
    counter[cck] *= -1
    pos = np.cumsum(counter)/4/ppr*2*pi*radius
    return pos

def ReadRotaryEncoder_oneCH(A_orig:np.array, ppr=256, radius=8):
    A = ConvertToBinary(A_orig-np.min(A_orig))
    first = np.array([False])
    A_rise = np.append(first, A[1:] > A[:-1])
    A_fall = np.append(first, A[:-1]> A[1:])
    counter = np.array([0]*len(A))
    counter[A_rise | A_fall] = 1
    pos = np.cumsum(counter)/2/ppr*2*pi*radius
    return pos
    
def CalibratePosition(hall_sensor:np.array,position:np.array,default_lap_distance=80):
    '''
    Calibrate position using the hall sensor

    Parameters
    ----------
    hall_sensor: np.array
        Hall sensor data from a analog channel
    position: np.array
        Position data decoded from rotary encoder
    default_lap_distance: float
        Default lap distance used in case there are less than two laps

    Returns
    -------
    pos_calibrated: np.array
        calibrated position using the hall sensor
    lapIndex: np.array
        time indices for the start of each lap
    '''
    # invert signal and substract baseline
    data = -hall_sensor+np.max(hall_sensor)    
    onsets_all,offsets_all = signal_timing(data)
    onsets, _ = combineEvents(onsets_all, offsets_all, isi_thr=1, sampfreq=30000)
    if len(onsets)<2:
        lap_distance = default_lap_distance
        pos_calibrated = position.copy()
        if len(onsets)>0:
           # begin condition
           pos_calibrated[:onsets[0]] = pos_calibrated[:onsets[0]]-position[onsets[0]]+lap_distance
           # end condition
           pos_calibrated[onsets[-1]:] = pos_calibrated[onsets[-1]:]-position[onsets[-1]]
    else:
        interlap_distance = np.diff(position[onsets])
        lap_distance = np.median(interlap_distance)   
        
        # Check missing hall sensor events
        missings = np.where(interlap_distance>1.5*lap_distance)[0]
        if len(missings)>0:
            print('There are '+str(len(missings))+' missing Hall sensor events')
            print('Fill in missing events')
            onsets_fill = list(onsets.copy())
            for missing in missings:
                pos_relative = position-position[onsets[missing]]
                tIndex = np.argmin(np.abs(pos_relative-lap_distance))
                onsets_fill.insert(missing+1,tIndex)
            onsets = np.array(onsets_fill)    
        
        # Check duplicate hall sensor events
        interlap_distance = np.diff(position[onsets])
        duplicates =  np.where(interlap_distance<.5*lap_distance)[0]
        if len(duplicates)>0:
            print('There are '+str(len(duplicates))+' duplicated Hall sensor events')
            print('Remove duplicated events')
            onsets_removed = list(onsets.copy())
            onsets_removed.remove(duplicates+1)
            onsets = np.array(onsets_removed)
            
        # Normalize position in between hall sensor events to correct integration error
        pos_calibrated = position.copy()
        onset1 = onsets[0]
        for onset in onsets[1:]:
            pos = pos_calibrated[onset1:onset]
            pos_norm = (pos-np.min(pos))/(np.max(pos)-np.min(pos))*lap_distance
            pos_calibrated[onset1:onset] = pos_norm
            onset1 = onset
        # begin condition
        onsets_first = onsets[0]
        pos_calibrated[:onsets_first] = pos_calibrated[:onsets_first]-position[onsets_first]+lap_distance
        if  np.any(pos_calibrated[:onsets_first]<0):
              pos_begin = pos_calibrated[:onsets_first]-np.min(pos_calibrated[:onsets_first])
              n = int(np.floor(np.max(pos_begin))/lap_distance)
              print('There are '+str(n)+' missing Hall sensor events at the start')
              print('Fill in missing events at the start')
              for ii in range(n):
                  onsets = np.concatenate(([np.where(pos_begin>lap_distance*ii)[0][0]],onsets))
              pos_calibrated[:onsets_first] = pos_begin%lap_distance
        # end condition
        onsets_last = onsets[-1]
        pos_calibrated[onsets_last:] = pos_calibrated[onsets_last:]-position[onsets_last]
        if  np.any(pos_calibrated[onsets_last:]>lap_distance):
            pos_end = pos_calibrated[onsets_last:]
            n = int(np.floor(np.max(pos_end)/lap_distance))
            print('There are '+str(n)+' missing Hall sensor events at the end')
            print('Fill in missing events at the end')
            for ii in range(n):
                onsets = np.concatenate((onsets,[np.where(pos_calibrated[onsets_last:]>lap_distance*(ii+1))[0][0]+onsets_last]))
            pos_calibrated[onsets_last:] = pos_end%lap_distance 
    lapIndex = onsets
    return pos_calibrated, lapIndex
    
def compute_distances(speed:np.array, sampfreq=3000, smoothBinWidth=1):
    # Adapted from Barna's code
    smspd = np.asarray(pandas.DataFrame(speed).ewm(span=int(sampfreq*smoothBinWidth)).mean())[:, 0]
    events = np.zeros(len(speed))
    anyevent = np.zeros(len(speed), dtype='bool')
    durations = []
    durations_times = []
    eid, md = 0, 0
    for t in range(len(speed)):
        if speed[t] > 2:
            if md == 0:
                md = 1
                eid += 1
                durations.append(1)
                durations_times.append(t)
                events[t] = eid
                anyevent[t] = 1
            else:
                events[t] = eid
                durations[-1] += 1
                anyevent[t] = 1
        elif speed[t] > 0.1 and md == 1:
            events[t] = eid
            durations[-1] += 1
            anyevent[t] = 1
        else:
            md = 0
    # expand movement events with one sec
    movement = np.zeros(anyevent.shape, dtype='bool')
    l = len(anyevent)
    for i in range(l):
        start, stop = max(0, int(i - sampfreq/2)), min(int(i + sampfreq/2), l)
        if True in anyevent[start:stop]:
            movement[i] = 1
    # gapless_move
    gapless = np.copy(anyevent)
    ready = False
    n = sampfreq
    while not ready:
        ready = True
        for t, m in enumerate(gapless):
            if not m:
                if np.any(gapless[t - n:t]) and np.any(gapless[t:t + n]):
                    gapless[t] = 1
                    ready = False
    return gapless, smspd

def startstop(gapless, smspd, speed, pos, sampfreq,
              duration=5, gap=3, ret_loc='actual', span=None):
    # Adapted from Barna's code
    mov = gapless
    duration *= sampfreq
    gap *= sampfreq
    duration = int(duration)
    gap = int(gap)
    if span is None:
        span = gap, len(pos) - gap
    # collect stops
    stops, starts = [], []
    t = span[0] + duration
    while t < span[1] - gap:
        if not np.any(mov[t:t + gap]) and np.all(mov[t - duration:t]):
            t0 = t
            if ret_loc == 'peak':
                while smspd[t0] < smspd[t0 - 1]:
                    t0 -= 1
                stops.append(t0)
                while mov[t0]:
                    t0 -= 1
                starts.append(t0)
            elif ret_loc == 'stopped':
                stops.append(t)
                while mov[t0 - 1]:
                    t0 -= 1
                starts.append(t0)
            elif ret_loc == 'actual':
                # go back while raw speed is zero
                while speed[t - 1] == 0 and t > duration:
                    t -= 1
                if t > duration:
                    stops.append(t)
                    while mov[t0 - 1] and t0 > duration:
                        t0 -= 1
                    while speed[t0] == 0:
                        t0 += 1
                    starts.append(t0)
            t += gap
        t += 1
    return starts, stops

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'halfnorm']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'halfnorm'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    elif window == 'halfnorm':
        xx = np.linspace(scipy.stats.halfnorm.ppf(0.01),scipy.stats.halfnorm.ppf(0.99),window_len)
        w=scipy.stats.halfnorm.pdf(xx)
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[window_len//2:-window_len//2+1]


def loadSpeed(mat_name:str, roi_name:str):
    mat_dic = si.loadmat(mat_name)
    roi = mat_dic[roi_name]
    speed = np.squeeze(abs(roi))
    return speed           

#function to downsample LFP
def reducebyave(arr, n):         
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)
        
def EventTrigTFR(recording,onsets,ch_ids,fname,t_win=[-0.1,0.1],freqRange=[1,200],freqScale='linear'):
    sampfreq = recording.get_sampling_frequency()
    taxis = np.arange(t_win[0],t_win[1],1/sampfreq)
    if freqScale == 'linear':
        freq = np.linspace(freqRange[0],freqRange[1])
    elif freqScale == 'log':
        freq = np.logspace(np.log10(freqRange[0]), np.log10(freqRange[1]))
    w = 6. # wavelevt width 
    widths = w*sampfreq / (2*freq*np.pi)
    traces = recording.get_traces(channel_ids=ch_ids,start_frame=onsets+int(t_win[0]*sampfreq),
                                 end_frame=onsets+int(t_win[1]*sampfreq))
    fig, axes = plt.subplots(len(ch_ids),2,figsize=[10,15],sharex=True)
    for idx,ch_id in enumerate(ch_ids):
        cwtm = scipy.signal.cwt(np.squeeze(traces[idx,:]), scipy.signal.morlet2, widths, w=w)
        tfr_aligned = np.abs(cwtm)
        im = axes[idx,0].pcolormesh(taxis, freq, tfr_aligned, cmap='viridis',shading='gouraud')
        fig.colorbar(im,ax=axes[idx,0])       
        axes[idx,1].plot(taxis,np.squeeze(traces[idx,:]))
    
    plt.savefig(fname+'.png',format='png')
    plt.close()
    return tfr_aligned,taxis,freq

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
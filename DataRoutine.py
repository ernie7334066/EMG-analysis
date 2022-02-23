#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:34:03 2020

@author: ernie
"""

import os, pickle
import spikeextractors as se
import spikeinterface.toolkit as st
import numpy as np
import scipy.io, csv
import matplotlib.pyplot as plt
import pyopenephys as pyo
import OpenEphys as oe

import DirsInput as di
import helper as h
from spikesorting.spikesorting_pipeline import ProbeInfo

# =============================================================================
# Open ephys related functions
# =============================================================================
def loadSpikeData(phy_folder:str, ch_loc_reset=True, reload=False, probeType=None):
    sorting = se.PhySortingExtractor(folder_path=phy_folder)
    if probeType is None:
        probeType = get_probe_info(phy_folder)
    if probeType == 'Probe128J':
       recording = se.BinDatRecordingExtractor(phy_folder+'/Recording.bin',30000,128,'int16')
       if not reload:
           recording = st.preprocessing.bandpass_filter(recording)
       ch_list,clusterIDs = extract_max_ch(phy_folder)
       sorting.set_units_property(unit_ids=clusterIDs,property_name='max_channel',values=ch_list)       
    elif (probeType != 'Probe128J') and reload:
        ind = phy_folder.find('/phy2')
        recording = se.OpenEphysRecordingExtractor(phy_folder[:ind])
    else:
        recording = se.PhyRecordingExtractor(phy_folder)
    if ch_loc_reset:       
       ch_ids, groups, ch_loc = ProbeInfo(probeType)
       recording.set_channel_locations(ch_loc, channel_ids=ch_ids)
       recording.set_channel_groups(groups=groups,channel_ids=ch_ids)       
    sampfreq = recording.get_sampling_frequency()    
    # convert group id from str to int
    unit_ids = sorting.get_unit_ids()
    for unit_id in unit_ids:
        unit_property_names = sorting.get_unit_property_names(unit_id)
        if 'group' in unit_property_names:
            group = sorting.get_unit_property(unit_id, 'group')
        elif 'ch_group' in unit_property_names:
            group = sorting.get_unit_property(unit_id, 'ch_group')
        else:
            max_channel = st.postprocessing.get_unit_max_channels(recording,sorting,unit_ids=unit_id)[0]
            group = groups[max_channel]
        sorting.set_unit_property(unit_id, 'group', int(group))
    return recording, sorting, sampfreq

def loadEventMessage(fname):
    f=open(fname,'rb')
    lines = f.read(1024).decode().split('\n')
    Event = {'timeindices':[],'messages':[]}
    for line in lines:
        if len(line)==0:
            continue
        parts = line.split()
        Event['timeindices'].append(int(parts[0]))
        message=" "
        Event['messages'].append(message.join(parts[1:]))
    f.close()
    return Event

def loadEventChannels(fname:str):
    Event = oe.load(fname)
    return Event

def loadADC(folder_dir:str, channel:int):
    # obtain adc trace
    adcFile = pyo.File(folder_dir,'ADC',ch_selected=channel).experiments[0].recordings[0]
    if adcFile.format == 'binary':
       signal = np.squeeze(adcFile.analog_signals[0].signal[32+channel,:])       
    else:
       signal = adcFile.analog_signals[0].signal[0]
    return signal

def EventTimeIndex(Event:dict,channelID:int):
    selected = Event['channel']==channelID
    oneset = Event['eventId']==1
    offset = Event['eventId']==0
    
    onsetInd = Event['timestamps'][selected & oneset]
    offsetInd = Event['timestamps'][selected & offset]
    if onsetInd.size>offsetInd.size:
        onsetInd = onsetInd[:-1]
    return onsetInd,offsetInd
    
def GetFirstTP(Event:dict):
    messages = Event['messages']
    timestamps = Event['timeindices']
    for timestamp,message in zip(timestamps,messages):
        if 'start time' in message:
            return timestamp
        
# =============================================================================
# Phy related functions        
# =============================================================================
def extract_max_ch(phy_folder:str):
    tsvfile = phy_folder+'/cluster_info.tsv'
    ch_list=[]
    clusterIDs=[]
    with open(tsvfile) as tsv:
        reader = csv.DictReader(tsv, dialect='excel-tab')
        for row in reader:
            ch_list.append(int(row['ch']))
            clusterIDs.append(int(row['id']))
    return ch_list,clusterIDs

def selectCluster(phy_folder:str, groupLabel='good',tsv='cluster_group.tsv',columnName='group'):    
    # read cluster group info
    tsvfile = phy_folder+'/'+tsv
    cluster_id_selected = h.select_unit_id(tsvfile, groupLabel,columnName=columnName)
    cluster_id_selected = list(set(cluster_id_selected))    
    return cluster_id_selected

def get_probe_info(phy_folder:str):
    projectNames = di.GetProjectNames()
    for projectName in projectNames:
        if projectName in phy_folder:
           dirs, probeTypes = di.dirs(projectName,Rotary=None) 
           break
    for idx,dir1 in enumerate(dirs):
        if dir1 in phy_folder:
           probeType = probeTypes[idx]
    return probeType

# =============================================================================
# General I/O functions
# =============================================================================
def save_data(fname,Data):
    fid = open( fname, 'wb' ) 
    pickle.dump( Data, fid)
    fid.close()

def load_data(fname):
    fid = open( fname, 'rb' ) 
    Data = pickle.load(fid)
    fid.close()
    return Data
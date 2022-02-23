#!/usr/bin/env python3

import csv, os
import pandas as pd

from RunZeta import loadData as ld

filepath = os.path.realpath(__file__)
ind = filepath.find('Python')
parent_path = filepath[:ind-1]
csvfile = parent_path+'/SessionInfo.csv'
miceInfo_file = parent_path+'/Mice list.xlsx'
    
def datafd(ProjectName='SNCGProject'):   
    return parent_path+'/'+ProjectName+'/Data'

def analysisfd(ProjectName='SNCGProject'):
    return parent_path+'/'+ProjectName+'/Analysis'

def optoTaggedUnit(ProjectName='SNCGProject',exp_group=None):    
    if ProjectName == 'SNCGProject':
        OptoTagSessionID = ['EH12_006','EH12_007','EH14_001','EH21_001','EH22_002','EH22_003','EH22_004']
        OptoTagUnitIDs =   [18,        48,         35,        12,        69,        36,        43       ]
    
    elif ProjectName == 'AxaxProject_opto':
        if (exp_group is None) or (exp_group == 'experiment'):
            OptoTagSessionID = ['EH18_001','EH18_003','EH19_002','EH19_002','EH19_002','EH20_001','EH20_001','EH30_003','EH31_001','EH31_005','EH33_002','EH33_002']
            OptoTagUnitIDs =   [ 5,         17,        20,        33,        37,        6,         9,         8,         6,         33,        172,       200]          
        elif exp_group == 'experiment_inhibit':           
            SortedData = ld(outfd=analysisfd(ProjectName)+'/OptoStim_zeta_experiment_inhibit')
            unitInfo = [temp for idx,temp in enumerate(SortedData['unitInfo_sort']) if SortedData['InFR_sign_sort'][idx]==-1]
            OptoTagSessionID = [unitSessionID[0][1:9] for unitSessionID in unitInfo]
            OptoTagUnitIDs = [unitID[1] for unitID in unitInfo]
    return OptoTagSessionID,OptoTagUnitIDs

def dirs(ProjectName='SNCGProject', Rotary='TRUE', sorting='TRUE', 
         exp_group=None, expression_level=['','good'], Hall=None):    
    dirs_list, probeType = [], []
    if Hall is None:
        Hall = ['TRUE','FALSE']
    elif isinstance(Hall,str):
        Hall = [Hall]
    if Rotary is None:
        Rotary = ['TRUE','FALSE']
    with open(csvfile) as csv_i:
        reader = csv.DictReader(csv_i, delimiter=',')
        for row in reader:
            if (row['Project_folder_name'] == ProjectName and
                row['Rotary_encoder'] in Rotary and
                row['Sorting_completed'] == sorting and
                row['Expression_level'] in expression_level and
                row['Hall_sensor'] in Hall):
                if exp_group is not None:
                    if row['Experiment_group'] == exp_group:
                       if row['Session_full_name'] not in RepeatedSessions():
                           dirs_list.append('/'+row['Session_full_name'])
                           probeType.append(row['Probe_type'])
                else:
                    if row['Session_full_name'] not in RepeatedSessions():
                        dirs_list.append('/'+row['Session_full_name'])
                        probeType.append(row['Probe_type'])

    return dirs_list, probeType

def MiceID(dir1:str):
    ind = dir1.find('EH')
    ind2 = dir1.find('_')
    return dir1[ind:ind2]

def MiceGender(dir1:str):
    m_id = MiceID(dir1)
    miceInfo = pd.read_excel(miceInfo_file, index_col=0)
    return miceInfo.loc[m_id,'Gender']
    
def SessionID(dir1:str):
    ind = dir1.find('_')
    ind2 = dir1.find('_',ind+1)
    return dir1[ind+1:ind2]

def Get_hemisphere(dir1:str):
    if dir1[0] == '/':
        dir1 = dir1[1:]
    with open(csvfile) as csv_i:
        reader = csv.DictReader(csv_i, delimiter=',')
        for row in reader:
            if row['Session_full_name'] == dir1:
                return row['Hemisphere']

def GetProjectNames():            
    ProjectNames=[]
    with open(csvfile) as csv_i:
        reader = csv.DictReader(csv_i, delimiter=',')
        for row in reader:
            if row['Project_folder_name'] == 'practice':
                continue
            ProjectNames.append(row['Project_folder_name'])
    return list(set(ProjectNames))
        
def ExtractSessionInfo(dir1:str,ColumnName:str):
    if dir1[0] == '/':
        dir1 = dir1[1:]
    with open(csvfile) as csv_i:
        reader = csv.DictReader(csv_i, delimiter=',')
        for row in reader:
            if row['Session_full_name'] == dir1:
                return row[ColumnName]
            
def RepeatedSessions():
    repeated_sessions = ['EH18_002_2020-05-15_10-05-40',
                         'EH18_004_2020-05-16_10-24-11',
                         'EH27_001_2020-10-25_10-13-07']
    return repeated_sessions
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 17:18:42 2022

@author: gniew
"""
import easygui
import pandas as pd
import numpy as np
from termcolor import colored
import math 
# VARIABLES

group = ["SAL", "CNO"]
df = None


class Joystick_analyzer:
    def __init__(self):
        self.column_name = ["Time_in_min", "Event_Marker", "TrialCt", "JoyPos_X", "JoyPos_Y", "Amplitude_Pos", "Base_JoyPos_X", "Base_JoyPos_Y", "SolOpenDuration", "DelayToRew", "ITI", "Threshold", "Fail_attempts", "Sum_of_fail_attempts", "Lick_state", "Sum_licks", "Type_move"]
        self.list_of_files = easygui.fileopenbox(title="Select a file", filetypes= "*.csv",  multiple=True)
        self.gen_df = (pd.read_csv(i,  header=None) for i in self.list_of_files)
        self.list_of_files = [i.split("\\")[-1] for i in  self.list_of_files]
        self.list_of_files = [i.split(".")[0] for i in  self.list_of_files]
        self.list_of_df = []
        self.group = ["SAL", "CNO"]
    def pre_proccesing(self):
        global df
        for i,j in enumerate(self.gen_df):
           j[0]= j[0].apply(lambda x : round(((x/(1000*60))%60),2))
           j[0].columns = self.column_name
           df = j
           df.columns = self.column_name
           #max_row = df["Time_in_min"].iloc[np.argmax(df["Time_in_min"])]
           max_row = df["Time_in_min"].tolist()
           answer = max_row.index(30.01 + max_row[0])
           #max_row = df["Time_in_min"].apply(lambda x : x.index())
           print(answer)
           df.columns = self.column_name
           df = df.drop(df.index[answer:])
           self.list_of_df.append(j)
           
           #print(self.list_of_df)
    #def find_bugs(self):
        #for i in self.list_of_df:
            
    
    def files_name(self):
        for i,j in enumerate(self.list_of_files):
            if self.group[0] in str(j):
                print(colored(i,"red"), colored(j, "red"))
            elif self.group[1] in str(j):
                print(colored(i,"green"), colored(j, "green"))
object_joy = Joystick_analyzer()
object_joy.pre_proccesing()
object_joy.files_name()
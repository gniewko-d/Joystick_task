# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 17:34:20 2022

@author: malgo
"""

import easygui
import pandas as pd
import numpy as np
from termcolor import colored
import math 
import tkinter as tk
import seaborn as sns;# sns.set_theme()
from tkinter import messagebox
import matplotlib.pyplot as plt
# VARIABLES
df = None
from scipy.signal import savgol_filter
class Joystick_analyzer:
    def __init__(self):
        self.column_name = ["Time_in_sec", "Event_Marker", "TrialCt", "JoyPos_X", "JoyPos_Y", "Amplitude_Pos", "Base_JoyPos_X", "Base_JoyPos_Y", "SolOpenDuration", "DelayToRew", "ITI", "Threshold", "Fail_attempts", "Sum_of_fail_attempts", "Lick_state", "Sum_licks", "Type_move"]
        self.list_of_files = easygui.fileopenbox(title="Select a file", filetypes= "*.csv",  multiple=True)
        self.gen_df = (pd.read_csv(i,  header=None) for i in self.list_of_files)
        self.list_of_files = [i.split("\\")[-1] for i in  self.list_of_files]
        self.list_of_files = [i.split(".")[0] for i in  self.list_of_files]
        self.list_of_df = []
        self.group = ["SAL", "CNO"]
    
    
    def pre_proccesing(self):
        global df
        for i,j in enumerate(self.gen_df):
           j[0]= j[0].apply(lambda x : round((x/1000),2))
           j.columns = self.column_name
           df = j
           df.columns = self.column_name
           max_row = df["Time_in_sec"].tolist()
           max_row = [round(i) for i in max_row]
           answer = max_row.index(1801 + max_row[0])
           j = j.drop(j.index[answer-4:])
           df.columns = self.column_name
           df = df.drop(df.index[answer-4:])
           self.list_of_df.append(j)
           
           
    def amplitude(self, event_markers = [0,1,2,3,4], hue = None, kde = False):
        global df
        amplitude_all =[]
        assert len(self.list_of_df) == len(self.list_of_files)
        df_amplitude = pd.DataFrame(columns= ["TrialCt", "Mouse_ID", "Amplitude_Pos", "Event_Marker"])
        for l,i in enumerate(self.list_of_df):
            trial_max = i["TrialCt"].max()
            for k in event_markers:
                amplitude_ = [i.loc[(i["TrialCt"] == j) & (i["Event_Marker"] == k), "Amplitude_Pos"].max() for j in range(1, trial_max + 1)]
                mouse_id = [self.list_of_files[l] for m in range(1, trial_max + 1)]
                event_marker = [k for j in range(1, trial_max + 1)]
                dict_to_add = {"TrialCt": range(1, trial_max + 1), "Mouse_ID": mouse_id, "Amplitude_Pos": amplitude_, "Event_Marker": event_marker}
                df_amplitude = df_amplitude.append(pd.DataFrame(dict_to_add))
                df_amplitude.reset_index(inplace = True, drop = True)
        sns.set_style('ticks')
        sns.displot(df_amplitude, x = "Amplitude_Pos", hue = hue, col = "Mouse_ID", kde = kde, color = "green", palette = "tab10")
        main = tk.Tk()
        msg = tk.messagebox.askquestion ('Save window','Do you want to save graphs?',icon = 'warning')
        if msg == "yes":
            main.destroy()
            save_file_v1 = easygui.diropenbox(msg = "Select folder for a save location", title = "Typical window")
            save_file_v1 = save_file_v1 + "//" + self.list_of_files[0] + "__" + self.list_of_files[-1] + ".svg"
            plt.savefig(save_file_v1)
        else:
            main.destroy()
        return df_amplitude
    
    def lick_histogram(self, pre_stim = 2, post_stim = 2, group = "all", marker= "ro"): 
       assert len(self.list_of_df) == len(self.list_of_files)
       
       start, stop = pre_stim * 19, post_stim * 19
       columns = np.linspace(start = -pre_stim, stop = post_stim, num = start + stop + 1)
       x = [round(i,2) for i in columns]
       columns_ = [str(i) for i in x]
       x = np.array(x)

       columns_.append("Animal_ID")
       df_licks_group = pd.DataFrame(columns= columns_, index = [i for i in range(0, len(self.list_of_df)+1)])
       list_do_hist = []
       for l,i in enumerate(self.list_of_df):
            index_events = i.index[i['Event_Marker'] == 2].tolist()
            trial_max = i["TrialCt"].max()
            list_value = [i.iloc[j-start:j+stop + 1, 14].tolist() for j in index_events]
            df_licks = pd.DataFrame(list_value,columns= [str(round(k,2)) for k in columns])
            prob_lick = df_licks.apply(lambda x: x.value_counts())
            prob_lick = round(prob_lick / trial_max,2)
            prob_lick = prob_lick.iloc[1, :].tolist()
            prob_lick.append(self.list_of_files[l])
            df_licks_group.iloc[l] = prob_lick
       df_licks_group.iloc[len(self.list_of_df), 0: len(columns_)-1] = df_licks_group.mean()
       df_licks_group.loc[len(self.list_of_df), "Animal_ID"] = "Mean"
       y =  df_licks_group.iloc[:,0:-1].values.tolist()
       y = np.array(y[0])
       
       yhat = savgol_filter(y, 9, 3)
       #fig, axs = plt.subplots(2,1)
       plt.plot(x,yhat, marker)
       plt.show()
       plt.plot(x,yhat)
        
    def files_name(self):
        for i,j in enumerate(self.list_of_files):
            if self.group[0] in str(j):
                print(colored(i,"red"), colored(j, "red"))
            elif self.group[1] in str(j):
                print(colored(i,"green"), colored(j, "green"))
            else:
                print(colored(i,"blue"), colored(j, "blue"))



object_joy = Joystick_analyzer()
object_joy.pre_proccesing()
object_joy.lick_histogram()
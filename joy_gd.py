# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 16:41:17 2023

@author: malgo
"""

import easygui
import pandas as pd
import numpy as np
from itertools import chain
from termcolor import colored
import math 
import progressbar
import tkinter as tk
import seaborn as sns;# sns.set_theme()
from tkinter import messagebox
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

df_porb_reward = None

# VARIABLES
df = None
df_result_amplitude = None
df_result_lick = None
df_result_type_move = None
df_lick_counts = None
df_result_veloctiy = None
xd = None 
xd1 = None
switcher = False
u_to_mm = 10  # 10 units t0 1 mm <--- <---
base_joypos_X = 505
base_joypos_Y = 515

class Joystick_analyzer:
    def __init__(self, opto = False):
        self.opto = opto
        if self.opto:
            print("--------> Opto mode activated <--------")
            self.column_name = ["Time_in_sec", "Event_Marker", "TrialCt", "JoyPos_X", "JoyPos_Y", "Amplitude_Pos", "Base_JoyPos_X", "Base_JoyPos_Y", "SolOpenDuration", "DelayToRew", "ITI", "Threshold", "Fail_attempts", "Sum_of_fail_attempts", "Lick_state", "Sum_licks", "Trial_type", "stim", "NormalTrialCt", "StimTrialCt"]
            self.list_of_files = easygui.fileopenbox(title="Select a file", filetypes= "*.csv",  multiple=True)
            self.gen_df = (pd.read_csv(i, header=None) for i in self.list_of_files)
            self.list_of_files = [i.split("\\")[-1] for i in  self.list_of_files]
            self.list_of_files = [i.split(".")[0] for i in  self.list_of_files]
            self.list_of_files = [[i +"_stim", i + "_nostim"] for i in  self.list_of_files]
            self.list_of_files = list(chain.from_iterable(self.list_of_files))
            self.list_of_files_stim = [i for i in self.list_of_files if "_stim" in i]
            self.list_of_files_nostim = [i for i in self.list_of_files if "_nostim" in i]
            self.list_of_df = []
            self.group = ["SAL", "CNO"]
        else:
            self.column_name = ["Time_in_sec", "Event_Marker", "TrialCt", "JoyPos_X", "JoyPos_Y", "Amplitude_Pos", "Base_JoyPos_X", "Base_JoyPos_Y", "SolOpenDuration", "DelayToRew", "ITI", "Threshold", "Fail_attempts", "Sum_of_fail_attempts", "Lick_state", "Sum_licks", "Type_move"]
            self.list_of_files = easygui.fileopenbox(title="Select a file", filetypes= "*.csv",  multiple=True)
            self.gen_df = (pd.read_csv(i, header=None) for i in self.list_of_files)
            self.list_of_files = [i.split("\\")[-1] for i in  self.list_of_files]
            self.list_of_files = [i.split(".")[0] for i in  self.list_of_files]
            self.list_of_df = []
            self.group = ["SAL", "CNO"]
    
    def pre_proccesing(self, cut_to = False):
        global df, switcher
        self.new_max_v2 = []
        self.trials_conteiner = []
        for i,j in enumerate(self.gen_df):
           j[0]= j[0].apply(lambda x : round((x/1000),2))
           j.columns = self.column_name
           df = j
           df.columns = self.column_name
           max_row = df["Time_in_sec"].tolist()
           max_row = [round(i) for i in max_row]
           #answer = max_row.index(1801 + max_row[0])
           #j = j.drop(j.index[answer-4:])
           trial_max = j["TrialCt"].max()
           bugged_index = [j.loc[j["TrialCt"] == l, "Event_Marker"].lt(0).idxmax() for l in range(1, trial_max + 1)]
           j.iloc[bugged_index, 5] = 0
           df.columns = self.column_name
           df["Amplitude_Pos_mm"] = j["Amplitude_Pos"].apply(lambda x: x / u_to_mm)
           
           delta_dist = df["Amplitude_Pos_mm"].tolist()
           delta_dist = [abs(delta_dist[ff] - delta_dist[ff-1]) for ff in range(1, len(delta_dist))]
           delta_dist.insert(0, 0)
           df["delta_Amplitude_Pos_mm"] = delta_dist
           df.iloc[bugged_index, 5] = 0
           j["Amplitude_Pos_mm" ] = j["Amplitude_Pos"].apply(lambda x: x/ u_to_mm)
           j["delta_Amplitude_Pos_mm"] = delta_dist
           
           time_list = df["Time_in_sec"].tolist()
           current_velocity = [ round((delta_dist[ii] / (time_list[ii] - time_list[ii-1]))/10 ,2) for ii in range(1, len(time_list))]
           current_velocity.insert(0, 0)
           df["Current_velocity_cm_s"] = current_velocity
           j["Current_velocity_cm_s"] = current_velocity
           if cut_to:
               df_sort = df.iloc[(df["Time_in_sec"]- cut_to).abs().argsort()[:1]]
               drop_start = df_sort.index.tolist()[0]
               df = df.iloc[0:drop_start]
               j = j.iloc[0: drop_start]
           if self.opto:
               j_stim = j.loc[j["Trial_type"] == 2, :]
               j_stim.reset_index(inplace = True, drop = True)
               j_stim_index_len = len(set(j.loc[ j["Trial_type"] == 2, "TrialCt"].tolist()))
               j_index_len = len(set(j.loc[ j["Trial_type"] == 1, "TrialCt"].tolist()))
               self.trials_conteiner.append(sorted(set(j.loc[ j["Trial_type"] == 2, "TrialCt"].tolist())))
               self.trials_conteiner.append(sorted(set(j.loc[j["Trial_type"] == 1, "TrialCt"].tolist())))
               j = j.loc[ j["Trial_type"] == 1, :]
               j.reset_index(inplace = True, drop = True)
               self.list_of_df.append(j_stim)
               self.new_max_v2.append(j_stim_index_len)
               self.new_max_v2.append(j_index_len)
           self.list_of_df.append(j)
        switcher = False
    
    def amplitude(self, event_markers = [0,1,2,3,4], hue = None, kde = False, group = "Mouse_ID", fill_nan = True, y_stat = "count", x_axis = "units", x_lim = False, y_lim = False):
        global df, df_result_amplitude, switcher, xd1, xd
        amplitude_all =[]
        assert len(self.list_of_df) == len(self.list_of_files)
        if x_axis == "units":
            df_amplitude = pd.DataFrame(columns= ["TrialCt", "Mouse_ID", "Amplitude_Pos", "Event_Marker"])
        elif x_axis == "mm":
            df_amplitude = pd.DataFrame(columns= ["TrialCt", "Mouse_ID", "Amplitude_Pos_mm", "Event_Marker"])
        for l,i in enumerate(self.list_of_df):
            if switcher:
                self.good_index = self.good_index_list[l]
                for k in event_markers:
                    if x_axis == "units":
                        
                        amplitude_ = [i.loc[(i["TrialCt"] == j) & (i["Event_Marker"] == k), "Amplitude_Pos"].max() for j in self.good_index]
                        mouse_id = [self.list_of_files[l] for m in range(1, len(self.good_index) +1)]
                        event_marker = [k for j in range(1, len(self.good_index) +1)]
                        dict_to_add = {"TrialCt": self.good_index, "Mouse_ID": mouse_id, "Amplitude_Pos": amplitude_, "Event_Marker": event_marker}
                        df_amplitude = df_amplitude.append(pd.DataFrame(dict_to_add))
                        df_amplitude.reset_index(inplace = True, drop = True)
                    elif x_axis == "mm":
                        
                        amplitude_ = [i.loc[(i["TrialCt"] == j) & (i["Event_Marker"] == k), "Amplitude_Pos_mm"].max() for j in self.good_index]
                        mouse_id = [self.list_of_files[l] for m in range(1, len(self.good_index) +1)]
                        event_marker = [k for j in range(1, len(self.good_index) +1)]
                        dict_to_add = {"TrialCt": self.good_index, "Mouse_ID": mouse_id, "Amplitude_Pos_mm": amplitude_, "Event_Marker": event_marker}
                        df_amplitude = df_amplitude.append(pd.DataFrame(dict_to_add))
                        df_amplitude.reset_index(inplace = True, drop = True)
            else:
                trial_max = i["TrialCt"].max()
                if self.opto:
                    self.good_index = self.trials_conteiner[l]
                for k in event_markers:
                    if x_axis == "units":
                        if self.opto:
                            amplitude_ = [i.loc[(i["TrialCt"] == j) & (i["Event_Marker"] == k), "Amplitude_Pos"].max() for j in self.good_index]
                            mouse_id = [self.list_of_files[l] for m in range(1, len(self.good_index) +1)]
                            event_marker = [k for j in range(1, len(self.good_index) +1)]
                            dict_to_add = {"TrialCt": self.good_index, "Mouse_ID": mouse_id, "Amplitude_Pos": amplitude_, "Event_Marker": event_marker}
                            df_amplitude = df_amplitude.append(pd.DataFrame(dict_to_add))
                            df_amplitude.reset_index(inplace = True, drop = True)
                        
                        else:
                            amplitude_ = [i.loc[(i["TrialCt"] == j) & (i["Event_Marker"] == k), "Amplitude_Pos"].max() for j in range(1, trial_max + 1)]
                            mouse_id = [self.list_of_files[l] for m in range(1, trial_max + 1)]
                            event_marker = [k for j in range(1, trial_max + 1)]
                            dict_to_add = {"TrialCt": range(1, trial_max + 1), "Mouse_ID": mouse_id, "Amplitude_Pos": amplitude_, "Event_Marker": event_marker}
                            df_amplitude = df_amplitude.append(pd.DataFrame(dict_to_add))
                            df_amplitude.reset_index(inplace = True, drop = True)
                    elif x_axis == "mm":
                        if self.opto:
                            amplitude_ = [i.loc[(i["TrialCt"] == j) & (i["Event_Marker"] == k), "Amplitude_Pos_mm"].max() for j in self.good_index]
                            mouse_id = [self.list_of_files[l] for m in range(1, len(self.good_index) +1)]
                            event_marker = [k for j in range(1, len(self.good_index) +1)]
                            dict_to_add = {"TrialCt": self.good_index, "Mouse_ID": mouse_id, "Amplitude_Pos_mm": amplitude_, "Event_Marker": event_marker}
                            df_amplitude = df_amplitude.append(pd.DataFrame(dict_to_add))
                            df_amplitude.reset_index(inplace = True, drop = True)
                        
                        else:
                            amplitude_ = [i.loc[(i["TrialCt"] == j) & (i["Event_Marker"] == k), "Amplitude_Pos_mm"].max() for j in range(1, trial_max + 1)]
                            mouse_id = [self.list_of_files[l] for m in range(1, trial_max + 1)]
                            event_marker = [k for j in range(1, trial_max + 1)]
                            dict_to_add = {"TrialCt": range(1, trial_max + 1), "Mouse_ID": mouse_id, "Amplitude_Pos_mm": amplitude_, "Event_Marker": event_marker}
                            df_amplitude = df_amplitude.append(pd.DataFrame(dict_to_add))
                            df_amplitude.reset_index(inplace = True, drop = True)
        
        if fill_nan:
            if x_axis == "units":
                null_sum = df_amplitude["Amplitude_Pos"].isnull().sum()
            elif x_axis == "mm":
                null_sum = df_amplitude["Amplitude_Pos_mm"].isnull().sum()
            for l,i in enumerate(self.list_of_files):
                for k in event_markers:
                    mask = (df_amplitude["Mouse_ID"] == i) & (df_amplitude["Event_Marker"] == k)
                    if x_axis == "units":
                        mean = round(df_amplitude.loc[mask, "Amplitude_Pos"].mean(),2)
                        df_amplitude.loc[mask, "Amplitude_Pos"] = df_amplitude.loc[mask, "Amplitude_Pos"].fillna(mean)
                    elif x_axis == "mm":
                        mean = round(df_amplitude.loc[mask, "Amplitude_Pos_mm"].mean(),2)
                        df_amplitude.loc[mask, "Amplitude_Pos_mm"] = df_amplitude.loc[mask, "Amplitude_Pos_mm"].fillna(mean)
                    
            print(f"I filled {null_sum} data points")
        sns.set_style('ticks')
        if x_axis == "units":
            bins= [i for i in range(0,270,10)]
        elif x_axis == "mm":
            bins= [i for i in range(0,27,1)]
        if group == None:
            if x_axis == "units":
                if self.opto:
                    mean_highta_stim = [round(len(df_amplitude.loc[(df_amplitude["Amplitude_Pos"] >= bins[ii]) & (df_amplitude["Amplitude_Pos"] < bins[ii+1]) & (df_amplitude["Event_Marker"].isin(event_markers)) & (df_amplitude["Mouse_ID"].isin(self.list_of_files_stim)), "TrialCt"]) / len(self.list_of_df)) for ii in range(0, len(bins)-1)]
                    mean_highta_nostim = [round(len(df_amplitude.loc[(df_amplitude["Amplitude_Pos"] >= bins[ii]) & (df_amplitude["Amplitude_Pos"] < bins[ii+1]) & (df_amplitude["Event_Marker"].isin(event_markers)) & (df_amplitude["Mouse_ID"].isin(self.list_of_files_nostim)), "TrialCt"]) / len(self.list_of_df)) for ii in range(0, len(bins)-1)]
                    bins.remove(260)
                    list_of_list_stim = []
                    list_of_list_nostim = []
                    for i,j in enumerate(mean_highta_stim):
                        fake_data = [bins[i] + 5 for value in range(0, j)]
                        list_of_list_stim.append(fake_data)
                    flatten_matrix_stim = [val for sublist in list_of_list_stim for val in sublist]
                    for i,j in enumerate(mean_highta_nostim):
                        fake_data = [bins[i] + 5 for value in range(0, j)]
                        list_of_list_nostim.append(fake_data)
                    flatten_matrix_nostim = [val for sublist in list_of_list_nostim for val in sublist]
                    bin_amplitude_stim = pd.Series(flatten_matrix_stim, name = "Amplitude_Pos")
                    bin_amplitude_nostim = pd.Series(flatten_matrix_nostim, name = "Amplitude_Pos")
                    fig, axs = plt.subplots(2)
                    sns.histplot(data = bin_amplitude_stim, bins = bins, stat = y_stat, fill = True, kde=True, ax= axs[0]).set(title = "Stimulation")
                    axs[0].set_ylabel("Mean count")
                    if x_lim:
                        axs[0].set_xlim(x_lim[0], x_lim[1])
                    if y_lim:
                        axs[0].set_ylim(y_lim[0], y_lim[1])
                    sns.histplot(data = bin_amplitude_nostim, bins = bins, stat = y_stat, fill = True, kde=True, ax= axs[1]).set(title = "No-stimulation")
                    axs[1].set_ylabel("Mean count")
                    if x_lim:
                        axs[1].set_xlim(x_lim[0], x_lim[1])
                    if y_lim:
                        axs[1].set_ylim(y_lim[0], y_lim[1])
                else:

                    mean_hight = [round(len(df_amplitude.loc[(df_amplitude["Amplitude_Pos"] >= bins[ii]) & (df_amplitude["Amplitude_Pos"] < bins[ii+1]) & (df_amplitude["Event_Marker"].isin(event_markers)), "TrialCt"]) / len(self.list_of_df)) for ii in range(0, len(bins)-1)]
                    bins.remove(260)
                    list_of_list = []
                    for i,j in enumerate(mean_hight):
                        fake_data = [bins[i] + 5 for value in range(0, j)]
                        list_of_list.append(fake_data)
                    flatten_matrix = [val for sublist in list_of_list for val in sublist]
                    bin_amplitude = pd.Series(flatten_matrix, name = "Amplitude_Pos")
                    sns.histplot(data = bin_amplitude, bins = bins, stat = y_stat, fill = True, kde=True)
                    plt.ylabel("Mean count")
                plt.legend(labels=[f"n = {len(self.list_of_df)}"])
            elif x_axis == "mm":
                if self.opto:
                     mean_highta_stim = [round(len(df_amplitude.loc[(df_amplitude["Amplitude_Pos_mm"] >= bins[ii]) & (df_amplitude["Amplitude_Pos_mm"] < bins[ii+1]) & (df_amplitude["Event_Marker"].isin(event_markers)) & (df_amplitude["Mouse_ID"].isin(self.list_of_files_stim)), "TrialCt"]) / len(self.list_of_df)) for ii in range(0, len(bins)-1)]
                     mean_highta_nostim = [round(len(df_amplitude.loc[(df_amplitude["Amplitude_Pos_mm"] >= bins[ii]) & (df_amplitude["Amplitude_Pos_mm"] < bins[ii+1]) & (df_amplitude["Event_Marker"].isin(event_markers)) & (df_amplitude["Mouse_ID"].isin(self.list_of_files_nostim)), "TrialCt"]) / len(self.list_of_df)) for ii in range(0, len(bins)-1)]
                     bins.remove(26)
                     list_of_list_stim = []
                     list_of_list_nostim = []
                     for i,j in enumerate(mean_highta_stim):
                         fake_data = [bins[i] + 0.5 for value in range(0, j)]
                         list_of_list_stim.append(fake_data)
                     flatten_matrix_stim = [val for sublist in list_of_list_stim for val in sublist]
                     for i,j in enumerate(mean_highta_nostim):
                         fake_data = [bins[i] + 0.5 for value in range(0, j)]
                         list_of_list_nostim.append(fake_data)
                     flatten_matrix_nostim = [val for sublist in list_of_list_nostim for val in sublist]
                     bin_amplitude_stim = pd.Series(flatten_matrix_stim, name = "Amplitude_Pos_mm")
                     bin_amplitude_nostim = pd.Series(flatten_matrix_nostim, name = "Amplitude_Pos_mm")
                     fig, axs = plt.subplots(2)
                     sns.histplot(data = bin_amplitude_stim, bins = bins, stat = y_stat, fill = True, kde=True, ax= axs[0]).set(title = "Stimulation")
                     axs[0].set_ylabel("Mean count")
                     if x_lim:
                         axs[0].set_xlim(x_lim[0], x_lim[1])
                         axs[1].set_xlim(x_lim[0], x_lim[1])
                     if y_lim:
                         axs[0].set_ylim(y_lim[0], y_lim[1])
                         axs[1].set_ylim(y_lim[0], y_lim[1])
                     sns.histplot(data = bin_amplitude_nostim, bins = bins, stat = y_stat, fill = True, kde=True, ax= axs[1]).set(title = "No-stimulation")
                     axs[1].set_ylabel("Mean count") 
                else:
                    mean_hight = [round(len(df_amplitude.loc[(df_amplitude["Amplitude_Pos_mm"] >= bins[ii]) & (df_amplitude["Amplitude_Pos_mm"] < bins[ii+1]) & (df_amplitude["Event_Marker"].isin(event_markers)), "TrialCt"]) / len(self.list_of_df)) for ii in range(0, len(bins)-1)]
                    bins.remove(26)
                    list_of_list = []
                    for i,j in enumerate(mean_hight):
                        fake_data = [bins[i] + 0.5 for value in range(0, j)]
                        list_of_list.append(fake_data)
                    flatten_matrix = [val for sublist in list_of_list for val in sublist]
                    bin_amplitude = pd.Series(flatten_matrix, name = "Amplitude_Pos_mm")
                    sns.histplot(data = bin_amplitude, bins = bins, stat = y_stat, fill = True, kde=True)
                    if x_lim:
                        plt.xlim(x_lim[0], x_lim[1])
                    if y_lim:
                        plt.ylim(y_lim[0], y_lim[1])
                    plt.ylabel("Mean count")
                plt.legend(labels=[f"n = {len(self.list_of_df)}"])
        else:
            if x_axis == "units":
                sns.displot(df_amplitude, x = "Amplitude_Pos", hue = hue, col = group, kde = kde, color = "green", palette = "tab10", bins = bins, stat = y_stat)
                plt.xlabel("Count")
                if x_lim:
                    plt.xlim(x_lim[0], x_lim[1])
                if y_lim:
                    plt.ylim(y_lim[0], y_lim[1])
            elif x_axis == "mm":
                sns.displot(df_amplitude, x = "Amplitude_Pos_mm", hue = hue, col = group, kde = kde, color = "green", palette = "tab10", bins = bins, stat = y_stat)
                if x_lim:
                    plt.xlim(x_lim[0], x_lim[1])
                if y_lim:
                    plt.ylim(y_lim[0], y_lim[1])
        if x_axis == "units":
            null_sum = df_amplitude["Amplitude_Pos"].isnull().sum()
        elif x_axis == "mm":
            null_sum = df_amplitude["Amplitude_Pos_mm"].isnull().sum()
        main = tk.Tk()
        msg = tk.messagebox.askquestion ('Save window','Do you want to save graphs and data?',icon = 'warning')
        if msg == "yes":
            main.destroy()
            save_file_v1 = easygui.diropenbox(msg = "Select folder for a save location", title = "Typical window")
            if group == None:
                save_file_v2 = save_file_v1 + "//" + self.list_of_files[0] + "__" + self.list_of_files[-1] + "_amplitude_mean" + ".svg"
            else:
                save_file_v2 = save_file_v1 + "//" + self.list_of_files[0] + "__" + self.list_of_files[-1] + "_amplitude_each_group" + ".svg"
            if self.opto and group == None:
                fig.tight_layout()
            plt.savefig(save_file_v2)
            plt.show()
            df_amplitude.to_excel(save_file_v1 + "//" + self.list_of_files[0] + "_" + self.list_of_files[-1] + "_amplitude_raw" + ".xlsx")
            event_marker_max = [str(k) + "_max" for k in event_markers]
            event_marker_mean = [str(k) + "_mean" for k in event_markers]
            event_marker_df = event_marker_mean + event_marker_max
            df_amplitude_summery = pd.DataFrame(columns= event_marker_df, index = self.list_of_files)
            
            for k in event_markers:
                if x_axis == "units":
                    ans1 = [df_amplitude.loc[(df_amplitude["Event_Marker"] == k) & (df_amplitude["Mouse_ID"] == kk), "Amplitude_Pos"].mean() for kk in self.list_of_files]
                    ans2 = [df_amplitude.loc[(df_amplitude["Event_Marker"] == k) & (df_amplitude["Mouse_ID"] == kk), "Amplitude_Pos"].max() for kk in self.list_of_files]
                    df_amplitude_summery[str(k) + "_max"] = ans2
                    df_amplitude_summery[str(k) + "_mean"] = ans1
                elif x_axis == "mm":
                    ans1 = [df_amplitude.loc[(df_amplitude["Event_Marker"] == k) & (df_amplitude["Mouse_ID"] == kk), "Amplitude_Pos_mm"].mean() for kk in self.list_of_files]
                    ans2 = [df_amplitude.loc[(df_amplitude["Event_Marker"] == k) & (df_amplitude["Mouse_ID"] == kk), "Amplitude_Pos_mm"].max() for kk in self.list_of_files]
                    df_amplitude_summery[str(k) + "_max"] = ans2
                    df_amplitude_summery[str(k) + "_mean"] = ans1
            
            df_amplitude_summery.to_excel(save_file_v1 + "//" + self.list_of_files[0] + "_" + self.list_of_files[-1] + "_amplitude_summery" + ".xlsx")
            
        else:
            if self.opto and group == None:
                fig.tight_layout()
            plt.show()
            main.destroy()
        df_result_amplitude = df_amplitude
    
    def lick_histogram(self, pre_stim = 2, post_stim = 2, group = "all", marker= "r", smooth = True, window_length = 9, polyorder = 3, stat = "prob", sem = True, x_lim = False, y_lim = False): 
       global df_result_lick, switcher, xd, df_lick_counts
       if self.opto:
           assert len(self.list_of_df) ==  len(self.list_of_files)
       else:
           assert len(self.list_of_df) == len(self.list_of_files)
       
       start, stop = pre_stim * 19, post_stim * 19
       columns = np.linspace(start = -pre_stim, stop = post_stim, num = start + stop + 1)
       x = [round(i,2) for i in columns]
       columns_ = [str(i) for i in x]
       x = np.array(x)

       columns_.append("Animal_ID")
       if self.opto:
           df_licks_group = pd.DataFrame(columns = columns_, index = [i for i in range(0, len(self.list_of_df)+3)])
       else:
           df_licks_group = pd.DataFrame(columns = columns_, index = [i for i in range(0, len(self.list_of_df)+1)])
       sem_lick_group = []
       for l,i in enumerate(self.list_of_df):
            lick_index = i.columns.tolist()
            lick_index = int(lick_index.index('Lick_state'))
            if self.opto:
                index_events = i.index[i['Event_Marker'] == 4].tolist()
            else:
                index_events = i.index[i['Event_Marker'] == 2].tolist()
            if switcher:
                print(f"len: {len(i)}")
                trial_max = self.new_max[l]
            elif self.opto:
                trial_max = self.new_max_v2[l]
            else:
                trial_max = i["TrialCt"].max()
            print(" ")
            list_value = [i.iloc[j-start:j+stop + 1, lick_index].tolist() for j in index_events]
            df_licks = pd.DataFrame(list_value,columns= [str(round(k,2)) for k in columns])
            lick_rate = df_licks
            prob_lick = df_licks.apply(lambda x: x.value_counts())
            prob_lick.fillna(0, inplace = True)
            df_lick_counts = prob_lick
            if stat == "prob":
                prob_lick = round(prob_lick / trial_max,2)
                prob_lick = prob_lick.iloc[1, :].tolist()
                prob_lick.append(self.list_of_files[l])
                df_licks_group.iloc[l] = prob_lick
            elif stat == "odds" :
                odds = [ round(a / b, 2) for a, b in zip(prob_lick.iloc[1, :].tolist(), prob_lick.iloc[0, :].tolist())] 
                odds.append(self.list_of_files[l])
                df_licks_group.iloc[l] = odds
            elif stat == "rate" :
                for i in range(0, lick_rate.shape[1]):
                    lick_rate.iloc[:,i] = lick_rate.iloc[:,i].apply(lambda x: (x * 1000)/50) # bin size 50 ms
                if sem:
                    lick_rate_sem = lick_rate.sem().tolist()
                    sem_lick_group.append(lick_rate_sem)
                
                lick_rate = lick_rate.mean().tolist()
                lick_rate.append(self.list_of_files[l])
                df_licks_group.iloc[l] = lick_rate
       if stat == "rate" and sem:
           sem_lick_group.append(df_licks_group.sem().tolist())
       df_licks_group.iloc[len(self.list_of_df), 0: len(columns_)-1] = df_licks_group.mean()
       df_licks_group.loc[len(self.list_of_df), "Animal_ID"] = "Mean"
       if self.opto:
           df_licks_group.loc[len(df_licks_group)-2, "Animal_ID"] = "Mean_stim"
           df_licks_group.loc[len(df_licks_group)-1, "Animal_ID"] = "Mean_nostim"
       df_licks_group.set_index("Animal_ID", inplace = True)
       if self.opto:
           columns_stim = [str(column) for column in df_licks_group.index.values.tolist() if "_stim" in str(column)]
           df_licks_group.loc["Mean_stim", :] = df_licks_group.loc[columns_stim, :].mean()
           columns_nostim = [str(column) for column in df_licks_group.index.values.tolist() if "_nostim" in str(column)]
           df_licks_group.loc["Mean_nostim", :] = df_licks_group.loc[columns_nostim, :].mean()
           sem_lick_group.append(df_licks_group.loc[columns_stim, :].sem().tolist())
           sem_lick_group.append(df_licks_group.loc[columns_nostim, :].sem().tolist())      
       df_result_lick = df_licks_group
       if stat == "prob":
           y_label = "Probability of lick"
       elif stat == "odds":
           y_label = "Lick/No-lick ratio"
       elif stat == "rate":
           y_label = "Mean lick rate (per sec)"
       main = tk.Tk()
       msg = tk.messagebox.askquestion ('Save window','Do you want to save graphs and data?',icon = 'warning')
       if msg == "yes":
           main.destroy()
           save_file_v1 = easygui.diropenbox(msg = "Select folder for a save location", title = "Typical window")
       else:
           main.destroy()
       if group == "all":
           bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
           n = 0
           iterator_v1 = 0 
           for index, row in df_licks_group.iterrows():
               n += 1
               bar.update(n)
               if sem and stat == "rate":
                   y = np.array(row)
                   error_sem = np.array(sem_lick_group[iterator_v1])
                   upper_band = np.asfarray(y + error_sem)
                   lower_band = np.asfarray(y - error_sem)
                   plt.fill_between(x, upper_band, lower_band, alpha = 0.4, color = "r")
                   iterator_v1 +=1
               if stat == "rate":
                   if x_lim:
                       plt.xlim(x_lim[0], x_lim[1])
                   if y_lim:
                       plt.ylim(y_lim[0], y_lim[1])
               plt.plot(x,np.array(row), marker, label = index)
               plt.title("Original")
               plt.ylabel(y_label)
               plt.xlabel("Time [s]")
               plt.annotate("Max", xy = (float(row[row == row.max()].index[0]), row.max()), xytext=(-1.0, row.max()),arrowprops = dict(facecolor='blue', shrink=0.1))
               plt.annotate("Reward start", xy = (0, row.min()), xytext=(0, (row.max() + row.min())/2),arrowprops = dict(facecolor='green', shrink=0.1))
               plt.legend()
               
               if msg == "yes":
                   save_file_v2 = save_file_v1 + "//" + index + "_" + "orginal_lick" + ".svg"
                   plt.savefig(save_file_v2)
                   plt.show()
               else:
                   plt.show()
               if smooth:
                   yhat = savgol_filter(np.array(row), window_length, polyorder)
                   if sem and stat == "rate":
                       yhat_error_sem = savgol_filter(np.array(sem_lick_group[iterator_v1-1]), window_length, polyorder)
                       upper_band = np.asfarray(yhat + yhat_error_sem)
                       lower_band = np.asfarray(yhat - yhat_error_sem)
                       plt.fill_between(x, upper_band, lower_band, alpha = 0.4, color = "r")
                   if stat == "rate":
                       if x_lim:
                           plt.xlim(x_lim[0], x_lim[1])
                       if y_lim:
                           plt.ylim(y_lim[0], y_lim[1])
                   plt.plot(x,yhat, marker, label = index)
                   plt.annotate("Max", xy = (x[np.where(yhat == max(yhat))[0]], max(yhat)), xytext=(-1.0, max(yhat)),arrowprops = dict(facecolor='blue', shrink=0.1))
                   plt.annotate("Reward start", xy = (0, min(yhat)), xytext=(0, (max(yhat) + min(yhat))/2),arrowprops = dict(facecolor='green', shrink=0.1))
                   plt.title("Smoothed")
                   plt.ylabel(y_label)
                   plt.xlabel("Time [s]")
                   plt.legend()
                   if msg == "yes":
                       save_file_v2 = save_file_v1 + "//" + index + "_" + "smoothed_lick" + ".svg"
                       plt.savefig(save_file_v2)
                       plt.show()
                   else:
                       plt.show()
       elif group == "mean":
           if sem and stat == "rate":
               if self.opto:
                   y = np.array(df_licks_group.iloc[-3])
                   error_sem = np.array(sem_lick_group[-3])
               else:
                    y = np.array(df_licks_group.iloc[-1])
                    error_sem = np.array(sem_lick_group[-1])
               upper_band = np.asfarray(y + error_sem)
               lower_band = np.asfarray(y - error_sem)
               plt.fill_between(x, upper_band, lower_band, alpha = 0.4, color = "r")
           if stat == "rate":
               if x_lim:
                   plt.xlim(x_lim[0], x_lim[1])
               if y_lim:
                   plt.ylim(y_lim[0], y_lim[1])
           plt.plot(x,df_licks_group.iloc[-1], marker, label = "Mean_lick")
           plt.title("Original")
           plt.ylabel(y_label)
           plt.xlabel("Time [s]")
           plt.legend()
           if msg == "yes":
                save_file_v2 = save_file_v1 + "//" + "Mean" + "_" + "orginal_lick" + ".svg"
                plt.savefig(save_file_v2)
                plt.show()
           else:
                plt.show()
           if smooth:
               if self.opto:
                   yhat = savgol_filter(np.array(df_licks_group.iloc[-3]), window_length, polyorder)
               else:
                   yhat = savgol_filter(np.array(df_licks_group.iloc[-1]), window_length, polyorder)
               if sem and stat == "rate":
                   if self.opto:
                        yhat_error_sem = savgol_filter(np.array(sem_lick_group[-3]), window_length, polyorder)
                   else:
                       yhat_error_sem = savgol_filter(np.array(sem_lick_group[-1]), window_length, polyorder)
                   upper_band_sem = np.asfarray(yhat + yhat_error_sem)
                   lower_band_sem = np.asfarray(yhat - yhat_error_sem)
                   plt.fill_between(x, upper_band_sem , lower_band_sem, alpha = 0.4, color = "r")
               if stat == "rate":
                   if x_lim:
                       plt.xlim(x_lim[0], x_lim[1])
                   if y_lim:
                       plt.ylim(y_lim[0], y_lim[1])
               plt.annotate("Max", xy = (x[np.where(yhat == max(yhat))[0]], max(yhat)), xytext=(-1.0, max(yhat)),arrowprops = dict(facecolor='blue', shrink=0.1))
               plt.annotate("Reward start", xy = (0, min(yhat)), xytext=(0, max(yhat)/2),arrowprops = dict(facecolor='green', shrink=0.1))
               plt.plot(x,yhat, marker, label = "Mean")
               plt.title("Smoothed")
               plt.ylabel(y_label)
               plt.xlabel("Time [s]")
               plt.legend()
            
           if msg == "yes":
               save_file_v2 = save_file_v1 + "//" + "Mean" + "_" + "smoothed_lick" + ".svg"
               plt.savefig(save_file_v2)
               plt.show()
           else:
               plt.show()
           
           if self.opto:
               fig, axs = plt.subplots(2)
               y_Mean_stim = np.array(df_licks_group.iloc[-2])
               y_Mean_nostim = np.array(df_licks_group.iloc[-1])
               if stat == "rate" and sem:
                    error_sem_stim = np.array(sem_lick_group[-2])
                    error_sem_nostim = np.array(sem_lick_group[-1])
                    upper_band_stim = np.asfarray(y_Mean_stim + error_sem_stim)
                    lower_band_stim = np.asfarray(y_Mean_stim - error_sem_stim)
                    upper_band_nostim = np.asfarray(y_Mean_nostim + error_sem_nostim)
                    lower_band_nostim = np.asfarray(y_Mean_nostim - error_sem_nostim)
                    axs[0].fill_between(x, upper_band_stim, lower_band_stim, alpha = 0.4, color = "r")
                    axs[1].fill_between(x, upper_band_nostim, lower_band_nostim, alpha = 0.4, color = "r")
               if stat == "rate":
                    if x_lim:
                         axs[0].set_xlim(x_lim[0], x_lim[1])
                         axs[1].set_xlim(x_lim[0], x_lim[1])
                    if y_lim:
                         axs[0].set_ylim(y_lim[0], y_lim[1])
                         axs[1].set_ylim(y_lim[0], y_lim[1])
               axs[0].plot(x,df_licks_group.iloc[-2], marker, label = "Mean_lick")
               axs[1].plot(x,df_licks_group.iloc[-1], marker, label = "Mean_lick")
               fig.suptitle("Original")
               axs[0].set_title('Stimulation')
               axs[1].set_title('No-stimulation')
               axs[0].set_ylabel(y_label)
               axs[0].set_xlabel("Time [s]")
               axs[1].set_ylabel(y_label)
               axs[1].set_xlabel("Time [s]")
               axs[0].annotate("Max", xy = (float(df_licks_group.iloc[-2][df_licks_group.iloc[-2] == df_licks_group.iloc[-2].max()].index[0]), df_licks_group.iloc[-2].max()), xytext=(-1.0, df_licks_group.iloc[-2].max()),arrowprops = dict(facecolor='blue', shrink=0.1))
               axs[0].annotate("Reward start", xy = (0, df_licks_group.iloc[-2].min()), xytext=(0, (df_licks_group.iloc[-2].max() + df_licks_group.iloc[-2].min())/2),arrowprops = dict(facecolor='green', shrink=0.1))
               
               axs[1].annotate("Max", xy = (float(df_licks_group.iloc[-2][df_licks_group.iloc[-2] == df_licks_group.iloc[-2].max()].index[0]), df_licks_group.iloc[-2].max()), xytext=(-1.0, df_licks_group.iloc[-2].max()),arrowprops = dict(facecolor='blue', shrink=0.1))
               axs[1].annotate("Reward start", xy = (0, df_licks_group.iloc[-2].min()), xytext=(0, (df_licks_group.iloc[-2].max() + df_licks_group.iloc[-2].min())/2),arrowprops = dict(facecolor='green', shrink=0.1))
               axs[0].legend()
               axs[1].legend()
               fig.tight_layout()
           if msg == "yes":
                save_file_v3 = save_file_v1 + "//" + "Mean" + "_" + "orginal_lick_stim_nostim" + ".svg"
                plt.savefig(save_file_v3)
                plt.show()
           else:
                plt.show()
           
           if self.opto and smooth:
               fig, axs = plt.subplots(2)
               yhat_stim = savgol_filter(np.array(df_licks_group.iloc[-2]), window_length, polyorder)
               yhat_nostim = savgol_filter(np.array(df_licks_group.iloc[-1]), window_length, polyorder)
               if sem and stat == "rate":
                   yhat_error_sem_stim = savgol_filter(np.array(sem_lick_group[-2]), window_length, polyorder)
                   yhat_error_sem_nostim = savgol_filter(np.array(sem_lick_group[-2]), window_length, polyorder)
                   upper_band_stim_sem = np.asfarray(yhat_stim + yhat_error_sem_stim)
                   lower_band_stim_sem = np.asfarray(yhat_stim - yhat_error_sem_stim)
                   upper_band_nostim_sem = np.asfarray(yhat_nostim + yhat_error_sem_nostim)
                   lower_band_nostim_sem = np.asfarray(yhat_nostim - yhat_error_sem_nostim)
                   axs[0].fill_between(x, upper_band_stim_sem, lower_band_stim_sem, alpha = 0.4, color = "r")
                   axs[1].fill_between(x, upper_band_nostim_sem, lower_band_nostim_sem, alpha = 0.4, color = "r")
               if stat == "rate":
                    if x_lim:
                         axs[0].set_xlim(x_lim[0], x_lim[1])
                         axs[1].set_xlim(x_lim[0], x_lim[1])
                    if y_lim:
                         axs[0].set_ylim(y_lim[0], y_lim[1])
                         axs[1].set_ylim(y_lim[0], y_lim[1])
               axs[0].plot(x,yhat_stim, marker, label = "Mean_lick")
               axs[1].plot(x,yhat_nostim, marker, label = "Mean_lick")
               fig.suptitle("Smoothed")
               axs[0].set_title('Stimulation')
               axs[1].set_title('No-stimulation')
               axs[0].set_ylabel(y_label)
               axs[0].set_xlabel("Time [s]")
               axs[1].set_ylabel(y_label)
               axs[1].set_xlabel("Time [s]")
               axs[0].annotate("Max", xy = (x[np.where(yhat_stim == max(yhat_stim))[0]], max(yhat_stim)), xytext=(-1.0, max(yhat_stim)),arrowprops = dict(facecolor='blue', shrink=0.1))
               axs[0].annotate("Reward start", xy = (0, yhat_stim.min()), xytext=(0, (yhat_stim.max() + yhat_stim.min())/2),arrowprops = dict(facecolor='green', shrink=0.1))
               
               axs[1].annotate("Max", xy = (x[np.where(yhat_nostim == max(yhat_nostim))[0]], max(yhat_nostim)), xytext=(-1.0, max(yhat_nostim)),arrowprops = dict(facecolor='blue', shrink=0.1))
               axs[1].annotate("Reward start", xy = (0, yhat_nostim.min()), xytext=(0, (yhat_nostim.max() + yhat_nostim.min())/2),arrowprops = dict(facecolor='green', shrink=0.1))
               
               axs[0].legend()
               axs[1].legend()
               fig.tight_layout()
           if msg == "yes":
                save_file_v4 = save_file_v1 + "//" + "Mean" + "_" + "smoothed_lick_stim_nostim" + ".svg"
                plt.savefig(save_file_v4)
                plt.show()
           else:
                plt.show()

       if stat == "rate":
           df_licks_group["Mean"] = df_licks_group.mean(axis =1)
           df_licks_group["Max"] = df_licks_group.iloc[:, 0: df_licks_group.shape[1] -1].max(axis=1)
       if msg == "yes":
            if self.opto:
                df_licks_group.to_excel(save_file_v1 + "//" + self.list_of_files[0] + "_" + self.list_of_files[-1] + "_lick" + ".xlsx")
            else:
                df_licks_group.to_excel(save_file_v1 + "//" + self.list_of_files[0] + "_" + self.list_of_files[-1] + "_lick" + ".xlsx")
               
    def files_name(self):
        for i,j in enumerate(self.list_of_files):
            if self.group[0] in str(j):
                print(colored(i,"red"), colored(j, "red"))
            elif self.group[1] in str(j):
                print(colored(i,"green"), colored(j, "green"))
            else:
                print(colored(i,"blue"), colored(j, "blue"))
    
    def find_bugs(self, alfa = 0.10, automatic = True):
        global xd, xd1, switcher, base_joypos_X, base_joypos_Y
        self.list_of_df_v1 = []
        print("Test started\n")
        main = tk.Tk()
        msg2 = tk.messagebox.askquestion ('Delete window','Do you want to delete bugged data?',icon = 'warning')
        main.destroy()
        self.new_max = []
        self.good_index_list = []
        if not automatic:
            base_x_upper = base_joypos_X + base_joypos_X * alfa
            base_y_upper = base_joypos_Y + base_joypos_Y * alfa
            base_x_lower = base_joypos_X - base_joypos_X * alfa
            base_y_lower = base_joypos_Y - base_joypos_Y * alfa
        for l,i in enumerate(self.list_of_df):
            print(f"len before: {len(i)}")
            if automatic:
                base_x_upper = i.loc[0, "Base_JoyPos_X"] + i.loc[0, "Base_JoyPos_X"] * alfa
                base_y_upper = i.loc[0, "Base_JoyPos_Y"] + i.loc[0, "Base_JoyPos_Y"] * alfa
                base_x_lower = i.loc[0, "Base_JoyPos_X"] - i.loc[0, "Base_JoyPos_X"] * alfa
                base_y_lower = i.loc[0, "Base_JoyPos_Y"] - i.loc[0, "Base_JoyPos_Y"] * alfa
            if self.opto:
                    ans_x = [set(i.loc[i["TrialCt"] == ii, "Base_JoyPos_X"]).pop() for ii in self.trials_conteiner[l]]
                    ans_y = [set(i.loc[i["TrialCt"] == ii, "Base_JoyPos_Y"]).pop() for ii in self.trials_conteiner[l]]
            else:
                trial_max = i["TrialCt"].max()
                ans_x = [set(i.loc[i["TrialCt"] == ii, "Base_JoyPos_X"]).pop() for ii in range(1,trial_max + 1)]
                ans_y = [set(i.loc[i["TrialCt"] == ii, "Base_JoyPos_Y"]).pop() for ii in range(1,trial_max + 1)]
            x_bool = [True if base_x_lower < jj < base_x_upper else False for jj in ans_x]
            y_bool = [True if base_y_lower < kk < base_y_upper else False for kk in ans_y]
            x_y_bool = ["ok" if x == True and y == True else "not ok" if x == True or y == True else "very bad" for x, y in zip(x_bool, y_bool)]
            if self.opto:
                df_result = pd.DataFrame({"x_bool": x_bool, "y_bool": y_bool, "x_y_bool": x_y_bool}, index = [i for i in self.trials_conteiner[l]])
            else:
                df_result = pd.DataFrame({"x_bool": x_bool, "y_bool": y_bool, "x_y_bool": x_y_bool}, index = range(1,trial_max +1))
            print("Mice ID: ", self.list_of_files[l],"\n", df_result["x_y_bool"].value_counts(), "\n" ,round(df_result["x_y_bool"].value_counts(normalize=True),2))
            print(" ")
            trial_max = i["TrialCt"].max()
            if msg2 == "yes":
                buggs_index = df_result.index[(df_result['x_y_bool'] == "not ok") | (df_result['x_y_bool'] == "very bad")].tolist()
                self.good_index = df_result.index[(df_result['x_y_bool'] == "ok")].tolist()
                self.good_index_list.append(self.good_index)
                self.new_max.append(trial_max - len(buggs_index))
                i.drop(i[i["TrialCt"].isin(buggs_index)].index, inplace = True)
                i.reset_index(inplace = True, drop = True)
                self.list_of_df_v1.append(i)
                switcher = True
                xd = i
        print("Test completed")
        print(f"len after: {len(i)}")
        if msg2 == "yes":
            self.list_of_df = self.list_of_df_v1
    
    def move_type(self, hue = None, event_markers = [0,1,2,3,4], group = "all"):
        global df_result_type_move, xd
        df_type_move = pd.DataFrame(columns= ["Pull", "Push", "None", "Id"])
        if group == "all": 
            main = tk.Tk()
            msg = tk.messagebox.askquestion ('Save window','Do you want to save graphs and data?',icon = 'warning')
            main.destroy()
            if msg == "yes":
                save_file_v1 = easygui.diropenbox(msg = "Select folder for a save location", title = "Typical window")
            for l,i in enumerate(self.list_of_df):
                i = i.loc[i["Event_Marker"].isin(event_markers), :]
                sns.countplot(x = 'Type_move', data=i, hue = hue, palette = "tab10").set(title= self.list_of_files[l])
                if hue != None:
                    for iii in  event_markers:
                        values = i.loc[i["Event_Marker"] == iii, "Type_move"].value_counts().tolist()
                        values.append(iii)
                        keys = i.loc[i["Event_Marker"] == iii, "Type_move"].value_counts().index.tolist()
                        keys.append("Id")
                        dict_to_add = {keys[ii]: values[ii] for ii in range(len(values))}
                        df_type_move = df_type_move.append(dict_to_add, ignore_index= True)
                    if msg == "yes":
                        save_file_v2 = save_file_v1 + "//" + self.list_of_files[l] + ".svg"
                        plt.savefig(save_file_v2)
                        plt.show()
                    else:
                        plt.show()
                else:
                    values = i.loc[:, "Type_move"].value_counts().tolist()
                    keys = i.loc[:, "Type_move"].value_counts().index.tolist()
                    dict_to_add = {keys[ii]: values[ii] for ii in range(len(values))}
                    df_type_move = df_type_move.append(dict_to_add, ignore_index=True)
                    if msg == "yes":
                        save_file_v2 = save_file_v1 + "//" + self.list_of_files[l] + ".svg"
                        plt.savefig(save_file_v2)
                        plt.show()
                    else:
                        plt.show()
            if hue != None:
                columns_v1 = [o for o in event_markers]
                columns_v1.append("Type_move")
                templet = np.zeros((3, len(columns_v1)))
                df_for_plot = pd.DataFrame(templet, columns= columns_v1)
                for i in event_markers:
                    #df_result_type_move = df_type_move
                    df_type_move.loc[len(df_type_move), :] = df_type_move.loc[df_type_move["Id"] == i, :].mean()
                    #df_for_plot[str(i)] = df_type_move.loc[df_type_move["Id"] == i, :].mean().tolist()
                    df_type_move.iloc[-1, 3] = "mean_" + str(i)
                    
                    columns = df_type_move.columns.tolist()
                    columns.remove("Id")
                    #df_for_plot[i] = df_type_move.loc[df_type_move["Id"] == "mean_" + str(i), "Pull": "None"].values
                    ans =  df_type_move.loc[df_type_move["Id"] == "mean_" + str(i), "Pull": "None"].values.tolist()
                    df_for_plot[i] = ans[0]
                df_for_plot["Type_move"] = ["Pull", "Push", "None"]
                df_result_type_move = df_for_plot
                xd = df_type_move
                ax = df_for_plot.plot(x="Type_move", y=event_markers, kind="bar", rot=0, title = "Mean").get_figure()
                #ax.savefig('test.pdf')
                if msg == "yes":
                    save_file_v3 = save_file_v1 + "//" + "Mean" + ".svg"
                    ax.savefig(save_file_v3)
                    #plt.savefig(save_file_v2)
                    #plt.show()
                #else:
                    #plt.show()
            else:
                df_type_move.loc[len(self.list_of_df), :] = df_type_move.mean()
                df_result_type_move = df_type_move
                columns = df_result_type_move.columns.tolist()
                height =  [round(r) for r in df_result_type_move.iloc[-1].tolist()]
                plt.bar(columns, height)
                plt.title("Mean")
                if msg == "yes":
                    save_file_v2 = save_file_v1 + "//" + "Mean" + ".svg"
                    plt.savefig(save_file_v2)
                    plt.show()
                else:
                    plt.show()
        elif group == "mean":
            main = tk.Tk()
            msg = tk.messagebox.askquestion ('Save window','Do you want to save graphs and data?',icon = 'warning')
            main.destroy()
            if msg == "yes":
                save_file_v1 = easygui.diropenbox(msg = "Select folder for a save location", title = "Typical window")
            for l,i in enumerate(self.list_of_df):
                i = i.loc[i["Event_Marker"].isin(event_markers), :]
                values = i.loc[:, "Type_move"].value_counts().tolist()
                keys = i.loc[:, "Type_move"].value_counts().index.tolist()
                dict_to_add = {keys[ii]: values[ii] for ii in range(len(values))}
                df_type_move = df_type_move.append(dict_to_add, ignore_index=True)
            df_type_move.loc[len(self.list_of_df), :] = df_type_move.mean()
            df_result_type_move = df_type_move
            columns = df_result_type_move.columns.tolist()
            height =  [round(r) for r in df_result_type_move.iloc[-1].tolist()]
            plt.bar(columns, height)
            plt.title("Mean")
            if msg == "yes":
                save_file_v2 = save_file_v1 + "//" + "Mean" + ".svg"
                plt.savefig(save_file_v2)
                plt.show()
            else:
                plt.show()

    def help_me(self):
        print("amplitude \n function parameters:\n event_markers - which events will be included in graph [value: 0,1,2,3,4 (int)] \n hue - events markers will be presented separately or jointly [value: Event_Marker (string), None (key word)]\n kde - data will be presented as an output of kernel density estimation [value: True (bool), False (bool)]\n group - graphs will be created for each mice separately or together [value: Mouse_ID (string), None (key word)]\n fill_nan - fill missing data [value: True (bool), False (bool)]\n y_stat - type of statistic used to create y axis - check seaborn API to use it correctly [value: count (string), frequency (string), probability (string), percent (string), density (string)]\n x_axis - x axis values will be presented as units or mmx axis values will be presented as units or mm [value: units (string), mm (string)]]")
        print("")
        print("lick_histogram \n function parameters:\n pre_stim - how much time in sec you want to include in graph, up to reward onset [value: (int)] \n post_stim - how much time in sec you want to include in graph, after reward onset [value: (int)] \n group - graphs will be created for each mice separately or together [value: all, mean (string)]\n marker - color and type of line on graph [value: (string)]\n smooth - smoothing algorithm that go through data [value: True (bool), False (bool)]\n window_length - (only if smooth = True) how long will be the polynomial that will be fitted to data [value: (int)]\n polyorder - (only if smooth = True) the degree of a polynomial that will be fitted to data [value: (int)]\n stat - y axis will be presetned as probability of lick or ratio lick/no-lick [value: prob, odds (string, )")
        print("")
        print("veloctiy \n function parameters:\n pre_stim - how much time in sec you want to include in graph, up to reward onset [value: (int)] \n post_stim - how much time in sec you want to include in graph, after reward onset [value: (int)] \n group - graphs will be created for each mice separately or together [value: all, mean (string)]\n marker - color and type of line on graph [value: (string)]\n smooth - smoothing algorithm that go through data [value: True (bool), False (bool)]\n window_length - (only if smooth = True) how long will be the polynomial that will be fitted to data [value: (int)]\n polyorder - (only if smooth = True) the degree of a polynomial that will be fitted to data [value: (int)]\n sem - standard error of mean will be added to graph [value: True (bool), False (bool)]")
    
    def trajectory(self, move_range = "0_to_max", calibrate = False, x_lim = False, y_lim = False, polar = False, starting_point = True):
        global xd1
        df_trajectory = pd.DataFrame(columns= ["TrialCt", "Mouse_ID", "x_coordinate_start_move_to_max", "y_coordinate_start_move_to_max", "time_start [sec]", "time_end [sec]", "move_duration [sec]", "time_trial_start [sec]", "move_duration_from_trial_start [sec]" ])
        df_polar = pd.DataFrame(columns= ["TrialCt", "Mouse_ID", "Radial coordinate", "Polar angle"])
        main = tk.Tk()
        msg = tk.messagebox.askquestion ('Save window','Do you want to save graphs and data?',icon = 'warning')
        if msg == "yes":
            main.destroy()
            save_file_v1 = easygui.diropenbox(msg = "Select folder for a save location", title = "Typical window")
        else:
            main.destroy()
        for l,i in enumerate(self.list_of_df):
            trial_list = sorted(set(i.loc[:, "TrialCt"].tolist()))
            
            xd1 = i
            if self.opto:
                start_index = [i.loc[i["TrialCt"] == hh].first_valid_index() for hh in trial_list if i.loc[(i["TrialCt"] == hh) & (i["Event_Marker"] == 3), "Amplitude_Pos"].tolist()]
                movment_end = [i.loc[(i["TrialCt"] == ii) & (i["Event_Marker"] == 3), "Amplitude_Pos"].idxmax() for ii in trial_list if i.loc[(i["TrialCt"] == ii) & (i["Event_Marker"] == 3), "Amplitude_Pos"].tolist()]
                trial_list_v2 = [trial for trial in trial_list if i.loc[(i["TrialCt"] == trial) & (i["Event_Marker"] == 3), "Amplitude_Pos"].tolist()]
            else:
                start_index = [i.loc[i["TrialCt"] == hh].first_valid_index() for hh in trial_list if i.loc[(i["TrialCt"] == hh) & (i["Event_Marker"] == 1), "Amplitude_Pos"].tolist()]
                movment_end = [i.loc[(i["TrialCt"] == ii) & (i["Event_Marker"] == 1), "Amplitude_Pos"].idxmax() for ii in trial_list if i.loc[(i["TrialCt"] == ii) & (i["Event_Marker"] == 1), "Amplitude_Pos"].tolist()]
                trial_list_v2 = [trial for trial in trial_list if i.loc[(i["TrialCt"] == trial) & (i["Event_Marker"] == 1), "Amplitude_Pos"].tolist()]
            
            movment_list_x_start_to_max = [i.iloc[range_index_1[0]: range_index_1[1]+1, 3].tolist() for range_index_1 in zip(start_index, movment_end)]
            movment_list_y_start_to_max = [i.iloc[range_index_1[0]: range_index_1[1]+1, 4].tolist() for range_index_1 in zip(start_index, movment_end)]
            
            list_supporter = []
            
            if self.opto:
                for range_index in zip(movment_end, start_index):
                    for current_index in range(range_index[0], range_index[1]-1,-1):
                        if (i.iloc[current_index, 5] < 10 and i.iloc[current_index, 1] == 1) or (i.iloc[current_index, 5] < 10 and i.iloc[current_index, 1] == 2):
                            list_supporter.append(current_index)
                            break
            else: 
                for range_index in zip(movment_end, start_index):
                    for current_index in range(range_index[0], range_index[1]-1,-1):
                        if i.iloc[current_index, 5] < 10 and i.iloc[current_index, 1] == 0:
                            list_supporter.append(current_index)
                            break


            movment_list_x_0_to_max = [i.iloc[range_index_1[0]: range_index_1[1]+1, 3].tolist() for range_index_1 in zip(list_supporter, movment_end)]
            movment_list_y_0_to_max = [i.iloc[range_index_1[0]: range_index_1[1]+1, 4].tolist() for range_index_1 in zip(list_supporter, movment_end)]
            time_of_movement_start = [i.iloc[range_index_1, 0].tolist() for range_index_1 in list_supporter]
            time_of_movement_stop = [i.iloc[range_index_1, 0].tolist() for range_index_1 in movment_end]
            trial_time_start = [i.iloc[range_index_1, 0].tolist() for range_index_1 in start_index]   
            movement_duration = [[range_index_1[1] - range_index_1[0]][0] for range_index_1 in zip(time_of_movement_start, time_of_movement_stop)]
            movement_duration_from_trial = [[range_index_1[1] - range_index_1[0]][0] for range_index_1 in zip(trial_time_start, time_of_movement_stop)]
            movment_list_x_0_to_max_backup = movment_list_x_0_to_max
            movment_list_y_0_to_max_backup = movment_list_y_0_to_max
            
            
            if calibrate:
                movment_list_x_0_to_max = [list(np.array(list_x) - list_x[0]) for list_x in movment_list_x_0_to_max]
                movment_list_y_0_to_max = [list(np.array(list_y) - list_y[0]) for list_y in movment_list_y_0_to_max]
                movment_list_x_start_to_max = [list(np.array(list_x) - list_x[0]) for list_x in movment_list_x_start_to_max]
                movment_list_y_start_to_max = [list(np.array(list_y) - list_y[0]) for list_y in movment_list_y_start_to_max]
            
            movment_list_y_0_to_max = [list(np.array(list_index) * -1) for list_index in movment_list_y_0_to_max]
            movment_list_y_start_to_max = [list(np.array(list_index) * -1) for list_index in movment_list_y_start_to_max]
                
            xd = movment_list_x_0_to_max
            if move_range == "start_move_to_max":
                fig, axs = plt.subplots(2,  figsize=(6, 8))
                fig.tight_layout(pad=5.0)
                fig.suptitle(self.list_of_files[l] + "_start_move_to_max", fontsize=14)
                [axs[0].plot(ll[0], ll[1], c = "g", alpha = 0.2) for ll in zip(movment_list_x_0_to_max, movment_list_y_0_to_max)]
                [axs[0].scatter(lll[0][-1], lll[1][-1] , c = "r", alpha = 1, s = 10) if index > 0 else  axs[0].scatter(lll[0][-1], lll[1][-1] , c = "r", alpha = 1, s = 10, label = "move end") for index, lll in enumerate(zip(movment_list_x_0_to_max, movment_list_y_0_to_max))]
                axs[0].set_title("Trajecotry plot")
                uni_list_x = []
                uni_list_y = []
                [uni_list_x.append(x) for list_x in movment_list_x_0_to_max for x in list_x]
                [uni_list_y.append(y) for list_y in movment_list_y_0_to_max for y in list_y]
                ########
                sns.kdeplot(x = uni_list_x, y = uni_list_y, ax = axs[1], fill= True, bw_adjust= 0.3)
                axs[1].set_title("Density plot")
                if calibrate:
                    x_annotate, y_annotate = round(abs(max(movment_list_x_0_to_max[round(len(movment_list_x_0_to_max)/2)])) /2), round(abs(max(movment_list_y_0_to_max[round(len(movment_list_y_0_to_max)/2)])) /2)
                    x_annotate +=10
                    axs[1].annotate('Starting point', xy =(0, 0), xytext =(x_annotate, y_annotate), arrowprops = dict(facecolor ='purple', shrink = 0.05),  )
                    axs[0].annotate('Starting point', xy =(0, 0), xytext =(x_annotate, y_annotate), arrowprops = dict(facecolor ='purple', shrink = 0.05),  )
                elif starting_point:
                    [axs[0].scatter(lll[0][0], lll[1][0] , c = "b", alpha = 1, s = 10) if index > 0 else  axs[0].scatter(lll[0][0], lll[1][0] , c = "b", alpha = 1, s = 10, label = "move start") for index, lll in enumerate(zip(movment_list_x_0_to_max, movment_list_y_0_to_max))]
                axs[0].legend()
                if x_lim:
                    axs[0].set_xlim(x_lim[0], x_lim[1])
                    axs[1].set_xlim(x_lim[0], x_lim[1])
                if y_lim:
                    axs[0].set_ylim(y_lim[0], y_lim[1])
                    axs[1].set_ylim(y_lim[0], y_lim[1])
                axs[0].set_ylabel("Y coordinate")
                axs[0].set_xlabel("X coordinate")
                axs[1].set_ylabel("Y coordinate")
                axs[1].set_xlabel("X coordinate")
                if msg == "yes":
                    save_file_v5 = save_file_v1 + "//" + "start_move_to_max" + "_" + self.list_of_files[l] + ".svg"
                    plt.savefig(save_file_v5)
                    plt.show()
                else:
                    plt.show()
                
            elif move_range == "start_trial_to_max":

                fig, axs = plt.subplots(2,  figsize=(6, 8))
                fig.tight_layout(pad=5.0)
                fig.suptitle(self.list_of_files[l] + "_start_trial_to_max", fontsize=14)
                [axs[0].plot(lll[0], lll[1] , c = "g", alpha = 0.2) for lll in zip(movment_list_x_start_to_max, movment_list_y_start_to_max)]
                [axs[0].scatter(lll[0][-1], lll[1][-1] , c = "r", alpha = 1, s = 10) if index > 0 else  axs[0].scatter(lll[0][-1], lll[1][-1] , c = "r", alpha = 1, s = 10, label = "move end") for index, lll in enumerate(zip(movment_list_x_start_to_max, movment_list_y_start_to_max))]
                axs[0].set_title("Trajecotry plot")
                uni_list_x = []
                uni_list_y = []
                [uni_list_x.append(x) for list_x in movment_list_x_start_to_max for x in list_x]
                [uni_list_y.append(y) for list_y in movment_list_y_start_to_max for y in list_y]
                sns.kdeplot(x = uni_list_x, y = uni_list_y, ax = axs[1], fill= True, bw_adjust= 0.3)
                axs[1].set_title("Density plot")
                if calibrate:
                    x_annotate, y_annotate = round(abs(max(movment_list_x_start_to_max[round(len(movment_list_x_start_to_max)/2)])) /2), round(abs(max(movment_list_y_start_to_max[round(len( movment_list_y_start_to_max)/2)])) /2)
                    x_annotate +=10
                    axs[1].annotate('Starting point', xy =(0, 0), xytext =(x_annotate, y_annotate), arrowprops = dict(facecolor ='purple', shrink = 0.05),  )
                    axs[0].annotate('Starting point', xy =(0, 0), xytext =(x_annotate, y_annotate), arrowprops = dict(facecolor ='purple', shrink = 0.05),  )
                elif starting_point:
                    [axs[0].scatter(lll[0][0], lll[1][0] , c = "b", alpha = 1, s = 10) if index > 0 else  axs[0].scatter(lll[0][0], lll[1][0] , c = "b", alpha = 1, s = 10, label = "move end") for index, lll in enumerate(zip(movment_list_x_start_to_max, movment_list_y_start_to_max))]
                axs[0].legend()
                if x_lim:
                    axs[0].set_xlim(x_lim[0], x_lim[1])
                    axs[1].set_xlim(x_lim[0], x_lim[1])
                if y_lim:
                    axs[0].set_ylim(y_lim[0], y_lim[1])
                    axs[1].set_ylim(y_lim[0], y_lim[1])
                axs[0].set_ylabel("Y coordinate")
                axs[0].set_xlabel("X coordinate")
                axs[1].set_ylabel("Y coordinate")
                axs[1].set_xlabel("X coordinate")
                if msg == "yes":
                    save_file_v5 = save_file_v1 + "//" + "start_trial_to_max" +"_" + self.list_of_files[l] + ".svg"
                    plt.savefig(save_file_v5)
                    plt.show()
                else:
                    plt.show()
            if calibrate and polar:
                movment_list_x_max = [list_x[-1] for list_x in movment_list_x_0_to_max]
                movment_list_y_max = [list_y[-1] for list_y in movment_list_y_0_to_max]
                r_list = []
                theta_list = []
                for max_index in zip(movment_list_x_max, movment_list_y_max):
                    ans= self.cart2pol(max_index[0], max_index[1])
                    r_list.append(ans[0])
                    theta_list.append(ans[1])
                width = np.pi/15
                ax = plt.subplot(111, projection='polar')
                ax.set_title(self.list_of_files[l] + "_start_move_to_max")
                
                bars = ax.bar( theta_list , r_list, width=width, bottom=0.0)
                for r, bar in zip(r_list, bars):
                    bar.set_alpha(0.3)
                dict_to_add = {"TrialCt": trial_list_v2, "Mouse_ID": self.list_of_files[l], "Radial coordinate": r_list, "Polar angle": theta_list}
                df_polar = df_polar.append(pd.DataFrame(dict_to_add))
                if msg == "yes":
                    save_file_v6 = save_file_v1 + "//" + "polar_start_move_to_max" + "_" + self.list_of_files[l] + ".svg"
                    plt.savefig(save_file_v6)
                    plt.show()
            
                else:
                    plt.show()
            # Create final result file 
            movment_list_y_0_to_max_backup = [list(np.array(list_index) * -1) for list_index in movment_list_y_0_to_max_backup]
            dict_to_add = {"TrialCt": trial_list_v2, "Mouse_ID": self.list_of_files[l], "x_coordinate_start_move_to_max": movment_list_x_0_to_max_backup, "y_coordinate_start_move_to_max": movment_list_y_0_to_max_backup, "move_duration [sec]": movement_duration, "time_trial_start [sec]": trial_time_start, "time_start [sec]": time_of_movement_start, "time_end [sec]": time_of_movement_stop, "move_duration_from_trial_start [sec]" : movement_duration_from_trial}
            df_trajectory = df_trajectory.append(pd.DataFrame(dict_to_add))
        if msg == "yes":
            df_trajectory.to_excel(save_file_v1 + "//" + self.list_of_files[0] + "_" + self.list_of_files[-1] + "_trajectory" + ".xlsx")
            if calibrate and polar:
                df_polar.to_excel(save_file_v1 + "//" + self.list_of_files[0] + "_" + self.list_of_files[-1] + "_polar" + ".xlsx")
    
    def cart2pol(self, x, y):
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return [rho, theta]
    
    def prob_reward(self, group = "one"):
        global xd, xd1, df_porb_reward
        df_porb_reward = pd.DataFrame(columns= ["Probability", "ID"])
        for l,i in enumerate(self.list_of_df):
            trial_list = sorted(set(i.loc[:, "TrialCt"].tolist()))
            all_movment = i.loc[((i["Event_Marker"] == 1) | (i["Event_Marker"] == 4)), "Event_Marker"].index.tolist()
            all_movment.append(all_movment[-1] + 100)
            xd = all_movment
            ans1 = [True for indexx in range(len(all_movment)-1) if all_movment[indexx+1] - all_movment[indexx] > 1]
            all_movment = len(ans1)
            porb = round(len(trial_list)/all_movment,2)
            dict_to_add = {"Probability": porb,"ID": self.list_of_files[l]}
            df_porb_reward = df_porb_reward.append(dict_to_add, ignore_index = True)
        mean_prob= float(df_porb_reward.mean().values)
        xd = mean_prob
        if group == "one":
            df_porb_reward.sort_values("Probability", inplace = True)
            
            ax = df_porb_reward.plot(x="ID", y="Probability", kind="bar", rot=-70, title = "Reward probability",figsize = (10,8), grid = True).get_figure()
            ax.tight_layout()
            dict_to_add = {"Probability": mean_prob,"ID": "Mean"}
            df_porb_reward = df_porb_reward.append(dict_to_add, ignore_index = True)
            main = tk.Tk()
            msg = tk.messagebox.askquestion ('Save window','Do you want to save graphs and data?',icon = 'warning')
            main.destroy()
            if msg == "yes":
                save_file_v3 = easygui.diropenbox(msg = "Select folder for a save location", title = "Typical window")
                df_porb_reward.to_excel(save_file_v3 + "//" + self.list_of_files[0] + self.list_of_files[-1] + "_prob_reward" + ".xlsx")
                save_file_v3_graph = save_file_v3 + "//" + "prob_reward_all" + ".svg"
                ax.savefig(save_file_v3_graph)
    
    def amplitude_time(self, pre_stim = 2, post_stim = 2, normalize = False, marker= "r", group = "All"):
        global xd,xd1
        assert len(self.list_of_df) == len(self.list_of_files)
        noramlizer = MinMaxScaler()
        start, stop = pre_stim * 19, post_stim * 19
        columns = np.linspace(start = -pre_stim, stop = post_stim, num = start + stop + 1)
        x = [round(i,2) for i in columns]
        columns_ = [str(i) for i in x]
        x = np.array(x)
        
        columns_.append("Animal_ID")
        df_amplitude_time_group = pd.DataFrame(columns= columns_, index = [i for i in range(0, len(self.list_of_df)+1)])
        for l,i in enumerate(self.list_of_df):
             index_events = i.index[i['Event_Marker'] == 2].tolist()
             if switcher:
                 trial_max = self.new_max
                 
             else:
                 trial_max = i["TrialCt"].max()
             list_value = [i.iloc[j-start:j+stop + 1, 5].tolist() for j in index_events]
             df_amplitude_time = pd.DataFrame(list_value,columns= [str(round(k,2)) for k in columns])
             if normalize:
                 df_amplitude_time = noramlizer.fit_transform(df_amplitude_time)
                 df_amplitude_time = pd.DataFrame(df_amplitude_time,columns= [str(round(k,2)) for k in columns])
             mice_mean = df_amplitude_time.mean(axis=0).tolist()
             mice_mean.append(self.list_of_files[l])
             df_amplitude_time_group.iloc[l]= mice_mean
        df_amplitude_time_group.iloc[len(self.list_of_df), 0: len(columns_)-1] = df_amplitude_time_group.mean()
        df_amplitude_time_group.loc[len(self.list_of_df), "Animal_ID"] = "Mean"
        df_amplitude_time_group.set_index("Animal_ID", inplace = True)
        xd = df_amplitude_time_group
        
        main = tk.Tk()
        msg = tk.messagebox.askquestion ('Save window','Do you want to save graphs and data?',icon = 'warning')
        if msg == "yes":
            main.destroy()
            save_file_v1 = easygui.diropenbox(msg = "Select folder for a save location", title = "Typical window")
            df_amplitude_time_group.to_excel(save_file_v1 + "//" + self.list_of_files[0] + self.list_of_files[-1] + "_amplitude_time" + ".xlsx")
        else:
            main.destroy()
        
        n = 0
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        if group == "all":
            for mice_id, row in df_amplitude_time_group.iterrows():
                n += 1
                bar.update(n)
                plt.plot(x,np.array(row), marker, label = mice_id)
                if normalize:
                    plt.title("Normalized data")
                else:
                    plt.title("Original")
            
                plt.ylabel("Movment amplitude")
                plt.xlabel("Time [s]")
                plt.annotate("Max amplitude", xy = (float(row[row == row.max()].index[0]), row.max()), xytext=(-1.0, row.max()),arrowprops = dict(facecolor='blue', shrink=0.1))
                plt.annotate("Reward start", xy = (0, row.min()), xytext=(0, (row.max() + row.min())/2),arrowprops = dict(facecolor='green', shrink=0.1))
                plt.legend()
                if msg == "yes":
                    save_file_v2 = save_file_v1 + "//" + mice_id + "_amplitude_time" + ".svg"
                    plt.savefig(save_file_v2)
                    plt.show()
                else:
                    plt.show()
        elif group == "mean":
            plt.plot(x, df_amplitude_time_group.iloc[-1], marker, label = "Mean")
            if normalize:
                plt.title("Normalized data")
            else:
                plt.title("Original")
            plt.ylabel("Movment amplitude")
            plt.xlabel("Time [s]")
            plt.annotate("Max amplitude", xy = (float(df_amplitude_time_group.iloc[-1][df_amplitude_time_group.iloc[-1] == df_amplitude_time_group.iloc[-1].max()].index[0]), df_amplitude_time_group.iloc[-1].max()), xytext=(-1.0, df_amplitude_time_group.iloc[-1].max()),arrowprops = dict(facecolor='blue', shrink=0.1))
            plt.annotate("Reward start", xy = (0, df_amplitude_time_group.iloc[-1].min()), xytext=(0, (df_amplitude_time_group.iloc[-1].max() + df_amplitude_time_group.iloc[-1].min())/2),arrowprops = dict(facecolor='green', shrink=0.1))
            plt.legend()
            if msg == "yes":
                 save_file_v2 = save_file_v1 + "//" + "Mean" + "_" + "amplitude_time" + ".svg"
                 plt.savefig(save_file_v2)
                 plt.show()
    
    def veloctiy(self, pre_stim = 2, post_stim = 0, group = "all", marker= "r", smooth = True, window_length = 9, polyorder = 3, sem = True, x_lim = False, y_lim = False):
        global df_result_veloctiy, switcher, xd, xd1
        assert len(self.list_of_df) == len(self.list_of_files)
        
        start, stop = pre_stim * 19, post_stim * 19
        columns = np.linspace(start = -pre_stim, stop = post_stim, num = start + stop + 1)
        x = [round(i,2) for i in columns]
        columns_ = [str(i) for i in x]
        x = np.array(x)
        
        columns_.append("Animal_ID")
        sem_velocity_list = []
        if self.opto:
           df_velocity_group = pd.DataFrame(columns = columns_, index = [i for i in range(0, len(self.list_of_df)+3)])
        else:
           df_velocity_group = pd.DataFrame(columns = columns_, index = [i for i in range(0, len(self.list_of_df)+1)])
    
        for l,i in enumerate(self.list_of_df):
             velocity_index = i.columns.tolist()
             velocity_index = int(velocity_index.index('Current_velocity_cm_s'))
             if self.opto:
                index_events = i.index[i['Event_Marker'] == 4].tolist()
             else:
                index_events = i.index[i['Event_Marker'] == 2].tolist()
             if switcher:
                 print(f"len: {len(i)}")
                 trial_max = self.new_max[l]
             else:
                 trial_max = i["TrialCt"].max()
             print(" ")
             list_value = [i.iloc[j-start:j+stop + 1, velocity_index].tolist() for j in index_events]
             df_velocity = pd.DataFrame(list_value,columns= [str(round(k,2)) for k in columns])
             
             if sem:
                 sem_velocity = df_velocity.sem().tolist()
                 sem_velocity_list.append(sem_velocity)
             df_velocity = df_velocity.mean().tolist()
             df_velocity.append(self.list_of_files[l])
             df_velocity_group.iloc[l] = df_velocity
        sem_velocity_list.append(df_velocity_group.sem().tolist())
        xd = sem_velocity_list
        df_velocity_group.iloc[len(self.list_of_df), 0: len(columns_)-1] = df_velocity_group.mean()
        df_velocity_group.loc[len(self.list_of_df), "Animal_ID"] = "Mean"
        if self.opto:
           df_velocity_group.loc[len(df_velocity_group)-2, "Animal_ID"] = "Mean_stim"
           df_velocity_group.loc[len(df_velocity_group)-1, "Animal_ID"] = "Mean_nostim"
        df_velocity_group.set_index("Animal_ID", inplace = True)
        if self.opto:
           columns_stim = [str(column) for column in df_velocity_group.index.values.tolist() if "_stim" in str(column)]
           df_velocity_group.loc["Mean_stim", :] = df_velocity_group.loc[columns_stim, :].mean()
           columns_nostim = [str(column) for column in df_velocity_group.index.values.tolist() if "_nostim" in str(column)]
           df_velocity_group.loc["Mean_nostim", :] = df_velocity_group.loc[columns_nostim, :].mean()
           sem_velocity_list.append(df_velocity_group.loc[columns_stim, :].sem().tolist())
           sem_velocity_list.append(df_velocity_group.loc[columns_nostim, :].sem().tolist())      
        main = tk.Tk()
        msg = tk.messagebox.askquestion ('Save window','Do you want to save graphs and data?',icon = 'warning')
        if msg == "yes":
            main.destroy()
            save_file_v1 = easygui.diropenbox(msg = "Select folder for a save location", title = "Typical window")
        else:
            main.destroy()
        y_label = "Velocity [cm/s]"
        if group == "all":
            bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
            n = 0
            iterator = 0
            for index, row in df_velocity_group.iterrows():
                n += 1
                bar.update(n)
                plt.plot(x,np.array(row), marker, label = index)
                xd1 = np.array(row)
                if sem:
                    y = np.array(row)
                    error_sem = np.array(sem_velocity_list[iterator])
                    upper_band = np.asfarray(y + error_sem)
                    lower_band = np.asfarray(y - error_sem)
                    plt.fill_between(x, upper_band, lower_band, alpha = 0.4, color = "r")
                    iterator += 1
                if x_lim:
                    plt.xlim(x_lim[0], x_lim[1])
                if y_lim:
                    plt.ylim(y_lim[0], y_lim[1])
                plt.title("Original")
                plt.ylabel(y_label)
                plt.xlabel("Time [s]")
                plt.annotate("Max velocity", xy = (float(row[row == row.max()].index[0]), row.max()), xytext=(-1.0, row.max()),arrowprops = dict(facecolor='blue', shrink=0.1))
                plt.annotate("Reward start", xy = (0, row.min()), xytext=(0, (row.max() + row.min())/2),arrowprops = dict(facecolor='green', shrink=0.1))
                plt.legend()
                
                if msg == "yes":
                    save_file_v2 = save_file_v1 + "//" + index + "_" + "orginal_velocity" + ".svg"
                    plt.savefig(save_file_v2)
                    plt.show()
                else:
                    plt.show()
                if smooth:
                    yhat = savgol_filter(np.array(row), window_length, polyorder)
                    if sem:
                        yhat_error_sem = savgol_filter(np.array(sem_velocity_list[iterator-1]), window_length, polyorder)
                        upper_band_sem = np.asfarray(yhat + yhat_error_sem)
                        lower_band_sem = np.asfarray(yhat - yhat_error_sem)
                        plt.fill_between(x, upper_band_sem, lower_band_sem, alpha = 0.4, color = "r")
                    if x_lim:
                        plt.xlim(x_lim[0], x_lim[1])
                    if y_lim:
                        plt.ylim(y_lim[0], y_lim[1])
                    plt.plot(x,yhat, marker, label = index)
                    plt.annotate("Max velocity", xy = (x[np.where(yhat == max(yhat))[0]], max(yhat)), xytext=(-1.0, max(yhat)),arrowprops = dict(facecolor='blue', shrink=0.1))
                    plt.annotate("Reward start", xy = (0, min(yhat)), xytext=(0, (max(yhat) + min(yhat))/2),arrowprops = dict(facecolor='green', shrink=0.1))
                    plt.title("Smoothed")
                    plt.ylabel(y_label)
                    plt.xlabel("Time [s]")
                    plt.legend()
                    if msg == "yes":
                        save_file_v2 = save_file_v1 + "//" + index + "_" + "smoothed_velocity" + ".svg"
                        plt.savefig(save_file_v2)
                        plt.show()
                    else:
                        plt.show()
        elif group == "mean":
            plt.plot(x, df_velocity_group.iloc[-1], marker, label = "Mean")
            if sem:
                y = np.array(df_velocity_group.iloc[-1])
                error_sem = np.array(sem_velocity_list[-1])
                upper_band = np.asfarray(y + error_sem)
                lower_band = np.asfarray(y - error_sem)
                plt.fill_between(x, upper_band, lower_band, alpha = 0.4, color = "r")
            if x_lim:
                plt.xlim(x_lim[0], x_lim[1])
            if y_lim:
                plt.ylim(y_lim[0], y_lim[1])
            plt.title("Original")
            plt.ylabel(y_label)
            plt.xlabel("Time [s]")
            plt.legend()
            if msg == "yes":
                 save_file_v2 = save_file_v1 + "//" + "Mean" + "_" + "orginal_velocity" + ".svg"
                 plt.savefig(save_file_v2)
                 plt.show()
            else:
                 plt.show()
            if smooth:
                yhat = savgol_filter(np.array(df_velocity_group.iloc[-1]), window_length, polyorder)
                if sem:
                    yhat_error_sem = savgol_filter(np.array(sem_velocity_list[-1]), window_length, polyorder)
                    upper_band_sem = np.asfarray(yhat + yhat_error_sem)
                    lower_band_sem = np.asfarray(yhat - yhat_error_sem)
                    plt.fill_between(x, upper_band_sem , lower_band_sem, alpha = 0.4, color = "r")
                if x_lim:
                    plt.xlim(x_lim[0], x_lim[1])
                if y_lim:
                    plt.ylim(y_lim[0], y_lim[1])
                plt.annotate("Max velocity", xy = (x[np.where(yhat == max(yhat))[0]], max(yhat)), xytext=(-1.0, max(yhat)),arrowprops = dict(facecolor='blue', shrink=0.1))
                plt.annotate("Reward start", xy = (0, min(yhat)), xytext=(0, max(yhat)/2),arrowprops = dict(facecolor='green', shrink=0.1))
                plt.plot(x,yhat, marker, label = "Mean")
                plt.title("Smoothed")
                plt.ylabel(y_label)
                plt.xlabel("Time [s]")
                plt.legend()
            if msg == "yes":
                save_file_v2 = save_file_v1 + "//" + "Mean" + "_" + "smoothed_velocity" + ".svg"
                plt.savefig(save_file_v2)
                plt.show()
            else:
                plt.show()
            
            if self.opto:
               fig, axs = plt.subplots(2)
               y_Mean_stim = np.array(df_velocity_group.iloc[-2])
               y_Mean_nostim = np.array(df_velocity_group.iloc[-1])
               if sem:
                    error_sem_stim = np.array(sem_velocity_list[-2])
                    error_sem_nostim = np.array(sem_velocity_list[-1])
                    upper_band_stim = np.asfarray(y_Mean_stim + error_sem_stim)
                    lower_band_stim = np.asfarray(y_Mean_stim - error_sem_stim)
                    upper_band_nostim = np.asfarray(y_Mean_nostim + error_sem_nostim)
                    lower_band_nostim = np.asfarray(y_Mean_nostim - error_sem_nostim)
                    axs[0].fill_between(x, upper_band_stim, lower_band_stim, alpha = 0.4, color = "r")
                    axs[1].fill_between(x, upper_band_nostim, lower_band_nostim, alpha = 0.4, color = "r")
               if x_lim:
                    axs[0].set_xlim(x_lim[0], x_lim[1])
                    axs[1].set_xlim(x_lim[0], x_lim[1])
               if y_lim:
                    axs[0].set_ylim(y_lim[0], y_lim[1])
                    axs[1].set_ylim(y_lim[0], y_lim[1])
               axs[0].plot(x,df_velocity_group.iloc[-2], marker, label = "Mean_velocity")
               axs[1].plot(x,df_velocity_group.iloc[-1], marker, label = "Mean_velocity")
               fig.suptitle("Original")
               axs[0].set_title('Stimulation')
               axs[1].set_title('No-stimulation')
               axs[0].set_ylabel(y_label)
               axs[0].set_xlabel("Time [s]")
               axs[1].set_ylabel(y_label)
               axs[1].set_xlabel("Time [s]")
               axs[0].annotate("Max", xy = (float(df_velocity_group.iloc[-2][df_velocity_group.iloc[-2] == df_velocity_group.iloc[-2].max()].index[0]), df_velocity_group.iloc[-2].max()), xytext=(-1.0, df_velocity_group.iloc[-2].max()),arrowprops = dict(facecolor='blue', shrink=0.1))
               axs[0].annotate("Reward start", xy = (0, df_velocity_group.iloc[-2].min()), xytext=(0, (df_velocity_group.iloc[-2].max() + df_velocity_group.iloc[-2].min())/2),arrowprops = dict(facecolor='green', shrink=0.1))
               
               axs[1].annotate("Max", xy = (float(df_velocity_group.iloc[-2][df_velocity_group.iloc[-2] == df_velocity_group.iloc[-2].max()].index[0]), df_velocity_group.iloc[-2].max()), xytext=(-1.0, df_velocity_group.iloc[-2].max()),arrowprops = dict(facecolor='blue', shrink=0.1))
               axs[1].annotate("Reward start", xy = (0, df_velocity_group.iloc[-2].min()), xytext=(0, (df_velocity_group.iloc[-2].max() + df_velocity_group.iloc[-2].min())/2),arrowprops = dict(facecolor='green', shrink=0.1))
               axs[0].legend()
               axs[1].legend()
               fig.tight_layout()

               if msg == "yes":
                    save_file_v5 = save_file_v1 + "//" + "Mean" + "_" + "orginal_velocity_stim_nostim" + ".svg"
                    plt.savefig(save_file_v5)
                    plt.show()
               else:
                    plt.show()
            
            if self.opto and smooth:
               fig, axs = plt.subplots(2)
               yhat_stim = savgol_filter(np.array(df_velocity_group.iloc[-2]), window_length, polyorder)
               yhat_nostim = savgol_filter(np.array(df_velocity_group.iloc[-1]), window_length, polyorder)
               if sem :
                   yhat_error_sem_stim = savgol_filter(np.array(sem_velocity_list[-2]), window_length, polyorder)
                   yhat_error_sem_nostim = savgol_filter(np.array(sem_velocity_list[-2]), window_length, polyorder)
                   upper_band_stim_sem = np.asfarray(yhat_stim + yhat_error_sem_stim)
                   lower_band_stim_sem = np.asfarray(yhat_stim - yhat_error_sem_stim)
                   upper_band_nostim_sem = np.asfarray(yhat_nostim + yhat_error_sem_nostim)
                   lower_band_nostim_sem = np.asfarray(yhat_nostim - yhat_error_sem_nostim)
                   axs[0].fill_between(x, upper_band_stim_sem, lower_band_stim_sem, alpha = 0.4, color = "r")
                   axs[1].fill_between(x, upper_band_nostim_sem, lower_band_nostim_sem, alpha = 0.4, color = "r")
                   if x_lim:
                         axs[0].set_xlim(x_lim[0], x_lim[1])
                         axs[1].set_xlim(x_lim[0], x_lim[1])
                   if y_lim:
                         axs[0].set_ylim(y_lim[0], y_lim[1])
                         axs[1].set_ylim(y_lim[0], y_lim[1])
               axs[0].plot(x,yhat_stim, marker, label = "Mean_velocity")
               axs[1].plot(x,yhat_nostim, marker, label = "Mean_velocity")
               fig.suptitle("Smoothed")
               axs[0].set_title('Stimulation')
               axs[1].set_title('No-stimulation')
               axs[0].set_ylabel(y_label)
               axs[0].set_xlabel("Time [s]")
               axs[1].set_ylabel(y_label)
               axs[1].set_xlabel("Time [s]")
               axs[0].annotate("Max", xy = (x[np.where(yhat_stim == max(yhat_stim))[0]], max(yhat_stim)), xytext=(-1.0, max(yhat_stim)),arrowprops = dict(facecolor='blue', shrink=0.1))
               axs[0].annotate("Reward start", xy = (0, yhat_stim.min()), xytext=(0, (yhat_stim.max() + yhat_stim.min())/2),arrowprops = dict(facecolor='green', shrink=0.1))
               
               axs[1].annotate("Max", xy = (x[np.where(yhat_nostim == max(yhat_nostim))[0]], max(yhat_nostim)), xytext=(-1.0, max(yhat_nostim)),arrowprops = dict(facecolor='blue', shrink=0.1))
               axs[1].annotate("Reward start", xy = (0, yhat_nostim.min()), xytext=(0, (yhat_nostim.max() + yhat_nostim.min())/2),arrowprops = dict(facecolor='green', shrink=0.1))
               axs[0].legend()
               axs[1].legend()
               fig.tight_layout()

               if msg == "yes":
                    save_file_v6 = save_file_v1 + "//" + "Mean" + "_" + "smoothed_velocity_stim_nostim" + ".svg"
                    plt.savefig(save_file_v6)
                    plt.show()
               else:
                    plt.show()

        df_velocity_group["Mean"] = df_velocity_group.mean(axis =1)
        df_velocity_group["Max"] = df_velocity_group.iloc[:, 0: df_velocity_group.shape[1] -1].max(axis =1)
        df_result_veloctiy = df_velocity_group
        if msg == "yes":
             df_velocity_group.to_excel(save_file_v1 + "//" + self.list_of_files[0] + "_" + self.list_of_files[-1] + "_velocity" + ".xlsx")

object_joy = Joystick_analyzer(opto= False)
object_joy.pre_proccesing(cut_to = 10000)
object_joy.find_bugs(alfa = 0.05, automatic = False)
#object_joy.veloctiy(group = "all", y_lim = [0, 10])
#object_joy.amplitude(hue = "Event_Marker", event_markers = [0,1,2,3,4], x_lim= [0,25], x_axis = "mm") 
#object_joy.move_type(event_markers = [0,1,3,4], hue = "Event_Marker", group = "all")
#object_joy.help_me()
#object_joy.prob_reward()
#object_joy.amplitude_time(group = "mean", x_axis == "mm")
#object_joy.lick_histogram(stat = "rate", group = "mean", y_lim = [0 , 20], sem = True)
object_joy.trajectory(move_range = "start_move_to_max", calibrate= True, polar = True)

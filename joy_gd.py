# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 16:41:17 2023

@author: malgo
"""

import easygui
import pandas as pd
import numpy as np
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
    def __init__(self):
        self.column_name = ["Time_in_sec", "Event_Marker", "TrialCt", "JoyPos_X", "JoyPos_Y", "Amplitude_Pos", "Base_JoyPos_X", "Base_JoyPos_Y", "SolOpenDuration", "DelayToRew", "ITI", "Threshold", "Fail_attempts", "Sum_of_fail_attempts", "Lick_state", "Sum_licks", "Type_move"]
        self.list_of_files = easygui.fileopenbox(title="Select a file", filetypes= "*.csv",  multiple=True)
        self.gen_df = (pd.read_csv(i, header=None) for i in self.list_of_files)
        self.list_of_files = [i.split("\\")[-1] for i in  self.list_of_files]
        self.list_of_files = [i.split(".")[0] for i in  self.list_of_files]
        self.list_of_df = []
        self.group = ["SAL", "CNO"]
    
    
    def pre_proccesing(self, cut_to = False):
        global df, switcher
        for i,j in enumerate(self.gen_df):
           j[0]= j[0].apply(lambda x : round((x/1000),2))
           j.columns = self.column_name
           df = j
           df.columns = self.column_name
           max_row = df["Time_in_sec"].tolist()
           max_row = [round(i) for i in max_row]
           answer = max_row.index(1801 + max_row[0])
           j = j.drop(j.index[answer-4:])
           trial_max = j["TrialCt"].max()
           bugged_index = [j.loc[j["TrialCt"] == l, "Event_Marker"].lt(0).idxmax() for l in range(1, trial_max + 1)]
           j.iloc[bugged_index, 5] = 0
           df.columns = self.column_name
           df = df.drop(df.index[answer-4:])
           df["Amplitude_Pos_mm"] = j["Amplitude_Pos"].apply(lambda x: x/ u_to_mm)
           
           delta_dist = df["Amplitude_Pos_mm"].tolist()
           delta_dist = [abs(delta_dist[ff] - delta_dist[ff-1]) for ff in range(1, len(delta_dist))]
           delta_dist.insert(0, 0)
           #delta_dist.append(0)
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
           self.list_of_df.append(j)
           print(j.info())
        switcher = False
           
    def amplitude(self, event_markers = [0,1,2,3,4], hue = None, kde = False, group = "Mouse_ID", fill_nan = True, y_stat = "count", x_axis = "units"):
        global df, df_result_amplitude, switcher, xd1, xd
        amplitude_all =[]
        assert len(self.list_of_df) == len(self.list_of_files)
        if x_axis == "units":
            df_amplitude = pd.DataFrame(columns= ["TrialCt", "Mouse_ID", "Amplitude_Pos", "Event_Marker"])
        elif x_axis == "mm":
            df_amplitude = pd.DataFrame(columns= ["TrialCt", "Mouse_ID", "Amplitude_Pos_mm", "Event_Marker"])
        for l,i in enumerate(self.list_of_df):
            if switcher:
                self.good_index_list
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
                for k in event_markers:
                    if x_axis == "units":
                        amplitude_ = [i.loc[(i["TrialCt"] == j) & (i["Event_Marker"] == k), "Amplitude_Pos"].max() for j in range(1, trial_max + 1)]
                        mouse_id = [self.list_of_files[l] for m in range(1, trial_max + 1)]
                        event_marker = [k for j in range(1, trial_max + 1)]
                        dict_to_add = {"TrialCt": range(1, trial_max + 1), "Mouse_ID": mouse_id, "Amplitude_Pos": amplitude_, "Event_Marker": event_marker}
                        df_amplitude = df_amplitude.append(pd.DataFrame(dict_to_add))
                        df_amplitude.reset_index(inplace = True, drop = True)
                    elif x_axis == "mm":
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
                mean_hight = [round(len(df_amplitude.loc[(df_amplitude["Amplitude_Pos"] >= bins[ii]) & (df_amplitude["Amplitude_Pos"] < bins[ii+1]) & (df_amplitude["Event_Marker"].isin(event_markers)), "TrialCt"]) / len(self.list_of_df)) for ii in range(0, len(bins)-1)]
                mean_event = [df_amplitude.loc[(df_amplitude["Amplitude_Pos"] >= bins[ii]) & (df_amplitude["Amplitude_Pos"] < bins[ii+1]) & (df_amplitude["Event_Marker"].isin(event_markers)), "Event_Marker"]  for ii in range(0, len(bins)-1)]
            #mean_amp = [df_amplitude.loc[(df_amplitude["Amplitude_Pos"] >= bins[ii]) & (df_amplitude["Amplitude_Pos"] < bins[ii+1]) & (df_amplitude["Event_Marker"].isin(event_markers)), "Event_Marker"]  for ii in range(0, len(bins)-1)]
                xd1 = mean_hight
                bins.remove(260)
                list_of_list = []
                for i,j in enumerate(mean_hight):
                    fake_data = [bins[i] + 5 for value in range(0, j)]
                    list_of_list.append(fake_data)
                flatten_matrix = [val for sublist in list_of_list for val in sublist]
                bin_amplitude = pd.Series(flatten_matrix, name = "Amplitude_Pos")
                xd = bin_amplitude
                sns.histplot(data = bin_amplitude, bins = bins, stat = y_stat, fill = True, kde=True)
                plt.legend(labels=[f"n = {len(self.list_of_df)}"])
            elif x_axis == "mm":
                mean_hight = [round(len(df_amplitude.loc[(df_amplitude["Amplitude_Pos_mm"] >= bins[ii]) & (df_amplitude["Amplitude_Pos_mm"] < bins[ii+1]) & (df_amplitude["Event_Marker"].isin(event_markers)), "TrialCt"]) / len(self.list_of_df)) for ii in range(0, len(bins)-1)]
                mean_event = [df_amplitude.loc[(df_amplitude["Amplitude_Pos_mm"] >= bins[ii]) & (df_amplitude["Amplitude_Pos_mm"] < bins[ii+1]) & (df_amplitude["Event_Marker"].isin(event_markers)), "Event_Marker"]  for ii in range(0, len(bins)-1)]
            #mean_amp = [df_amplitude.loc[(df_amplitude["Amplitude_Pos"] >= bins[ii]) & (df_amplitude["Amplitude_Pos"] < bins[ii+1]) & (df_amplitude["Event_Marker"].isin(event_markers)), "Event_Marker"]  for ii in range(0, len(bins)-1)]
                xd1 = mean_hight
                bins.remove(26)
                list_of_list = []
                for i,j in enumerate(mean_hight):
                    fake_data = [bins[i] + 0.5 for value in range(0, j)]
                    list_of_list.append(fake_data)
                flatten_matrix = [val for sublist in list_of_list for val in sublist]
                bin_amplitude = pd.Series(flatten_matrix, name = "Amplitude_Pos_mm")
                xd = bin_amplitude
                sns.histplot(data = bin_amplitude, bins = bins, stat = y_stat, fill = True, kde=True)
                plt.legend(labels=[f"n = {len(self.list_of_df)}"])
        else:
            xd = df_amplitude
            if x_axis == "units":
                sns.displot(df_amplitude, x = "Amplitude_Pos", hue = hue, col = group, kde = kde, color = "green", palette = "tab10", bins = bins, stat = y_stat)
            elif x_axis == "mm":
                sns.displot(df_amplitude, x = "Amplitude_Pos_mm", hue = hue, col = group, kde = kde, color = "green", palette = "tab10", bins = bins, stat = y_stat)
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
                bin_amplitude.to_excel(save_file_v1 + "//" + self.list_of_files[0] + "_" + self.list_of_files[-1] + "_amplitude_mean" + ".xlsx")
            else:
                save_file_v2 = save_file_v1 + "//" + self.list_of_files[0] + "__" + self.list_of_files[-1] + "_amplitude_each_group" + ".svg"
            plt.savefig(save_file_v2)
            plt.show()
            df_amplitude.to_excel(save_file_v1 + "//" + self.list_of_files[0] + "_" + self.list_of_files[-1] + "_amplitude" + ".xlsx")
            
        else:
            plt.show()
            main.destroy()
        df_result_amplitude = df_amplitude
    
    def lick_histogram(self, pre_stim = 2, post_stim = 2, group = "all", marker= "r", smooth = True, window_length = 9, polyorder = 3, stat = "prob", sem = True, x_lim = False, y_lim = False): 
       global df_result_lick, switcher, xd, df_lick_counts
       assert len(self.list_of_df) == len(self.list_of_files)
       
       start, stop = pre_stim * 19, post_stim * 19
       columns = np.linspace(start = -pre_stim, stop = post_stim, num = start + stop + 1)
       x = [round(i,2) for i in columns]
       columns_ = [str(i) for i in x]
       x = np.array(x)

       columns_.append("Animal_ID")
       df_licks_group = pd.DataFrame(columns = columns_, index = [i for i in range(0, len(self.list_of_df)+1)])
       sem_lick_group = []
       for l,i in enumerate(self.list_of_df):
            index_events = i.index[i['Event_Marker'] == 2].tolist()
            if switcher:
                print(f"len: {len(i)}")
                trial_max = self.new_max[l]
            else:
                trial_max = i["TrialCt"].max()
            print(" ")
            list_value = [i.iloc[j-start:j+stop + 1, 14].tolist() for j in index_events]
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
                
                lick_rate= lick_rate.mean().tolist()
                lick_rate.append(self.list_of_files[l])
                df_licks_group.iloc[l] = lick_rate
       if stat == "rate":
           sem_lick_group.append(df_licks_group.sem().tolist())
       df_licks_group.iloc[len(self.list_of_df), 0: len(columns_)-1] = df_licks_group.mean()
       df_licks_group.loc[len(self.list_of_df), "Animal_ID"] = "Mean"
       df_licks_group.set_index("Animal_ID", inplace = True)
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
           xd = sem_lick_group
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
               yhat = savgol_filter(np.array(df_licks_group.iloc[-1]), window_length, polyorder)
               if sem and stat == "rate":
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
       if stat == "rate":
           df_licks_group["Mean"] = df_licks_group.mean(axis =1)
           df_licks_group["Max"] = df_licks_group.iloc[:, 0: df_licks_group.shape[1] -1].max(axis=1)
       if msg == "yes":
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
            trial_max = i["TrialCt"].max()
            ans_x = [set(i.loc[i["TrialCt"] == ii, "Base_JoyPos_X"]).pop() for ii in range(1,trial_max +1)]
            ans_y = [set(i.loc[i["TrialCt"] == ii, "Base_JoyPos_Y"]).pop() for ii in range(1,trial_max +1)]
            x_bool = [True if base_x_lower < jj < base_x_upper else False for jj in ans_x]
            y_bool = [True if base_y_lower < kk < base_y_upper else False for kk in ans_y]
            x_y_bool = ["ok" if x == True and y == True else "not ok" if x == True or y == True else "very bad" for x, y in zip(x_bool, y_bool)]
            df_result = pd.DataFrame({"x_bool": x_bool, "y_bool": y_bool, "x_y_bool": x_y_bool}, index = range(1,trial_max +1))
            xd1 = df_result
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
    
    def trajectory(self, move_range = "0_to_max", calibrate = False):
        global xd1
        noramlizer = MinMaxScaler()
        
        
        for l,i in enumerate(self.list_of_df):
            trial_list = sorted(set(i.loc[:, "TrialCt"].tolist()))
            if calibrate:
                i["JoyPos_X"] = i["Base_JoyPos_X"] - i["JoyPos_X"]
                i["JoyPos_Y"] = i["Base_JoyPos_Y"] - i["JoyPos_Y"] 
            xd1 = i
            start_index = [i.loc[i["TrialCt"] == hh].first_valid_index() for hh in trial_list if i.loc[(i["TrialCt"] == hh) & (i["Event_Marker"] == 1), "Amplitude_Pos"].tolist()]
            
            movment_end = [i.loc[(i["TrialCt"] == ii) & (i["Event_Marker"] == 1), "Amplitude_Pos"].idxmax() for ii in trial_list if i.loc[(i["TrialCt"] == ii) & (i["Event_Marker"] == 1), "Amplitude_Pos"].tolist()]
            
            cropp_movment = [index for range_index in zip(movment_end, start_index) for index in range(range_index[0], range_index[1]-1,-1)]

            movment_list_x = [i.loc[i["TrialCt"] == g , "JoyPos_X"].tolist() for g in trial_list]
            movment_list_x_norm = [ noramlizer.fit_transform(np.array(gg).reshape(-1, 1)) for gg in movment_list_x]
            movment_list_y = [i.loc[i["TrialCt"] == g , "JoyPos_Y"].tolist() for g in trial_list]
            movment_list_y_norm = [ noramlizer.fit_transform(np.array(gg).reshape(-1, 1)) for gg in movment_list_y]
            
            movment_list_x_start_to_max = [i.iloc[range_index_1[0]: range_index_1[1]+1, 3].tolist() for range_index_1 in zip(start_index, movment_end)]
            movment_list_y_start_to_max = [i.iloc[range_index_1[0]: range_index_1[1]+1, 4].tolist() for range_index_1 in zip(start_index, movment_end)]
            
            list_final = []
            list_supporter = []
            
            for range_index in zip(movment_end, start_index):
                for current_index in range(range_index[0], range_index[1]-1,-1):
                    if i.iloc[current_index, 5] < 10 and i.iloc[current_index, 1] == 0:
                        list_supporter.append(current_index)
                        break
            for range_index_2 in zip(list_supporter, start_index):
                for current_index in range(range_index_2[0], range_index_2[1]-1,-1):
                    if i.iloc[current_index, 5] >= 10:
                        list_final.append(current_index+1)
                        break
                    elif current_index == range_index_2[1]:
                        list_final.append(current_index)
                        break
            movment_list_x_0_to_max = [i.iloc[range_index_1[0]: range_index_1[1]+1, 3].tolist() for range_index_1 in zip(list_final, movment_end)]
            movment_list_y_0_to_max = [i.iloc[range_index_1[0]: range_index_1[1]+1, 4].tolist() for range_index_1 in zip(list_final, movment_end)]
            
            xd = movment_list_x_0_to_max
            
            if move_range == "0_to_max":
                [plt.plot(ll[0], ll[1], c = "g", alpha = 0.2) for ll in zip(movment_list_x_0_to_max, movment_list_y_0_to_max)]
                plt.title(self.list_of_files[l])
                plt.show()
            elif move_range == "start_to_max":
                [plt.plot(lll[0], lll[1], c = "g", alpha = 0.2) for lll in zip(movment_list_x_start_to_max, movment_list_y_start_to_max)]
                plt.title(self.list_of_files[l])
                plt.show()
            elif move_range == "x_norm":
                [plt.plot(llll[0], llll[1], c = "g", alpha = 0.2) for llll in zip(movment_list_x, movment_list_y)]
                plt.title(self.list_of_files[l])
                plt.show()
    
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
        df_velocity_group = pd.DataFrame(columns = columns_, index = [i for i in range(0, len(self.list_of_df)+1)])
        
        for l,i in enumerate(self.list_of_df):
             index_events = i.index[i['Event_Marker'] == 2].tolist()
             if switcher:
                 print(f"len: {len(i)}")
                 trial_max = self.new_max[l]
             else:
                 trial_max = i["TrialCt"].max()
             print(" ")
             list_value = [i.iloc[j-start:j+stop + 1, 19].tolist() for j in index_events]
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
        df_velocity_group.set_index("Animal_ID", inplace = True)
        
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
                        plt.fill_between(x, upper_band, lower_band, alpha = 0.4, color = "r")
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
        
        df_velocity_group["Mean"] = df_velocity_group.mean(axis =1)
        df_velocity_group["Max"] = df_velocity_group.iloc[:, 0: df_velocity_group.shape[1] -1].max(axis =1)
        df_result_veloctiy = df_velocity_group
        if msg == "yes":
             df_velocity_group.to_excel(save_file_v1 + "//" + self.list_of_files[0] + "_" + self.list_of_files[-1] + "_velocity" + ".xlsx")

#object_joy = Joystick_analyzer()
#object_joy.pre_proccesing(cut_to = 100700)
#object_joy.find_bugs(alfa = 0.36, automatic = False)

#object_joy.veloctiy(group = "mean", y_lim = [0, 20])
#object_joy.amplitude(event_markers = [1,3,2], x_axis = "mm", hue = "Event_Marker", group = None)
#object_joy.move_type(event_markers = [0,1,3,4], hue = "Event_Marker", group = "all")
#object_joy.help_me()
#object_joy.prob_reward()
#object_joy.amplitude_time(group = "mean", x_axis == "mm")
#object_joy.lick_histogram(stat = "prob", group = "all", y_lim = [0 , 8])
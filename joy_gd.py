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
xd = None 
xd1 = None
switcher = False
class Joystick_analyzer:
    def __init__(self):
        self.column_name = ["Time_in_sec", "Event_Marker", "TrialCt", "JoyPos_X", "JoyPos_Y", "Amplitude_Pos", "Base_JoyPos_X", "Base_JoyPos_Y", "SolOpenDuration", "DelayToRew", "ITI", "Threshold", "Fail_attempts", "Sum_of_fail_attempts", "Lick_state", "Sum_licks", "Type_move"]
        self.list_of_files = easygui.fileopenbox(title="Select a file", filetypes= "*.csv",  multiple=True)
        self.gen_df = (pd.read_csv(i, header=None) for i in self.list_of_files)
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
           trial_max = j["TrialCt"].max()
           #bugged_index = [j.loc[j["Event_Marker"] == l, "Event_Marker"].lt(0).idxmax() for l in range(1, trial_max + 1)]
           bugged_index = [j.loc[j["TrialCt"] == l, "Event_Marker"].lt(0).idxmax() for l in range(1, trial_max + 1)]
           j.iloc[bugged_index, 5] = 0
           df.columns = self.column_name
           df = df.drop(df.index[answer-4:])
           df.iloc[bugged_index, 5] = 0
           self.list_of_df.append(j)
           
           
    def amplitude(self, event_markers = [0,1,2,3,4], hue = None, kde = False, group = "Mouse_ID", fill_nan = True):
        global df, df_result_amplitude, switcher
        amplitude_all =[]
        assert len(self.list_of_df) == len(self.list_of_files)
        df_amplitude = pd.DataFrame(columns= ["TrialCt", "Mouse_ID", "Amplitude_Pos", "Event_Marker"])
        for l,i in enumerate(self.list_of_df):
            if switcher:
                self.good_index
                for k in event_markers:
                    amplitude_ = [i.loc[(i["TrialCt"] == j) & (i["Event_Marker"] == k), "Amplitude_Pos"].max() for j in self.good_index]
                    mouse_id = [self.list_of_files[l] for m in range(1, len(self.good_index) +1)]
                    event_marker = [k for j in range(1, len(self.good_index) +1)]
                    dict_to_add = {"TrialCt": self.good_index, "Mouse_ID": mouse_id, "Amplitude_Pos": amplitude_, "Event_Marker": event_marker}
                    df_amplitude = df_amplitude.append(pd.DataFrame(dict_to_add))
                    df_amplitude.reset_index(inplace = True, drop = True)
            
            else:
                trial_max = i["TrialCt"].max()
                for k in event_markers:
                    amplitude_ = [i.loc[(i["TrialCt"] == j) & (i["Event_Marker"] == k), "Amplitude_Pos"].max() for j in range(1, trial_max + 1)]
                    mouse_id = [self.list_of_files[l] for m in range(1, trial_max + 1)]
                    event_marker = [k for j in range(1, trial_max + 1)]
                    dict_to_add = {"TrialCt": range(1, trial_max + 1), "Mouse_ID": mouse_id, "Amplitude_Pos": amplitude_, "Event_Marker": event_marker}
                    df_amplitude = df_amplitude.append(pd.DataFrame(dict_to_add))
                    df_amplitude.reset_index(inplace = True, drop = True)
        if fill_nan:
            null_sum = df_amplitude["Amplitude_Pos"].isnull().sum()
            for l,i in enumerate(self.list_of_files):
                for k in event_markers:
                    mask = (df_amplitude["Mouse_ID"] == i) & (df_amplitude["Event_Marker"] == k)
                    mean = round(df_amplitude.loc[mask, "Amplitude_Pos"].mean(),2)
                    df_amplitude.loc[mask, "Amplitude_Pos"] = df_amplitude.loc[mask, "Amplitude_Pos"].fillna(mean)
                    
            print(f"I filled {null_sum} data points")
        sns.set_style('ticks')
        sns.displot(df_amplitude, x = "Amplitude_Pos", hue = hue, col = group, kde = kde, color = "green", palette = "tab10")
        null_sum = df_amplitude["Amplitude_Pos"].isnull().sum()
        main = tk.Tk()
        msg = tk.messagebox.askquestion ('Save window','Do you want to save graphs and data?',icon = 'warning')
        if msg == "yes":
            main.destroy()
            save_file_v1 = easygui.diropenbox(msg = "Select folder for a save location", title = "Typical window")
            save_file_v2 = save_file_v1 + "//" + self.list_of_files[0] + "__" + self.list_of_files[-1] + ".svg"
            plt.savefig(save_file_v2)
            plt.show()
            df_amplitude.to_excel(save_file_v1 + "//" + self.list_of_files[0] + "_" + self.list_of_files[-1] + "_amplitude" + ".xlsx")
        else:
            plt.show()
            main.destroy()
        df_result_amplitude = df_amplitude
    
    def lick_histogram(self, pre_stim = 2, post_stim = 2, group = "all", marker= "r", smooth = True, window_length = 9, polyorder = 3): 
       global df_result_lick, switcher
       assert len(self.list_of_df) == len(self.list_of_files)
       
       start, stop = pre_stim * 19, post_stim * 19
       columns = np.linspace(start = -pre_stim, stop = post_stim, num = start + stop + 1)
       x = [round(i,2) for i in columns]
       columns_ = [str(i) for i in x]
       x = np.array(x)

       columns_.append("Animal_ID")
       df_licks_group = pd.DataFrame(columns= columns_, index = [i for i in range(0, len(self.list_of_df)+1)])
       for l,i in enumerate(self.list_of_df):
            index_events = i.index[i['Event_Marker'] == 2].tolist()
            if switcher:
                trial_max = self.new_max
                
            else:
                trial_max = i["TrialCt"].max()
            list_value = [i.iloc[j-start:j+stop + 1, 14].tolist() for j in index_events]
            df_licks = pd.DataFrame(list_value,columns= [str(round(k,2)) for k in columns])
            prob_lick = df_licks.apply(lambda x: x.value_counts())
            prob_lick.fillna(0, inplace = True)
            prob_lick = round(prob_lick / trial_max,2)
            prob_lick = prob_lick.iloc[1, :].tolist()
            prob_lick.append(self.list_of_files[l])
            df_licks_group.iloc[l] = prob_lick

       df_licks_group.iloc[len(self.list_of_df), 0: len(columns_)-1] = df_licks_group.mean()
       df_licks_group.loc[len(self.list_of_df), "Animal_ID"] = "Mean"
       df_licks_group.set_index("Animal_ID", inplace = True)
       df_result_lick = df_licks_group
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
           for index, row in df_licks_group.iterrows():
               n += 1
               bar.update(n)
               plt.plot(x,np.array(row), marker, label = index)
               plt.title("Original")
               plt.ylabel("Probability density")
               plt.xlabel("Time [s]")
               plt.annotate("Max prob", xy = (float(row[row == row.max()].index[0]), row.max()), xytext=(-1.0, row.max()),arrowprops = dict(facecolor='blue', shrink=0.1))
               plt.annotate("Reward start", xy = (0, row.min()), xytext=(0, (row.max() + row.min())/2),arrowprops = dict(facecolor='green', shrink=0.1))
               plt.legend()
               
               if msg == "yes":
                   save_file_v2 = save_file_v1 + "//" + index + "_" + "orginal" + ".svg"
                   plt.savefig(save_file_v2)
                   plt.show()
               else:
                   plt.show()
               if smooth:
                   yhat = savgol_filter(np.array(row), window_length, polyorder)
                   plt.plot(x,yhat, marker, label = index)
                   plt.annotate("Max prob", xy = (x[np.where(yhat == max(yhat))[0]], max(yhat)), xytext=(-1.0, max(yhat)),arrowprops = dict(facecolor='blue', shrink=0.1))
                   plt.annotate("Reward start", xy = (0, min(yhat)), xytext=(0, (max(yhat) + min(yhat))/2),arrowprops = dict(facecolor='green', shrink=0.1))
                   plt.title("Smoothed")
                   plt.ylabel("Probability density")
                   plt.xlabel("Time [s]")
                   plt.legend()
                   if msg == "yes":
                       save_file_v2 = save_file_v1 + "//" + index + "_" + "smoothed" + ".svg"
                       plt.savefig(save_file_v2)
                       plt.show()
                   else:
                       plt.show()
       elif group == "mean":
           plt.plot(x,df_licks_group.iloc[-1], marker, label = "Mean")
           plt.title("Original")
           plt.ylabel("Probability density")
           plt.xlabel("Time [s]")
           plt.legend()
           if msg == "yes":
                save_file_v2 = save_file_v1 + "//" + "Mean" + "_" + "orginal" + ".svg"
                plt.savefig(save_file_v2)
                plt.show()
           else:
                plt.show()
           if smooth:
               yhat = savgol_filter(np.array(df_licks_group.iloc[-1]), window_length, polyorder)
               plt.annotate("Max prob", xy = (x[np.where(yhat == max(yhat))[0]], max(yhat)), xytext=(-1.0, max(yhat)),arrowprops = dict(facecolor='blue', shrink=0.1))
               plt.annotate("Reward start", xy = (0, min(yhat)), xytext=(0, max(yhat)/2),arrowprops = dict(facecolor='green', shrink=0.1))
               plt.plot(x,yhat, marker, label = "Mean")
               plt.title("Smoothed")
               plt.ylabel("Probability density")
               plt.xlabel("Time [s]")
               plt.legend()
           if msg == "yes":
               save_file_v2 = save_file_v1 + "//" + "Mean" + "_" + "smoothed" + ".svg"
               plt.savefig(save_file_v2)
               plt.show()
           else:
               plt.show()
        
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
    def find_bugs(self, alfa = 0.10):
        global xd, xd1, switcher
        self.list_of_df_v1 = []
        print("Test started\n")
        main = tk.Tk()
        msg2 = tk.messagebox.askquestion ('Delete window','Do you want to delete bugged data?',icon = 'warning')
        main.destroy()
        for l,i in enumerate(self.list_of_df):
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
            print("Mice ID: ", self.list_of_files[l],"\n", df_result["x_y_bool"].value_counts(), "\n" ,round(df_result["x_y_bool"].value_counts(normalize=True),2))
            print(" ")
            trial_max = i["TrialCt"].max()
            if msg2 == "yes":
                buggs_index = df_result.index[(df_result['x_y_bool'] == "not ok") | (df_result['x_y_bool'] == "very bad")].tolist()
                self.good_index = df_result.index[(df_result['x_y_bool'] == "ok")].tolist()
                self.new_max =  trial_max - len(buggs_index)
                i.drop(i[i["TrialCt"].isin(buggs_index)].index, inplace = True)
                self.list_of_df_v1.append(i)
                switcher = True
        print("Test completed")
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
        print("amplitude \n function parameters:\n event_markers - which events will be included in graph [value: 0,1,2,3,4 (int)] \n hue - events markers will be presented separately or jointly [value: Event_Marker (string), None]\n kde - data will be presented as an output of kernel density estimation [value: True (bool), False (bool)]\n group - graphs will be created for each mice separately or together [value: Mouse_ID (string), None]\n fill_nan - fill missing data [value: True (bool), False (bool)]")
        print("")
        print("lick_histogram \n function parameters:\n pre_stim - how much time in sec you want to include in graph, up to reward onset [value: (int)] \n post_stim - how much time in sec you want to include in graph, after reward onset [value: (int)] \n group - graphs will be created for each mice separately or together [value: all, mean (string)]\n marker - color and type of line on graph [value: (string)]\n smooth - smoothing algorithm that go through data [value: True (bool), False (bool)]\n window_length - (only if smooth = True) how long will be the polynomial that will be fitted to data [value: (int)]\n polyorder - (only if smooth = True) the degree of a polynomial that will be fitted to data")
        print("")
        
    def trajectory(self, move_range = "0_to_max"):
        noramlizer = MinMaxScaler()
        
        
        for l,i in enumerate(self.list_of_df):
            trial_list = sorted(set(i.loc[:, "TrialCt"].tolist()))
            
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
            xd1= list_supporter
            xd = movment_list_x_0_to_max
            
            if move_range == "0_to_max":
                [plt.plot(ll[0], ll[1], c = "g", alpha = 0.2) for ll in zip(movment_list_x_0_to_max, movment_list_y_0_to_max)]
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
    
    def amplitude_time(self, pre_stim = 2, post_stim = 2, normalize = False, marker= "r"):
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
        else:
            main.destroy()
        
        n = 0
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        
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
            plt.show()
            

object_joy = Joystick_analyzer()
object_joy.pre_proccesing()
#object_joy.find_bugs(alfa = 0.05)
#object_joy.trajectory()
#object_joy.lick_histogram()
#object_joy.amplitude(event_markers = [0,1], hue = "Event_Marker", fill_nan = True, group = None)
#object_joy.move_type(event_markers = [0,1,3,4], hue = "Event_Marker", group = "all")
#object_joy.help_me()
#object_joy.prob_reward()
object_joy.amplitude_time()

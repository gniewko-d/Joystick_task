import easygui
import pandas as pd
import numpy as np
#from termcolor import colored
import math 
#import progressbar
import tkinter as tk
import seaborn as sns;# sns.set_theme()
from tkinter import messagebox
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from scipy.integrate import quad
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller


u_to_mm = 10
class Fiber_photometry:
    def __init__(self, granger = False):
        if not granger:
            ok_btn_txt = "Continue"
            title = "Information box"
            message_first = ["Upload CSV files for first type of signal"] 
            output = easygui.msgbox(message_first, title, ok_btn_txt)
            
            self.list_of_files = easygui.fileopenbox(title="Select a file", filetypes= "*.csv",  multiple=True)
            self.gen_df = (pd.read_csv(i) for i in self.list_of_files)
            self.list_of_files = [i.split("\\")[-1] for i in  self.list_of_files]
            self.list_of_files = [i.split(".")[0] for i in  self.list_of_files]
            
            message_second = ["Upload CSV files for second type of signal"] 
            output = easygui.msgbox(message_second, title, ok_btn_txt)
            
            self.list_of_files_2 = easygui.fileopenbox(title="Select a file", filetypes= "*.csv",  multiple=True)
            self.gen_df_2 = (pd.read_csv(i) for i in self.list_of_files_2)
            self.list_of_files_2 = [i.split("\\")[-1] for i in  self.list_of_files_2]
            self.list_of_files_2 = [i.split(".")[0] for i in  self.list_of_files_2]

            
            print("Uploaded files:")
            ans = [print(i) for i in self.list_of_files]
            ans = [print(i) for i in self.list_of_files_2]
            self.list_of_df = []
            self.list_of_df_v2 = []
        main = tk.Tk()
        self.msg = tk.messagebox.askquestion ('Save window','Do you want to save graphs and data?',icon = 'warning')
        if self.msg == "yes":
            main.destroy()
            self.save_file_v1 = easygui.diropenbox(msg = "Select folder for a save location", title = "Typical window")
        else:
            main.destroy()  

    def pre_process(self, data_type = "DeltaF", start_time = False, stop_time = False, trial_start = False, trial_stop = False, trials_numbers = False):
        self.start_time = start_time
        self.stop_time = stop_time
        self.trial_start = trial_start
        self.trial_stop = trial_stop
        self.trials_numbers = trials_numbers
        self.data_type = data_type
        filter_file = None
        if trials_numbers:
                ok_btn_txt = "Continue"
                title = "Information box"
                message_third = ["Upload Excel file with additional informations about channels processing: "] 
                output = easygui.msgbox(message_third, title, ok_btn_txt)
                filter_file = easygui.fileopenbox(title="Select a file", filetypes= "*.xslx",  multiple=False)
                filter_file = pd.read_excel(filter_file, header=None)
                self.filter_file = filter_file
        for index, df in enumerate(zip(self.gen_df,  self.gen_df_2)):
            curent_title_v1 = self.list_of_files[index] 
            curent_title_v2 = self.list_of_files_2[index]
            df_1 = self.__do_pre_process(start_time, stop_time, trial_start, trial_stop, data_type, df[0], trials_numbers, curent_title_v1,filter_file)
            df_2 = self.__do_pre_process(start_time, stop_time, trial_start, trial_stop, data_type, df[1], trials_numbers, curent_title_v2,filter_file)
            self.list_of_df.append(df_1)
            self.list_of_df_v2.append(df_2)
    
    def __do_pre_process(self,start_time, stop_time, trial_start, trial_stop, data_type, df, trials_numbers, current_title, filter_file):
        stoper = True
        columns_ = df.columns.tolist()
        columns_ = [i for i in columns_ if data_type in i or i == "timestamp"]
        df = df.loc[:, columns_]
        if start_time:
                start_idx = abs(df["timestamp"]- start_time).idxmin()
                df = df.iloc[start_idx:, :]
                df.reset_index(drop=True, inplace=True)

        if stop_time:
            stop_idx = abs(df["timestamp"]- stop_time).idxmin()
            df = df.iloc[:stop_idx +1, :]
            df.reset_index(drop=True, inplace=True)

        if trial_start or trial_stop:
            time_ = df["timestamp"]
            df = df.iloc[:,1:]
            if not trial_start:
                trial_start = 0  
            if not trial_stop:
                trial_stop = len(df.columns)
            if trial_start != 0 and stoper:
                trial_start -=1
                stoper = False
        
            
            df = df.iloc[:, trial_start: trial_stop ]
            df.insert(0, "timestamp", time_)
        if trials_numbers:
                trials_to_save = filter_file.loc[filter_file[0] == current_title, 1]
                if len(trials_to_save):
                    trials_to_save = trials_to_save.values[0].split(",")
                    trials_to_save = [int(val) for val in trials_to_save]
                    if 0 not in trials_to_save:
                        trials_to_save.insert(0,0)
                    df = df.iloc[:, trials_to_save]
                else:
                    pass
        return df

    def plot(self, smooth = True, window_length_fibre = 12, polyorder_fibre = 3, extra_decoration = True,  window_length_behaviour = 12, polyorder_behaviour = 3, heat_ticks_n = [-2,-1,0,1,2,3,4,5,6],  y1 =[],y2=[], y3=[],y4=[]):
        
        self.__pre_process_behaviour()

        columns_ = self.list_of_df[0]["timestamp"].tolist()
        df_plot_group_velocity = pd.DataFrame(columns= columns_, index = [i for i in range(0, len(self.list_of_df)+1)])
        df_sem_group_velocity = pd.DataFrame(columns= columns_, index = [i for i in range(0, len(self.list_of_df)+1)])
        
        df_plot_group_lick = pd.DataFrame(columns= columns_, index = [i for i in range(0, len(self.list_of_df)+1)])
        df_sem_group_lick = pd.DataFrame(columns= columns_, index = [i for i in range(0, len(self.list_of_df)+1)])

        df_plot_group_signal_1 = pd.DataFrame(columns= columns_, index = [i for i in range(0, len(self.list_of_df)+1)])
        df_sem_group_signal_1 = pd.DataFrame(columns= columns_, index = [i for i in range(0, len(self.list_of_df))])
        
        df_plot_group_signal_2 = pd.DataFrame(columns= columns_, index = [i for i in range(0, len(self.list_of_df)+1)])
        df_sem_group_signal_2 = pd.DataFrame(columns= columns_, index = [i for i in range(0, len(self.list_of_df))])
        
        
        index_signal_1 = self.list_of_files
        index_signal_1.append("Mean")
        index_signal_2 = self.list_of_files_2
        index_signal_2.append("Mean")
        index_beh = self.list_of_files_behaviour
        index_beh.append("Mean")
        for i,df in enumerate(zip(self.list_of_df, self.list_of_df_v2)):
            

            time_ = np.array(df[0]["timestamp"] /1000)
            self.time_ = np.array(df[0]["timestamp"])

            ############# First Signal ###############
            data_signal_1 = df[0].iloc[:,1:]    
            data_mean_signal_1 = np.array(data_signal_1.mean(axis = 1))
            data_sem_signal_1  = np.array(data_signal_1.sem(axis = 1))
            upper_band_signal_1 = np.asfarray(data_mean_signal_1 + data_sem_signal_1)
            lower_band_signal_1 = np.asfarray(data_mean_signal_1 - data_sem_signal_1)

            df_plot_group_signal_1.iloc[i, :] = data_mean_signal_1
            df_sem_group_signal_1.iloc[i, :] = data_sem_signal_1
            
            ############# Second Signal ###############
            data_signal_2 = df[1].iloc[:,1:]    
            data_mean_signal_2 = np.array(data_signal_2.mean(axis = 1))
            data_sem_signal_2  = np.array(data_signal_2.sem(axis = 1))
            upper_band_signal_2 = np.asfarray(data_mean_signal_2 + data_sem_signal_2)
            lower_band_signal_2 = np.asfarray(data_mean_signal_2 - data_sem_signal_2)

            df_plot_group_signal_2.iloc[i, :] = data_mean_signal_2
            df_sem_group_signal_2.iloc[i, :] = data_sem_signal_2
            
            ################## FIGURE #######################
            fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(8,6), sharex= True)
            fig.suptitle(self.list_of_files[i])
            fig.tight_layout()
            
            mean_velocity, sem_velocity, mean_lick, sem_lick, velocity_conteriner, lick_conteriner = self.__process_behaviour(self.list_of_df_behaviour[i])
            upper_band_velocity = np.asfarray(mean_velocity + sem_velocity)
            lower_band_velocity  = np.asfarray(mean_velocity - sem_velocity)
            
            upper_band_lick = np.asfarray(mean_lick + sem_lick)
            lower_band_lick  = np.asfarray(mean_lick - sem_lick)
                       
            # 1st plot
            ax[0].plot(time_, data_mean_signal_1, color = "r", label = "Mean signal_1 +/- SEM", zorder=10)
            ax[0].fill_between(time_, upper_band_signal_1, lower_band_signal_1, alpha = 0.4, color = "r", zorder=10) 
            ax[0].set_ylabel(self.data_type)
            ax[0].axhline(0, color = "black")
            if y1:
                ax[0].set_ylim(y1[0], y1[1])
            # 2nd plot
            ax[1].plot(time_, data_mean_signal_2, color = "r", label = "Mean signal_2 +/- SEM", zorder=10)
            ax[1].fill_between(time_, upper_band_signal_2, lower_band_signal_2, alpha = 0.4, color = "r", zorder=10) 
            ax[1].set_ylabel(self.data_type)
            ax[1].axhline(0, color = "black")
            if y2:
                ax[1].set_ylim(y2[0], y2[1])
            # 3rd plot
            ax[2].plot(time_, mean_velocity, color = "r", label = "Mean velocity +/- SEM", zorder=10)
            ax[2].set_ylabel("Velocity [cm/s]")
            ax[2].axhline(0, color = "black")
            if y3:
                ax[2].set_ylim(y3[0], y3[1])
            df_plot_group_velocity.iloc[i, :] = mean_velocity
            df_sem_group_velocity.iloc[i, :] = sem_velocity
            # 4th plot
            ax[3].plot(time_, mean_lick, color = "r", label = "Mean lick +/- SEM", zorder=10)
            ax[3].fill_between(time_, upper_band_lick, lower_band_lick, alpha = 0.4, color = "r", zorder=10) 
            ax[3].set_ylabel("Mean lick rate (per sec)")
            ax[3].axhline(0, color = "black")
            ax[3].set_xlabel("Time [s]")
            if y4:
                ax[3].set_ylim(y4[0], y4[1])
            plt.tight_layout()
            df_plot_group_lick.iloc[i, :] = mean_lick
            df_sem_group_lick.iloc[i, :] = sem_lick


            
            if 0 in time_:
                ax[0].axvline(0, color = 'black', label = 'Treshold reached', linestyle = "dashed")
                ax[1].axvline(0, color = 'black', label = 'Treshold reached', linestyle = "dashed")
                ax[2].axvline(0, color = 'black', label = 'Treshold reached', linestyle = "dashed")
                ax[3].axvline(0, color = 'black', label = 'Treshold reached', linestyle = "dashed")
                 
            if extra_decoration:
                ans = [ax[0].plot(time_,np.array(data_signal_1[i]), color = "g", alpha = 0.4) if index > 0 else  ax[0].plot(time_,np.array(data_signal_1[i]), color = "g", alpha = 0.4, label = "Each trial") for index, i in enumerate(data_signal_1) ]
                ans = [ax[1].plot(time_,np.array(data_signal_2[i]), color = "g", alpha = 0.4) if index > 0 else  ax[1].plot(time_,np.array(data_signal_2[i]), color = "g", alpha = 0.4, label = "Each trial") for index, i in enumerate(data_signal_2) ]
                ans = [ax[2].plot(time_,np.array(i), color = "g", alpha = 0.4) if index > 0 else  ax[2].plot(time_,np.array(i), color = "g", alpha = 0.4, label = "Each trial") for index, i in  velocity_conteriner.iterrows()]
                ans = [ax[3].plot(time_,np.array(i), color = "g", alpha = 0.4) if index > 0 else  ax[3].plot(time_,np.array(i), color = "g", alpha = 0.4, label = "Each trial") for index, i in  lick_conteriner.iterrows()]

            ax[0].legend(loc=2)
            ax[1].legend(loc=2)
            ax[2].legend(loc= 2)
            ax[3].legend(loc= 2)

            
            #fig.colorbar(im2)
            plt.legend()
            if self.msg == "yes":
                    plt.savefig(self.save_file_v1 + "//" + self.list_of_files_behaviour[i] +"_behaviour_fibre.svg" )
                   
            plt.show()
            
            ############### FIGURE HEAT MAP ###############
            fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(10,6), sharex= True)
            fig.suptitle(self.list_of_files[i] + "_heat_map")
            fig.tight_layout()
            
            # 1st plot
            df_T_1 = df[0].iloc[:,1:].T
            df_T_1.columns = df[0]["timestamp"] /1000
            im_1 = ax[0].imshow(df_T_1, cmap = "hot",aspect='auto')
            ax[0].set_ylabel("Trial [n]")
            ax[0].set_title("Signal 1")
            #ax[0].set_xticks([])
            fig.colorbar(im_1)
            
            # 2st plot
            df_T_2 = df[1].iloc[:,1:].T
            df_T_2.columns = df[1]["timestamp"] /1000
            im_2 = ax[1].imshow(df_T_2, cmap = "hot", aspect='auto')
            ax[1].set_ylabel("Trial [n]")
            ax[1].set_title("Signal 2")
            #ax[1].set_xticks([])
            fig.colorbar(im_2)

            # 3st plot
            velocity_conteriner.columns = df[1]["timestamp"] /1000 
            im_3 = ax[2].imshow(velocity_conteriner, cmap = "hot", aspect='auto')
            ax[2].set_ylabel("Trial [n]")
            ax[2].set_title("Velocity")
            #ax[2].set_xticks([])
            fig.colorbar(im_3)
            
            # 4th plot
            lick_conteriner.columns = df[1]["timestamp"] /1000 
            im_3 = ax[3].imshow(lick_conteriner, cmap = "hot", aspect='auto')
            ax[3].set_ylabel("Trial [n]")
            #ax[3].set_xticks(list(df[1]["timestamp"]))
            ax[3].set_title("lick rate (per sec)")
            fig.colorbar(im_3)
            range_ = [list(time_).index(element) for element in heat_ticks_n]
            ax[3].set_xticks([i for i in range_], labels=[i for i in heat_ticks_n])
            plt.tight_layout()
            if 0 in time_:
                ax[0].axvline(np.where(time_ == 0), color = 'black', label = 'Treshold reached', linestyle = "dashed")
                ax[1].axvline(np.where(time_ == 0), color = 'black', label = 'Treshold reached', linestyle = "dashed")
                ax[2].axvline(np.where(time_ == 0), color = 'black', label = 'Treshold reached', linestyle = "dashed")
                ax[3].axvline(np.where(time_ == 0), color = 'black', label = 'Treshold reached', linestyle = "dashed")
            ax[0].legend(loc=2)
            ax[1].legend(loc=2)
            ax[2].legend(loc= 2)
            ax[3].legend(loc= 2)
            if self.msg == "yes":
                plt.savefig(self.save_file_v1 + "//" + self.list_of_files_behaviour[i] +"_heatmap_behaviour_fibre.svg" )
            plt.show()


            if smooth:
                # First Signal #
                yhat_signal_1 = savgol_filter(data_mean_signal_1, window_length_fibre, polyorder_fibre)
                yhat_error_sem_signal_1 = savgol_filter(data_sem_signal_1, window_length_fibre, polyorder_fibre)
                upper_band_signal_1 = np.asfarray(yhat_signal_1 + yhat_error_sem_signal_1)
                lower_band_signal_1 = np.asfarray(yhat_signal_1 - yhat_error_sem_signal_1)
                
                # Second Signal #
                yhat_signal_2 = savgol_filter(data_mean_signal_2, window_length_fibre, polyorder_fibre)
                yhat_error_sem_signal_2 = savgol_filter(data_sem_signal_2, window_length_fibre, polyorder_fibre)
                upper_band_signal_2 = np.asfarray(yhat_signal_2 + yhat_error_sem_signal_2)
                lower_band_signal_2 = np.asfarray(yhat_signal_2 - yhat_error_sem_signal_2)

                # Velocity #
                yhat_velocity = savgol_filter(mean_velocity, window_length_behaviour, polyorder_behaviour)
                yhat_error_sem_velocity = savgol_filter(sem_velocity, window_length_behaviour, polyorder_behaviour)
                upper_band_velocity = np.asfarray(yhat_velocity + yhat_error_sem_velocity)
                lower_band_velocity = np.asfarray(yhat_velocity - yhat_error_sem_velocity)
                
                # Licks #
                yhat_lick = savgol_filter(mean_lick, window_length_behaviour, polyorder_behaviour)
                yhat_error_sem_lick = savgol_filter(sem_lick, window_length_behaviour, polyorder_behaviour)
                upper_band_lick = np.asfarray(yhat_lick + yhat_error_sem_lick)
                lower_band_lick = np.asfarray(yhat_lick - yhat_error_sem_lick)

             ################## FIGURE #######################
                fig, ax = plt.subplots(ncols=1, nrows=4, sharex= True, figsize=(10,6))
                fig.suptitle(self.list_of_files[i] + " smoothed")
                fig.tight_layout()
                    
                # 1st plot
                ax[0].plot(time_, yhat_signal_1, color = "r", label = "Mean signal_1 +/- SEM", zorder=10)
                ax[0].fill_between(time_, upper_band_signal_1, lower_band_signal_1, alpha = 0.4, color = "r", zorder=10) 
                #ax[0].set_xlabel("Time [s]")
                ax[0].set_ylabel(self.data_type)
                ax[0].axhline(0, color = "black")
                if y1:
                    ax[0].set_ylim(y1[0], y1[1])
                # 2nd plot
                ax[1].plot(time_, yhat_signal_2, color = "r", label = "Mean signal_2 +/- SEM", zorder=10)
                ax[1].fill_between(time_, upper_band_signal_2, lower_band_signal_2, alpha = 0.4, color = "r", zorder=10) 
                #ax[1].set_xlabel("Time [s]")
                ax[1].set_ylabel(self.data_type)
                ax[1].axhline(0, color = "black")
                if y2:
                    ax[1].set_ylim(y2[0], y2[1])
                # 3rd plot
                ax[2].plot(time_, yhat_velocity, color = "r", label = "Mean velocity +/- SEM", zorder=10)
                ax[2].fill_between(time_, upper_band_velocity, lower_band_velocity, alpha = 0.4, color = "r", zorder=10) 
                ax[2].set_ylabel("Velocity [cm/s]")
                ax[2].axhline(0, color = "black")
                if y3:
                    ax[2].set_ylim(y3[0], y3[1])
                
                # 4th plot
                ax[3].plot(time_, yhat_lick, color = "r", label = "Mean lick +/- SEM", zorder=10)
                ax[3].fill_between(time_, upper_band_lick, lower_band_lick, alpha = 0.4, color = "r", zorder=10) 
                ax[3].set_xlabel("Time [s]")
                ax[3].set_ylabel("Mean lick rate (per sec)")
                ax[3].axhline(0, color = "black")
                if y4:
                    ax[3].set_ylim(y4[0], y4[1])
                
                
                if 0 in time_:
                    ax[0].axvline(0, color = 'black', label = 'Treshold reached', linestyle = "dashed")
                    ax[1].axvline(0, color = 'black', label = 'Treshold reached', linestyle = "dashed")
                    ax[2].axvline(0, color = 'black', label = 'Treshold reached', linestyle = "dashed")
                    ax[3].axvline(0, color = 'black', label = 'Treshold reached', linestyle = "dashed")
                if extra_decoration:
                     ans = [ax[0].plot(time_,np.array(savgol_filter(data_signal_1[i], window_length_fibre, polyorder_fibre)), color = "g", alpha = 0.4) if index > 0 else  ax[0].plot(time_,np.array(savgol_filter(data_signal_1[i], window_length_fibre, polyorder_fibre)), color = "g", alpha = 0.4, label = "Each trial") for index, i in enumerate(data_signal_1) ]
                     ans = [ax[1].plot(time_,np.array(savgol_filter(data_signal_2[i], window_length_fibre, polyorder_fibre)), color = "g", alpha = 0.4) if index > 0 else  ax[1].plot(time_,np.array(savgol_filter(data_signal_2[i], window_length_fibre, polyorder_fibre)), color = "g", alpha = 0.4, label = "Each trial") for index, i in enumerate(data_signal_2) ]
                     ans = [ax[2].plot(time_,np.array(savgol_filter(i, window_length_behaviour, polyorder_behaviour)), color = "g", alpha = 0.4) if index > 0 else  ax[2].plot(time_,np.array(savgol_filter(i, window_length_behaviour, polyorder_behaviour)), color = "g", alpha = 0.4, label = "Each trial") for index, i in  velocity_conteriner.iterrows()]
                     ans = [ax[3].plot(time_,np.array(savgol_filter(i, window_length_behaviour, polyorder_behaviour)), color = "g", alpha = 0.4) if index > 0 else  ax[3].plot(time_,np.array(savgol_filter(i, window_length_behaviour, polyorder_behaviour)), color = "g", alpha = 0.4, label = "Each trial") for index, i in  lick_conteriner.iterrows()]
                ax[0].legend(loc=2)
                ax[1].legend(loc=2)
                ax[2].legend(loc= 2)
                ax[3].legend(loc= 2)
                #im2 = ax[0].imshow(df_T, cmap = "hot")
                #ax[0].set_ylabel("Trial [n]")
                #ax[0].set_xticks([])
                #fig.colorbar(im2)
                plt.legend()
                if self.msg == "yes":
                    
                    plt.savefig(self.save_file_v1 + "//" + self.list_of_files_behaviour[i] +"_smoothed_behaviour_fibre.svg" )
                plt.show()

                    ############### FIGURE HEAT MAP ###############
                fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(10,6), sharex= True)
                fig.suptitle(self.list_of_files[i] + "_heat_map_smoothed")
                fig.tight_layout()
                
                # 1st plot
                yhat_signal_1 = savgol_filter(df_T_1, window_length_fibre, polyorder_fibre, axis=1)
                im_1 = ax[0].imshow(yhat_signal_1, cmap = "hot",aspect='auto')
                ax[0].set_ylabel("Trial [n]")
                ax[0].set_title("Signal 1")
                #ax[0].set_xticks([])
                fig.colorbar(im_1)
                
                # 2st plot
                yhat_signal_2 = savgol_filter(df_T_2, window_length_fibre, polyorder_fibre,axis=1)
                im_2 = ax[1].imshow(df_T_2, cmap = "hot", aspect='auto')
                ax[1].set_ylabel("Trial [n]")
                ax[1].set_title("Signal 2")
                #ax[1].set_xticks([])
                fig.colorbar(im_2)

                # 3st plot
                yhat_velocity = savgol_filter(velocity_conteriner, window_length_behaviour, polyorder_behaviour, axis=1) 
                im_3 = ax[2].imshow(yhat_velocity, cmap = "hot", aspect='auto')
                ax[2].set_ylabel("Trial [n]")
                ax[2].set_title("Velocity")
                #ax[2].set_xticks([])
                fig.colorbar(im_3)
                
                # 4th plot
                yhat_lick = savgol_filter(lick_conteriner, window_length_behaviour, polyorder_behaviour, axis=1) 
                im_3 = ax[3].imshow(yhat_lick, cmap = "hot", aspect='auto')
                ax[3].set_ylabel("Trial [n]")
                #ax[3].set_xticks(list(df[1]["timestamp"]))
                ax[3].set_title("lick rate (per sec)")
                fig.colorbar(im_3)
                range_ = [list(time_).index(element) for element in heat_ticks_n]
                ax[3].set_xticks([i for i in range_], labels=[i for i in heat_ticks_n])
                plt.tight_layout()
                if 0 in time_:
                    ax[0].axvline(np.where(time_ == 0), color = 'black', label = 'Treshold reached', linestyle = "dashed")
                    ax[1].axvline(np.where(time_ == 0), color = 'black', label = 'Treshold reached', linestyle = "dashed")
                    ax[2].axvline(np.where(time_ == 0), color = 'black', label = 'Treshold reached', linestyle = "dashed")
                    ax[3].axvline(np.where(time_ == 0), color = 'black', label = 'Treshold reached', linestyle = "dashed")
                ax[0].legend(loc=2)
                ax[1].legend(loc=2)
                ax[2].legend(loc= 2)
                ax[3].legend(loc= 2)
                if self.msg == "yes":
                    plt.savefig(self.save_file_v1 + "//" + self.list_of_files_behaviour[i] +"_heatmap_smoothed_behaviour_fibre.svg" )
                plt.show()




        ########### ALL GROUP ############
        # First Signal #
        df_plot_group_signal_1.iloc[-1, :] = df_plot_group_signal_1.iloc[:-1, :].mean(axis=0)
        df_plot_group_signal_1.index = index_signal_1
        df_plot_group_signal_1[columns_] = df_plot_group_signal_1[columns_].apply(pd.to_numeric, errors='coerce')
        min_ = df_plot_group_signal_1.min(axis = 1)
        max_ = df_plot_group_signal_1.max(axis = 1)
        min_idx = df_plot_group_signal_1.idxmin(axis = 1)
        max_idx = df_plot_group_signal_1.idxmax(axis = 1)
        df_plot_group_signal_1["Min value"] = min_
        df_plot_group_signal_1["Max value"] = max_
        df_plot_group_signal_1["Min value time [ms]"] = min_idx
        df_plot_group_signal_1["Max value time [ms]"] = max_idx
        
        data_mean_signal_1 = np.array(df_plot_group_signal_1.iloc[:-1, :-4].mean(axis=0))
        data_sem_signal_1  = np.array(df_sem_group_signal_1.mean(axis=0))
        upper_band_signal_1 = np.asfarray(data_mean_signal_1 + data_sem_signal_1)
        lower_band_signal_1 = np.asfarray(data_mean_signal_1 - data_sem_signal_1)
        # Second Signal #
        
        df_plot_group_signal_2.iloc[-1, :] = df_plot_group_signal_2.iloc[:-1, :].mean(axis=0)
        df_plot_group_signal_2.index = index_signal_2
        df_plot_group_signal_2[columns_] = df_plot_group_signal_2[columns_].apply(pd.to_numeric, errors='coerce')
        min_ = df_plot_group_signal_2.min(axis = 1)
        max_ = df_plot_group_signal_2.max(axis = 1)
        min_idx = df_plot_group_signal_2.idxmin(axis = 1)
        max_idx = df_plot_group_signal_2.idxmax(axis = 1)
        df_plot_group_signal_2["Min value"] = min_
        df_plot_group_signal_2["Max value"] = max_
        df_plot_group_signal_2["Min value time [ms]"] = min_idx
        df_plot_group_signal_2["Max value time [ms]"] = max_idx
        
        data_mean_signal_2 = np.array(df_plot_group_signal_2.iloc[:-1, :-4].mean(axis=0))
        data_sem_signal_2  = np.array(df_sem_group_signal_2.mean(axis=0))
        upper_band_signal_2 = np.asfarray(data_mean_signal_2 + data_sem_signal_2)
        lower_band_signal_2 = np.asfarray(data_mean_signal_2 - data_sem_signal_2)
        # Velocity #

        df_plot_group_velocity.iloc[-1, :] = df_plot_group_velocity.iloc[:-1, :].mean(axis=0)
        df_plot_group_velocity.index = index_beh
        df_plot_group_velocity[columns_] = df_plot_group_velocity[columns_].apply(pd.to_numeric, errors='coerce')
        df_sem_group_velocity.iloc[-1, :] = df_sem_group_velocity.mean(axis=0)
        min_ = df_plot_group_velocity.min(axis = 1)
        max_ = df_plot_group_velocity.max(axis = 1)
        min_idx = df_plot_group_velocity.idxmin(axis = 1)
        max_idx = df_plot_group_velocity.idxmax(axis = 1)
        df_plot_group_velocity["Min value"] = min_
        df_plot_group_velocity["Max value"] = max_
        df_plot_group_velocity["Min value time [ms]"] = min_idx
        df_plot_group_velocity["Max value time [ms]"] = max_idx
        
        
        data_mean_v  = np.array(df_plot_group_velocity.iloc[-1, :-4])
        data_sem_v  = np.array(df_sem_group_velocity.iloc[-1, :])
        upper_band_v = np.asfarray(data_mean_v + data_sem_v)
        lower_band_v = np.asfarray(data_mean_v - data_sem_v)

        # Licks #

        df_plot_group_lick.iloc[-1, :] = df_plot_group_lick.iloc[:-1, :].mean(axis=0)
        df_plot_group_lick.index = index_beh
        df_plot_group_lick[columns_] = df_plot_group_lick[columns_].apply(pd.to_numeric, errors='coerce')
        df_sem_group_lick.iloc[-1, :] = df_sem_group_lick.mean(axis=0)
        min_ = df_plot_group_lick.min(axis = 1)
        max_ = df_plot_group_lick.max(axis = 1)
        min_idx = df_plot_group_lick.idxmin(axis = 1)
        max_idx = df_plot_group_lick.idxmax(axis = 1)
        df_plot_group_lick["Min value"] = min_
        df_plot_group_lick["Max value"] = max_
        df_plot_group_lick["Min value time [ms]"] = min_idx
        df_plot_group_lick["Max value time [ms]"] = max_idx
        
        
        data_mean_l  = np.array(df_plot_group_lick.iloc[-1, :-4])
        data_sem_l  = np.array(df_sem_group_velocity.iloc[-1, :])
        upper_band_l = np.asfarray(data_mean_l + data_sem_l)
        lower_band_l = np.asfarray(data_mean_l - data_sem_l)

        
        ########### FIGURE ##########

        fig, ax = plt.subplots(ncols=1, nrows=4, sharex= True, figsize=(10,6))
        fig.suptitle("Mean")
        fig.tight_layout()

        # 1st plot
        ax[0].plot(time_, data_mean_signal_1, color = "r", label = "Mean signal_1 +/- SEM", zorder=10)
        ax[0].fill_between(time_, upper_band_signal_1, lower_band_signal_1, alpha = 0.4, color = "r", zorder=10) 
        #ax[0].set_xlabel("Time [s]")
        ax[0].set_ylabel(self.data_type)
        ax[0].axhline(0, color = "black")
        if y1:
            ax[0].set_ylim(y1[0], y1[1])
        
        # 2nd plot
        ax[1].plot(time_, data_mean_signal_2, color = "r", label = "Mean signal_2 +/- SEM", zorder=10)
        ax[1].fill_between(time_, upper_band_signal_2, lower_band_signal_2, alpha = 0.4, color = "r", zorder=10) 
        #ax[1].set_xlabel("Time [s]")
        ax[1].set_ylabel(self.data_type)
        ax[1].axhline(0, color = "black")
        if y2:
            ax[1].set_ylim(y2[0], y2[1])
        
        # 3rd plot
        ax[2].plot(time_, data_mean_v, color = "r", label = "Mean velocity +/- SEM", zorder=10)
        ax[2].fill_between(time_, upper_band_v, lower_band_v, alpha = 0.4, color = "r", zorder=10) 
        ax[2].set_ylabel("Velocity [cm/s]")
        ax[2].axhline(0, color = "black")
        if y3:
            ax[2].set_ylim(y3[0], y3[1])

        # 4th plot
        ax[3].plot(time_, data_mean_l, color = "r", label = "Mean lick +/- SEM", zorder=10)
        ax[3].fill_between(time_, upper_band_l, lower_band_l, alpha = 0.4, color = "r", zorder=10) 
        ax[3].set_ylabel("Mean lick rate (per sec)")
        ax[3].set_xlabel("Time [s]")
        ax[3].axhline(0, color = "black")
        if y4:
            ax[3].set_ylim(y4[0], y4[1])

        if 0 in time_:
            ax[0].axvline(0, color = 'black', label = 'Treshold reached', linestyle = "dashed")
            ax[1].axvline(0, color = 'black', label = 'Treshold reached', linestyle = "dashed")
            ax[2].axvline(0, color = 'black', label = 'Treshold reached', linestyle = "dashed")
            ax[3].axvline(0, color = 'black', label = 'Treshold reached', linestyle = "dashed")
        
        if extra_decoration:
            ans = [ax[0].plot(time_,list(row[1]), color = "g", alpha = 0.4) if index > 0 else ax[0].plot(time_,list(row[1]), color = "g", alpha = 0.4, label = "Each mice") for  index, row in enumerate(df_plot_group_signal_1.iloc[:-1, :-4].iterrows())]
            ans = [ax[1].plot(time_,np.array(row[1]), color = "g", alpha = 0.4) if index > 0 else ax[1].plot(time_,np.array(row[1]), color = "g", alpha = 0.4, label = "Each mice") for  index, row in  enumerate(df_plot_group_signal_2.iloc[:-1, :-4].iterrows())]
            ans = [ax[2].plot(time_,list(row[1]), color = "g", alpha = 0.4) if index > 0 else ax[2].plot(time_,list(row[1]), color = "g", alpha = 0.4, label = "Each mice") for  index, row in enumerate(df_plot_group_velocity.iloc[:-1, :-4].iterrows())]
            ans = [ax[3].plot(time_,np.array(row[1]), color = "g", alpha = 0.4) if index > 0 else ax[3].plot(time_,np.array(row[1]), color = "g", alpha = 0.4, label = "Each mice") for  index, row in  enumerate(df_plot_group_lick.iloc[:-1, :-4].iterrows())]
            
        
        ax[0].legend(loc=2)
        ax[1].legend(loc=2)
        ax[2].legend(loc= 2)
        ax[3].legend(loc= 2)
        #im2 = ax[0].imshow(df_T, cmap = "hot")
        #ax[0].set_ylabel("Mice ")
        #ax[0].set_xticks([])
        #ax[1].legend(loc=2)
        #fig.colorbar(im2)
        plt.legend()
        if self.msg == "yes":
            plt.savefig(self.save_file_v1 + "//" + "_all_group_behaviour_fibre.svg" )
            df_plot_group_signal_1.to_excel(self.save_file_v1 + "//" + "signal_1.xlsx")
            df_plot_group_signal_2.to_excel(self.save_file_v1 + "//" + "signal_2.xlsx")
            df_plot_group_velocity.to_excel(self.save_file_v1 + "//" + "velocity.xlsx")
            df_plot_group_lick.to_excel(self.save_file_v1 + "//" + "lick.xlsx")
        plt.show()
        ############### FIGURE HEAT MAP ###############
        fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(10,6), sharex= True)
        fig.suptitle("Mean heatmap")
        fig.tight_layout()
        # 1st plot
        
        im_1 = ax[0].imshow(df_plot_group_signal_1.iloc[:-1, :-4], cmap = "hot",aspect='auto')
        ax[0].set_ylabel("Mice [n]")
        ax[0].set_title("Signal 1")
        #ax[0].set_xticks([])
        fig.colorbar(im_1)
        
        # 2st plot

        im_2 = ax[1].imshow(df_plot_group_signal_2.iloc[:-1, :-4], cmap = "hot", aspect='auto')
        ax[1].set_ylabel("Mice [n]")
        ax[1].set_title("Signal 2")
        fig.colorbar(im_2)

        # 3st plot
       
        im_3 = ax[2].imshow(df_plot_group_velocity.iloc[:-1, :-4], cmap = "hot", aspect='auto')
        ax[2].set_ylabel("Mice [n]")
        ax[2].set_title("Velocity")
        
        fig.colorbar(im_3)
        
        # 4th plot
        
        im_3 = ax[3].imshow(df_plot_group_lick.iloc[:-1, :-4], cmap = "hot", aspect='auto')
        ax[3].set_ylabel("Mice [n]")
        ax[3].set_title("lick rate (per sec)")
        fig.colorbar(im_3)
        range_ = [list(time_).index(element) for element in heat_ticks_n]
        ax[3].set_xticks([i for i in range_], labels=[i for i in heat_ticks_n])        
        plt.tight_layout()
        if 0 in time_:
            ax[0].axvline(np.where(time_ == 0), color = 'black', label = 'Treshold reached', linestyle = "dashed")
            ax[1].axvline(np.where(time_ == 0), color = 'black', label = 'Treshold reached', linestyle = "dashed")
            ax[2].axvline(np.where(time_ == 0), color = 'black', label = 'Treshold reached', linestyle = "dashed")
            ax[3].axvline(np.where(time_ == 0), color = 'black', label = 'Treshold reached', linestyle = "dashed")
        ax[0].legend(loc=2)
        ax[1].legend(loc=2)
        ax[2].legend(loc= 2)
        ax[3].legend(loc= 2)
        if self.msg == "yes":
            plt.savefig(self.save_file_v1 + "//" + "_all_group_heatmap_behaviour_fibre.svg" )
        plt.show()


        if smooth:
                # First Signal #
                yhat_signal_1 = savgol_filter(data_mean_signal_1, window_length_fibre, polyorder_fibre)
                yhat_error_sem_signal_1 = savgol_filter(data_sem_signal_1, window_length_fibre, polyorder_fibre)
                upper_band_signal_1 = np.asfarray(yhat_signal_1 + yhat_error_sem_signal_1)
                lower_band_signal_1 = np.asfarray(yhat_signal_1 - yhat_error_sem_signal_1)
                # Second Signal #
                yhat_signal_2 = savgol_filter(data_mean_signal_2, window_length_fibre, polyorder_fibre)
                yhat_error_sem_signal_2 = savgol_filter(data_sem_signal_2, window_length_fibre, polyorder_fibre)
                upper_band_signal_2 = np.asfarray(yhat_signal_2 + yhat_error_sem_signal_2)
                lower_band_signal_2 = np.asfarray(yhat_signal_2 - yhat_error_sem_signal_2)
                # Velocity #
                yhat_velocity = savgol_filter(data_mean_v, window_length_behaviour, polyorder_behaviour)
                yhat_error_sem_velocity = savgol_filter(data_sem_v, window_length_behaviour, polyorder_behaviour)
                upper_band_velocity = np.asfarray(yhat_velocity + yhat_error_sem_velocity)
                lower_band_velocity = np.asfarray(yhat_velocity - yhat_error_sem_velocity)
                # Licks #
                yhat_lick = savgol_filter(data_mean_l, window_length_behaviour, polyorder_behaviour)
                yhat_error_sem_lick = savgol_filter(data_sem_l, window_length_behaviour, polyorder_behaviour)
                upper_band_lick = np.asfarray(yhat_lick + yhat_error_sem_lick)
                lower_band_lick = np.asfarray(yhat_lick - yhat_error_sem_lick)

             ################## FIGURE #######################
                fig, ax = plt.subplots(ncols=1, nrows=4, sharex= True, figsize=(10,6))
                fig.suptitle("Mean smoothed")
                fig.tight_layout()
                   
                # 1st plot
                ax[0].plot(time_, yhat_signal_1, color = "r", label = "Mean signal_1 +/- SEM", zorder=10)
                ax[0].fill_between(time_, upper_band_signal_1, lower_band_signal_1, alpha = 0.4, color = "r", zorder=10) 
                #ax[0].set_xlabel("Time [s]")
                ax[0].set_ylabel(self.data_type)
                ax[0].axhline(0, color = "black")
                if y1:
                    ax[0].set_ylim(y1[0], y1[1])
                
                # 2nd plot
                ax[1].plot(time_, yhat_signal_2, color = "r", label = "Mean signal_2 +/- SEM", zorder=10)
                ax[1].fill_between(time_, upper_band_signal_2, lower_band_signal_2, alpha = 0.4, color = "r", zorder=10) 
                #ax[1].set_xlabel("Time [s]")
                ax[1].set_ylabel(self.data_type)
                ax[1].axhline(0, color = "black")
                if y2:
                    ax[1].set_ylim(y2[0], y2[1])
               
               # 3rd plot
                ax[2].plot(time_, yhat_velocity, color = "r", label = "Mean velocity +/- SEM", zorder=10)
                ax[2].fill_between(time_, upper_band_velocity, lower_band_velocity, alpha = 0.4, color = "r", zorder=10) 
                ax[3].axhline(0, color = "black")
                ax[2].set_ylabel("Velocity [cm/s]")
                if y3:
                    ax[2].set_ylim(y3[0], y3[1])
                
                # 4th plot
                ax[3].plot(time_, yhat_lick, color = "r", label = "Mean lick +/- SEM", zorder=10)
                ax[3].fill_between(time_, upper_band_lick, lower_band_lick, alpha = 0.4, color = "r", zorder=10) 
                ax[3].axhline(0, color = "black")
                ax[3].set_ylabel("Mean lick rate (per sec)")
                ax[3].set_xlabel("Time [s]")
                if y4:
                    ax[3].set_ylim(y4[0], y4[1])

                if 0 in time_:
                    ax[0].axvline(0, color = 'black', label = 'Treshold reached', linestyle = "dashed")
                    ax[1].axvline(0, color = 'black', label = 'Treshold reached', linestyle = "dashed")
                    ax[2].axvline(0, color = 'black', label = 'Treshold reached', linestyle = "dashed")
                    ax[3].axvline(0, color = 'black', label = 'Treshold reached', linestyle = "dashed")
                if extra_decoration:
                    ans = [ax[0].plot(time_,np.array(savgol_filter(row[1], window_length_fibre, polyorder_fibre)), color = "g", alpha = 0.4) if index > 0 else  ax[0].plot(time_,np.array(savgol_filter(row[1], window_length_fibre, polyorder_fibre)), color = "g", alpha = 0.4, label = "Each mice") for index, row in enumerate(df_plot_group_signal_1.iloc[:-1, :-4].iterrows()) ]
                    ans = [ax[1].plot(time_,np.array(savgol_filter(row[1], window_length_fibre, polyorder_fibre)), color = "g", alpha = 0.4) if index > 0 else  ax[1].plot(time_,np.array(savgol_filter(row[1], window_length_fibre, polyorder_fibre)), color = "g", alpha = 0.4, label = "Each mice") for index, row in enumerate(df_plot_group_signal_2.iloc[:-1, :-4].iterrows()) ]
                    ans = [ax[2].plot(time_,np.array(savgol_filter(row[1], window_length_behaviour, polyorder_behaviour)), color = "g", alpha = 0.4) if index > 0 else  ax[2].plot(time_,np.array(savgol_filter(row[1], window_length_behaviour, polyorder_behaviour)), color = "g", alpha = 0.4, label = "Each mice") for index,  row in enumerate(df_plot_group_velocity.iloc[:-1, :-4].iterrows())]
                    ans = [ax[3].plot(time_,np.array(savgol_filter(row[1], window_length_behaviour, polyorder_behaviour)), color = "g", alpha = 0.4) if index > 0 else  ax[3].plot(time_,np.array(savgol_filter(row[1], window_length_behaviour, polyorder_behaviour)), color = "g", alpha = 0.4, label = "Each mice") for index, row in enumerate(df_plot_group_lick.iloc[:-1, :-4].iterrows())]

                
                ax[0].legend(loc=2)
                ax[1].legend(loc=2)
                ax[2].legend(loc= 2)
                ax[3].legend(loc= 2)
                plt.legend()
                if self.msg == "yes":
                    plt.savefig(self.save_file_v1 + "//" + "_all_group_behaviour_fibre_smoothed.svg" )
                    
                plt.show()

                        ############### FIGURE HEAT MAP ###############
                fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(10,6), sharex= True)
                fig.suptitle("Mean heatmap smoothed")
                fig.tight_layout()
                # 1st plot
                yhat_signal_1 = savgol_filter(df_plot_group_signal_1.iloc[:-1, :-4], window_length_fibre, polyorder_fibre, axis=1)
                im_1 = ax[0].imshow(yhat_signal_1, cmap = "hot",aspect='auto')
                ax[0].set_ylabel("Mice [n]")
                ax[0].set_title("Signal 1")
                #ax[0].set_xticks([])
                fig.colorbar(im_1)
                
                # 2st plot
                yhat_signal_2 = savgol_filter(df_plot_group_signal_2.iloc[:-1, :-4], window_length_fibre, polyorder_fibre, axis=1)
                im_2 = ax[1].imshow(yhat_signal_2, cmap = "hot", aspect='auto')
                ax[1].set_ylabel("Mice [n]")
                ax[1].set_title("Signal 2")
                fig.colorbar(im_2)

                # 3st plot
                yhat_velocity = savgol_filter(df_plot_group_velocity.iloc[:-1, :-4], window_length_behaviour, polyorder_behaviour, axis=1)
                im_3 = ax[2].imshow(yhat_velocity, cmap = "hot", aspect='auto')
                ax[2].set_ylabel("Mice [n]")
                ax[2].set_title("Velocity")
                fig.colorbar(im_3)
                
                # 4th plot
                yhat_lick = savgol_filter(df_plot_group_lick.iloc[:-1, :-4], window_length_behaviour, polyorder_behaviour, axis=1)
                im_3 = ax[3].imshow(yhat_lick, cmap = "hot", aspect='auto')
                ax[3].set_ylabel("Mice [n]")
                ax[3].set_title("lick rate (per sec)")
                fig.colorbar(im_3)
                range_ = [list(time_).index(element) for element in heat_ticks_n]
                ax[3].set_xticks([i for i in range_], labels=[i for i in heat_ticks_n])                
                plt.tight_layout()
                if 0 in time_:
                    ax[0].axvline(np.where(time_ == 0), color = 'black', label = 'Treshold reached', linestyle = "dashed")
                    ax[1].axvline(np.where(time_ == 0), color = 'black', label = 'Treshold reached', linestyle = "dashed")
                    ax[2].axvline(np.where(time_ == 0), color = 'black', label = 'Treshold reached', linestyle = "dashed")
                    ax[3].axvline(np.where(time_ == 0), color = 'black', label = 'Treshold reached', linestyle = "dashed")
                ax[0].legend(loc=2)
                ax[1].legend(loc=2)
                ax[2].legend(loc= 2)
                ax[3].legend(loc= 2)
                if self.msg == "yes":
                    plt.savefig(self.save_file_v1 + "//" + "_all_group_heatmap_behaviour_fibre_smoothed.svg" )
                plt.show()

    def auc(self, time_range_signal_1 = [0, 1000], time_range_signal_2 = [0, 1000], time_range_velocity = [0, 1000], time_range_lick = [0, 1000],  method = 'poly'):
        
        self.__pre_process_behaviour()
        columns_ = self.list_of_df[0]["timestamp"].tolist()
        columns_2 = self.list_of_df_v2[0]["timestamp"].tolist()
        df_plot_group_signal_1 = pd.DataFrame(columns= columns_, index = [i for i in range(0, len(self.list_of_df)+1)])
        df_plot_group_signal_2 = pd.DataFrame(columns= columns_2, index = [i for i in range(0, len(self.list_of_df_v2)+1)])
        df_plot_group_velocity = pd.DataFrame(columns= columns_2, index = [i for i in range(0, len(self.list_of_df_behaviour)+1)])
        df_plot_group_lick = pd.DataFrame(columns= columns_2, index = [i for i in range(0, len(self.list_of_df_behaviour)+1)])
        columns_v1 = ["Mice id", "bin_start", "bin_stop", "AUC", "Max value in bin", "Max value time", "Min value in bin", "Min value time"]
        df_auc_result_signal_1 = pd.DataFrame(columns= columns_v1, index = [i for i in range(0, len(self.list_of_df)+1)])
        df_auc_result_signal_2 = pd.DataFrame(columns= columns_v1, index = [i for i in range(0, len(self.list_of_df_v2)+1)])
        df_auc_result_velocity = pd.DataFrame(columns= columns_v1, index = [i for i in range(0, len(self.list_of_df_behaviour)+1)])
        df_auc_result_lick = pd.DataFrame(columns= columns_v1, index = [i for i in range(0, len(self.list_of_df_behaviour)+1)])
        for i,df in enumerate(zip(self.list_of_df, self.list_of_df_v2)):
            self.time_ = np.array(df[0]["timestamp"])
            mean_velocity, sem_velocity, mean_lick, sem_lick, velocity_conteriner, lick_conteriner = self.__process_behaviour(self.list_of_df_behaviour[i])
            
            velocity_conteriner = velocity_conteriner.T
            velocity_conteriner.insert(0, "timestamp", velocity_conteriner.index)
            velocity_conteriner.reset_index(drop=True, inplace=True)
            
            lick_conteriner = lick_conteriner.T
            lick_conteriner.insert(0, "timestamp", lick_conteriner.index)
            lick_conteriner.reset_index(drop=True, inplace=True)

            result = [ item for item in [self.list_of_files[i], time_range_signal_1[0], time_range_signal_1[1]]]
            result_2 = [ item for item in [self.list_of_files_2[i], time_range_signal_2[0], time_range_signal_2[1]]]
            result_3 = [ item for item in [self.list_of_files_behaviour[i], time_range_velocity[0], time_range_velocity[1]]]
            result_4 = [ item for item in [self.list_of_files_behaviour[i], time_range_lick[0], time_range_lick[1]]]

            df_copy_signal_1 = df[0]
            time_orginal_signal_1 = np.array(df_copy_signal_1["timestamp"])
            df_copy_signal_1 = df_copy_signal_1.iloc[:,1:]
            data_mean_all_signal_1 = np.array(df_copy_signal_1.mean(axis = 1))
            df_signal_1 = df[0]

            df_copy_signal_2 = df[1]
            time_orginal_signal_2 = np.array(df_copy_signal_2["timestamp"])
            df_copy_signal_2 = df_copy_signal_2.iloc[:,1:]
            data_mean_all_signal_2 = np.array(df_copy_signal_2.mean(axis = 1))
            df_signal_2 = df[1]

            velocity_conteriner_copy = velocity_conteriner
            time_orginal_velocity_conteriner = np.array(velocity_conteriner_copy["timestamp"])
            velocity_conteriner_copy = velocity_conteriner_copy.iloc[:,1:]
            data_mean_all_velocity_conteriner = np.array(velocity_conteriner_copy.mean(axis = 1))
            df_velocity = velocity_conteriner

            lick_conteriner_copy = lick_conteriner
            time_orginal_lick_conteriner = np.array(lick_conteriner_copy["timestamp"])
            lick_conteriner_copy = lick_conteriner_copy.iloc[:,1:]
            data_mean_all_lick_conteriner = np.array(lick_conteriner_copy.mean(axis = 1))
            df_lick = lick_conteriner
            

            if method == "poly":
                ############## Signal 1 ##############
                if time_range_signal_1:
                    start_idx_signal_1 = abs(df[0]["timestamp"]- time_range_signal_1[0]).idxmin()
                    stop_idx_signal_1 = abs(df[0]["timestamp"]- time_range_signal_1[1]).idxmin()
                    df_signal_1  = df_signal_1.iloc[start_idx_signal_1:stop_idx_signal_1 +1, :]
                time_signal_1 = np.array(df_signal_1["timestamp"] )
                df_signal_1 = df_signal_1.iloc[:,1:]
                data_mean_signal_1 = np.array(df_signal_1.mean(axis = 1))
                degree = 80
                coefficients_signal_1 = np.polyfit(time_orginal_signal_1, data_mean_all_signal_1, degree)
                polynomial_signal_1 = np.poly1d(coefficients_signal_1)
                # Calculate the area under the curve
                area_signal_1, error = quad(polynomial_signal_1, time_range_signal_1[0], time_range_signal_1[1])
                ############## Signal 2 ##############
                if time_range_signal_2:
                    start_idx_signal_2 = abs(df[1]["timestamp"]- time_range_signal_2[0]).idxmin()
                    stop_idx_signal_2 = abs(df[1]["timestamp"]- time_range_signal_2[1]).idxmin()
                    df_signal_2 = df_signal_2.iloc[start_idx_signal_2:stop_idx_signal_2 +1, :]
                time_signal_2 = np.array(df_signal_2["timestamp"] )
                df_signal_2 = df_signal_2.iloc[:,1:]
                data_mean_signal_2 = np.array(df_signal_2.mean(axis = 1))
                degree = 80
                coefficients_signal_2 = np.polyfit(time_orginal_signal_2, data_mean_all_signal_2, degree)
                polynomial_signal_2 = np.poly1d(coefficients_signal_2)
                # Calculate the area under the curve
                area_signal_2, error = quad(polynomial_signal_2, time_range_signal_2[0], time_range_signal_2[1])
                ############## Velocity ##############
                if time_range_velocity:
                    start_idx_velocity = abs(velocity_conteriner["timestamp"]- time_range_velocity[0]).idxmin()
                    stop_idx_velocity = abs(velocity_conteriner["timestamp"]- time_range_velocity[1]).idxmin()
                    df_velocity  = df_velocity.iloc[start_idx_velocity:stop_idx_velocity +1, :]
                time_velocity = np.array(df_velocity["timestamp"] )
                df_velocity = df_velocity.iloc[:,1:]
                data_mean_velocity = np.array(df_velocity.mean(axis = 1))
                degree = 80
                coefficients_velocity = np.polyfit(time_orginal_velocity_conteriner, data_mean_all_velocity_conteriner, degree)
                polynomial_velocity= np.poly1d(coefficients_velocity)
                # Calculate the area under the curve
                area_velocity, error = quad(polynomial_velocity, time_range_velocity[0], time_range_velocity[1])
                ############## Lick ##############
                if time_range_lick:
                    start_idx_lick = abs(lick_conteriner["timestamp"]- time_range_lick[0]).idxmin()
                    stop_idx_lick = abs(lick_conteriner["timestamp"]- time_range_lick[1]).idxmin()
                    df_lick  = df_lick.iloc[start_idx_lick:stop_idx_lick +1, :] 
                time_lick = np.array(df_lick["timestamp"] )
                df_lick = df_lick.iloc[:,1:]
                data_mean_lick = np.array(df_lick.mean(axis = 1))
                degree = 80
                coefficients_lick = np.polyfit(time_orginal_lick_conteriner, data_mean_all_lick_conteriner, degree)
                polynomial_lick = np.poly1d(coefficients_lick)
                # Calculate the area under the curve
                area_lick, error = quad(polynomial_lick, time_range_lick[0], time_range_lick[1])

            elif method == "trap":
                if time_range_signal_1:
                    start_idx_signal_1 = abs(df[0]["timestamp"]- time_range_signal_1[0]).idxmin()
                    stop_idx_signal_1 = abs(df[0]["timestamp"]- time_range_signal_1[1]).idxmin()
                    df_signal_1  = df_signal_1.iloc[start_idx_signal_1:stop_idx_signal_1 +1, :]
                time_signal_1 = np.array(df_signal_1["timestamp"] )
                df_signal_1 = df_signal_1.iloc[:,1:]
                data_mean_signal_1 = np.array(df_signal_1.mean(axis = 1))
                #data_mean = np.array(df.mean(axis = 1))
                area_signal_1 = np.trapz(data_mean_signal_1, time_signal_1)
                if time_range_signal_2:
                    start_idx_signal_2 = abs(df[1]["timestamp"]- time_range_signal_2[0]).idxmin()
                    stop_idx_signal_2 = abs(df[1]["timestamp"]- time_range_signal_2[1]).idxmin()
                    df_signal_2 = df_signal_2.iloc[start_idx_signal_2:stop_idx_signal_2 +1, :]
                time_signal_2 = np.array(df_signal_2["timestamp"] )
                df_signal_2 = df_signal_2.iloc[:,1:]
                data_mean_signal_2 = np.array(df_signal_2.mean(axis = 1))
                #data_mean = np.array(df.mean(axis = 1))
                area_signal_2 = np.trapz(data_mean_signal_2, time_signal_2)
                if time_range_velocity:
                    start_idx_velocity = abs(velocity_conteriner["timestamp"]- time_range_velocity[0]).idxmin()
                    stop_idx_velocity = abs(velocity_conteriner["timestamp"]- time_range_velocity[1]).idxmin()
                    df_velocity  = df_velocity.iloc[start_idx_velocity:stop_idx_velocity +1, :]
                time_velocity = np.array(df_velocity["timestamp"] )
                df_velocity = df_velocity.iloc[:,1:]
                data_mean_velocity = np.array(df_velocity.mean(axis = 1))
                area_velocity = np.trapz(data_mean_velocity, time_velocity)
                if time_range_lick:
                    start_idx_lick = abs(lick_conteriner["timestamp"]- time_range_lick[0]).idxmin()
                    stop_idx_lick = abs(lick_conteriner["timestamp"]- time_range_lick[1]).idxmin()
                    df_lick  = df_lick.iloc[start_idx_lick:stop_idx_lick +1, :]
                time_lick = np.array(df_lick["timestamp"] )
                df_lick = df_lick.iloc[:,1:]
                data_mean_lick = np.array(df_lick.mean(axis = 1))
                area_lick = np.trapz(data_mean_lick, time_lick)

            df_plot_group_signal_1.iloc[i, :] = data_mean_all_signal_1
            df_plot_group_signal_2.iloc[i, :] = data_mean_all_signal_2
            df_plot_group_velocity.iloc[i, :] =  data_mean_all_velocity_conteriner
            df_plot_group_lick.iloc[i, :] =  data_mean_all_lick_conteriner
            
            
            __ = [ result.append(item) for item in [area_signal_1, data_mean_signal_1.max(), time_signal_1[data_mean_signal_1.argmax()], data_mean_signal_1.min(), time_signal_1[data_mean_signal_1.argmin()]]]
            __ = [ result_2.append(item) for item in [area_signal_2, data_mean_signal_2.max(), time_signal_2[data_mean_signal_2.argmax()], data_mean_signal_2.min(), time_signal_2[data_mean_signal_2.argmin()]]]
            __ = [ result_3.append(item) for item in [area_velocity, data_mean_velocity.max(), time_velocity[data_mean_velocity.argmax()], data_mean_velocity.min(), time_velocity[data_mean_velocity.argmin()]]] 
            __ = [result_4.append(item) for item in [area_lick, data_mean_lick.max(), time_lick[data_mean_lick.argmax()], data_mean_lick.min(), time_lick[data_mean_lick.argmin()]]]
            
            df_auc_result_signal_1.iloc[i, :] = result
            df_auc_result_signal_2.iloc[i, :] = result_2
            df_auc_result_velocity.iloc[i, :] = result_3
            df_auc_result_lick.iloc[i, :] = result_4

            fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(8,6), sharex= True)
            fig.suptitle("Results of AUC")
            fig.tight_layout()

            ax[0].scatter(time_orginal_signal_1, data_mean_all_signal_1, label='Data points_signal_1')
            ax[1].scatter(time_orginal_signal_2, data_mean_all_signal_2, label='Data points_signal_2')
            ax[2].scatter(time_orginal_velocity_conteriner, data_mean_all_velocity_conteriner, label='Data points_velocity')
            ax[3].scatter(time_orginal_lick_conteriner, data_mean_all_lick_conteriner, label='Data points_lick')
            if method == "poly":
                x_fit_signal_1 = np.linspace(time_range_signal_1[0], time_range_signal_1[1], 5000)
                y_fit_signal_1 = polynomial_signal_1(x_fit_signal_1)
                x_fit_signal_2 = np.linspace(time_range_signal_2[0], time_range_signal_2[1], 5000)
                y_fit_signal_2 = polynomial_signal_2(x_fit_signal_2)
                x_fit_velocity = np.linspace(time_range_velocity[0], time_range_velocity[1], 5000)
                y_fit_velocity = polynomial_velocity(x_fit_velocity)
                x_fit_lick = np.linspace(time_range_lick[0], time_range_lick[1], 5000)
                y_fit_lick = polynomial_lick(x_fit_lick)
                ax[0].plot(x_fit_signal_1, y_fit_signal_1, label=f'Polynomial fit (degree {degree})', color='red')
                ax[1].plot(x_fit_signal_2, y_fit_signal_2, label=f'Polynomial fit (degree {degree})', color='red')
                ax[2].plot(x_fit_velocity, y_fit_velocity, label=f'Polynomial fit (degree {degree})', color='red')
                ax[3].plot(x_fit_lick, y_fit_lick, label=f'Polynomial fit (degree {degree})', color='red')
            elif method == "trap":
                x_fit_signal_1 = time_signal_1
                y_fit_signal_1 = data_mean_signal_1
                x_fit_signal_2 = time_signal_2
                y_fit_signal_2 = data_mean_signal_2
                x_fit_velocity = time_velocity
                y_fit_velocity = data_mean_velocity
                x_fit_lick = time_lick
                y_fit_lick = data_mean_lick

            ax[0].fill_between(x_fit_signal_1, y_fit_signal_1, alpha=0.4, color = "red")
            ax[1].fill_between(x_fit_signal_2, y_fit_signal_2, alpha=0.4, color = "red")
            ax[2].fill_between(x_fit_velocity, y_fit_velocity, alpha=0.4, color = "red")
            ax[3].fill_between(x_fit_lick, y_fit_lick, alpha=0.4, color = "red")

            ax[0].set_title(self.list_of_files[i])
            ax[1].set_title(self.list_of_files_2[i])
            ax[2].set_title(self.list_of_files_behaviour[i] +"_velocity")
            ax[3].set_title(self.list_of_files_behaviour[i] +"_lick")
            
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('y')
            ax[0].axhline(0, color = "black")
            ax[0].legend(loc=2)
            
            ax[1].set_xlabel('x')
            ax[1].set_ylabel('y')
            ax[1].axhline(0, color = "black")
            ax[1].legend(loc=2)

            ax[2].set_xlabel('x')
            ax[2].set_ylabel('y')
            ax[2].axhline(0, color = "black")
            ax[2].legend(loc=2)

            ax[3].set_xlabel('x')
            ax[3].set_ylabel('y')
            ax[3].axhline(0, color = "black")
            ax[3].legend(loc=2)
            plt.show() 
            ############### ALL GROUP ##############
        df_plot_group_signal_1.iloc[-1, :] = df_plot_group_signal_1.iloc[:-1, :].mean(axis=0)
        result = [item for item in ["Mean", time_range_signal_1[0], time_range_signal_1[1]]]
        df_plot_group_signal_2.iloc[-1, :] = df_plot_group_signal_2.iloc[:-1, :].mean(axis=0)
        result_2 = [item for item in ["Mean", time_range_signal_2[0], time_range_signal_2[1]]]
        df_plot_group_velocity.iloc[-1, :] = df_plot_group_velocity.iloc[:-1, :].mean(axis=0)
        result_3 = [item for item in ["Mean", time_range_velocity[0], time_range_velocity[1]]]
        df_plot_group_lick.iloc[-1, :] = df_plot_group_lick.iloc[:-1, :].mean(axis=0)
        result_4 = [item for item in ["Mean", time_range_lick[0], time_range_lick[1]]]

        data_mean_signal_1 = pd.to_numeric(df_plot_group_signal_1.iloc[-1, :])
        data_mean_signal_2 = pd.to_numeric(df_plot_group_signal_2.iloc[-1, :])
        data_mean_velocity = pd.to_numeric(df_plot_group_velocity.iloc[-1, :])
        data_mean_lick = pd.to_numeric(df_plot_group_lick.iloc[-1, :])
        if method == "poly":
            degree = 80
            coefficients_signal_1 = np.polyfit(time_orginal_signal_1, data_mean_signal_1, degree)
            polynomial_signal_1 = np.poly1d(coefficients_signal_1)
            # Calculate the area under the curve
            area_signal_1, error = quad(polynomial_signal_1, time_range_signal_1[0], time_range_signal_1[1])       
            df_signal_1 = np.array(data_mean_signal_1.iloc[ start_idx_signal_1:stop_idx_signal_1 +1])
            
            coefficients_signal_2 = np.polyfit(time_orginal_signal_2, data_mean_signal_2, degree)
            polynomial_signal_2 = np.poly1d(coefficients_signal_2)
            # Calculate the area under the curve
            area_signal_2, error = quad(polynomial_signal_2, time_range_signal_2[0], time_range_signal_2[1])       
            df_signal_2 = np.array(data_mean_signal_2.iloc[ start_idx_signal_2:stop_idx_signal_2 +1])
        
            coefficients_velocity = np.polyfit(time_orginal_velocity_conteriner, data_mean_velocity, degree)
            polynomial_velocity = np.poly1d(coefficients_velocity)
           # Calculate the area under the curve
            area_velocity, error = quad(polynomial_velocity, time_range_velocity[0], time_range_velocity[1])       
            df_velocity = np.array(data_mean_velocity.iloc[start_idx_velocity:stop_idx_velocity +1])

            coefficients_lick = np.polyfit(time_orginal_lick_conteriner, data_mean_lick, degree)
            polynomial_lick = np.poly1d(coefficients_lick)
            # Calculate the area under the curve
            area_lick, error = quad(polynomial_lick, time_range_lick[0], time_range_lick[1])       
            df_lick = np.array(data_mean_lick.iloc[start_idx_lick:stop_idx_lick +1])
        
        
        elif method == "trap":
            df_signal_1 = np.array(data_mean_signal_1.iloc[start_idx_signal_1: stop_idx_signal_1 +1])
            area_signal_1 = np.trapz(df_signal_1, time_signal_1)
            
            df_signal_2 = np.array(data_mean_signal_2.iloc[ start_idx_signal_2:stop_idx_signal_2 +1])
            area_signal_2= np.trapz(df_signal_2, time_signal_2)

            df_velocity = np.array(data_mean_velocity.iloc[start_idx_velocity:stop_idx_velocity +1])
            area_velocity = np.trapz(df_velocity, time_velocity)

            df_lick = np.array(data_mean_lick.iloc[start_idx_lick:stop_idx_lick +1])
            area_lick = np.trapz(df_lick, time_lick)
        
        fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(8,6), sharex= True)
        fig.suptitle("Mean")
        fig.tight_layout()

        ax[0].scatter(time_orginal_signal_1, data_mean_signal_1, label='Data points')
        ax[1].scatter(time_orginal_signal_2, data_mean_signal_2, label='Data points')
        ax[2].scatter(time_orginal_velocity_conteriner, data_mean_velocity, label='Data points_velocity')
        ax[3].scatter(time_orginal_lick_conteriner, data_mean_lick, label='Data points_lick')

        if method == "poly":
            x_fit_signal_1 = np.linspace(time_range_signal_1[0], time_range_signal_1[1], 5000)
            y_fit_signal_1 = polynomial_signal_1(x_fit_signal_1)
            x_fit_signal_2 = np.linspace(time_range_signal_2[0], time_range_signal_2[1], 5000)
            y_fit_signal_2 = polynomial_signal_2(x_fit_signal_2)
            x_fit_velocity = np.linspace(time_range_velocity[0], time_range_velocity[1], 5000)
            y_fit_velocity = polynomial_velocity(x_fit_velocity)
            x_fit_lick = np.linspace(time_range_lick[0], time_range_lick[1], 5000)
            y_fit_lick= polynomial_lick(x_fit_lick)
            
            ax[0].plot(x_fit_signal_1, y_fit_signal_1, label=f'Polynomial fit (degree {degree})', color='red')
            ax[1].plot(x_fit_signal_2, y_fit_signal_2, label=f'Polynomial fit (degree {degree})', color='red')
            ax[2].plot(x_fit_velocity, y_fit_velocity, label=f'Polynomial fit (degree {degree})', color='red')
            ax[3].plot(x_fit_lick, y_fit_lick, label=f'Polynomial fit (degree {degree})', color='red')
        elif method == "trap":
            x_fit_signal_1 = time_signal_1
            y_fit_signal_1 =  df_signal_1
            x_fit_signal_2 = time_signal_2
            y_fit_signal_2 =  df_signal_2
            x_fit_velocity = time_velocity
            y_fit_velocity = df_velocity
            x_fit_lick = time_lick
            y_fit_lick = df_lick


        ax[0].fill_between(x_fit_signal_1, y_fit_signal_1, alpha=0.4, color = "red")
        ax[1].fill_between(x_fit_signal_2, y_fit_signal_2, alpha=0.4, color = "red")
        ax[2].fill_between(x_fit_velocity, y_fit_velocity, alpha=0.4, color = "red")
        ax[3].fill_between(x_fit_lick, y_fit_lick, alpha=0.4, color = "red")
        
        ax[0].axhline(0, color = "black")
        ax[1].axhline(0, color = "black")
        ax[2].axhline(0, color = "black")
        ax[3].axhline(0, color = "black")

        ax[0].set_xlabel("Time [s]")
        ax[1].set_xlabel("Time [s]")
        ax[2].set_xlabel("Time [s]")
        ax[3].set_xlabel("Time [s]")
        
        ax[0].set_title("Signal 1")
        ax[1].set_title("Signal 2")
        ax[2].set_title("Velocity")
        ax[3].set_title("Lick")
        
        ax[0].set_ylabel(self.data_type)
        ax[1].set_ylabel(self.data_type)
        ax[2].set_ylabel("Velocity [cm/s]")
        ax[3].set_ylabel("Lickes per sec")

        ax[0].legend(loc=2)
        ax[1].legend(loc=2)
        ax[2].legend(loc=2)
        ax[3].legend(loc=2)
        
        plt.show()
        __ = [ result.append(item) for item in [area_signal_1, df_signal_1.max(), time_signal_1[df_signal_1.argmax()], df_signal_1.min(), time_signal_1[df_signal_1.argmin()]]] 
        __ = [ result_2.append(item) for item in [area_signal_2, df_signal_2.max(), time_signal_2[df_signal_2.argmax()], df_signal_2.min(), time_signal_2[df_signal_2.argmin()]]] 
        __ = [ result_3.append(item) for item in [area_velocity, df_velocity.max(), time_velocity[df_velocity.argmax()], df_signal_2.min(), time_velocity[df_velocity.argmin()]]] 
        __ = [ result_4.append(item) for item in [area_lick, df_lick.max(), time_lick[df_lick.argmax()], df_signal_2.min(), time_velocity[df_velocity.argmin()]]] 


        df_auc_result_signal_1.iloc[-1, :] = result
        df_auc_result_signal_2.iloc[-1, :] = result_2
        df_auc_result_velocity.iloc[-1, :] = result_3
        df_auc_result_lick.iloc[-1, :] = result_4
        if self.msg == "yes":
            df_auc_result_signal_1.to_excel(self.save_file_v1 + "//" + self.list_of_files[0] + "_" + self.list_of_files[-1] + "_signal_1_auc" + ".xlsx")
            df_auc_result_signal_2.to_excel(self.save_file_v1 + "//" + self.list_of_files_2[0] + "_" + self.list_of_files_2[-1] + "_signal_2_auc" + ".xlsx")
            df_auc_result_velocity.to_excel(self.save_file_v1 + "//" + self.list_of_files_behaviour[0] + "_" + self.list_of_files_behaviour[-1] + "_velocity_auc" + ".xlsx")
            df_auc_result_lick.to_excel(self.save_file_v1 + "//" + self.list_of_files_behaviour[0] + "_" + self.list_of_files_behaviour[-1] + "_lick_auc" + ".xlsx")

    def __pre_process_behaviour(self):
        self.column_name = ["Time_in_msec", "Event_Marker", "TrialCt", "JoyPos_X", "JoyPos_Y", "Amplitude_Pos", "Base_JoyPos_X", "Base_JoyPos_Y", "SolOpenDuration", "DelayToRew", "ITI", "Threshold", "Fail_attempts", "Sum_of_fail_attempts", "Lick_state", "Sum_licks"]
        title = "Information box"
        ok_btn_txt = "Continue"
        message = [i + "  " + z +" \n" for i, z in zip(self.list_of_files, self.list_of_files_2)]
        
        message.insert(0, "Upload CSV files for given  file pairs: \n")
        output = easygui.msgbox(message, title, ok_btn_txt)
        self.list_of_df_behaviour = []
        self.list_of_files_behaviour = easygui.fileopenbox(title="Select a file", filetypes= "*.csv",  multiple=True)
        self.gen_df_behaviour = (pd.read_csv(i,  encoding='latin-1') for i in self.list_of_files_behaviour )
        self.list_of_files_behaviour = [i.split("\\")[-1] for i in  self.list_of_files_behaviour]
        self.list_of_files_behaviour = [i.split(".")[0] for i in  self.list_of_files_behaviour]
        print("Uploaded files:")
        ans = [print(i) for i in self.list_of_files_behaviour]
        for uploaded_signal, upoladed_velocity in zip(self.list_of_files, self.list_of_files_behaviour):
            try:
                assert upoladed_velocity in uploaded_signal
            except AssertionError:
                print(f"The wrong file was uploaded:\n {uploaded_signal} != {upoladed_velocity}")
            else:
                print("File was uploaded correctly")
        for i,df in enumerate(self.gen_df_behaviour):
            if df.shape[1] == 17:
                df = df.iloc[:, :-1]
            time_list = df.iloc[:, 0].apply(lambda x : round((x / 1000),2)) # Convert time to seconds
            df.columns = self.column_name
            trial_max = df["TrialCt"].unique()
            bugged_index = [df.loc[df["TrialCt"] == l, "Event_Marker"].lt(0).idxmax() for l in trial_max]
            df.iloc[bugged_index, 5] = 0
            df["Amplitude_Pos_mm" ] = df["Amplitude_Pos"].apply(lambda x: x/ u_to_mm)
            delta_dist = df["Amplitude_Pos_mm"].tolist()
            delta_dist = [abs(delta_dist[ff] - delta_dist[ff-1]) for ff in range(1, len(delta_dist))]
            delta_dist.insert(0, 0)
            df["delta_Amplitude_Pos_mm"] = delta_dist
            delta_dist_negative = df["Amplitude_Pos_mm"].tolist()
            delta_dist_negative = [delta_dist_negative[ff] - delta_dist_negative[ff-1] for ff in range(1, len(delta_dist_negative))]
            delta_dist_negative.insert(0, 0)
            df["delta_neg_Amplitude_Pos_mm"] = delta_dist_negative
            
            current_velocity = [ round(((delta_dist[ii] / 10) / (time_list[ii] - time_list[ii-1])) ,2) for ii in range(1, len(time_list))]
            current_velocity.insert(0, 0)
            df["Current_velocity_cm_s"] = current_velocity
            if 0 in df["TrialCt"].tolist():
                last_index = df.loc[df["TrialCt"] == 0].last_valid_index()
                drop_list = list(range(0, last_index + 1))
                df.drop(drop_list, inplace = True)
                df.reset_index(inplace = True, drop = True)
            if self.trial_start or self.trial_stop:
                if not self.trial_start:
                    self.trial_start = 0
                else:
                    self.trial_start =   df.loc[df["TrialCt"] == self.trial_start].first_valid_index()
                if not self.trial_stop:
                    self.trial_stop = df.shape[0] + 1
                else:
                    self.trial_stop =   df.loc[df["TrialCt"] == self.trial_stop].last_valid_index()
                
                df = df.iloc[self.trial_start: self.trial_stop +1, :]
                df.reset_index(inplace = True, drop = True)
            if self.trials_numbers:
                trials_to_save = self.filter_file.loc[self.filter_file[0] == self.list_of_files[i], 1]
                if len(trials_to_save):
                    trials_to_save = trials_to_save.values[0].split(",")
                    trials_to_save = [int(val) for val in trials_to_save]
                    if 0 not in trials_to_save:
                        trials_to_save.insert(0,0)
                    df = df.loc[df["TrialCt"].isin(trials_to_save), :]
                    df.reset_index(drop=True, inplace=True)
                else:
                    pass
            #df.to_excel("C:\\Users\\gniew\\Desktop\\PRACA\\Przemek\\test\\test.xlsx")
            self.list_of_df_behaviour.append(df)
      
    def __process_behaviour(self, data):
        bin_ms = 50
        trials_ = data["TrialCt"].unique()
        #self.time_
        time_over_tresh = [data.loc[(data["TrialCt"] == i) & (data["Event_Marker"] == 1), "Time_in_msec"].first_valid_index() for i in trials_]
        time_over_tresh = [i for i in time_over_tresh if i != None]
        time_over_tresh = [data.iloc[i, 0] for i in time_over_tresh]
        time_over_tresh_range = [self.time_ + i for i in time_over_tresh]
        df_mean_velocity = pd.DataFrame(columns= self.time_)
        df_mean_lick = pd.DataFrame(columns= self.time_)
        for time_range in time_over_tresh_range:
            
            idx_ = [abs(data["Time_in_msec"]- time).idxmin() for time in time_range]
            velocity = [data.iloc[index, 19] for index in idx_]
            licks = [data.iloc[index, 14] for index in idx_]
            assert len(velocity) == len(self.time_)
            assert len(licks) == len(self.time_)
            
            df_mean_velocity.loc[len(df_mean_velocity)] =  velocity 
            df_mean_lick.loc[len(df_mean_velocity)] = licks
        for i in range(0, df_mean_lick.shape[1]):
                df_mean_lick.iloc[:,i] = df_mean_lick.iloc[:,i].apply(lambda x: (x * 1000)/bin_ms) 
        
        ans_mean = np.array(df_mean_velocity.mean(axis = 0))
        ans_sem = np.array(df_mean_velocity.sem(axis= 0))
        ans_mean_v2 = np.array(df_mean_lick.mean(axis = 0))
        ans_sem_v2 = np.array(df_mean_lick.sem(axis = 0))
        return ans_mean, ans_sem, ans_mean_v2, ans_sem_v2, df_mean_velocity, df_mean_lick 
            
    def granger_causality(self, start_time = False, stop_time = False, filter_channels = False): 
        ok_btn_txt = "Continue"
       
        title = "Information box"
        message_first = ["Upload CSV files for first type of signal"] 
        output = easygui.msgbox(message_first, title, ok_btn_txt)
        self.list_of_files_first = easygui.fileopenbox(title="Select a file", filetypes= "*.csv",  multiple=True)
        self.gen_df_first = (pd.read_csv(i) for i in self.list_of_files_first)
        self.list_of_files_first = [i.split("\\")[-1] for i in  self.list_of_files_first]
        self.list_of_files_first = [i.split(".")[0] for i in  self.list_of_files_first]
        

        message_second = ["Upload CSV files for second type of signal"] 
        output = easygui.msgbox(message_second, title, ok_btn_txt)
        self.list_of_files_second = easygui.fileopenbox(title="Select a file", filetypes= "*.csv",  multiple=True)
        self.gen_df_second = (pd.read_csv(i) for i in self.list_of_files_second)
        self.list_of_files_second = [i.split("\\")[-1] for i in  self.list_of_files_second]
        self.list_of_files_second = [i.split(".")[0] for i in  self.list_of_files_second]
        
        if filter_channels:
            message_third = ["Upload Excel file with additional informations about channels processing: "] 
            output = easygui.msgbox(message_third, title, ok_btn_txt)
            filter_file = easygui.fileopenbox(title="Select a file", filetypes= "*.xslx",  multiple=False)
            filter_file = pd.read_excel(filter_file, header=None)

        print("I will do granger causality test for: ")
        ans = [print(f"{j[0]} vs {j[1]}") for i,j in enumerate(zip(self.list_of_files_first,  self.list_of_files_second ))]
        
        df_results = pd.DataFrame(columns=["Test", "Statistic", "p-value", "lag", "Compared groups"])
        for index, df in enumerate(zip(self.gen_df_first,  self.gen_df_second )):
            df_1 = self.__pre_process_granger_causality(start_time, stop_time, df[0])
            df_2 = self.__pre_process_granger_causality(start_time, stop_time, df[1]) 
            
            if filter_channels:
                df_1 = self.__filter_channels(filter_file, df_1, index=index)
                df_2 = self.__filter_channels(filter_file, df_2, index=index)
            
            df_1.columns = [i+"_signal_v1" for i in df_1.columns]
            df_2.columns = [i+"_signal_v2" for i in df_2.columns]
            df_finall = pd.concat([df_1,df_2], axis = 1)
            df_finall.drop("Timestamp_signal_v2", axis=1, inplace=True)
            df_finall = self.__stationary_transform(df_finall)
            self.__do_granger_(df_finall, index, df_results)
            #########
        if self.msg == "yes":
            df_results.to_excel(self.save_file_v1+ "//" +self.list_of_files_first[0] + "_" + self.list_of_files_second[0] + "_" + "granger_causality"  + ".xlsx")

    def __pre_process_granger_causality(self, start_time, stop_time, df):
        if start_time:
            start_idx = abs(df["Timestamp"]- start_time).idxmin()
            df = df.iloc[start_idx:, :]
            df.reset_index(drop=True, inplace=True)

        if stop_time:
            stop_idx = abs(df["Timestamp"]- stop_time).idxmin()
            df = df.iloc[:stop_idx +1, :]
            df.reset_index(drop=True, inplace=True)
    
        return df
    
    def __stationary_transform(self, df):
        
        columns = df.columns
        columns = columns[1:]
        for i in columns:
            result = adfuller(df[i])
            if result[1] > 0.05:
                df[i] = df[i].diff().dropna()
                print(f"Column {i} is not stationary")
            else:
                print(f"Column {i} is stationary")
        return df
    
    def __do_granger_(self,df, index, df_finall):
        columns = df.columns
        columns = columns[1:]
        if len(columns) == 2:
            check = pd.DataFrame(grangercausalitytests(df[[columns[0], columns[1]]], maxlag=3, verbose= False))
            for item, value in check.items():
                    for item_v2, value_2 in value[0].items():
                        results = list(value_2)[:2]
                        results.insert(0, item_v2)
                        results.append(item)
                        results.append(f"{self.list_of_files_second[index]}_{columns[1]} --> {self.list_of_files_first[index]}_{columns[0]}")

                        df_finall.loc[len(df_finall)] = results
            
            check_reverse = pd.DataFrame(grangercausalitytests(df[[columns[1], columns[0]]], maxlag=3, verbose= False))
            for item_reverse, value_reverse in check_reverse.items():
                    for item_v2_reverse, value_2_reverse in value_reverse[0].items():
                        results = list(value_2_reverse)[:2]
                        results.insert(0, item_v2_reverse)
                        results.append(item_reverse)
                        results.append(f"{self.list_of_files_first[index]}_{columns[0]} --> {self.list_of_files_second[index]}_{columns[1]}")
                        
                        df_finall.loc[len(df_finall)] = results
                    

            
        elif len(columns) == 4:
    
            for i in range(0,len(columns)-2):
                check = pd.DataFrame(grangercausalitytests(df[[columns[i], columns[i+2]]], maxlag=3, verbose= False))
                for item, value in check.items():
                    for item_v2, value_2 in value[0].items():
                        results = list(value_2)[:2]
                        results.insert(0, item_v2)
                        results.append(item)
                        results.append(f"{self.list_of_files_second[index]}_{columns[i+2]} --> {self.list_of_files_first[index]}_{columns[i]}")
                        
                        df_finall.loc[len(df_finall)] = results
                check_reverse = pd.DataFrame(grangercausalitytests(df[[columns[i+2], columns[i]]], maxlag=3, verbose= False))
                for item_reverse, value_reverse in check_reverse.items():
                    for item_v2_reverse, value_2_reverse in value_reverse[0].items():
                        results = list(value_2_reverse)[:2]
                        results.insert(0, item_v2_reverse)
                        results.append(item_reverse)
                        results.append(f"{self.list_of_files_first[index]}_{columns[i]} --> {self.list_of_files_second[index]}_{columns[i+2]}")
                        
                        df_finall.loc[len(df_finall)] = results

    def __filter_channels(self, df_filter, df_data, index):
        
        try:
            channel_to_del = df_filter.loc[df_filter[0] == self.list_of_files_first[index], 1]
            if int(channel_to_del) == 1:
                df_data = df_data.drop("CH1", axis = 1)
                print(f"Channel 1 in file {self.list_of_files_first[index]}, will be drop")
                return df_data
            elif int(channel_to_del) == 2:
               df_data =  df_data.drop("CH2", axis = 1)
               print(f"Channel 2 in file {self.list_of_files_first[index]}, will be drop")
               return df_data
        except TypeError:
            print(f"Channels in file {self.list_of_files_first[index]}, will not be filter")
            return df_data
        
########################################################
object_ff = Fiber_photometry(granger=False)
object_ff.pre_process(trials_numbers=False)
#object_ff.plot( extra_decoration= True)
object_ff.auc(method= "poly", time_range_signal_2=[-666, 2345])
#object_ff.granger_causality(start_time= 10000, filter_channels= False)

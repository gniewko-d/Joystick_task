# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 17:18:42 2022

@author: gniew
"""
import easygui
import pandas as pd
df = None



class Joystick_analyzer:
    def __init__(self):
        self.list_of_files = easygui.fileopenbox(title="Select a file", filetypes= "*.csv",  multiple=True)
        self.gen_df = (pd.read_csv(i, header=None) for i in self.list_of_files)
    
    
    def pre_proccesing(self):
        global df
        for i in self.gen_df:
           i[0]= i[0].apply(lambda x : round(((x/(1000*60))%60),2))
           i[]
           df = i
           
           #yield i
           
    #def init_df(self):
        


object_joy = Joystick_analyzer()
object_joy.pre_proccesing()
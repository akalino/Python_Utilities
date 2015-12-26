# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 09:48:15 2015

@author: kalial01
"""

from Tkinter import * 
import time 
import pandas as pd 
import datetime

class swApp(Frame):     
    
    global prjNames 
    global startTimes 
    global endTimes 
    
    prjNames = [] 
    startTimes = [] 
    endTimes = []     
    
    def __init__(self, master): 
        Frame.__init__(self, master)  
        self.grid()   
        self.create_app()  
        
    def create_app(self):  
        self.instruction = Label(self, text = "Enter project name:")  
        self.instruction.grid(row = 1, column = 1, sticky = W)          
        
        self.task = Entry(self)  
        self.task.grid(row = 2, column = 1, columnspan = 8, sticky = W)   
        
        self.start_button = Button(self, text = "Start Task", command = self.sw_start)  
        self.start_button.grid(row=5, column = 1, sticky = W)    
        
        self.pause_button = Button(self, text = "End Task", command = self.sw_pause) 
        self.pause_button.grid(row=5, column = 3, sticky = W) 
        
        self.end_button = Button(self, text = "Generate Daily Report", command = self.sw_end) 
        self.end_button.grid(row=7, column = 1, sticky = W)
        
    def sw_start(self): 
        #print('Press ENTER to begin. Afterwards, press ENTER to "click" the stopwatch. Press Ctrl-C to quit.')
        #raw_input()                    # press Enter to begin
        #print('Started.')
        curName = self.task.get() 
        prjNames.append(curName)        
        startTime = datetime.datetime.fromtimestamp(time.time())    # get the first lap's start time
        startTimes.append(startTime)        

    def sw_pause(self): 
        endTime = datetime.datetime.fromtimestamp(time.time()) 
        endTimes.append(endTime)
        
    def sw_end(self):   
        stSer = pd.Series(startTimes) 
        endSer = pd.Series(endTimes)
        ts = endSer - stSer 
        pn = pd.Series(prjNames)
        endOfDay = pd.DataFrame({'Project':pn,'Start':stSer,'End':endSer,'Time Spent':ts})    
        dt = datetime.datetime.now()
        today = 'timesheet_' + str(dt.year) + str(dt.month) + str(dt.day) + '.csv'        
        endOfDay.to_csv(today, sep=",", index=False)
        
root = Tk() 
root.title("Time Tracker") 
root.geometry("250x100") 
app = swApp(root) 

root.mainloop()

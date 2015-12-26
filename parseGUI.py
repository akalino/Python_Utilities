# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 14:15:01 2015

@author: alkal
"""

class IRapp(Frame): 
	
    def __init__(self, master): 
        Frame.__init__(self, master)  
        self.grid()   
        self.create_app()   
	
	
    def create_app(self):   

        self.parser = IntVar()  
        #self.pos = IntVar() 
        self.builder = Label(self, text = "Choose your preferred parsing package")  
        self.builder.grid(row = 2, column = 0, columnspan = 3, sticky = W)  
        
        Radiobutton(self, text = "TextBlob Noun Phrases", variable = self.parser, value = 1).grid(row = 3, column = 1, sticky = W)  
        Radiobutton(self, text = "Pattern Parser", variable = self.parser, value = 2).grid(row = 4, column = 1, sticky = W)  
        Radiobutton(self, text = "Stanford PCFG", variable = self.parser, value = 3).grid(row = 5, column = 1, sticky = W)  
        Radiobutton(self, text = "Stanford Factored", variable = self.parser, value = 4).grid(row = 5, column = 1, sticky = W)  
        Radiobutton(self, text = "Stanford Caseless", variable = self.parser, value = 5).grid(row = 5, column = 1, sticky = W) 
        
        self.build_button = Button(self, text = "Build New Index", command = self.indexing) 
        self.build_button.grid(row = 3, column = 3, sticky = W)  
        self.save_button = Button(self, text = "Save Index", command = self.saving)  
        self.save_button.grid(row = 4, column = 3, sticky = W)	  
        self.load_button = Button(self, text = "Load Existing Index", command = self.loading) 
        self.load_button.grid(row = 5, column = 3, sticky = W) 	 
        
        self.instruction = Label(self, text = "Enter the query")  
        self.instruction.grid(row = 8, column = 0, sticky = W)   

        self.query = Entry(self)  
        self.query.grid(row = 8, column = 2, columnspan = 3, sticky = W)   
        
        self.submit_button = Button(self, text = "Submit", command = self.querying)  
        self.submit_button.grid(row=9, column = 1, sticky = W)   
        
        self.text = Text(self, width = 60, height = 30, wrap = WORD)    
        self.instruction2 = Label(self, text = "Displaying top 10 results")  
        self.instruction2.grid(row = 11, column = 1, columnspan = 2, sticky = W)   
        self.text.grid(row = 12, column = 1, columnspan = 2, sticky = W)   

    def indexing(self):   
        meas1 = self.sim.get()  
        toggle1 = self.pos.get()  
        if toggle1 == 1:  
            inputPath = 'data/CranfieldDocsPOS/*.txt' 
        else:  
            inputPath = 'data/CranfieldDocsParse/*.txt' 
            
        if meas1 == 1:  
            s = CosineIRSystem()     
            global s  
            start = time.clock() 
            s.index_collection(glob.glob(inputPath))   
            end = time.clock()  
            runtime = (end-start) 
            message2 = "Index construction complete - using cosine similarity"  
            message3 = "Index construction took " + str(runtime) + " seconds" 
        elif meas1 == 2:  
            s = OkapiIRSystem()    
            global s  
            start = time.clock() 
            s.index_collection(glob.glob(inputPath))   
            end = time.clock()  
            runtime = (end-start) 
            message2 = "Index construction complete - using Okapi similarity"  
            message3 = "Index construction took " + str(runtime) + " seconds"   
        else:  
            message2 = "Needs user parameter selections"  
            message3 = "Please try again" 
            
    def set_text_newline(s):	 
        self.text.insert(INSERT, '\n' + s) 
        self.text.delete(0.0, END) 
        set_text_newline(message2)  
        set_text_newline(message3) 
    
    def saving(self):    
        def set_text_newline(s):	 
            self.text.insert(INSERT, '\n' + s) 
            with open('saved_index.pkl', 'wb') as output:  
                pickle.dump(s, output, pickle.HIGHEST_PROTOCOL)  
            self.text.delete(0.0, END)  
            set_text_newline('The index was saved') 
            
    def loading(self):   
        def set_text_newline(s):	 
            self.text.insert(INSERT, '\n' + s) 
            with open('saved_index.pkl', 'rb') as input:  
                s = pickle.load(input)   
            global s 
            self.text.delete(0.0, END)  
            set_text_newline('An index was loaded, please query') 
    
    def querying(self):  
        def set_text_newline(s):	 
            self.text.insert(INSERT, '\n' + s) 
            queryStr = self.query.get()      
            queryStr = str(queryStr) 
            splitVar = re.split(" ", queryStr) 
            prepQuery = tuple(splitVar)   
            s.present_results(*prepQuery)  
            self.text.delete(0.0, END) 
            res = top10   
            for i in range(len(res)):  
                set_text_newline(res[i]) 


root = Tk() 
root.title("Retrieval System") 
root.geometry("750x550") 
app = IRapp(root) 

root.mainloop()

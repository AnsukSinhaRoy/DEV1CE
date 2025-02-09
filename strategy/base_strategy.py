#base_strategy.py
from abc import ABC, abstractmethod

class BaseStrategy(ABC):#Abstract Base Class
    #Abstract Base Class for all strategies means that all strategies must inherit from this class
    #and implement the apply_strategy method
    #This class will have a constructor that will take the dataframe as input and initialize the shares, amount, entered, and stop_loss attributes.
    #The apply_strategy method will be an abstract method that will be implemented by the child classes.
    #The apply_strategy method will be responsible for applying the strategy logic to the dataframe.

    def __init__(self, df):#what is this self? self is a reference to the current instance of the class.
        self.df = df #self.df means that the dataframe is an attribute of the class and can be accessed by any method of the class.
        self.shares = 0
        self.amount = df['close'][0]
        self.entered = False
        self.stop_loss = None
    
    @abstractmethod
    def run_strategy(self):
        pass

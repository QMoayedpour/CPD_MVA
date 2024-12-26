import numpy as np

class Cost:
    def __init__(self):
        pass

    def __call__(self, l):
        return 0

class LCost(Cost):
    def __init__(self, c):
        self.c = c
    
    def __call__(self, l):
        return self.c * l

class LogCost(Cost):
    def __init__(self, c):
        self.c = c
    
    def __call__(self, l):
        return self.c * float(np.log(l)) if l>0 else 0

class SquareCost(Cost):
    def __init__(self, c):
        self.c = c
    
    def __call__(self, l):
        return self.c * l**2

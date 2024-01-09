from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score




# Kernel Functions 
def uniform(x,y):
    return (np.abs(x/y) <= 1) and 1/2 or 0

def triangle(x,y):
    return (np.abs(x/y) <= 1) and  (1 - np.abs(x/y)) or 0

def gaussian(x,y):
    return (1.0/np.sqrt(2*np.pi))* np.exp(-.5*(x/y)**2)

def laplacian(x,y):
    return (1.0/(2*y))* np.exp(-np.abs(x/y))

def epanechnikov(x,y):
    return (np.abs(x/y)<=1) and ((3/4)*(1-(x/y)**2)) or 0
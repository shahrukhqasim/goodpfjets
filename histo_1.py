import matplotlib.pyplot as plt
import numpy as np





x = np.loadtxt('dump.txt')

x = x[x>0]


print(np.histogram(x,bins=[0,10,20,50,100,200,300,400,500,600,700,800,900,10000]))


print(x)
import numpy as np
import math
from dataset import ti,yi
#------------------------------------------------------------------------------
def yhatresult(x):
    return 5+(1+(20/(0.18+x)))
#------------------------------------------------------------------------------    
def plotresult(ti,yi,fvalidation,yhat, t):
    import matplotlib.pyplot as plt
    
    T = np.arange(min(ti),max(ti),0.1)
    
    
    plt.scatter(ti, yi, color='darkred', marker='x')
    plt.plot(t, yhat, color='green', linestyle='solid', linewidth = 1)
    plt.xlabel('Resistance (kOhm)')
    plt.ylabel('Voltage Gain')
    plt.title('Theoretical Graph | FV:'+str(fvalidation),fontstyle='italic')
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.1)
    plt.legend(['theoretical curve','collected data'])
    plt.show()    
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------ 
trainingindices = np.arange(0,len(ti),2)    
traininginput = np.array(ti)[trainingindices]
trainingoutput = np.array(yi)[trainingindices]
validationindices = np.arange(1,len(ti),2)    
validationinput = np.array(ti)[validationindices]
validationoutput = np.array(yi)[validationindices]

print(validationindices)
print(trainingindices)

T=[]
for k in range(len(validationindices)):
    T.append(ti[validationindices[k]])

Y = []

fvalidation=0
for i in range (len(validationindices)):
    result=yhatresult(ti[i])
    Y.append(result)
    e = yi[validationindices[i]] - yhatresult(ti[i])
    fvalidation = fvalidation + e**2
plotresult(ti,yi,fvalidation, Y, T)

print(fvalidation)


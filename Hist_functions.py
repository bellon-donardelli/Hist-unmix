import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.special import gamma
from scipy.special import gammainc
from numpy import trapz
from scipy.integrate import simpson



def gradient(x,y,pars='yes'):
    
    '''
    Calculates the ∂y/∂x through a finite-differences method.
    
    Parameters: 
        x : a 1D-array containing the values of the independent variable
        y : a 1D-array containing the values of the dependent variable
        pars: string, 'yes' or 'no', which indicates if the border values will be ignored.
        
    returns:
        grad: a 1D-array of values. 
        '''
        
    grad=np.zeros(np.size(x))
    for i in range(np.size(x)):
        if i==0:
            grad[i]=np.array((y[i+1]-y[i])/(x[i+1]-x[i]))
        if i>0 and i<(np.size(x)-1):
            grad[i]=np.array((y[i+1]-y[i-1])/(x[i+1]-x[i-1]))
        else:
            grad[i]=np.array((y[i]-y[i-1])/(x[i]-x[i-1]))
    
    if pars=='yes' or pars=='y' or pars=='YES' or pars =='Yes':
        grad[-1]=grad[-2]
        grad[0]=grad[1]
    elif pars=='no' or pars=='n' or pars=='NO' or pars =='No':
        grad=grad
    return(grad)



def region_selection(x,y,yi,yf):
    '''
    Sample the data (x,y) in a given interval.
    
    Parameters: 
        x : a 1D-array containing the values of the independent variable
        y : a 1D-array containing the values of the dependent variable
        yi: a integer, the left interval delimeter
        yf: a integer, the right interval delimiter
        
    returns:
        x_new: a 1D-array of values;
        y_new: a 1D-array of values.
    '''

    y_new=[]
    x_new=[]
    for i in range(np.size(y)):
        if i>=yi and i<=yf:
            y_new=np.append(y_new,y[i])
            x_new=np.append(x_new,x[i])
    
    return x_new,y_new


def region_selection2(y,yi,yf):
    '''
    Sample the data (y) in a given interval.
    
    Parameters: 
        x : a 1D-array containing the values of the independent variable
        yi: a integer, the left interval delimeter
        yf: a integer, the right interval delimiter
        
    returns:
        x_new: a 1D-array of values.
    '''
    y_new=[]
    for i in range(np.size(y)):
        if i>=yi and i<=yf:
            y_new=np.append(y_new,y[i])

    return y_new
    

def line_inv(x,y):
    '''
    Perform a linear regression (y=ax+b) through a least squares matrix methodology.
    Parameters: 
        x : a 1D-array containing the values of the independent variable
        y : a 1D-array containing the values of the dependent variable

    returns:
        a: a float, the slope;
        b: a float, the linear coefficient.

    '''
    gradb=np.ones(np.size(x))
    G=np.column_stack([x,gradb])
    a,b=(np.linalg.inv(G.T@G))@G.T@y
    

    return a,b

def ferro_area(x,data,data2,selecting_function,lineinv_function):
    '''
    Firstly subtracts the high-field para/dia susceptibility, then calculates the total area corresponding to the ferromagnetic components.

    Parameters: 
        x : a 1D-array containing the values of the independent variable;
        data : a 1D-array containing the values of the dependent variable;
        selecting_function: a function that will sample the high field region, region_selection;
        lineinv_function: a function that provides linear regression parameters (y=ax+b), line_inv;
        numerical_int: a function that calculate numerical integrals, numerical_int.


    returns:
        Area: a float.

    '''
    xs1,ys1=selecting_function(x,data,0,int(np.size(data)/18)) #constraining the values at high fields to invert a line representing para/dia contribution

    xs2,ys2=selecting_function(x,data,(int(np.size(data)/2)+int(np.size(data)/2.2)),int(np.size(data)))#constraining the values at high fields to invert a line representing para/dia contribution

    x3,y3=np.concatenate((xs1,xs2)),np.concatenate((ys1,ys2)) #concatening new section to invert a line at high-fields
    
    a,b=lineinv_function(x3,y3) #invertion of a line to find the slope
    y_line=(a*x)+b #calculating the line inverted for high-field values


    xs4,ys4=selecting_function(x,data2,0,int(np.size(data2)*0.1)) #constraining the values at high fields to invert a line representing para/dia contribution

    a2,b2=lineinv_function(xs4,ys4) #invertion of a line to find the slope
    y_line2=(a2*x)+b2 #calculating the line inverted for high-field values

    
    Area=simpson(data-y_line,x) 

    return Area,a2

def rev_ferro_area(x,data,u,X0):
    '''
    Firstly subtracts the high-field para/dia susceptibility, then calculates the total area corresponding to the ferromagnetic components.

    Parameters: 
        x : a 1D-array containing the values of the independent variable;
        data : a 1D-array containing the values of the dependent variable;
        u: the mean value of a Gamma-Cauchy distribution (float)
        X0: the dia/paramagnetic slope (float)


    returns:
        Area: a float.

    '''

    xnew=[]

    for i in range(np.size(x)):
        if x[i]>u:
            xnew=np.append(xnew,x[i])
    
    
    Area=np.trapz(data-X0,xnew,dx=1e-7) #calculating the Area under the smoothed curve

    Area=np.array([Area])
  
    return Area


def moving_mean(x,group_size=30):

    '''
    Computes a moving mean filter on a given interval.

    Parameters: 
        x : a 1D-array containing the values of the independent variable;
        group_size: an integer >=1, delimiting the number of points;


    returns:
        moving_meanx: a 1D-array of values (float).

    '''
    i = 0
    moving_meanx=[]
    
    # Calculate the means:
    while i < len(x) - group_size  + 1:
        group = x[i : i + group_size ]
        mean_group = sum(group) / group_size 
        moving_meanx.append(mean_group)
        i +=1
    

    moving_meanx=np.array(moving_meanx)

    return moving_meanx


def GGCD_1C(u,theta,alfa,beta,I,x):

    '''
    Computes modified Gamma-Cauchy exponential model for a single ferromagnetic component

    Parameters: 
        u : the mean coercivity of the population, where the component peaks (float);
        theta: the dispersion parameter, suggested to be between 0 and 1 (float);
        alfa,beta: skewness/kurtosis controling parameters (float);
        I: scaling factor (float);
        x: 1D-array of applied field values (float). 


    returns:
        y: a 1D-array of values of a Gamma-Cauchy exponential model.

    '''

    f_term=(-np.log10((1/2)-((1/np.pi)*np.arctan((x-abs(u))/abs(theta)))))**(abs(alfa)-1)
    s_term=((1/2)-((1/np.pi)*np.arctan((x-abs(u))/abs(theta))))**((1/abs(beta))-1)
    t_term=((np.pi*abs(theta)*(abs(beta)**abs(alfa))*gamma(abs(alfa)))*(1+(((x-abs(u))/abs(theta))**2)))
    
    y=(((f_term*s_term)/(t_term))*abs(I))
    
    return y



def GGCD_2C(u1,theta1,alfa1,beta1,I1,u2,theta2,alfa2,beta2,I2,x):

    '''
    Computes modified Gamma-Cauchy exponential model for two ferromagnetic component.

    Parameters: 
        u1/u2: the mean coercivities of the population, where the component peaks (float);
        theta1/theta2: the dispersion parameters, suggested to be between 0 and 1 (float);
        alfa1/alfa2/beta1/beta2: skewness/kurtosis controling parameters (float);
        I1/I2: scaling factors (float);
        x: 1D-array of applied field values (float). 


    returns:
        y: a 1D-array of values of a Gamma-Cauchy exponential model (float).

    '''
    f_term1=(-np.log10((1/2)-((1/np.pi)*np.arctan((x-abs(u1))/abs(theta1)))))**(abs(alfa1)-1)
    s_term1=((1/2)-((1/np.pi)*np.arctan((x-abs(u1))/abs(theta1))))**((1/abs(beta1))-1)
    t_term1=((np.pi*abs(theta1)*(abs(beta1)**abs(alfa1))*gamma(abs(alfa1)))*(1+(((x-abs(u1))/abs(theta1))**2)))

    f_term2=(-np.log10((1/2)-((1/np.pi)*np.arctan((x-abs(u2))/abs(theta2)))))**(abs(alfa2)-1)
    s_term2=((1/2)-((1/np.pi)*np.arctan((x-abs(u2))/abs(theta2))))**((1/abs(beta2))-1)
    t_term2=((np.pi*abs(theta2)*(abs(beta2)**abs(alfa2))*gamma(abs(alfa2)))*(1+(((x-abs(u2))/abs(theta2))**2)))
    
    y1=(((f_term1*s_term1)/(t_term1))*abs(I1))
    y2=(((f_term2*s_term2)/(t_term2))*abs(I2))

    y=(y1+y2)
    
    return y


def GGCD_3C(u1,theta1,alfa1,beta1,I1,u2,theta2,alfa2,beta2,I2,u3,theta3,alfa3,beta3,I3,x):

    '''
    Computes modified Gamma-Cauchy exponential model for three ferromagnetic component.

    Parameters: 
        u1/u2/u3 : the mean coercivities of the population, where the component peaks (float);
        theta1/theta2/theta3: the dispersion parameters, suggested to be between 0 and 1 (float);
        alfa1/alfa2/alfa3/beta1/beta2/beta3: skewness/kurtosis controling parameters (float);
        I1/I2/I3: scaling factors (float);
        x: 1D-array of applied field values (float). 


    returns:
        y: a 1D-array of values of a Gamma-Cauchy exponential model (float).

    '''

    f_term1=(-np.log10((1/2)-((1/np.pi)*np.arctan((x-abs(u1))/abs(theta1)))))**(abs(alfa1)-1)
    s_term1=((1/2)-((1/np.pi)*np.arctan((x-abs(u1))/abs(theta1))))**((1/abs(beta1))-1)
    t_term1=((np.pi*abs(theta1)*(abs(beta1)**abs(alfa1))*gamma(abs(alfa1)))*(1+(((x-abs(u1))/abs(theta1))**2)))

    f_term2=(-np.log10((1/2)-((1/np.pi)*np.arctan((x-abs(u2))/abs(theta2)))))**(abs(alfa2)-1)
    s_term2=((1/2)-((1/np.pi)*np.arctan((x-abs(u2))/abs(theta2))))**((1/abs(beta2))-1)
    t_term2=((np.pi*abs(theta2)*(abs(beta2)**abs(alfa2))*gamma(abs(alfa2)))*(1+(((x-abs(u2))/abs(theta2))**2)))

    f_term3=(-np.log10((1/2)-((1/np.pi)*np.arctan((x-abs(u3))/abs(theta3)))))**(abs(alfa3)-1)
    s_term3=((1/2)-((1/np.pi)*np.arctan((x-abs(u3))/abs(theta3))))**((1/abs(beta3))-1)
    t_term3=((np.pi*abs(theta3)*(abs(beta3)**abs(alfa3))*gamma(abs(alfa3)))*(1+(((x-abs(u3))/abs(theta3))**2)))
    
    y1=(((f_term1*s_term1)/(t_term1))*abs(I1))
    y2=(((f_term2*s_term2)/(t_term2))*abs(I2))
    y3=(((f_term3*s_term3)/(t_term3))*abs(I3))

    y=(y1+y2+y3)
    
    return y


def Levenberg_Marquardt_1C(function,u1,theta1,alfa1,beta1,I1,x,data,constrain,sample_name,eps=1e-15,maxiter=200):

    '''
    Computes an optimization procedure to invert (Levenberg_Marquardt) parameters of a Gamma-Cauchy exponential model with one ferromagnetic component.

    Parameters:
        function: the foward model for one ferromagnetic component;
        u1 : starting guess, the mean coercivity of the population, where the component peaks (float);
        theta1: starting guess, the dispersion parameter, suggested to be between 0 and 1 (float);
        alfa1,beta1: starting guess, skewness/kurtosis controling parameters (float);
        I1: starting guess, scaling factor (float);
        x: 1D-array of applied field values (float). 
        data: 1D-array of the gradient of the lower branched magnetic hysteresis (float);
        constrain: string, force minimization in around the starting guess for the coercivity (Y/N);
        sample_name: string, keeps the name of the sample in order to name the saved figures;
        eps: small value that is used both in the central differences calculation and for convergence criteria (float);
        maxiter: maximum number of iterations per inversion procedure (integer).



    returns:
        euclidean_norm: the squared error euclidean norm (float);
        p_0: array with the inverted parameters. 

    '''

    a=0.1  # decreasing dumping factor rate
    damping=0.1 # dumping factor

    bubble_deltad=[] #empty array
    bubble_error=[] #empty array
    bubble_parameters=[] #empty array

    if constrain=='N' or constrain=='n':
        p_0=[u1,theta1,alfa1,beta1,I1]# array containing the parameters
        p_cor=np.zeros(np.shape(p_0)) # array of zeros that will hold the corrected parameters
        

        #calculating Jacobian Matrix with the initial guesses
        grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],x))/(2*(eps))
        grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],x))/(2*(eps))
        grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],x))/(2*(eps))
        grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],x))/(2*(eps))
        grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),x))/(2*(eps))
        J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1])

        #difference between observed and calculated data

        delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],x)

        #parameters correction

        h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

        p_cor=[p_0[0]+h[0],p_0[1]+h[1],
            p_0[2]+h[2],p_0[3]+h[3],
            p_0[4]+h[4]]

        error=np.sum((delta_d)**2)

        for i in range(maxiter):
            if i==0:
                bubble_error=error
                bubble_deltad=delta_d
                p_0=p_cor
                grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],x))/(2*(eps))
                grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],x))/(2*(eps))
                grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],x))/(2*(eps))
                grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],x))/(2*(eps))
                grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),x))/(2*(eps))
                J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1])

                delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],x)

                h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                p_cor=[p_0[0]+h[0],p_0[1]+h[1],
                    p_0[2]+h[2],p_0[3]+h[3],
                    p_0[4]+h[4]]

                error=np.sum((delta_d)**2)

                criteria=abs(bubble_error-error)

                if criteria>(eps):
                    p_0=p_cor
                    
                else:
                    p_cor=p_0
                    

                tolerance=np.linalg.norm(J.T@delta_d,2)

                if tolerance<=(eps) or i==maxiter: #testing the condition
                    break

            if ((i % 2) == 0) or (error>bubble_error):
                bubble_error=error
                bubble_deltad=delta_d
                p_0=p_cor
                grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],x))/(2*(eps))
                grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],x))/(2*(eps))
                grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],x))/(2*(eps))
                grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],x))/(2*(eps))
                grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),x))/(2*(eps))
                J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1])

                delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],x)

                h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                p_cor=[p_0[0]+h[0],p_0[1]+h[1],
                    p_0[2]+h[2],p_0[3]+h[3],
                    p_0[4]+h[4]]

                error=np.sum((delta_d)**2)

                        

                criteria=abs(bubble_error-error)

                if criteria>(eps):
                    wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                    damping=damping*(a**wip)
                    p_0=p_cor
                else:
                    p_cor=p_0
                    damping=a*damping
                

                tolerance=np.linalg.norm(J.T@delta_d,2)

                if tolerance<=(eps) or i==maxiter: #testing the condition
                    break

            else:
                J=J
                delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],x)

                #parameters correction

                h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                p_cor=[p_0[0]+h[0],p_0[1]+h[1],
                    p_0[2]+h[2],p_0[3]+h[3],
                    p_0[4]+h[4]]

                error=np.sum((delta_d)**2)

                criteria=abs(bubble_error-error)
                

                if criteria>(eps):
                    wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                    damping=damping*(a**wip)
                    p_0=p_cor
                else:
                    p_cor=p_0
                    damping=a*damping

                tolerance=np.linalg.norm(J.T@delta_d,2)

                if tolerance<=(eps) or i==maxiter: #testing the condition
                    break

        p_0[0]=abs(p_0[0])
        p_0[1]=abs(p_0[1])
        p_0[2]=abs(p_0[2])
        p_0[3]=abs(p_0[3])
        p_0[4]=abs(p_0[4])

    if constrain=='Y' or constrain=='y':
        p_0=[u1,theta1,alfa1,beta1,I1] # array containing the parameters
        p_cor=np.zeros(np.shape(p_0)) # array of zeros that will hold the corrected parameters
        

        #calculating Jacobian Matrix with the initial guesses
        grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],x))/(2*(eps))
        grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],x))/(2*(eps))
        grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],x))/(2*(eps))
        grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),x))/(2*(eps))
        J=np.column_stack([grad_theta1,grad_alfa1,grad_beta1,grad_I1])

        delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],x)

        h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

        p_cor=[p_0[0],p_0[1]+h[0],
            p_0[2]+h[1],p_0[3]+h[2],
            p_0[4]+h[3]]

        error=np.sum((delta_d)**2)

        for i in range(maxiter):
            if i==0:
                bubble_error=error
                bubble_deltad=delta_d
                p_0=p_cor
                grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],x))/(2*(eps))
                grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],x))/(2*(eps))
                grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],x))/(2*(eps))
                grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),x))/(2*(eps))
                J=np.column_stack([grad_theta1,grad_alfa1,grad_beta1,grad_I1])

                delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],x)

                h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                p_cor=[p_0[0],p_0[1]+h[0],
                    p_0[2]+h[1],p_0[3]+h[2],
                    p_0[4]+h[3]]

                error=np.sum((delta_d)**2)

                criteria=abs(bubble_error-error)

                if criteria>(eps):
                    p_0=p_cor
                    
                else:
                    p_cor=p_0
                    

                tolerance=np.linalg.norm(J.T@delta_d,2)

                if tolerance<=(eps) or i==maxiter: #testing the condition
                    break

            if ((i % 2) == 0) or (error>bubble_error):
                bubble_error=error
                bubble_deltad=delta_d
                p_0=p_cor
                grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],x))/(2*(eps))
                grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],x))/(2*(eps))
                grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],x))/(2*(eps))
                grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),x))/(2*(eps))
                J=np.column_stack([grad_theta1,grad_alfa1,grad_beta1,grad_I1])

                delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],x)

                h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                p_cor=[p_0[0],p_0[1]+h[0],
                    p_0[2]+h[1],p_0[3]+h[2],
                    p_0[4]+h[3]]

                error=np.sum((delta_d)**2)


                criteria=abs(bubble_error-error)

                if criteria>(eps):
                    wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                    damping=damping*(a**wip)
                    p_0=p_cor
                else:
                    p_cor=p_0
                    damping=a*damping
                
                tolerance=np.linalg.norm(J.T@delta_d,2)

                if tolerance<=(eps) or i==maxiter: #testing the condition
                    break

            else:
                J=J
                delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],x)

                h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                p_cor=[p_0[0],p_0[1]+h[0],
                    p_0[2]+h[1],p_0[3]+h[2],
                    p_0[4]+h[3]]

                error=np.sum((delta_d)**2)

                
                

                criteria=abs(bubble_error-error)
                

                if criteria>(eps):
                    wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                    damping=damping*(a**wip)
                    p_0=p_cor
                else:
                    p_cor=p_0
                    damping=a*damping

                tolerance=np.linalg.norm(J.T@delta_d,2)

                if tolerance<=(eps) or i==maxiter: #testing the condition
                    break

        p_0[0]=abs(p_0[0])
        p_0[1]=abs(p_0[1])
        p_0[2]=abs(p_0[2])
        p_0[3]=abs(p_0[3])
        p_0[4]=abs(p_0[4])


    
    yt=function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],x)
    df=pd.DataFrame(data={'Bc1':p_0[0],
                      'theta1':np.round(p_0[1],5),
                      'alfa1':np.round(p_0[2],5),
                      'beta1':np.round(p_0[3],5),
                      'I1':p_0[4]},index=[0])

    df1=pd.DataFrame(data={'x1':x,
              'Ca':yt,
              'Data gradient':data})


    df.to_csv('inversion_parameters_'+str(sample_name)+'.csv', index=False)
    df1.to_csv('inversion_components_'+str(sample_name)+'.csv', index=False)


    euclidean_norm=np.linalg.norm(delta_d**2,2)
    return euclidean_norm,np.array(p_0)






def Levenberg_Marquardt_2C(function,u1,theta1,alfa1,beta1,I1,u2,theta2,alfa2,beta2,I2,x,data,constrain,option,sample_name,eps=1e-15,maxiter=200):

    '''
    Computes an optimization procedure to invert (Levenberg_Marquardt) parameters of a Gamma-Cauchy exponential model with two ferromagnetic components.

    Parameters:
        function: the foward model for one ferromagnetic component;
        u1/u2 : starting guess, the mean coercivity of the population, where the component peaks (float);
        theta1/theta2: starting guess, the dispersion parameter, suggested to be between 0 and 1 (float);
        alfa1/alfa2,beta1/beta2: starting guess, skewness/kurtosis controling parameters (float);
        I1/I2: starting guess, scaling factor (float);
        x: 1D-array of applied field values (float). 
        data: 1D-array of the gradient of the lower branched magnetic hysteresis (float);
        constrain: string, force minimization in around the starting guess for the coercivity (Y/N);
        sample_name: string, keeps the name of the sample in order to name the saved figures;
        eps: small value that is used both in the central differences calculation and for convergence criteria (float);
        maxiter: maximum number of iterations per inversion procedure (integer).



    returns:
        euclidean_norm: the squared error euclidean norm (float);
        p_0: array with the inverted parameters. 

    '''

    a=0.1  # decreasing dumping factor rate
    damping=0.1 # dumping factor

    bubble_deltad=[] #empty array
    bubble_error=[] #empty array
    bubble_parameters=[] #empty array

    if constrain=='N' or constrain=='n':
        p_0=[u1,theta1,alfa1,beta1,I1,u2,theta2,alfa2,beta2,I2] # array containing the parameters
        p_cor=np.zeros(np.shape(p_0)) # array of zeros that will hold the corrected parameters

         # array with the analytical measurement error


        #calculating Jacobian Matrix with the initial guesses
        grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],x))/(2*(eps))
        grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],x))/(2*(eps))
        grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],x))/(2*(eps))
        grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],x))/(2*(eps))
        grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],x))/(2*(eps))
        grad_u2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]+(eps),p_0[6],
            p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]-(eps),p_0[6],
            p_0[7],p_0[8],p_0[9],x))/(2*(eps))
        grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
            p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
            p_0[7],p_0[8],p_0[9],x))/(2*(eps))
        grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7]+(eps),p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7]-(eps),p_0[8],p_0[9],x))/(2*(eps))
        grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8]+(eps),p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8]-(eps),p_0[9],x))/(2*(eps))
        grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9]-(eps),x))/(2*(eps))

        J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1,grad_u2,grad_theta2,grad_alfa2,grad_beta2,grad_I2])

        #difference between observed and calculated data

        delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)

        #parameters correction

        h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

        p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5]+h[5],
            p_0[6]+h[6],p_0[7]+h[7],p_0[8]+h[8],p_0[9]+h[9]]

        error=np.sum((delta_d)**2)

        for i in range(maxiter):
            if i==0:
                bubble_error=error
                bubble_deltad=delta_d
                p_0=p_cor
                grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                grad_u2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]+(eps),p_0[6],
                    p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]-(eps),p_0[6],
                    p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                    p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                    p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7]+(eps),p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7]-(eps),p_0[8],p_0[9],x))/(2*(eps))
                grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8]+(eps),p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8]-(eps),p_0[9],x))/(2*(eps))
                grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9]-(eps),x))/(2*(eps))

                J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1,grad_u2,grad_theta2,grad_alfa2,grad_beta2,grad_I2])

                delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)

                h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5]+h[5],
                    p_0[6]+h[6],p_0[7]+h[7],p_0[8]+h[8],p_0[9]+h[9]]

                error=np.sum((delta_d)**2)

                
                

                criteria=abs(bubble_error-error)

                if criteria>(eps):
                    p_0=p_cor
                    
                else:
                    p_cor=p_0
                    

                tolerance=np.linalg.norm(J.T@delta_d,2)
                

                if tolerance<=(eps) or i==maxiter: #testing the condition
                    break

            if ((i % 2) == 0) or (error>bubble_error):
                bubble_error=error
                bubble_deltad=delta_d
                p_0=p_cor
                grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                grad_u2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]+(eps),p_0[6],
                    p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]-(eps),p_0[6],
                    p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                    p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                    p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7]+(eps),p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7]-(eps),p_0[8],p_0[9],x))/(2*(eps))
                grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8]+(eps),p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8]-(eps),p_0[9],x))/(2*(eps))
                grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9]-(eps),x))/(2*(eps))

                J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1,grad_u2,grad_theta2,grad_alfa2,grad_beta2,grad_I2])

                delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)

                h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5]+h[5],
                    p_0[6]+h[6],p_0[7]+h[7],p_0[8]+h[8],p_0[9]+h[9]]

                error=np.sum((delta_d)**2)

                
                

                criteria=abs(bubble_error-error)

                if criteria>(eps):
                    wip=(np.array(p_cor)*np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                    damping=damping*(a**wip)
                    p_0=p_cor
                else:
                    p_cor=p_0
                    damping=a*damping
                    

                tolerance=np.linalg.norm(J.T@delta_d,2)
                

                if tolerance<=(eps) or i==maxiter: #testing the condition
                    break

            else:
                J=J
                delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)

                h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5]+h[5],
                    p_0[6]+h[6],p_0[7]+h[7],p_0[8]+h[8],p_0[9]+h[9]]

                error=np.sum((delta_d)**2)

                
                

                criteria=abs(bubble_error-error)

                if criteria>(eps):
                    wip=(np.array(p_cor)*np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                    damping=damping*(a**wip)
                    p_0=p_cor
                    
                else:
                    p_cor=p_0
                    damping=a*damping
                    

                tolerance=np.linalg.norm(J.T@delta_d,2)
                
                if tolerance<=(eps) or i==maxiter: #testing the condition
                    break

        p_0[0]=abs(p_0[0])
        p_0[1]=abs(p_0[1])
        p_0[2]=abs(p_0[2])
        p_0[3]=abs(p_0[3])
        p_0[4]=abs(p_0[4])
        p_0[5]=abs(p_0[5])
        p_0[6]=abs(p_0[6])
        p_0[7]=abs(p_0[7])
        p_0[8]=abs(p_0[8])
        p_0[9]=abs(p_0[9])

    if constrain=='Y' or constrain=='y':
        if option=="A":
            p_0=[u1,theta1,alfa1,beta1,I1,u2,theta2,alfa2,beta2,I2] # array containing the parameters
            p_cor=np.zeros(np.shape(p_0)) # array of zeros that will hold the corrected parameters

             # array with the analytical measurement error
    

            #calculating Jacobian Matrix with the initial guesses
            grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_u2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]+(eps),p_0[6],
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]-(eps),p_0[6],
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7]+(eps),p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7]-(eps),p_0[8],p_0[9],x))/(2*(eps))
            grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8]+(eps),p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8]-(eps),p_0[9],x))/(2*(eps))
            grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9]-(eps),x))/(2*(eps))

            J=np.column_stack([grad_theta1,grad_alfa1,grad_beta1,grad_I1,grad_u2,grad_theta2,grad_alfa2,grad_beta2,grad_I2])

            delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)

            h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

            p_cor=[p_0[0],p_0[1]+h[0],p_0[2]+h[1],p_0[3]+h[2],p_0[4]+h[3],p_0[5]+h[4],
                p_0[6]+h[5],p_0[7]+h[6],p_0[8]+h[7],p_0[9]+h[8]]

            error=np.sum((delta_d)**2)

            for i in range(maxiter):
                if i==0:
                    bubble_error=error
                    bubble_deltad=delta_d
                    p_0=p_cor
                    grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_u2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]+(eps),p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]-(eps),p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]+(eps),p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]-(eps),p_0[8],p_0[9],x))/(2*(eps))
                    grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]+(eps),p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]-(eps),p_0[9],x))/(2*(eps))
                    grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]-(eps),x))/(2*(eps))

                    J=np.column_stack([grad_theta1,grad_alfa1,grad_beta1,grad_I1,grad_u2,grad_theta2,grad_alfa2,grad_beta2,grad_I2])

                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0],p_0[1]+h[0],p_0[2]+h[1],p_0[3]+h[2],p_0[4]+h[3],p_0[5]+h[4],
                        p_0[6]+h[5],p_0[7]+h[6],p_0[8]+h[7],p_0[9]+h[8]]

                    error=np.sum((delta_d)**2)

                    
                    

                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        p_0=p_cor
                        
                    else:
                        p_cor=p_0
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)
                    

                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

                if ((i % 2) == 0) or (error>bubble_error):
                    bubble_error=error
                    bubble_deltad=delta_d
                    p_0=p_cor
                    grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_u2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]+(eps),p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]-(eps),p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]+(eps),p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]-(eps),p_0[8],p_0[9],x))/(2*(eps))
                    grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]+(eps),p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]-(eps),p_0[9],x))/(2*(eps))
                    grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]-(eps),x))/(2*(eps))

                    J=np.column_stack([grad_theta1,grad_alfa1,grad_beta1,grad_I1,grad_u2,grad_theta2,grad_alfa2,grad_beta2,grad_I2])

                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0],p_0[1]+h[0],p_0[2]+h[1],p_0[3]+h[2],p_0[4]+h[3],p_0[5]+h[4],
                        p_0[6]+h[5],p_0[7]+h[6],p_0[8]+h[7],p_0[9]+h[8]]

                    error=np.sum((delta_d)**2)

                    
                    
                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                        damping=damping*(a**wip)
                        p_0=p_cor
                    else:
                        p_cor=p_0
                        damping=a*damping
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)
                    

                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

                else:
                    J=J
                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0],p_0[1]+h[0],p_0[2]+h[1],p_0[3]+h[2],p_0[4]+h[3],p_0[5]+h[4],
                        p_0[6]+h[5],p_0[7]+h[6],p_0[8]+h[7],p_0[9]+h[8]]

                    error=np.sum((delta_d)**2)

                    
                    
                    criteria=abs(bubble_error-error)

                    
                    

                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                        damping=damping*(a**wip)
                        p_0=p_cor
                    else:
                        p_cor=p_0
                        damping=a*damping
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)
                    
                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

            p_0[0]=abs(p_0[0])
            p_0[1]=abs(p_0[1])
            p_0[2]=abs(p_0[2])
            p_0[3]=abs(p_0[3])
            p_0[4]=abs(p_0[4])
            p_0[5]=abs(p_0[5])
            p_0[6]=abs(p_0[6])
            p_0[7]=abs(p_0[7])
            p_0[8]=abs(p_0[8])
            p_0[9]=abs(p_0[9])


        if option=="B":
            p_0=[u1,theta1,alfa1,beta1,I1,u2,theta2,alfa2,beta2,I2] # array containing the parameters
            p_cor=np.zeros(np.shape(p_0)) # array of zeros that will hold the corrected parameters

             # array with the analytical measurement error
    

            #calculating Jacobian Matrix with the initial guesses
            grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7]+(eps),p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7]-(eps),p_0[8],p_0[9],x))/(2*(eps))
            grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8]+(eps),p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8]-(eps),p_0[9],x))/(2*(eps))
            grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9]-(eps),x))/(2*(eps))

            J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1,grad_theta2,grad_alfa2,grad_beta2,grad_I2])

            delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)

            h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

            p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5],
                p_0[6]+h[5],p_0[7]+h[6],p_0[8]+h[7],p_0[9]+h[8]]

            error=np.sum((delta_d)**2)

            for i in range(maxiter):
                if i==0:
                    bubble_error=error
                    bubble_deltad=delta_d
                    p_0=p_cor
                    grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]+(eps),p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]-(eps),p_0[8],p_0[9],x))/(2*(eps))
                    grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]+(eps),p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]-(eps),p_0[9],x))/(2*(eps))
                    grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]-(eps),x))/(2*(eps))

                    J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1,grad_theta2,grad_alfa2,grad_beta2,grad_I2])

                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5],
                        p_0[6]+h[5],p_0[7]+h[6],p_0[8]+h[7],p_0[9]+h[8]]

                    error=np.sum((delta_d)**2)

                    
                    

                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        p_0=p_cor
                        
                    else:
                        p_cor=p_0
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)
                    

                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

                if ((i % 2) == 0) or (error>bubble_error):
                    bubble_error=error
                    bubble_deltad=delta_d
                    p_0=p_cor
                    grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]+(eps),p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]-(eps),p_0[8],p_0[9],x))/(2*(eps))
                    grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]+(eps),p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]-(eps),p_0[9],x))/(2*(eps))
                    grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]-(eps),x))/(2*(eps))

                    J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1,grad_theta2,grad_alfa2,grad_beta2,grad_I2])

                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5],
                        p_0[6]+h[5],p_0[7]+h[6],p_0[8]+h[7],p_0[9]+h[8]]

                    error=np.sum((delta_d)**2)

                    
                    
                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                        damping=damping*(a**wip)
                        p_0=p_cor
                    else:
                        p_cor=p_0
                        damping=a*damping
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)
                    

                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

                else:
                    J=J
                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5],
                        p_0[6]+h[5],p_0[7]+h[6],p_0[8]+h[7],p_0[9]+h[8]]

                    error=np.sum((delta_d)**2)

                    
                    
                    criteria=abs(bubble_error-error)

                    
                    

                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                        damping=damping*(a**wip)
                        p_0=p_cor
                    else:
                        p_cor=p_0
                        damping=a*damping
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)
                    
                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

            p_0[0]=abs(p_0[0])
            p_0[1]=abs(p_0[1])
            p_0[2]=abs(p_0[2])
            p_0[3]=abs(p_0[3])
            p_0[4]=abs(p_0[4])
            p_0[5]=abs(p_0[5])
            p_0[6]=abs(p_0[6])
            p_0[7]=abs(p_0[7])
            p_0[8]=abs(p_0[8])
            p_0[9]=abs(p_0[9])

        if option=="C":
            p_0=[u1,theta1,alfa1,beta1,I1,u2,theta2,alfa2,beta2,I2] # array containing the parameters
            p_cor=np.zeros(np.shape(p_0)) # array of zeros that will hold the corrected parameters

             # array with the analytical measurement error
    

            #calculating Jacobian Matrix with the initial guesses
            grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                p_0[7],p_0[8],p_0[9],x))/(2*(eps))
            grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7]+(eps),p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7]-(eps),p_0[8],p_0[9],x))/(2*(eps))
            grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8]+(eps),p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8]-(eps),p_0[9],x))/(2*(eps))
            grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9]-(eps),x))/(2*(eps))

            J=np.column_stack([grad_theta1,grad_alfa1,grad_beta1,grad_I1,grad_theta2,grad_alfa2,grad_beta2,grad_I2])

            delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)

            h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

            p_cor=[p_0[0],p_0[1]+h[0],p_0[2]+h[1],p_0[3]+h[2],p_0[4]+h[3],p_0[5],
                p_0[6]+h[4],p_0[7]+h[5],p_0[8]+h[6],p_0[9]+h[7]]

            error=np.sum((delta_d)**2)

            for i in range(maxiter):
                if i==0:
                    bubble_error=error
                    bubble_deltad=delta_d
                    p_0=p_cor
                    grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]+(eps),p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]-(eps),p_0[8],p_0[9],x))/(2*(eps))
                    grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]+(eps),p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]-(eps),p_0[9],x))/(2*(eps))
                    grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]-(eps),x))/(2*(eps))

                    J=np.column_stack([grad_theta1,grad_alfa1,grad_beta1,grad_I1,grad_theta2,grad_alfa2,grad_beta2,grad_I2])

                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0],p_0[1]+h[0],p_0[2]+h[1],p_0[3]+h[2],p_0[4]+h[3],p_0[5],
                        p_0[6]+h[4],p_0[7]+h[5],p_0[8]+h[6],p_0[9]+h[7]]

                    error=np.sum((delta_d)**2)

                    
                    
                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        p_0=p_cor
                        
                    else:
                        p_cor=p_0
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)
                    

                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

                if ((i % 2) == 0) or (error>bubble_error):
                    bubble_error=error
                    bubble_deltad=delta_d
                    p_0=p_cor
                    grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                        p_0[7],p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                        p_0[7],p_0[8],p_0[9],x))/(2*(eps))
                    grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]+(eps),p_0[8],p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]-(eps),p_0[8],p_0[9],x))/(2*(eps))
                    grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]+(eps),p_0[9],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]-(eps),p_0[9],x))/(2*(eps))
                    grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]-(eps),x))/(2*(eps))

                    J=np.column_stack([grad_theta1,grad_alfa1,grad_beta1,grad_I1,grad_theta2,grad_alfa2,grad_beta2,grad_I2])

                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0],p_0[1]+h[0],p_0[2]+h[1],p_0[3]+h[2],p_0[4]+h[3],p_0[5],
                        p_0[6]+h[4],p_0[7]+h[5],p_0[8]+h[6],p_0[9]+h[7]]

                    error=np.sum((delta_d)**2)

                    
                    
                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                        damping=damping*(a**wip)
                        p_0=p_cor
                    else:
                        p_cor=p_0
                        damping=a*damping
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)
                    

                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

                else:
                    J=J
                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0],p_0[1]+h[0],p_0[2]+h[1],p_0[3]+h[2],p_0[4]+h[3],p_0[5],
                        p_0[6]+h[4],p_0[7]+h[5],p_0[8]+h[6],p_0[9]+h[7]]

                    error=np.sum((delta_d)**2)

                    
                    
                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                        damping=damping*(a**wip)
                        p_0=p_cor
                    else:
                        p_cor=p_0
                        damping=a*damping
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)
                    
                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

            p_0[0]=abs(p_0[0])
            p_0[1]=abs(p_0[1])
            p_0[2]=abs(p_0[2])
            p_0[3]=abs(p_0[3])
            p_0[4]=abs(p_0[4])
            p_0[5]=abs(p_0[5])
            p_0[6]=abs(p_0[6])
            p_0[7]=abs(p_0[7])
            p_0[8]=abs(p_0[8])
            p_0[9]=abs(p_0[9])

    
    yt=function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)
    Ca=function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],0,x)
    Cb=function(p_0[0],p_0[1],p_0[2],p_0[3],0,p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],x)
    df=pd.DataFrame(data={'Bc1':p_0[0],
                      'theta1':np.round(p_0[1],5),
                      'alfa1':np.round(p_0[2],5),
                      'beta1':np.round(p_0[3],5),
                      'I1':p_0[4],
                      'Bc2':p_0[5],
                      'theta2':np.round(p_0[6],5),
                      'alfa2':np.round(p_0[7],5),
                      'beta2':np.round(p_0[8],5),
                      'I2':p_0[9]},index=[0])

    df1=pd.DataFrame(data={'x1':x,
              'Ca+Cb':yt,
              'Ca':Ca,
              'Cb':Cb,
              'Data gradient':data})


    df.to_csv('inversion_parameters_'+str(sample_name)+'.csv', index=False)
    df1.to_csv('inversion_components_'+str(sample_name)+'.csv', index=False)

    euclidean_norm=np.linalg.norm(delta_d**2,2)

    return euclidean_norm,np.array(p_0)

def Levenberg_Marquardt_3C(function,u1,theta1,alfa1,beta1,I1,u2,theta2,alfa2,beta2,I2,u3,theta3,alfa3,beta3,I3,x,data,constrain,option,sample_name,eps=1e-15,maxiter=200):
    
    '''
    Computes an optimization procedure to invert (Levenberg_Marquardt) parameters of a Gamma-Cauchy exponential model with three ferromagnetic components.

    Parameters:
        function: the foward model for one ferromagnetic component;
        u1/u2/u3 : starting guess, the mean coercivity of the population, where the component peaks (float);
        theta1/theta2/theta3: starting guess, the dispersion parameter, suggested to be between 0 and 1 (float);
        alfa1/alfa2/alfa3,beta1/beta2/beta3: starting guess, skewness/kurtosis controling parameters (float);
        I1/I2/I3: starting guess, scaling factor (float);
        x: 1D-array of applied field values (float). 
        data: 1D-array of the gradient of the lower branched magnetic hysteresis (float);
        constrain: string, force minimization in around the starting guess for the coercivity (Y/N);
        sample_name: string, keeps the name of the sample in order to name the saved figures;
        eps: small value that is used both in the central differences calculation and for convergence criteria (float);
        maxiter: maximum number of iterations per inversion procedure (integer).



    returns:
        euclidean_norm: the squared error euclidean norm (float);
        p_0: array with the inverted parameters. 
    '''

    a=0.1
    damping=0.1 # dumping factor

    bubble_deltad=[] #empty array
    bubble_error=[] #empty array
    bubble_parameters=[] #empty array

    if constrain=='N' or constrain=='n':
        p_0=[u1,theta1,alfa1,beta1,I1,u2,theta2,alfa2,beta2,I2,u3,theta3,alfa3,beta3,I3] # array containing the parameters
        p_cor=np.zeros(np.shape(p_0)) # array of zeros that will hold the corrected parameters

         # array with the analytical measurement error


        #calculating Jacobian Matrix with the initial guesses
        grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
        grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
        grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
        grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
        grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
        grad_u2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]+(eps),p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]-(eps),p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
        grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
        grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7]+(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7]-(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
        grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8]+(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8]-(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
        grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9]+(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9]-(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
        grad_u3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10]+(eps),p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10]-(eps),p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
        grad_theta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]+(eps),p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]-(eps),p_0[12],p_0[13],p_0[14],x))/(2*(eps))
        grad_alfa3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]+(eps),p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]-(eps),p_0[13],p_0[14],x))/(2*(eps))
        grad_beta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]+(eps),p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]-(eps),p_0[14],x))/(2*(eps))
        grad_I3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
            p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]-(eps),x))/(2*(eps))

        J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1,
            grad_u2,grad_theta2,grad_alfa2,grad_beta2,grad_I2,
            grad_u3,grad_theta3,grad_alfa3,grad_beta3,grad_I3])

        #difference between observed and calculated data

        delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
            p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

        #parameters correction

        h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

        p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5]+h[5],
            p_0[6]+h[6],p_0[7]+h[7],p_0[8]+h[8],p_0[9]+h[9],p_0[10]+h[10],
            p_0[11]+h[11],p_0[12]+h[12],p_0[13]+h[13],p_0[14]+h[14]]

        error=np.sum((delta_d)**2)

        for i in range(maxiter):
            if i==0:
                bubble_error=error
                bubble_deltad=delta_d
                p_0=p_cor
                grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_u2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]+(eps),p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]-(eps),p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7]+(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7]-(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8]+(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8]-(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9]+(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9]-(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_u3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10]+(eps),p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10]-(eps),p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_theta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]+(eps),p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]-(eps),p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_alfa3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]+(eps),p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]-(eps),p_0[13],p_0[14],x))/(2*(eps))
                grad_beta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]+(eps),p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]-(eps),p_0[14],x))/(2*(eps))
                grad_I3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]-(eps),x))/(2*(eps))

                J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1,
                    grad_u2,grad_theta2,grad_alfa2,grad_beta2,grad_I2,
                    grad_u3,grad_theta3,grad_alfa3,grad_beta3,grad_I3])

                delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                    p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

                h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5]+h[5],
                    p_0[6]+h[6],p_0[7]+h[7],p_0[8]+h[8],p_0[9]+h[9],p_0[10]+h[10],
                    p_0[11]+h[11],p_0[12]+h[12],p_0[13]+h[13],p_0[14]+h[14]]

                error=np.sum((delta_d)**2)

                
                
                criteria=abs(bubble_error-error)

                if criteria>(eps):
                    p_0=p_cor
                    
                else:
                    p_cor=p_0
                    

                tolerance=np.linalg.norm(J.T@delta_d,2)


                if tolerance<=(eps) or i==maxiter: #testing the condition
                    break

            if ((i % 2) == 0) or (error>bubble_error):
                bubble_error=error
                bubble_deltad=delta_d
                p_0=p_cor
                grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_u2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]+(eps),p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]-(eps),p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7]+(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7]-(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8]+(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8]-(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9]+(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9]-(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_u3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10]+(eps),p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10]-(eps),p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_theta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]+(eps),p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]-(eps),p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                grad_alfa3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]+(eps),p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]-(eps),p_0[13],p_0[14],x))/(2*(eps))
                grad_beta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]+(eps),p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]-(eps),p_0[14],x))/(2*(eps))
                grad_I3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                    p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]-(eps),x))/(2*(eps))

                J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1,
                    grad_u2,grad_theta2,grad_alfa2,grad_beta2,grad_I2,
                    grad_u3,grad_theta3,grad_alfa3,grad_beta3,grad_I3])

                delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                    p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

                h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5]+h[5],
                    p_0[6]+h[6],p_0[7]+h[7],p_0[8]+h[8],p_0[9]+h[9],p_0[10]+h[10],
                    p_0[11]+h[11],p_0[12]+h[12],p_0[13]+h[13],p_0[14]+h[14]]

                error=np.sum((delta_d)**2)

                
                
                criteria=abs(bubble_error-error)

                if criteria>(eps):
                    wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                    damping=damping*(a**wip)
                    p_0=p_cor
                else:
                    p_cor=p_0
                    damping=a*damping
                    

                tolerance=np.linalg.norm(J.T@delta_d,2)

                if tolerance<=(eps) or i==maxiter: #testing the condition
                    break

            else:
                J=J
                delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                    p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

                h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5]+h[5],
                    p_0[6]+h[6],p_0[7]+h[7],p_0[8]+h[8],p_0[9]+h[9],p_0[10]+h[10],
                    p_0[11]+h[11],p_0[12]+h[12],p_0[13]+h[13],p_0[14]+h[14]]

                error=np.sum((delta_d)**2)

                
                

                criteria=abs(bubble_error-error)

                if criteria>(eps):
                    wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                    damping=damping*(a**wip)
                    p_0=p_cor
                else:
                    p_cor=p_0
                    damping=a*damping
                    

                tolerance=np.linalg.norm(J.T@delta_d,2)
                
                if tolerance<=(eps) or i==maxiter: #testing the condition
                    break
        
        p_0[0]=abs(p_0[0])
        p_0[1]=abs(p_0[1])
        p_0[2]=abs(p_0[2])
        p_0[3]=abs(p_0[3])
        p_0[4]=abs(p_0[4])
        p_0[5]=abs(p_0[5])
        p_0[6]=abs(p_0[6])
        p_0[7]=abs(p_0[7])
        p_0[8]=abs(p_0[8])
        p_0[9]=abs(p_0[9])
        p_0[10]=abs(p_0[10])
        p_0[11]=abs(p_0[11])
        p_0[12]=abs(p_0[12])
        p_0[13]=abs(p_0[13])
        p_0[14]=abs(p_0[14])


    if constrain=='Y' or constrain=='y':
        if option=="A":
            p_0=[u1,theta1,alfa1,beta1,I1,u2,theta2,alfa2,beta2,I2,u3,theta3,alfa3,beta3,I3] # array containing the parameters
            p_cor=np.zeros(np.shape(p_0)) # array of zeros that will hold the corrected parameters

             # array with the analytical measurement error
    

            #calculating Jacobian Matrix with the initial guesses
            grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_u2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]+(eps),p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]-(eps),p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7]+(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7]-(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8]+(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8]-(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9]+(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9]-(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_u3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10]+(eps),p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10]-(eps),p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_theta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]+(eps),p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]-(eps),p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_alfa3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]+(eps),p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]-(eps),p_0[13],p_0[14],x))/(2*(eps))
            grad_beta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]+(eps),p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]-(eps),p_0[14],x))/(2*(eps))
            grad_I3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]-(eps),x))/(2*(eps))

            J=np.column_stack([grad_theta1,grad_alfa1,grad_beta1,grad_I1,
                grad_u2,grad_theta2,grad_alfa2,grad_beta2,grad_I2,
                grad_u3,grad_theta3,grad_alfa3,grad_beta3,grad_I3])

            delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

            h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

            p_cor=[p_0[0],p_0[1]+h[0],p_0[2]+h[1],p_0[3]+h[2],p_0[4]+h[3],p_0[5]+h[4],
                p_0[6]+h[5],p_0[7]+h[6],p_0[8]+h[7],p_0[9]+h[8],p_0[10]+h[9],
                p_0[11]+h[10],p_0[12]+h[11],p_0[13]+h[12],p_0[14]+h[13]]


            error=np.sum((delta_d)**2)

            for i in range(maxiter):
                if i==0:
                    bubble_error=error
                    bubble_deltad=delta_d
                    p_0=p_cor
                    grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_u2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]+(eps),p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]-(eps),p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]+(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]-(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]+(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]-(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]+(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]-(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_u3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10]+(eps),p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10]-(eps),p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]+(eps),p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]-(eps),p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]+(eps),p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]-(eps),p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]+(eps),p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]-(eps),p_0[14],x))/(2*(eps))
                    grad_I3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]-(eps),x))/(2*(eps))

                    J=np.column_stack([grad_theta1,grad_alfa1,grad_beta1,grad_I1,
                        grad_u2,grad_theta2,grad_alfa2,grad_beta2,grad_I2,
                        grad_u3,grad_theta3,grad_alfa3,grad_beta3,grad_I3])

                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                        p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0],p_0[1]+h[0],p_0[2]+h[1],p_0[3]+h[2],p_0[4]+h[3],p_0[5]+h[4],
                        p_0[6]+h[5],p_0[7]+h[6],p_0[8]+h[7],p_0[9]+h[8],p_0[10]+h[9],
                        p_0[11]+h[10],p_0[12]+h[11],p_0[13]+h[12],p_0[14]+h[13]]


                    error=np.sum((delta_d)**2)

                    
                    
                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        p_0=p_cor
                        
                    else:
                        p_cor=p_0
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)


                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

                if ((i % 2) == 0) or (error>bubble_error):
                    bubble_error=error
                    bubble_deltad=delta_d
                    p_0=p_cor
                    grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_u2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]+(eps),p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]-(eps),p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]+(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]-(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]+(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]-(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]+(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]-(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_u3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10]+(eps),p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10]-(eps),p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]+(eps),p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]-(eps),p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]+(eps),p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]-(eps),p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]+(eps),p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]-(eps),p_0[14],x))/(2*(eps))
                    grad_I3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]-(eps),x))/(2*(eps))

                    J=np.column_stack([grad_theta1,grad_alfa1,grad_beta1,grad_I1,
                        grad_u2,grad_theta2,grad_alfa2,grad_beta2,grad_I2,
                        grad_u3,grad_theta3,grad_alfa3,grad_beta3,grad_I3])

                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                        p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0],p_0[1]+h[0],p_0[2]+h[1],p_0[3]+h[2],p_0[4]+h[3],p_0[5]+h[4],
                        p_0[6]+h[5],p_0[7]+h[6],p_0[8]+h[7],p_0[9]+h[8],p_0[10]+h[9],
                        p_0[11]+h[10],p_0[12]+h[11],p_0[13]+h[12],p_0[14]+h[13]]

                    error=np.sum((delta_d)**2)

                    
                    
                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                        damping=damping*(a**wip)
                        p_0=p_cor
                    else:
                        p_cor=p_0
                        damping=a*damping
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)

                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

                else:
                    J=J

                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                        p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0],p_0[1]+h[0],p_0[2]+h[1],p_0[3]+h[2],p_0[4]+h[3],p_0[5]+h[4],
                        p_0[6]+h[5],p_0[7]+h[6],p_0[8]+h[7],p_0[9]+h[8],p_0[10]+h[9],
                        p_0[11]+h[10],p_0[12]+h[11],p_0[13]+h[12],p_0[14]+h[13]]

                    
                    

                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                        damping=damping*(a**wip)
                        p_0=p_cor
                    else:
                        p_cor=p_0
                        damping=a*damping
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)
                    
                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break
            
            p_0[0]=abs(p_0[0])
            p_0[1]=abs(p_0[1])
            p_0[2]=abs(p_0[2])
            p_0[3]=abs(p_0[3])
            p_0[4]=abs(p_0[4])
            p_0[5]=abs(p_0[5])
            p_0[6]=abs(p_0[6])
            p_0[7]=abs(p_0[7])
            p_0[8]=abs(p_0[8])
            p_0[9]=abs(p_0[9])
            p_0[10]=abs(p_0[10])
            p_0[11]=abs(p_0[11])
            p_0[12]=abs(p_0[12])
            p_0[13]=abs(p_0[13])
            p_0[14]=abs(p_0[14])

        if option=="B":
            p_0=[u1,theta1,alfa1,beta1,I1,u2,theta2,alfa2,beta2,I2,u3,theta3,alfa3,beta3,I3] # array containing the parameters
            p_cor=np.zeros(np.shape(p_0)) # array of zeros that will hold the corrected parameters

             # array with the analytical measurement error
    

            #calculating Jacobian Matrix with the initial guesses
            grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7]+(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7]-(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8]+(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8]-(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9]+(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9]-(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_u3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10]+(eps),p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10]-(eps),p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_theta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]+(eps),p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]-(eps),p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_alfa3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]+(eps),p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]-(eps),p_0[13],p_0[14],x))/(2*(eps))
            grad_beta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]+(eps),p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]-(eps),p_0[14],x))/(2*(eps))
            grad_I3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]-(eps),x))/(2*(eps))

            J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1,
                grad_theta2,grad_alfa2,grad_beta2,grad_I2,
                grad_u3,grad_theta3,grad_alfa3,grad_beta3,grad_I3])

            delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

            h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

            p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5],
                p_0[6]+h[5],p_0[7]+h[6],p_0[8]+h[7],p_0[9]+h[8],p_0[10]+h[9],
                p_0[11]+h[10],p_0[12]+h[11],p_0[13]+h[12],p_0[14]+h[13]]

            error=np.sum((delta_d)**2)

            for i in range(maxiter):
                if i==0:
                    bubble_error=error
                    bubble_deltad=delta_d
                    p_0=p_cor
                    grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]+(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]-(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]+(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]-(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]+(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]-(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_u3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10]+(eps),p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10]-(eps),p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]+(eps),p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]-(eps),p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]+(eps),p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]-(eps),p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]+(eps),p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]-(eps),p_0[14],x))/(2*(eps))
                    grad_I3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]-(eps),x))/(2*(eps))

                    J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1,
                        grad_theta2,grad_alfa2,grad_beta2,grad_I2,
                        grad_u3,grad_theta3,grad_alfa3,grad_beta3,grad_I3])

                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                        p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5],
                        p_0[6]+h[5],p_0[7]+h[6],p_0[8]+h[7],p_0[9]+h[8],p_0[10]+h[9],
                        p_0[11]+h[10],p_0[12]+h[11],p_0[13]+h[12],p_0[14]+h[13]]

                    error=np.sum((delta_d)**2)

                    
                    
                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        p_0=p_cor
                        
                    else:
                        p_cor=p_0
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)


                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

                if ((i % 2) == 0) or (error>bubble_error):
                    bubble_error=error
                    bubble_deltad=delta_d
                    p_0=p_cor
                    grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]+(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]-(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]+(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]-(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]+(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]-(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_u3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10]+(eps),p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10]-(eps),p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]+(eps),p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]-(eps),p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]+(eps),p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]-(eps),p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]+(eps),p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]-(eps),p_0[14],x))/(2*(eps))
                    grad_I3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]-(eps),x))/(2*(eps))

                    J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1,
                        grad_theta2,grad_alfa2,grad_beta2,grad_I2,
                        grad_u3,grad_theta3,grad_alfa3,grad_beta3,grad_I3])

                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                        p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5],
                        p_0[6]+h[5],p_0[7]+h[6],p_0[8]+h[7],p_0[9]+h[8],p_0[10]+h[9],
                        p_0[11]+h[10],p_0[12]+h[11],p_0[13]+h[12],p_0[14]+h[13]]


                    error=np.sum((delta_d)**2)

                    
                    
                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                        damping=damping*(a**wip)
                        p_0=p_cor
                    else:
                        p_cor=p_0
                        damping=a*damping
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)

                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

                else:
                    J=J
                    
                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                        p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5],
                        p_0[6]+h[5],p_0[7]+h[6],p_0[8]+h[7],p_0[9]+h[8],p_0[10]+h[9],
                        p_0[11]+h[10],p_0[12]+h[11],p_0[13]+h[12],p_0[14]+h[13]]

                    error=np.sum((delta_d)**2)

                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                        damping=damping*(a**wip)
                        p_0=p_cor
                    else:
                        p_cor=p_0
                        damping=a*damping
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)
                    
                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

            p_0[0]=abs(p_0[0])
            p_0[1]=abs(p_0[1])
            p_0[2]=abs(p_0[2])
            p_0[3]=abs(p_0[3])
            p_0[4]=abs(p_0[4])
            p_0[5]=abs(p_0[5])
            p_0[6]=abs(p_0[6])
            p_0[7]=abs(p_0[7])
            p_0[8]=abs(p_0[8])
            p_0[9]=abs(p_0[9])
            p_0[10]=abs(p_0[10])
            p_0[11]=abs(p_0[11])
            p_0[12]=abs(p_0[12])
            p_0[13]=abs(p_0[13])
            p_0[14]=abs(p_0[14])

        if option=="C":
            p_0=[u1,theta1,alfa1,beta1,I1,u2,theta2,alfa2,beta2,I2,u3,theta3,alfa3,beta3,I3] # array containing the parameters
            p_cor=np.zeros(np.shape(p_0)) # array of zeros that will hold the corrected parameters

             # array with the analytical measurement error
    

            #calculating Jacobian Matrix with the initial guesses
            grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_u2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]+(eps),p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]-(eps),p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7]+(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7]-(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8]+(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8]-(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9]+(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9]-(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_theta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]+(eps),p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]-(eps),p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_alfa3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]+(eps),p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]-(eps),p_0[13],p_0[14],x))/(2*(eps))
            grad_beta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]+(eps),p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]-(eps),p_0[14],x))/(2*(eps))
            grad_I3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]-(eps),x))/(2*(eps))

            J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1,
                grad_u2,grad_theta2,grad_alfa2,grad_beta2,grad_I2,
                grad_theta3,grad_alfa3,grad_beta3,grad_I3])

            #difference between observed and calculated data

            delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

            #parameters correction

            h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

            p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5]+h[5],
                p_0[6]+h[6],p_0[7]+h[7],p_0[8]+h[8],p_0[9]+h[9],p_0[10],
                p_0[11]+h[10],p_0[12]+h[11],p_0[13]+h[12],p_0[14]+h[13]]

            error=np.sum((delta_d)**2)

            for i in range(maxiter):
                if i==0:
                    bubble_error=error
                    bubble_deltad=delta_d
                    p_0=p_cor
                    grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_u2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]+(eps),p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]-(eps),p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]+(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]-(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]+(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]-(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]+(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]-(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]+(eps),p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]-(eps),p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]+(eps),p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]-(eps),p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]+(eps),p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]-(eps),p_0[14],x))/(2*(eps))
                    grad_I3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]-(eps),x))/(2*(eps))

                    J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1,
                        grad_u2,grad_theta2,grad_alfa2,grad_beta2,grad_I2,
                        grad_theta3,grad_alfa3,grad_beta3,grad_I3])

                    #difference between observed and calculated data

                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                        p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

                    #parameters correction

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5]+h[5],
                        p_0[6]+h[6],p_0[7]+h[7],p_0[8]+h[8],p_0[9]+h[9],p_0[10],
                        p_0[11]+h[10],p_0[12]+h[11],p_0[13]+h[12],p_0[14]+h[13]]

                    error=np.sum((delta_d)**2)

                    
                    
                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        p_0=p_cor
                        
                    else:
                        p_cor=p_0
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)


                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

                if ((i % 2) == 0) or (error>bubble_error):
                    bubble_error=error
                    bubble_deltad=delta_d
                    p_0=p_cor
                    grad_u1=(function(p_0[0]+(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0]-(eps),p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_u2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]+(eps),p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5]-(eps),p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]+(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]-(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]+(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]-(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]+(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]-(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]+(eps),p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]-(eps),p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]+(eps),p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]-(eps),p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]+(eps),p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]-(eps),p_0[14],x))/(2*(eps))
                    grad_I3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]-(eps),x))/(2*(eps))

                    J=np.column_stack([grad_u1,grad_theta1,grad_alfa1,grad_beta1,grad_I1,
                        grad_u2,grad_theta2,grad_alfa2,grad_beta2,grad_I2,
                        grad_theta3,grad_alfa3,grad_beta3,grad_I3])

                    #difference between observed and calculated data

                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                        p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

                    #parameters correction

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5]+h[5],
                        p_0[6]+h[6],p_0[7]+h[7],p_0[8]+h[8],p_0[9]+h[9],p_0[10],
                        p_0[11]+h[10],p_0[12]+h[11],p_0[13]+h[12],p_0[14]+h[13]]

                    error=np.sum((delta_d)**2)

                    
                    
                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                        damping=damping*(a**wip)
                        p_0=p_cor
                    else:
                        p_cor=p_0
                        damping=a*damping
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)

                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

                else:
                    J=J
                    
                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                        p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

                    #parameters correction

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0]+h[0],p_0[1]+h[1],p_0[2]+h[2],p_0[3]+h[3],p_0[4]+h[4],p_0[5]+h[5],
                        p_0[6]+h[6],p_0[7]+h[7],p_0[8]+h[8],p_0[9]+h[9],p_0[10],
                        p_0[11]+h[10],p_0[12]+h[11],p_0[13]+h[12],p_0[14]+h[13]]

                    error=np.sum((delta_d)**2)

                    
                    

                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                        damping=damping*(a**wip)
                        p_0=p_cor
                    else:
                        p_cor=p_0
                        damping=a*damping
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)
                    
                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

            p_0[0]=abs(p_0[0])
            p_0[1]=abs(p_0[1])
            p_0[2]=abs(p_0[2])
            p_0[3]=abs(p_0[3])
            p_0[4]=abs(p_0[4])
            p_0[5]=abs(p_0[5])
            p_0[6]=abs(p_0[6])
            p_0[7]=abs(p_0[7])
            p_0[8]=abs(p_0[8])
            p_0[9]=abs(p_0[9])
            p_0[10]=abs(p_0[10])
            p_0[11]=abs(p_0[11])
            p_0[12]=abs(p_0[12])
            p_0[13]=abs(p_0[13])
            p_0[14]=abs(p_0[14])

        if option=="D":
            p_0=[u1,theta1,alfa1,beta1,I1,u2,theta2,alfa2,beta2,I2,u3,theta3,alfa3,beta3,I3] # array containing the parameters
            p_cor=np.zeros(np.shape(p_0)) # array of zeros that will hold the corrected parameters

             # array with the analytical measurement error
    

            #calculating Jacobian Matrix with the initial guesses
            grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7]+(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7]-(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8]+(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8]-(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9]+(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9]-(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_theta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]+(eps),p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]-(eps),p_0[12],p_0[13],p_0[14],x))/(2*(eps))
            grad_alfa3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]+(eps),p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]-(eps),p_0[13],p_0[14],x))/(2*(eps))
            grad_beta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]+(eps),p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]-(eps),p_0[14],x))/(2*(eps))
            grad_I3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]-(eps),x))/(2*(eps))

            J=np.column_stack([grad_theta1,grad_alfa1,grad_beta1,grad_I1,grad_theta2,
                grad_alfa2,grad_beta2,grad_I2,grad_theta3,grad_alfa3,grad_beta3,grad_I3])

            #difference between observed and calculated data

            delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

            #parameters correction

            h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

            p_cor=[p_0[0],p_0[1]+h[0],p_0[2]+h[1],p_0[3]+h[2],p_0[4]+h[3],p_0[5],
                p_0[6]+h[4],p_0[7]+h[5],p_0[8]+h[6],p_0[9]+h[7],p_0[10],
                p_0[11]+h[8],p_0[12]+h[9],p_0[13]+h[10],p_0[14]+h[11]]

            error=np.sum((delta_d)**2)

            for i in range(maxiter):
                if i==0:
                    bubble_error=error
                    bubble_deltad=delta_d
                    p_0=p_cor
                    grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]+(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]-(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]+(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]-(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]+(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]-(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]+(eps),p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]-(eps),p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]+(eps),p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]-(eps),p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]+(eps),p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]-(eps),p_0[14],x))/(2*(eps))
                    grad_I3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]-(eps),x))/(2*(eps))

                    J=np.column_stack([grad_theta1,grad_alfa1,grad_beta1,grad_I1,grad_theta2,
                        grad_alfa2,grad_beta2,grad_I2,grad_theta3,grad_alfa3,grad_beta3,grad_I3])

                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                        p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0],p_0[1]+h[0],p_0[2]+h[1],p_0[3]+h[2],p_0[4]+h[3],p_0[5],
                        p_0[6]+h[4],p_0[7]+h[5],p_0[8]+h[6],p_0[9]+h[7],p_0[10],
                        p_0[11]+h[8],p_0[12]+h[9],p_0[13]+h[10],p_0[14]+h[11]]

                    error=np.sum((delta_d)**2)

                    
                    
                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        p_0=p_cor
                        
                    else:
                        p_cor=p_0
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)


                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

                if ((i % 2) == 0) or (error>bubble_error):
                    bubble_error=error
                    bubble_deltad=delta_d
                    p_0=p_cor
                    grad_theta1=(function(p_0[0],p_0[1]+(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1]-(eps),p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa1=(function(p_0[0],p_0[1],p_0[2]+(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2]-(eps),p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta1=(function(p_0[0],p_0[1],p_0[2],p_0[3]+(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3]-(eps),p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_I1=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]+(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4]-(eps),p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]+(eps),
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6]-(eps),
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]+(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7]-(eps),p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]+(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8]-(eps),p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_I2=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]+(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9]-(eps),p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_theta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]+(eps),p_0[12],p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11]-(eps),p_0[12],p_0[13],p_0[14],x))/(2*(eps))
                    grad_alfa3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]+(eps),p_0[13],p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12]-(eps),p_0[13],p_0[14],x))/(2*(eps))
                    grad_beta3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]+(eps),p_0[14],x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13]-(eps),p_0[14],x))/(2*(eps))
                    grad_I3=(function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]+(eps),x)-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],
                        p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14]-(eps),x))/(2*(eps))

                    J=np.column_stack([grad_theta1,grad_alfa1,grad_beta1,grad_I1,grad_theta2,
                        grad_alfa2,grad_beta2,grad_I2,grad_theta3,grad_alfa3,grad_beta3,grad_I3])

                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                        p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0],p_0[1]+h[0],p_0[2]+h[1],p_0[3]+h[2],p_0[4]+h[3],p_0[5],
                        p_0[6]+h[4],p_0[7]+h[5],p_0[8]+h[6],p_0[9]+h[7],p_0[10],
                        p_0[11]+h[8],p_0[12]+h[9],p_0[13]+h[10],p_0[14]+h[11]]


                    error=np.sum((delta_d)**2)

                    
                    
                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                        damping=damping*(a**wip)
                        p_0=p_cor
                    else:
                        p_cor=p_0
                        damping=a*damping
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)

                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

                else:
                    J=J
                    delta_d=data-function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],
                        p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)

                    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

                    p_cor=[p_0[0],p_0[1]+h[0],p_0[2]+h[1],p_0[3]+h[2],p_0[4]+h[3],p_0[5],
                        p_0[6]+h[4],p_0[7]+h[5],p_0[8]+h[6],p_0[9]+h[7],p_0[10],
                        p_0[11]+h[8],p_0[12]+h[9],p_0[13]+h[10],p_0[14]+h[11]]

                    error=np.sum((delta_d)**2)

                    
                    

                    criteria=abs(bubble_error-error)

                    if criteria>(eps):
                        wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                        damping=damping*(a**wip)
                        p_0=p_cor
                    else:
                        p_cor=p_0
                        damping=a*damping
                        

                    tolerance=np.linalg.norm(J.T@delta_d,2)
                    
                    if tolerance<=(eps) or i==maxiter: #testing the condition
                        break

            p_0[0]=abs(p_0[0])
            p_0[1]=abs(p_0[1])
            p_0[2]=abs(p_0[2])
            p_0[3]=abs(p_0[3])
            p_0[4]=abs(p_0[4])
            p_0[5]=abs(p_0[5])
            p_0[6]=abs(p_0[6])
            p_0[7]=abs(p_0[7])
            p_0[8]=abs(p_0[8])
            p_0[9]=abs(p_0[9])
            p_0[10]=abs(p_0[10])
            p_0[11]=abs(p_0[11])
            p_0[12]=abs(p_0[12])
            p_0[13]=abs(p_0[13])
            p_0[14]=abs(p_0[14])


    yt=function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)
    Ca=function(p_0[0],p_0[1],p_0[2],p_0[3],p_0[4],p_0[5],p_0[6],p_0[7],p_0[8],0,p_0[10],p_0[11],p_0[12],p_0[13],0,x)
    Cb=function(p_0[0],p_0[1],p_0[2],p_0[3],0,p_0[5],p_0[6],p_0[7],p_0[8],p_0[9],p_0[10],p_0[11],p_0[12],p_0[13],0,x)
    Cc=function(p_0[0],p_0[1],p_0[2],p_0[3],0,p_0[5],p_0[6],p_0[7],p_0[8],0,p_0[10],p_0[11],p_0[12],p_0[13],p_0[14],x)
    df=pd.DataFrame(data={'Bc1':p_0[0],
                      'theta1':np.round(p_0[1],5),
                      'alfa1':np.round(p_0[2],5),
                      'beta1':np.round(p_0[3],5),
                      'I1':p_0[4],
                      'Bc2':p_0[5],
                      'theta2':np.round(p_0[6],5),
                      'alfa2':np.round(p_0[7],5),
                      'beta2':np.round(p_0[8],5),
                      'I2':p_0[9],
                      'Bc3':p_0[10],
                      'theta3':np.round(p_0[11],5),
                      'alfa3':np.round(p_0[12],5),
                      'beta3':np.round(p_0[13],5),
                      'I3':p_0[14]},index=[0])

    df1=pd.DataFrame(data={'x1':x,
              'Ca+Cb+Cc':yt,
              'Ca':Ca,
              'Cb':Cb,
              'Cc':Cc,
              'Data gradient':data})


    df.to_csv('inversion_parameters_'+str(sample_name)+'.csv', index=False)
    df1.to_csv('inversion_components_'+str(sample_name)+'.csv', index=False)

    euclidean_norm=np.linalg.norm(delta_d**2,2)
    return euclidean_norm,np.array(p_0)


def high_field(lamb,phi,x):
    '''
    Computes a modified high-field saturation approach (susceptibility).

    Parameters:
        lamb: linear coefficient, a float ;
        phi : power law coefficient, a float ;
        x: 1D-array of applied field values (float).


    returns:
        y: 1D-array values (float). 
    '''
    y=lamb*phi*(x**(phi-1))

    return y


def Levenberg_Marquardt_HF(function,lamb,phi,x,data,eps=1e-7,maxiter=200):

    '''
    Computes an optimization procedure to invert (Levenberg_Marquardt) parameters of a modified high-field saturation approach

    Parameters:
        function: the foward model for one ferromagnetic component;
        lamb: starting guess, the linear coefficient (float) ;
        phi : starting guess, the power law coefficient (float) ;
        x: 1D-array of applied field values (float).
        data: 1D-array of an inverted ferromagnetic component (float);
        eps: small value that is used both in the central differences calculation and for convergence criteria (float);
        maxiter: maximum number of iterations per inversion procedure (integer).



    returns:
        euclidean_norm: the squared error euclidean norm (float);
        p_0: array with the inverted parameters. 

    '''

    a=0.2  # decreasing dumping factor rate
    b=0.2 # increasing dumping factor rate
    damping=0.1 # dumping factor

    bubble_deltad=[] #empty array
    bubble_error=[] #empty array
    bubble_parameters=[] #empty array
    p_0=[lamb,phi]# array containing the parameters
    p_cor=np.zeros(np.shape(p_0)) # array of zeros that will hold the corrected parameters
    
     # weighted diagonal matrix
    #calculating Jacobian Matrix with the initial guesses
    
    grad_lamb=(function(p_0[0]+(eps),p_0[1],x)-function(p_0[0]-(eps),p_0[1],x))/(2*(eps))
    grad_phi=(function(p_0[0],p_0[1]+(eps),x)-function(p_0[0],p_0[1]-(eps),x))/(2*(eps))
    J=np.column_stack([grad_lamb,grad_phi])

    delta_d=data-function(p_0[0],p_0[1],x)

    h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

    p_cor=[p_0[0]+h[0],p_0[1]+h[1]]

    error=np.sum((delta_d)**2)

    for i in range(maxiter):
        if i==0:
            bubble_error=error
            bubble_deltad=delta_d
            p_0=p_cor
            grad_lamb=(function(p_0[0]+(eps),p_0[1],x)-function(p_0[0]-(eps),p_0[1],x))/(2*(eps))
            grad_phi=(function(p_0[0],p_0[1]+(eps),x)-function(p_0[0],p_0[1]-(eps),x))/(2*(eps))
            J=np.column_stack([grad_lamb,grad_phi])

            delta_d=data-function(p_0[0],p_0[1],x)

            h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

            p_cor=[p_0[0]+h[0],p_0[1]+h[1]]

            error=np.sum((delta_d)**2)

            criteria=abs(bubble_error-error)

            if criteria>(eps):
                p_0=p_cor

            else:
                p_cor=p_0


            tolerance=np.linalg.norm(J.T@delta_d,2)

            if tolerance<=(eps) or i==maxiter: #testing the condition
                break

        if ((i % 2) == 0) or (error>bubble_error):
            bubble_error=error
            bubble_deltad=delta_d
            p_0=p_cor
            grad_lamb=(function(p_0[0]+(eps),p_0[1],x)-function(p_0[0]-(eps),p_0[1],x))/(2*(eps))
            grad_phi=(function(p_0[0],p_0[1]+(eps),x)-function(p_0[0],p_0[1]-(eps),x))/(2*(eps))
            J=np.column_stack([grad_lamb,grad_phi])

            delta_d=data-function(p_0[0],p_0[1],x)

            h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

            p_cor=[p_0[0]+h[0],p_0[1]+h[1]]

            error=np.sum((delta_d)**2)

            criteria=abs(bubble_error-error)

            if criteria>(eps):
                wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                damping=damping*(a**wip)
                p_0=p_cor
            else:
                p_cor=p_0
                damping=a*damping


            tolerance=np.linalg.norm(J.T@delta_d,2)

            if tolerance<=(eps) or i==maxiter: #testing the condition
                break

        else:
            J=J
            delta_d=data-function(p_0[0],p_0[1],x)

            h=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@(J.T)@delta_d

            p_cor=[p_0[0]+h[0],p_0[1]+h[1]]

            error=np.sum((delta_d)**2)

            criteria=abs(bubble_error-error)


            if criteria>(eps):
                wip=np.dot(np.array(p_cor),np.array(p_0))/(np.linalg.norm(p_cor)*np.linalg.norm(p_0))
                damping=damping*(a**wip)
                p_0=p_cor
            else:
                p_cor=p_0
                damping=a*damping

            tolerance=np.linalg.norm(J.T@delta_d,2)

            if tolerance<=(eps) or i==maxiter: #testing the condition
                break


    euclidean_norm=np.linalg.norm(delta_d**2,2)

    return euclidean_norm,np.array(p_0)

def R2_calc(y,yc):
    '''
    Calculates the determination coefficient (R²) of an inverted model in comparison with the real data. 

    Parameters:
        y: 1D-array of the observed data (float);
        yc: 1D-array of an inverted model (float);

    returns:
        R2: the determination coefficient (float)

    '''
    SQR=np.sum((y-yc)**2)
    SQT=np.sum((y-np.mean(y))**2)
    R2=1-(SQR/SQT)
    
    return R2



def chi_squared(observed,calculated,parameter):

    '''
    Calculates the chi-squared value of an inverted model in comparison with the real data. 

    Parameters:
        observed: 1D-array of the observed data (float);
        calculated: 1D-array of an inverted model (float);

    returns:
        chi: the reduced chi-squared statistic of the final inverted model (float)

    '''

    chi=np.sum((observed-calculated)**2)/(np.size(observed)-np.size(parameter))
    
    return chi

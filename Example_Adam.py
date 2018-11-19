import math
import numpy as np
import matplotlib.pyplot as plt
alpha = 0.01
beta_1 = 0.9
beta_2 = 0.999						
epsilon = 1e-8
def func(x):
	return x*x + 10* math.sin(x)
def grad_func(x):					
	return 2*x +10*math.cos(x)
theta_0 = 0						
m_t = 0 
v_t = 0 
t = 0
theta_list = []

while (1):					
    t+=1
    g_t = grad_func(theta_0)		#computes the gradient of the stochastic function
    m_t = beta_1*m_t + (1-beta_1)*g_t	#updates the moving averages of the gradient
    v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)	#updates the moving averages of the squared gradient
    m_cap = m_t/(1-(beta_1**t))		#calculates the bias-corrected estimates
    v_cap = v_t/(1-(beta_2**t))		#calculates the bias-corrected estimates
    theta_0_prev = theta_0								
    theta_0 = theta_0 - (alpha*m_cap)/(math.sqrt(v_cap)+epsilon)	#updates the parameters
    theta_list.append(theta_0)
    if(theta_0 == theta_0_prev):		#checks if it is converged or not
        break
Set_test = np.arange(len(theta_list),dtype='i')
plt.plot(theta_list)
plt.show()
	

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 22:07:28 2017

A pendulum is constructed from a bob of mass m connected to a rod
that acts as a stiff spring with spring constant k.  Motion is in
the xy plane.  RK4 will be used to model the bob's motion.

Date last modified: 3/27/17          
@author: NEWCOMERGA1
"""

import numpy as np
import matplotlib.pyplot as plt

def pendul(s, t, param):
    '''
    Returns right-hand side of spring pendulum equations, used for RK4
      Inputs
        s      State vector [r  v  phi  w]
        t      Time (not used)
        param   Parameters: mass, length, spring constant, gravity  
      Output
        deriv  Derivatives [dr/dt dv/dt dphi/dt dw/dt]
    '''
    # Constants and variables
    m = param[0]
    L = param[1]
    k = param[2]
    g = param[3]
    
    r = s[0]
    v = s[1]
    phi = s[2]
    w = s[3]
    
    v_accel = r*w**2 - k/m*(r-L)
    w_accel = -(L*g/(r**2)*np.sin(phi) + 2*v/r*w)
    
    #  Return derivatives
    deriv = np.array([v, v_accel, w, w_accel])
    return deriv

    
def rk4(x, t, tau, derivsRK, param):
    """
    Runge-Kutta integrator (4th order)
    Inputs
      x          Array of current values of dependent variables
      t          Independent variable (usually time)
      tau        Step size (usually time step)
      derivsRK   Right hand side of the ODE; derivsRK is the
                 name of the function which returns dx/dt
                 Calling format derivsRK(x,t,param).
      param      Extra parameters passed to derivsRK
    Output
      x          New value of x after a step of size tau
    """

    # Evaluate F1 = f(x,t).
    F1 = derivsRK(x, t, param)  

    # Evaluate F2 = f( x+tau*F1/2, t+tau/2 )
    half_tau = 0.5*tau
    t_half = t + half_tau

    xtemp = x +half_tau*F1
    F2 = derivsRK(xtemp, t_half, param)  
    
    # Evaluate F3 = f( x+tau*F2/2, t+tau/2 )
    xtemp = x + half_tau*F2
    F3 = derivsRK(xtemp, t_half, param)
    
    # Evaluate F4 = f( x+tau*F3, t+tau )
    t_full = t + tau
    xtemp = x + tau*F3
    F4 = derivsRK(xtemp, t_full, param)
    
    # Return x(t+tau) computed from fourth-order R-K
    x += tau/6.*(F1 + 2.*(F2+F3) + F4)

    return x
    

    
print(    '''
    Models motion of a pendulum bob attached to a stiff spring of constant
    k with the fourth order Runge-Kutta method.
    '''
)
# Set constants
pi = np.pi
m = 0.1         # mass (kg)
L = 1.0         # length of rod (m)
k = 10**1       # spring constant (N/m)
g = 9.81        # graviational constant (m/s^2)
time = 0.0      # (s)
param = [m, L, k, g]    # list of parameters


# Setting initial values
phi0 = pi/180 * float(input("Enter initial angle(degrees): "))
r0 = L*1.5
w0 = 0.0             # initial angular velocity
v0 = 0.5             # initial radial velocity

state = np.array([r0, v0, phi0, w0])        # state vector


# Time step
nStep = int(input("Enter number of steps: "))
tau = float(input("Enter time step(s): "))
flag = 0

# Lists for plotting
r_plot = []             # radial values
phi_plot = []           # angular values
x_plot = []             # x position
y_plot = []             # y position
t_plot = []             # time

# Running RK4
for iStep in range(nStep):
    r = state[0]
    phi = state[2]
    
    # storing values
    r_plot.append(r)
    phi_plot.append(phi)
    x_plot.append(r*np.sin(phi))
    y_plot.append(-r*np.cos(phi))
    t_plot.append(time)
    
    # calculating next state
    phiold = phi
    state = rk4(state, time, tau, pendul, param)
    phi = state[2]
        
    # incrementing time
    time += tau    


# Output
print("\nWith constants:")
print("  m = "+str(m)+" kg")
print("  L = "+str(L)+" m")
print("  k = "+str(k)+" N/m")
print("The largest distance stretched was: "+str(np.max(r_plot)-L)+" m")


# Plotting cartesian motion, radial motion, and angular motion
curr_pos = -1
def updateChart():
    global ax1, fig1
    
    ax1.cla()
    
    if curr_pos == -1:
        ax1.plot(x_plot, y_plot)
        ax1.set_title("Spring Pendulum Motion")
        ax1.set_xlabel("x")
        ax1.set_xlim(np.min(x_plot),np.max(x_plot))
        ax1.set_ylabel("y")
        ax1.set_ylim(np.min(y_plot),-np.min(y_plot)*.01)
        ax1.grid(True)
            # Drawing pendulum and data
        ax1.plot([0,0],[0,np.min(y_plot)], color='k', linestyle='-', linewidth=2)
        ax1.plot([-L/2,L/2],[0,0], color='k', linestyle='-', linewidth=2)
        ax1.plot([0,r0*np.sin(phi0)],[0,-r0*np.cos(phi0)], color='r', linestyle='-')
    else:
        ax1.scatter(x_plot[curr_pos], y_plot[curr_pos], s=5, color='r')
        ax1.plot(x_plot[0:curr_pos], y_plot[0:curr_pos], lw=1, color='blue')
        ax1.set_title("Spring Pendulum Motion ({}/{})".format(curr_pos+1, len(x_plot)))
        ax1.set_xlabel("x")
        ax1.set_xlim(np.min(x_plot),np.max(x_plot))
        ax1.set_ylabel("y")
        #ax1.set_ylim(np.min(y_plot),-np.min(y_plot)*.01)
        ax1.grid(True)
            # Drawing pendulum and data
        ax1.plot([0,0],[0,np.min(y_plot)], color='k', linestyle='-', linewidth=2)
        ax1.plot([-L/2,L/2],[0,0], color='k', linestyle='-', linewidth=2)
        ax1.plot([0,x_plot[curr_pos]],[0,y_plot[curr_pos]], color='r', linestyle='-')
        
    
    str_phi = str(round(phi0*180/pi,2))
    str_k = str(k)
    str_L = str(L) 
    ax1.text(np.min(x_plot)*.90, np.min(y_plot)*.17,        # constants data
             r'$\phi_0$ = '+str_phi+r'$^o$'+'\n'
             +'k  = '+str_k+r' N/m'+'\n'
             +r'L$_0$ = '+str_L+' m', 
             style='italic', bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

    fig1.canvas.draw()
    
def press(e):
    global curr_pos
    
    if e.key == 'left':
        curr_pos -= 10
    elif e.key == 'right':
        curr_pos += 10
    else:
        return
    
    curr_pos = curr_pos % len(x_plot)

    updateChart()

fig1 = plt.figure(1, facecolor="0.98")
fig1.canvas.mpl_connect('key_press_event', press)
#fig1.clf()
ax1 = fig1.add_subplot(111)
updateChart()
#plt.show()

#fig2 = plt.figure(2, facecolor="0.98")
#fig2.clf()
#ax2 = fig2.add_subplot(111)
#ax2.plot(t_plot, r_plot)
#ax2.set_title("Radial Motion")
#ax2.set_xlabel("t")
#ax2.set_xlim(0.0, quart_per*6)
#ax2.set_xticks(np.arange(min(t_plot), max(t_plot), quart_per))
#ax2.set_ylabel("r")
#ax2.grid(True)
##plt.show()
#
#fig3 = plt.figure(3, facecolor="0.98")
#fig3.clf()
#ax3 = fig3.add_subplot(111)
#ax3.plot(t_plot, phi_plot)
#ax3.set_title("Angular Motion")
#ax3.set_xlabel("t")
#ax3.set_xlim(0, quart_per*6)
#ax3.set_xticks(np.arange(min(t_plot), max(t_plot), quart_per))
#ax3.set_ylabel(r"$\phi$")
#ax3.grid(True)
##plt.show()



    
    
    
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 22:07:28 2017

HW 5.2 -  A pendulum is constructed from a bob of mass m connected to a rod
          that acts as a stiff spring with spring constant k.  Motion is in
          the xy plane.  RK4 will be used to model the bob's motion.

Date last modified: 3/27/17          
@author: NEWCOMERGA1
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def pendulumBoard(s, t, param):
    '''
    Returns right-hand side of board pendulum equations, used for RK4
      Inputs
        s      State vector [r  v  phi  w]
        t      Time (not used)
        param   Parameters: mass, length, board mass, gravity  
      Output
        deriv  Derivatives [dr/dt dv/dt dphi/dt dw/dt]
    '''
    # Constants and variables
    m = param[0]
    L = param[1]
    M = param[2]
    g = param[3]
    
    r = s[0]
    v = s[1]
    phi = s[2]
    w = s[3]
    
    v_accel = m*np.sin(phi)/(m*(np.sin(phi)**2)+M) * (L*w**2 + g*np.cos(phi))
    w_accel = -np.sin(phi)/(m*(np.sin(phi)**2)+M) * (g/L*(m+M) + m*w**2*np.cos(phi))
    
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
    Models motion of a pendulum bob attached at the end to a board that can 
    translate freely in the X axis with the fourth order Runge-Kutta method.
    
    Motion Plot::   -Use R/L arrow keys to step forward/backward in time
                    -Use U/D arrow keys to toggle motion plot and parameter plot
    '''
)
# Set constants
pi = np.pi
m = 0.1         # mass (kg)
L = 1.0         # length of rod (m)
M = 0.1          # mass of the board (kg)
g = 9.81        # graviational constant (m/s^2)
time = 0.0      # (s)
param = [m, L, M, g]    # list of parameters


# Setting initial values
phi0 = pi/180 * 182
r0 = 0               # initial position of the board
w0 = 0.1             # initial angular velocity of the bob
v0 = -0.1             # initial velocity of the board

state = np.array([r0, v0, phi0, w0])        # state vector


# Time step
nStep = 1000    # number of steps
tau = 0.01       # time step (s)

# Lists for plotting
phi_plot = []           # angular values
x_plot = []             # x position of the bob
y_plot = []             # y position of the bob
X_plot = []             # x position of the board
Y_plot = []             # y position of the board
t_plot = []             # time

# Running RK4
for iStep in range(nStep):
    r = state[0]        # position of the board
    phi = state[2]      # angular position of the bob
    
    # storing values
    phi_plot.append(phi * 180/pi)
    x_plot.append(L*np.sin(phi) + r)
    y_plot.append(-L*np.cos(phi))
    X_plot.append(r)
    Y_plot.append(0)
    t_plot.append(time)
    
    # calculating next state
    state = rk4(state, time, tau, pendulumBoard, param)
        
    # incrementing time
    time += tau    


# Output
print("\nWith constants:")
print("  m = {} kg".format(m))
print("  L = {} m".format(L))
print("  M = {} kg".format(M))
print("Initial Conditions:")
print("  phi = {} deg".format(phi0*180/pi))
print("  w0 = {} deg/s".format(w0 * 180/pi))
print("  v0 = {} m/s".format(v0))


# Plotting cartesian motion, radial motion, and angular motion
curr_pos = -1
plotMotion = True
plotTrace = False
def updateChart():
    global ax1, fig1
    
    if plotMotion:
        if not ax1.get_visible():
            ax1.set_visible(True)
        if ax2.get_visible():
            ax2.set_visible(False)
        if ax3.get_visible():
            ax3.set_visible(False)
    
        ax1.cla()

        minX = np.min([np.min(x_plot), np.min(X_plot)])
        maxX = np.max([np.max(x_plot), np.max(X_plot)])
        
        ax1.scatter(x_plot[curr_pos], y_plot[curr_pos], s=5, color='r')
        ax1.plot(x_plot[0:curr_pos], y_plot[0:curr_pos], lw=1, color='blue')
        ax1.set_title("Spring Pendulum Motion ({}/{})".format(curr_pos+1, len(x_plot)))
        ax1.set_xlabel("x")
        ax1.set_xlim(minX, maxX)
        ax1.set_ylabel("y")
        ax1.grid(True)
        ax1.plot([X_plot[curr_pos],x_plot[curr_pos]],[Y_plot[curr_pos],y_plot[curr_pos]], color='r', linestyle='-')
        ax1.plot([X_plot[curr_pos]-L/4, X_plot[curr_pos]+L/4], [0,0], color="brown", lw=5)
            
        # Drawing pendulum and data
        ax1.plot([0,0],[0,np.min(y_plot)], color='k', linestyle='-', linewidth=2)
        ax1.plot([minX-L/2,maxX+L/2],[0,0], color='k', linestyle='-', linewidth=2)
        
    if not plotMotion:
        if ax1.get_visible() and not plotTrace:
            ax1.set_visible(False)
        if not ax2.get_visible():
            ax2.set_visible(True)
        if not ax3.get_visible():
            ax3.set_visible(True)

        ax2.cla()
        ax3.cla()

        ax2.plot(t_plot, X_plot)
        ax2.set_title("Translation of the Board")
        ax2.set_xlabel("time(s)")
        ax2.grid(True)
        
        ax3.plot(t_plot, phi_plot)
        ax3.set_title("Angular Motion of the Bob")
        ax3.set_xlabel("time(s)")
        ax3.set_ylabel(r"$\phi$")
        ax3.grid(True)


    fig1.tight_layout()
    fig1.canvas.draw()
    
def press(e):
    global curr_pos, plotMotion
    
    if e.key == 'left':
        curr_pos -= 10
    elif e.key == 'right':
        curr_pos += 10
    elif e.key == "up":
        plotMotion = not plotMotion
    elif e.key == "down":
        plotMotion = not plotMotion
    else:
        return
    
    curr_pos = curr_pos % len(x_plot)

    updateChart()

fig1 = plt.figure(1, facecolor="0.98")
fig1.canvas.mpl_connect('key_press_event', press)
spec1 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig1)
ax1 = fig1.add_subplot(spec1[:, 0])
ax2 = fig1.add_subplot(spec1[0, 0])
ax3 = fig1.add_subplot(spec1[1, 0])

updateChart()


    
    
    
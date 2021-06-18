# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 22:07:28 2017

Two pendulums of mass m1/m2 are connected to a board
that can translate left and right along the X axis (frictionlness).  
RK4 will be used to model the bob's motion.

Date last modified: 6/17/21          
@author: NEWCOMERGA1
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ===================================================
# //////////////////////// FUNCTIONS
# =======================================
def pendulumBoard(s, t, param):
    '''
    Returns right-hand side of board double-pendulum equations, used for RK4
      Inputs
        s      State vector [r  v  phi  w]
        t      Time (not used)
        param   Parameters: mass, length, board mass, gravity  
      Output
        deriv  Derivatives [dr/dt dv/dt dphi/dt dw/dt]
    '''
    # Constants and variables
    m1 = param[0]
    m2 = param[1]
    L1 = param[2]
    L2 = param[3]
    M = param[4]
    g = param[5]
    
    r = s[0]
    v = s[1]
    phi1 = s[2]
    w1 = s[3]
    phi2 = s[2]
    w2 = s[3]
    
    mass_denom = m1*(np.sin(phi1)**2) + m2*(np.sin(phi2)**2) + M
    
    # calculations
    v_accel = ( (m1*np.sin(phi1)* (L1*w1**2 + g*np.cos(phi1)) +  m2*np.sin(phi2)* (L2*w2**2 + g*np.cos(phi2)))
                    / mass_denom )
    w1_accel = -1 / (mass_denom*L1) * (g*np.sin(phi1)*(M+m1+m2) + m1*L1*w1**2*np.sin(phi1)*np.cos(phi1)
                                      + m2*L2*w2**2*np.sin(phi2)*np.cos(phi2) )
    w2_accel = -1 / (mass_denom*L2) * (g*np.sin(phi2)*(M+m1+m2) + m1*L1*w1**2*np.sin(phi1)*np.cos(phi1)
                                      + m2*L2*w2**2*np.sin(phi2)*np.cos(phi2) )
    
    #  Return derivatives
    deriv = np.array([v, v_accel, w1, w1_accel, w2, w2_accel])
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
    

# ===================================================
# //////////////////////// PROGRAM
# =======================================

print(    '''
    Models motion of a pendulum bob attached at the end to a board that can 
    translate freely in the X axis with the fourth order Runge-Kutta method.
    
    Motion Plot::   -Use R/L arrow keys to step forward/backward in time
                    -Use U/D arrow keys to toggle motion plot and parameter plot
    '''
)
# Set constants
pi = np.pi
m1 = 0.1         # mass of pendulum bob 1 (kg)
m2 = 0.1         # mass of pendulum bob 2 (kg)
L1 = 1.0         # length of pendulum 1 (m)
L2 = 1.0         # length of pendulum 2 (m)
c1 = -0.1        # distance of board COM to bob 2 pendulum pin (m)
c2 = 0.1         # distance of board COM to bob 2 pendulum pin (m)
M = 1          # mass of the board (kg)
g = 9.81        # graviational constant (m/s^2)
time = 0.0      # (s)
param = [m1, m2, L1, L2, M, g]    # list of parameters


# Setting initial values
phi10 = pi/180 * 45
phi20 = pi/180 * -45
w10 = 0.0             # initial angular velocity of bob 1
w20 = 0.0             # initial angular velocity of bob 2
r0 = 0               # initial position of the board
v0 = 0.0             # initial velocity of the board

state = np.array([r0, v0, phi10, w10, phi20, w20])        # state vector


# Time step
nStep = 1000    # number of steps
tau = 0.01       # time step (s)

# Lists for plotting
phi1_plot = []           # angular values of bob 1
phi2_plot = []           # angular values of bob 2
x1_plot = []             # x position of bob 1
y1_plot = []             # y position of bob 1
x2_plot = []             # x position of bob 2
y2_plot = []             # y position of bob 2
X_plot = []             # x position of the board
Y_plot = []             # y position of the board
t_plot = []             # time

# Running RK4
for iStep in range(nStep):
    r = state[0]        # position of the board
    phi1 = state[2]      # angular position of bob 1
    phi2 = state[4]      # angular position of bob 2
    
    # storing values
    phi1_plot.append(phi1 * 180/pi)
    phi2_plot.append(phi2 * 180/pi)
    x1_plot.append(L1*np.sin(phi1) + r + c1)
    y1_plot.append(-L1*np.cos(phi1))
    x2_plot.append(L2*np.sin(phi2) + r + c2)
    y2_plot.append(-L2*np.cos(phi2))
    X_plot.append(r)
    Y_plot.append(0)
    t_plot.append(time)
    
    # calculating next state
    state = rk4(state, time, tau, pendulumBoard, param)
        
    # incrementing time
    time += tau    


# Output
print("\nWith constants:")
print("  m1 = {} kg".format(m1))
print("  m2 = {} kg".format(m2))
print("  L1 = {} m".format(L1))
print("  L2 = {} m".format(L2))
print("  M = {} kg".format(M))
print("Initial conditions:")
print("  phi1 = {} deg".format(phi10*180/pi))
print("  phi2 = {} deg".format(phi20*180/pi))
print("  w10 = {} deg/s".format(w10 * 180/pi))
print("  w20 = {} deg/s".format(w20 * 180/pi))
print("  v0 = {} m/s".format(v0))


# Plotting cartesian motion, radial motion, and angular motion
curr_pos = -1
plotMotion = True
def updateChart():
    global ax1, fig1, ax2, ax3, ax4
    
    if plotMotion:
        if not ax1.get_visible():
            ax1.set_visible(True)
        if ax2.get_visible():
            ax2.set_visible(False)
        if ax3.get_visible():
            ax3.set_visible(False)
        if ax4.get_visible():
            ax4.set_visible(False)
    
        ax1.cla()

        minX = np.min([np.min(x1_plot), np.min(x2_plot), np.min(X_plot)])
        maxX = np.max([np.max(x1_plot), np.max(x2_plot), np.max(X_plot)])
        minY = np.min([np.min(y1_plot), np.min(y2_plot), np.min(Y_plot)])
        maxY = np.max([np.max(y1_plot), np.max(y2_plot), np.max(Y_plot)])
        L = np.max([L1, L2])
        
        ax1.scatter(x1_plot[curr_pos], y1_plot[curr_pos], s=5, color='r')
        ax1.scatter(x2_plot[curr_pos], y2_plot[curr_pos], s=5, color='r')
        ax1.plot(x1_plot[0:curr_pos], y1_plot[0:curr_pos], lw=1, color='blue')
        ax1.plot(x2_plot[0:curr_pos], y2_plot[0:curr_pos], lw=1, color='orange')
        ax1.set_title("Spring Pendulum Motion ({}/{})".format(curr_pos+1, len(x1_plot)))
        ax1.set_xlabel("x")
        ax1.set_xlim(minX, maxX)
        ax1.set_ylabel("y")
        ax1.grid(True)
        ax1.plot([X_plot[curr_pos]+c1,x1_plot[curr_pos]],[Y_plot[curr_pos],y1_plot[curr_pos]], color='r', linestyle='-')
        ax1.plot([X_plot[curr_pos]+c2,x2_plot[curr_pos]],[Y_plot[curr_pos],y2_plot[curr_pos]], color='r', linestyle='-')
        ax1.plot([X_plot[curr_pos]-L/4, X_plot[curr_pos]+L/4], [0,0], color="brown", lw=5)
            
        # Drawing pendulum and data
        ax1.plot([0,0],[0,minY], color='k', linestyle='-', linewidth=2)
        ax1.plot([minX-L/2,maxX+L/2],[0,0], color='k', linestyle='-', linewidth=2)
        
    if not plotMotion:
        if ax1.get_visible():
            ax1.set_visible(False)
        if not ax2.get_visible():
            ax2.set_visible(True)
        if not ax3.get_visible():
            ax3.set_visible(True)
        if not ax4.get_visible():
            ax4.set_visible(True)

        ax2.cla()
        ax3.cla()
        ax4.cla()

        ax2.plot(t_plot, X_plot)
        ax2.set_title("Translation of the Board")
        ax2.set_xlabel("time(s)")
        ax2.grid(True)
        
        ax3.plot(t_plot, phi1_plot)
        ax3.set_title("Angular Motion of the Bob 1")
        ax3.set_xlabel("time(s)")
        ax3.set_ylabel(r"$\phi$")
        ax3.grid(True)

        ax4.plot(t_plot, phi2_plot)
        ax4.set_title("Angular Motion of the Bob 2")
        ax4.set_xlabel("time(s)")
        ax4.set_ylabel(r"$\phi$")
        ax4.grid(True)


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
    
    curr_pos = curr_pos % len(X_plot)

    updateChart()

fig1 = plt.figure(1, facecolor="0.80")
fig1.canvas.mpl_connect('key_press_event', press)
spec1 = gridspec.GridSpec(ncols=1, nrows=3, figure=fig1)
ax1 = fig1.add_subplot(spec1[:, 0])
ax2 = fig1.add_subplot(spec1[0, 0])
ax3 = fig1.add_subplot(spec1[1, 0])
ax4 = fig1.add_subplot(spec1[2, 0])

updateChart()


    
    
    
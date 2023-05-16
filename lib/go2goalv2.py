"""
Trajectory planner to go directly to the goal

"""

__authors__ = "D. Knowles"
__date__ = "02 Apr 2022"

import os
import sys
import math
sys.path.append("..")

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

import lib.parameters as P
from lib.tools import wrap0to2pi


class GoToGoalv2():
    def __init__ (self, debug=False):
        """Simple planner that will always tell the agent to go to goal.

        """
        self.debug = debug

    def plan(self, b):
        """Chose new action plan.

        Parameters
        ----------
        b : belief.py class instance
            current belief state from which we make the action plan.

        Returns
        -------
        a : tuple of action tuples
            Tuple of length num_agents of actions that should be taken
            for each agent.

        """
        action = []
        for aa in range(len(b.mus)):
            aa_x = b.mus[aa][0]
            aa_y = b.mus[aa][1]
            aa_z = b.mus[aa][2]
            aa_theta = b.mus[aa][3]
            aa_vel = b.mus[aa][4]

            # traj, V, om = differential_flatness_trajectory()
            distance = np.sqrt((aa_x - P.goal_states[0,aa])**2 \
                             + (aa_y - P.goal_states[1,aa])**2)
            if distance < max(1,abs(aa_vel/P.acc_lb)):
                yawrate = 0.
                accel = -aa_vel
                action.append((yawrate, accel))
                continue

            tf = max(distance/P.vel_ub,2.)

            # time
            dt = 1.
            N = int(tf/dt)+1
            t = dt*np.array(range(N))

            # Initial conditions
            s_0 = State(x=aa_x, y=aa_y, V=aa_vel, th=aa_theta)

            # Final conditions
            s_f = State(x=P.goal_states[0,aa],
                        y=P.goal_states[1,aa],
                        V=P.goal_states[4,aa],
                        th=P.goal_states[3,aa]
                        )

            coeffs = compute_traj_coeffs(initial_state=s_0, final_state=s_f, tf=tf)
            t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
            V,om = compute_controls(traj=traj,t=t)

            part_b_complete = False
            s = compute_arc_length(V, t)
            if s is not None:
                part_b_complete = True
                V_tilde = rescale_V(V, om, P.vel_ub, P.yaw_ub)
                tau = compute_tau(V_tilde, s)
                om_tilde = rescale_om(V, om, V_tilde)
                t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)

                # Save trajectory data
                data = {'z': traj_scaled, 'V': V_scaled, 'om': om_scaled}

            # Plots
            # if self.debug:
            #     fig = plt.figure(figsize=(15, 7))
            #     plt.subplot(2, 2, 1)
            #     plt.plot(traj[:,0], traj[:,1], 'k-',linewidth=2)
            #     plt.grid('on')
            #     plt.plot(s_0.x, s_0.y, 'go', markerfacecolor='green', markersize=15)
            #     plt.plot(s_f.x, s_f.y, 'ro', markerfacecolor='red', markersize=15)
            #     plt.xlabel('X [m]')
            #     plt.ylabel('Y [m]')
            #     plt.title("Path (position)")
            #     # plt.axis([-1, 6, -1, 6])
            #
            #     ax = plt.subplot(2, 2, 2)
            #     plt.plot(t, V, linewidth=2)
            #     plt.plot(t, om, linewidth=2)
            #     plt.grid('on')
            #     plt.xlabel('Time [s]')
            #     plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
            #     plt.title('Original Control Input')
            #     plt.tight_layout()
            #
            #     plt.subplot(2, 2, 4, sharex=ax)
            #     if part_b_complete:
            #         plt.plot(t_new, V_scaled, linewidth=2)
            #         plt.plot(t_new, om_scaled, linewidth=2)
            #         plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
            #         plt.grid('on')
            #     else:
            #         plt.text(0.5,0.5,"[Problem iv not completed]", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            #     plt.xlabel('Time [s]')
            #     plt.title('Scaled Control Input')
            #     plt.tight_layout()
            #
            #     plt.subplot(2, 2, 3)
            #     if part_b_complete:
            #         h, = plt.plot(t, s, 'b-', linewidth=2)
            #         handles = [h]
            #         labels = ["Original"]
            #         h, = plt.plot(tau, s, 'r-', linewidth=2)
            #         handles.append(h)
            #         labels.append("Scaled")
            #         plt.legend(handles, labels, loc="best")
            #     else:
            #         plt.text(0.5,0.5,"[Problem iv not completed]", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            #     plt.grid('on')
            #     plt.xlabel('Time [s]')
            #     plt.ylabel('Arc-length [m]')
            #     plt.title('Original and scaled arc-length')
            #     plt.tight_layout()
            #     plt.show()
            #     plt.close(fig)

            # don't exceed limits
            accel = np.clip(V_scaled[1]-V_scaled[0], P.acc_lb, P.acc_ub)
            yawrate = np.clip(om_scaled[1], P.yaw_lb, P.yaw_ub)
            # accel = np.clip(V[1]-V[0], P.acc_lb, P.acc_ub)
            # yawrate = np.clip(om[1], P.yaw_lb, P.yaw_ub)

            action.append((yawrate, accel))

        return tuple(action), 0

    def choose_yaw(self, yaw_desired, yaw_current):
        """Converts yawrate between (-pi,pi)

        """
        while yaw_desired - yaw_current > np.pi:
            yaw_desired -= 2.*np.pi
        while yaw_desired - yaw_current < -np.pi:
            yaw_desired += 2.*np.pi
        return yaw_desired - yaw_current


    def reset(self):
        """Any reset necessary between timesteps

        """
        pass

class State:
    def __init__(self,x,y,V,th):
        self.x = x
        self.y = y
        self.V = V
        self.th = th

    @property
    def xd(self):
        return self.V*np.cos(self.th)

    @property
    def yd(self):
        return self.V*np.sin(self.th)


def compute_traj_coeffs(initial_state, final_state, tf):
    """
    Inputs:
        initial_state (State)
        final_state (State)
        tf (float) final time
    Output:
        coeffs (np.array shape [8]), coefficients on the basis functions

    Hint: Use the np.linalg.solve function.
    """
    ########## Code starts here ##########
    t0 = 0.0
    x_a = np.array([[1., t0, t0**2, t0**3],
                    [0., 1., 2.*t0, 3.*t0**2],
                    [1., tf, tf**2, tf**3],
                    [0., 1., 2.*tf, 3.*tf**2],])
    x_b = np.array([[initial_state.x],
                    [initial_state.xd],
                    [final_state.x],
                    [final_state.xd]])
    x_coeffs = np.linalg.solve(x_a,x_b)

    y_a = np.array([[1., t0, t0**2, t0**3],
                    [0., 1., 2.*t0, 3.*t0**2],
                    [1., tf, tf**2, tf**3],
                    [0., 1., 2.*tf, 3.*tf**2],])
    y_b = np.array([[initial_state.y],
                    [initial_state.yd],
                    [final_state.y],
                    [final_state.yd]])
    y_coeffs = np.linalg.solve(y_a,y_b)
    coeffs = np.concatenate((x_coeffs,y_coeffs),axis=0)
    ########## Code ends here ##########
    return coeffs

def compute_traj(coeffs, tf, N):
    """
    Inputs:
        coeffs (np.array shape [8]), coefficients on the basis functions
        tf (float) final_time
        N (int) number of points
    Output:
        traj (np.array shape [N,7]), N points along the trajectory, from t=0
            to t=tf, evenly spaced in time
    """
    t = np.linspace(0,tf,N) # generate evenly spaced points from 0 to tf
    traj = np.zeros((N,7))
    ########## Code starts here ##########
    def x_t(time_now):
        result = coeffs.item(0) + coeffs.item(1)*time_now + coeffs.item(2)*time_now**2 + coeffs.item(3)*time_now**3
        return result
    def x_dot_t(time_now):
        result = coeffs.item(1) + 2.*coeffs.item(2)*time_now + 3.*coeffs.item(3)*time_now**2
        return result
    def x_ddot_t(time_now):
        result = 2.*coeffs.item(2) + 6.*coeffs.item(3)*time_now
        return result
    def y_t(time_now):
        result = coeffs.item(4) + coeffs.item(5)*time_now + coeffs.item(6)*time_now**2 + coeffs.item(7)*time_now**3
        return result
    def y_dot_t(time_now):
        result = coeffs.item(5) + 2.*coeffs.item(6)*time_now + 3.*coeffs.item(7)*time_now**2
        return result
    def y_ddot_t(time_now):
        result = 2.*coeffs.item(6) + 6.*coeffs.item(7)*time_now
        return result
    traj[:,0] = x_t(t)
    traj[:,1] = y_t(t)
    traj[:,2] = np.unwrap(np.arctan2(y_dot_t(t),x_dot_t(t)),
                          discont=np.pi)
    traj[:,3] = x_dot_t(t)
    traj[:,4] = y_dot_t(t)
    traj[:,5] = x_ddot_t(t)
    traj[:,6] = y_ddot_t(t)

    ########## Code ends here ##########

    return t, traj

def compute_controls(traj,t):
    """
    Input:
        traj (np.array shape [N,7])
    Outputs:
        V (np.array shape [N]) V at each point of traj
        om (np.array shape [N]) om at each point of traj
    """
    ########## Code starts here #########
    N = traj.shape[0]
    V = np.zeros(N)
    om = np.zeros(N)
    for ii in range(N):
        V[ii] = np.sqrt(traj[ii][3]**2 + traj[ii][4]**2)
        if ii > 0:
            om[ii] = (traj[ii][2] - traj[ii-1][2])/(t[ii] - t[ii-1])

    ########## Code ends here ##########

    return V, om

def compute_arc_length(V, t):
    """
    This function computes arc-length s as a function of t.
    Inputs:
        V: a vector of velocities of length T
        t: a vector of time of length T
    Output:
        s: the arc-length as a function of time. s[i] is the arc-length at time
            t[i]. This has length T.

    Hint: Use the function cumtrapz. This should take one line.
    """
    ########## Code starts here ##########
    s = cumtrapz(V,t)
    s = np.insert(s, 0, 0.0)
    ########## Code ends here ##########
    return s

def rescale_V(V, om, V_max, om_max):
    """
    This function computes V_tilde, given the unconstrained solution V, and om.
    Inputs:
        V: vector of velocities of length T. Solution from the unconstrained,
            differential flatness problem.
        om: vector of angular velocities of length T. Solution from the
            unconstrained, differential flatness problem.
    Output:
        V_tilde: Rescaled velocity that satisfies the control constraints.

    Hint: At each timestep V_tilde should be computed as a minimum of the
    original value V, and values required to ensure _both_ constraints are
    satisfied.
    Hint: This should only take one or two lines.
    """
    ########## Code starts here ##########
    V_tilde = np.copy(V)
    for ii in range(len(V)):
        if V[ii] > 0:
            if om[ii] != 0.0:
                V_tilde[ii] = min(V[ii],V_max,V[ii]*om_max/np.abs(om[ii]))
            else:
                V_tilde[ii] = min(V[ii],V_max)
        else:
            if om[ii] != 0.0:
                V_tilde[ii] = max(V[ii],-V_max,V[ii]*om_max/np.abs(om[ii]))
            else:
                V_tilde[ii] = min(V[ii],V_max)
    ########## Code ends here ##########
    return V_tilde


def compute_tau(V_tilde, s):
    """
    This function computes the new time history tau as a function of s.
    Inputs:
        V_tilde: a sequence of scaled velocities of length T.
        s: a sequence of arc-length of length T.
    Output:
        tau: the new time history for the sequence. tau[i] is the time at s[i]. This has length T.

    Hint: Use the function cumtrapz. This should take one line.
    """
    ########## Code starts here ##########
    tau = cumtrapz(1./(V_tilde+0.0001),s)
    tau = np.insert(tau,0,0.0)
    ########## Code ends here ##########
    return tau

def rescale_om(V, om, V_tilde):
    """
    This function computes the rescaled om control.
    Inputs:
        V: vector of velocities of length T. Solution from the unconstrained, differential flatness problem.
        om:  vector of angular velocities of length T. Solution from the unconstrained, differential flatness problem.
        V_tilde: vector of scaled velocities of length T.
    Output:
        om_tilde: vector of scaled angular velocities

    Hint: This should take one line.
    """
    ########## Code starts here ##########
    om_tilde = np.multiply(np.divide(V_tilde,V+1E-10),om)
    ########## Code ends here ##########
    return om_tilde

def compute_traj_with_limits(z_0, z_f, tf, N, V_max, om_max):
    coeffs = compute_traj_coeffs(initial_state=z_0, final_state=z_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
    V,om = compute_controls(traj=traj,t=t)
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)

    return traj, tau, V_tilde, om_tilde

def interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f):
    """
    Inputs:
        traj (np.array [N,7]) original unscaled trajectory
        tau (np.array [N]) rescaled time at orignal traj points
        V_tilde (np.array [N]) new velocities to use
        om_tilde (np.array [N]) new rotational velocities to use
        dt (float) timestep for interpolation

    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    """
    # Get new final time
    tf_new = tau[-1]

    # Generate new uniform time grid
    N_new = int(tf_new/dt)
    t_new = dt*np.array(range(N_new+1))

    # Interpolate for state trajectory
    traj_scaled = np.zeros((N_new+1,7))
    traj_scaled[:,0] = np.interp(t_new,tau,traj[:,0])   # x
    traj_scaled[:,1] = np.interp(t_new,tau,traj[:,1])   # y
    traj_scaled[:,2] = np.interp(t_new,tau,traj[:,2])   # th
    # Interpolate for scaled velocities
    V_scaled = np.interp(t_new, tau, V_tilde)           # V
    om_scaled = np.interp(t_new, tau, om_tilde)         # om
    # Compute xy velocities
    traj_scaled[:,3] = V_scaled*np.cos(traj_scaled[:,2])    # xd
    traj_scaled[:,4] = V_scaled*np.sin(traj_scaled[:,2])    # yd
    # Compute xy acclerations
    traj_scaled[:,5] = np.append(np.diff(traj_scaled[:,3])/dt,-s_f.V*om_scaled[-1]*np.sin(s_f.th)) # xdd
    traj_scaled[:,6] = np.append(np.diff(traj_scaled[:,4])/dt, s_f.V*om_scaled[-1]*np.cos(s_f.th)) # ydd

    return t_new, V_scaled, om_scaled, traj_scaled

if __name__ == "__main__":
    # traj, V, om = differential_flatness_trajectory()
    # Constants
    tf = 15.
    V_max = 0.5
    om_max = 1

    # time
    dt = 0.005
    N = int(tf/dt)+1
    t = dt*np.array(range(N))

    # Initial conditions
    s_0 = State(x=0, y=0, V=V_max, th=-np.pi/2)

    # Final conditions
    s_f = State(x=5, y=5, V=V_max, th=-np.pi/2)

    coeffs = compute_traj_coeffs(initial_state=s_0, final_state=s_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
    V,om = compute_controls(traj=traj,t=t)

    part_b_complete = False
    s = compute_arc_length(V, t)
    if s is not None:
        part_b_complete = True
        V_tilde = rescale_V(V, om, V_max, om_max)
        tau = compute_tau(V_tilde, s)
        om_tilde = rescale_om(V, om, V_tilde)

        t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)

        # Save trajectory data
        data = {'z': traj_scaled, 'V': V_scaled, 'om': om_scaled}

    # Plots
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 2, 1)
    plt.plot(traj[:,0], traj[:,1], 'k-',linewidth=2)
    plt.grid('on')
    plt.plot(s_0.x, s_0.y, 'go', markerfacecolor='green', markersize=15)
    plt.plot(s_f.x, s_f.y, 'ro', markerfacecolor='red', markersize=15)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title("Path (position)")
    plt.axis([-1, 6, -1, 6])

    ax = plt.subplot(2, 2, 2)
    plt.plot(t, V, linewidth=2)
    plt.plot(t, om, linewidth=2)
    plt.grid('on')
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
    plt.title('Original Control Input')
    plt.tight_layout()

    plt.subplot(2, 2, 4, sharex=ax)
    if part_b_complete:
        plt.plot(t_new, V_scaled, linewidth=2)
        plt.plot(t_new, om_scaled, linewidth=2)
        plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
        plt.grid('on')
    else:
        plt.text(0.5,0.5,"[Problem iv not completed]", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.xlabel('Time [s]')
    plt.title('Scaled Control Input')
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    if part_b_complete:
        h, = plt.plot(t, s, 'b-', linewidth=2)
        handles = [h]
        labels = ["Original"]
        h, = plt.plot(tau, s, 'r-', linewidth=2)
        handles.append(h)
        labels.append("Scaled")
        plt.legend(handles, labels, loc="best")
    else:
        plt.text(0.5,0.5,"[Problem iv not completed]", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.grid('on')
    plt.xlabel('Time [s]')
    plt.ylabel('Arc-length [m]')
    plt.title('Original and scaled arc-length')
    plt.tight_layout()
    plt.show()

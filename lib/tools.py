"""Rotation Utilities

Code based on *Small Unmanned Aircraft: Theory and Practice* by Randal
Beard and Tim McLain. Reference: uavbook.byu.edu

"""
__authors__ = "R. Beard, D. Knowles"
__date__ = "11 Apr 2019"

import os
import sys
from math import cos, sin

import numpy as np

def RotationBody2Vehicle(phi,theta,psi):
    result = np.array([[cos(theta)*cos(psi),cos(theta)*sin(psi),-sin(theta)],
                       [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi),sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi),sin(phi)*cos(theta)],
                       [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi),cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi),cos(phi)*cos(theta)]])
    return result.T


def wrap0to2pi(angle):
    while angle >= 2*np.pi:
        angle -= 2*np.pi
    while angle <= 0:
        angle += 2*np.pi
    return angle

def mkdir(dir):
    # create directory if it doesn't yet exist
    if not os.path.isdir(dir):
        try:
            os.makedirs(dir)
        except OSError as e:
            print("e: ",e)
            sys.exit(1)

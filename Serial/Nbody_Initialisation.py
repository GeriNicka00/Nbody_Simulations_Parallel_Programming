# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 12:39:50 2021

@author: Geri Nicka
"""

# Python module to generate initial conditions

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np


# Center of mass velocities.
def COM(vx, vy, vz, m):
    vx_cm = vx - np.sum(m * vx) / np.sum(m)
    vy_cm = vy - np.sum(m * vy) / np.sum(m)
    vz_cm = vz - np.sum(m * vz) / np.sum(m)
    return vx_cm, vy_cm, vz_cm

# Creates initial positions and velocities.
def Glob(n_obj, m_tot):
    np.random.seed(11)           # To reproduce the same exact simulation.
    mu, sigma = 0, 2
    # Positions.
    x = np.random.normal(mu, sigma, size=n_obj).astype('d')
    y = np.random.normal(mu, sigma, size=n_obj).astype('d')
    z = np.random.normal(mu, sigma, size=n_obj).astype('d')
    
    # Velocities. 
    mu_v, sigma_v, scaling_v = mu, sigma / 5, 0.001
    vx = np.random.normal(mu_v, sigma_v, size=n_obj).astype('d') * scaling_v
    vy = np.random.normal(mu_v, sigma_v, size=n_obj).astype('d') * scaling_v
    vz = np.random.normal(mu_v, sigma_v, size=n_obj).astype('d') * scaling_v

    # Masses.
    m = np.random.rand(n_obj).astype('d')
    m *= m_tot / np.sum(m)

    vx, vy, vz = COM(vx, vy, vz, m)

    return x, y, z, vx, vy, vz, m



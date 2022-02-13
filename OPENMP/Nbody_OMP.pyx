# -*- coding: utf-8 -*-
"""
Created on Nov 11 2021 by Geri Nicka, University of Bristol
"""

# Run on Linux/Mac:
    
#    Extension must be .pyx    
#    Compile with CC=gcc-11 python OMP_Setup.py build_ext -fi.
#    Run with python OMP_Runner.py 1000 200 6 -sym -nopos.
#    Thats for 1000 objects for 200 time steps with 6 cores using symmetry.
#    Use -nonsym without symmetry.
#    Use -nopos not to save positions.

# Running on BC4: 
    
#    Extension must be .pyx
#    Compile with python OMP_Setup.py build_ext -fi
#    Run with python Multiple_Runs_OMP.py 2000 200 8 1.
#    Here 2000: objects, 200: size of time step, 8: time request, 1: sequential number.


import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import time
import pickle

import sys
cimport cython
cimport numpy as np
ctypedef np.float64_t dtype_t
from cython.parallel cimport prange
cimport openmp
from libc.math cimport sqrt

import Nbody_Initialisation as ninit



def Indices_Generation(n_obj):
    assigned_objs = ((n_obj - 1) // 2) + (n_obj - 1) % 2
    j_indices = np.zeros((n_obj, assigned_objs), dtype=np.int_)
    j_sizes = np.zeros(n_obj, dtype=np.int_)
    
    # Loop over each object among all objects.
    for i in range(n_obj):
        indices_list = list(range((i + 1) % 2, i, 2)) + list(range(i + 2, n_obj, 2))
        
        for ind, el in enumerate(indices_list):
            j_indices[i][ind] = indices_list[ind]
        j_sizes[i] = len(indices_list)

    return j_indices, j_sizes


@cython.boundscheck(False)
@cython.wraparound(False)

def Acceleration_with_Sym3(double [:] ax, double [:] ay, double [:] az,
                           double [:, :] ax_mat, double [:, :] ay_mat, double [:, :] az_mat,
                           double [:] x, double [:] y, double [:] z,
                           double [:] m, double r_min_2,
                           int n_obj, int n_threads):
    cdef:
        double dx = 0.0
        double dy = 0.0
        double dz = 0.0
        double norm2 = 0.0
        double norm3 = 0.0
        Py_ssize_t i, j, k

    for i in prange(0, n_obj, nogil=True, schedule = 'dynamic', num_threads=n_threads):
        # All accelerations are set to zero.
        ax[i] = 0.0
        ay[i] = 0.0
        az[i] = 0.0

        for j in range(i, n_obj):
            # Computing distances.
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j]

            norm2 = (dx**2 + dy**2 + dz**2 + r_min_2)
            norm3 = sqrt(norm2) * norm2

            # Computing accelerations for objects i and j.
            ax_mat[i][j] = -m[j] * (dx / norm3)
            ay_mat[i][j] = -m[j] * (dy / norm3)
            az_mat[i][j] = -m[j] * (dz / norm3)

            # Updating taking advantage of symmetry for symmetric j and i.
            ax_mat[j][i] = -ax_mat[i][j]
            ay_mat[j][i] = -ay_mat[i][j]
            az_mat[j][i] = -az_mat[i][j]

    for i in prange(0, n_obj, nogil=True, num_threads=n_threads):
        # Updating accelerations for object i.
        for j in range(n_obj):
            ax[i] += ax_mat[i][j]
            ay[i] += ay_mat[i][j]
            az[i] += az_mat[i][j]


@cython.boundscheck(False)
@cython.wraparound(False)

def Acceleration_with_Sym2(double [:] ax, double [:] ay, double [:] az,
                           double [:, :] ax_mat, double [:, :] ay_mat, double [:, :] az_mat,
                           double [:] x, double [:] y, double [:] z,
                           double [:] m, double r_min_2,
                           int n_obj, long [:, :] j_indices, long [:] j_sizes, int n_threads):
    cdef:
        double dx = 0.0
        double dy = 0.0
        double dz = 0.0
        double norm2 = 0.0
        double norm3 = 0.0
        Py_ssize_t i, j, k

    for i in prange(0, n_obj, nogil=True, num_threads=n_threads):
        # All accelerations are set to zero.
        ax[i] = 0.0
        ay[i] = 0.0
        az[i] = 0.0

        for k in range(j_sizes[i]):
            j = j_indices[i][k]
            # Computing distances.
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j]

            norm2 = (dx**2 + dy**2 + dz**2 + r_min_2)
            norm3 = sqrt(norm2) * norm2

            # Computing accelerations for objects i and j.
            ax_mat[i][j] = -m[j] * (dx / norm3)
            ay_mat[i][j] = -m[j] * (dy / norm3)
            az_mat[i][j] = -m[j] * (dz / norm3)

            # Updating taking advantage of symmetry for symmetric j and i.
            ax_mat[j][i] = -ax_mat[i][j]
            ay_mat[j][i] = -ay_mat[i][j]
            az_mat[j][i] = -az_mat[i][j]

    for i in prange(0, n_obj, nogil=True, num_threads=n_threads):
        # Updating accelerations for object i.
        for j in range(n_obj):
            ax[i] += ax_mat[i][j]
            ay[i] += ay_mat[i][j]
            az[i] += az_mat[i][j]

@cython.boundscheck(False)
@cython.wraparound(False)

def Acceleration_with_Sym1(double [:] ax, double [:] ay, double [:] az,
                           double [:] x, double [:] y, double [:] z,
                           double [:] m, double r_min_2,
                           int n_obj, long [:, :] j_indices, long [:] j_sizes, int n_threads):
    cdef:
        double dx = 0.0
        double dy = 0.0
        double dz = 0.0
        double norm2 = 0.0
        double norm3 = 0.0
        Py_ssize_t i, j, k
        double [:, :] ax_loc = np.zeros((n_obj, n_obj), dtype=np.double)
        double [:, :] ay_loc = np.zeros((n_obj, n_obj), dtype=np.double)
        double [:, :] az_loc = np.zeros((n_obj, n_obj), dtype=np.double)

    for i in prange(0, n_obj, nogil=True, num_threads=n_threads):
        # All accelerations are set to zero.
        ax[i] = 0.0
        ay[i] = 0.0
        az[i] = 0.0

        for k in range(j_sizes[i]):
            j = j_indices[i][k]
            # Computing distances.
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j]

            norm2 = (dx**2 + dy**2 + dz**2 + r_min_2)
            norm3 = sqrt(norm2) * norm2

            # Computing accelerations for objects i and j.
            ax_loc[i][j] = -m[j] * (dx / norm3)
            ay_loc[i][j] = -m[j] * (dy / norm3)
            az_loc[i][j] = -m[j] * (dz / norm3)

            # Updating taking advantage of symmetry for symmetric j and i.
            ax_loc[j][i] = -ax_loc[i][j]
            ay_loc[j][i] = -ay_loc[i][j]
            az_loc[j][i] = -az_loc[i][j]

    for i in prange(0, n_obj, nogil=True, num_threads=n_threads):
        # Updating accelerations for object i.
        for j in range(n_obj):
            ax[i] += ax_loc[i][j]
            ay[i] += ay_loc[i][j]
            az[i] += az_loc[i][j]


@cython.boundscheck(False)
@cython.wraparound(False)

def Acceleration_without_Sym(double [:] ax, double [:] ay, double [:] az,
                             double [:] x, double [:] y, double [:] z,
                             double [:] m, double r_min_2,
                             int n_obj, int n_threads):
    cdef:
        double dx = 0.0
        double dy = 0.0
        double dz = 0.0
        double norm2 = 0.0
        double norm3 = 0.0
        Py_ssize_t i, j

    for i in prange(0, n_obj, nogil=True, num_threads=n_threads):
        # All accelerations are set to zero.
        ax[i] = 0.0
        ay[i] = 0.0
        az[i] = 0.0
        
        for j in range(n_obj):
            # Computing distances.
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j]

            norm2 = (dx**2 + dy**2 + dz**2 + r_min_2)
            norm3 = sqrt(norm2) * norm2

            # Computing accelerations for objects i and j and updating acceleration for particle i.
            ax[i] += -m[j] * (dx / norm3)
            ay[i] += -m[j] * (dy / norm3)
            az[i] += -m[j] * (dz / norm3)


@cython.boundscheck(False)
@cython.wraparound(False)

def Velocity_Evolution(double [:] v1x, double [:] v1y, double [:] v1z,
           double [:] v2x, double [:] v2y, double [:] v2z, double dt, int n_obj, int n_threads):
    cdef:
        Py_ssize_t i
        
    for i in prange(0, n_obj, nogil=True, num_threads=n_threads):
        v1x[i] += v2x[i] * dt
        v1y[i] += v2y[i] * dt
        v1z[i] += v2z[i] * dt


def main(int n_obj, int n_t, int n_threads):
    
    cdef:
        double r_min_2 = 0.01  # Minimal radius squared. Avoids collisions and r=0 in the denominator.
        
        double dt = 0.01                                                 # Size of time step.
        double t_0 = 0.0                                                 # Initial time.
        double t_f = t_0 + ((n_t - 1) * dt)                              # Final time.
        double [:] times = np.linspace(t_0, t_f, n_t)                    # A list of time values.
        
        double m_tot = 20.0                                              # Total mass.
        
        double [:] x = np.zeros(n_obj, dtype=np.double)                  # Will store x values.
        double [:] y = np.zeros(n_obj, dtype=np.double)                  # Will store y values.
        double [:] z = np.zeros(n_obj, dtype=np.double)                  # Will store z values.
        
        double [:] vx = np.zeros(n_obj, dtype=np.double)                 # Will store vx values.
        double [:] vy = np.zeros(n_obj, dtype=np.double)                 # Will store vy values.
        double [:] vz = np.zeros(n_obj, dtype=np.double)                 # Will store vz values.
        
        double [:] ax = np.zeros(n_obj, dtype=np.double)                 # Will store ax values.
        double [:] ay = np.zeros(n_obj, dtype=np.double)                 # Will store ay values.
        double [:] az = np.zeros(n_obj, dtype=np.double)                 # Will store az values.
        
        double [:, :] ax_mat = np.zeros((n_obj, n_obj), dtype=np.double) # Stores accelerations along x.
        double [:, :] ay_mat = np.zeros((n_obj, n_obj), dtype=np.double) # Stores accelerations along y.
        double [:, :] az_mat = np.zeros((n_obj, n_obj), dtype=np.double) # Stores accelerations along z.
        
        double [:, :] pos_x = np.zeros((n_obj, n_t + 1), dtype='d')      # Stores positions along x.
        double [:, :] pos_y = np.zeros((n_obj, n_t + 1), dtype='d')      # Stores positions along y.
        double [:, :] pos_z = np.zeros((n_obj, n_t + 1), dtype='d')      # Stores positions along z.
        
        long [:, :] j_indices
        long [:] j_sizes
        double [:, :] exec_times = np.zeros((2, 3))
        Py_ssize_t i
        str device = ''
        int use_symmetry = 0
        int save_positions = 1

    for arg in sys.argv:
        # To use symmetry.
        if arg == "-sym":
            use_symmetry = 1
        # Not to save positions.  
        if arg == "-nopos":
            save_positions = 0

    device = 'BC4'

    # G=1 for simplicity.

    # Start of initialisation time.
    exec_times[0][0] = openmp.omp_get_wtime()

    # Initiating positions and velocities.
    x, y, z, vx, vy, vz, m = ninit.Glob(n_obj, m_tot)

    pos_x[:, 0] = x
    pos_y[:, 0] = y
    pos_z[:, 0] = z

    j_indices, j_sizes = Indices_Generation(n_obj)

    # End of initialisation time.
    exec_times[0][1] = openmp.omp_get_wtime()
    
    # Start of computation time.
    exec_times[1][0] = openmp.omp_get_wtime()

    for i in range(n_t):

        Velocity_Evolution(x, y, z, vx, vy, vz, dt / 2.0, n_obj, n_threads)
        
        if use_symmetry == 1:
        # Acceleration_with_Sym1(ax, ay, az, x, y, z, m, r_min_2, n_obj, j_indices, j_sizes, n_threads)
        # Acceleration_with_Sym2(ax, ay, az, ax_mat, ay_mat, az_mat, x, y, z, m, r_min_2, n_obj, j_indices, j_sizes, n_threads)
        
            Acceleration_with_Sym3(ax, ay, az, ax_mat, ay_mat, az_mat, x, y, z, m, r_min_2, n_obj, n_threads)
        else:
            
            Acceleration_without_Sym(ax, ay, az, x, y, z, m, r_min_2, n_obj, n_threads)
        Velocity_Evolution(vx, vy, vz, ax, ay, az, dt, n_obj, n_threads)
        Velocity_Evolution(x, y, z, vx, vy, vz, dt / 2.0, n_obj, n_threads)

        pos_x[:, i + 1] = x
        pos_y[:, i + 1] = y
        pos_z[:, i + 1] = z

    # End of computation time.
    exec_times[1][1] = openmp.omp_get_wtime()
    
    # Calculating execution time for initialisation.
    exec_times[0][2] = exec_times[0][1] - exec_times[0][0]
    # Calculating execution time for computation.
    exec_times[1][2] = exec_times[1][1] - exec_times[1][0]

    # Prints results.
    if use_symmetry == 1:
        symmetry_option = ""
        sym_txt = ""
        
    else:
        symmetry_option = "not "
        sym_txt = "non"
        
    out_file = f"output_omp_{n_obj}_{n_t}_{n_threads}_{sym_txt}sym.txt"
    
    out_text = f"Gravity simulation run on {device} using {n_threads} cores\n" \
                  f"    {n_obj} objects over {n_t} time steps of {dt} s\n" \
                  f"    algorithm {symmetry_option}taking advantage of symmetry\n" \
                  f"Execution time with openmp.omp_get_wtime():\n" \
                  f"    Initialisation time:  {exec_times[0][2]} s\n" \
                  f"    Computation time:     {exec_times[1][2]} s\n"
                  
    print(out_text)
    
    with open(out_file, 'a') as f:
        f.write(out_text)

    summary_list = [n_threads, n_obj, n_t, exec_times[0][2], exec_times[1][2]]

    # Save results in files for post-processing.
    save_dict = {'times.npy': times, 'exec_times.npy': exec_times}
    
    # Will save positions if set equal to 1. Makes data transfer slow.
    if save_positions == 1:
        save_dict['pos_x.npy'] = pos_x
        save_dict['pos_y.npy'] = pos_y
        save_dict['pos_z.npy'] = pos_z
        
    for key, value in save_dict.items():
        
        with open(key, 'wb') as f:
            np.save(f, value)

    pickle_filename = f"../summary_{n_threads:02d}_{n_obj:05d}_{n_t:04d}.pkl"

    with open(pickle_filename, 'wb') as f:
        pickle.dump(summary_list, f)










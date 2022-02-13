# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 13:48:51 2021

@author: Geri Nicka
"""

# Run on Linux/Mac machines using: 'mpiexec -n 8 python Nbody_MPI.py 500 200'.
# 8: number of cores, 500: number of objects, 200 time steps. CAN CHANGE THESE WHEN CALLING THE SCRIPT.
# Submit on BC4 using 'sbatch Nbody_MPI.sh'.

from mpi4py import MPI

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import numpy as np
import pickle

import Nbody_Initialisation as ninit 

# Acceleration of the masses.
def Acc(r_x, r_y, r_z, x_n, y_n, z_n, m, r_min_2):

    # 2D array storing r_j - r_i = -(r_i - r_j)
    dx_n = r_x - np.vstack(x_n)
    dy_n = r_y - np.vstack(y_n)
    dz_n = r_z - np.vstack(z_n)

    norm2 = (dx_n**2 + dy_n**2 + dz_n**2 + r_min_2)
    norm3 = np.sqrt(norm2) * norm2

    ax_n = (dx_n / norm3) @ m
    ay_n = (dy_n / norm3) @ m
    az_n = (dz_n / norm3) @ m
    return ax_n, ay_n, az_n

# Splitting the arrays in an optimised way. See in the report.
def Array_XYZ_Splitting(a):
    # a is the array to split
    a_split = np.split(a, 3)
    return a_split[0], a_split[1], a_split[2]

# Distributing tasks in an optimised way between the cores. See in the report. 
def Distribute_between_Tasks(a_split_x, a_split_y, a_split_z, numtasks, taskid):
    # array split for each nth task
    ax_n = np.array_split(a_split_x, numtasks)[taskid]
    ay_n = np.array_split(a_split_y, numtasks)[taskid]
    az_n = np.array_split(a_split_z, numtasks)[taskid]
    return ax_n, ay_n, az_n

# Evolution of velocity overtime.
def Vel_Evol(v1x, v1y, v1z, v2x, v2y, v2z, dt):
    v1x += v2x * dt
    v1y += v2y * dt
    v1z += v2z * dt

# Main function which makes the whole thing run.
def main(n_obj, n_t):

    comm = MPI.COMM_WORLD
    numtasks = comm.Get_size()
    taskid = comm.Get_rank()    # TaskID=0 for the root task.
    root = 0
    device = 'BC4'              # Can be BC4 or personal device. Not important.


    r_min_2 = 0.01      # Minimal radius squared. Avoids collision and 0 denominator.
    dt = 0.01           # Size of time step.

    if taskid == root:

        # G=1 for simplicity.

        exec_time = np.zeros((2, 3))
        # Start initialisation time.
        exec_time[0][0] = MPI.Wtime()


        t_0 = 0.  # Initial time.

        t_f = t_0 + ((n_t - 1) * dt)  # Final time.
        time_vals = np.linspace(t_0, t_f, n_t)  # Time values.

        m_tot = 20.0  # Total mass.

        x, y, z, vx, vy, vz, m = ninit.Glob(n_obj, m_tot)  # Initialising the positions and velocities.

        pos_x = np.zeros((n_obj, n_t + 1), dtype='d')
        pos_x[:, 0] = x
        
        pos_y = np.zeros((n_obj, n_t + 1), dtype='d')
        pos_y[:, 0] = y
        
        pos_z = np.zeros((n_obj, n_t + 1), dtype='d')
        pos_z[:, 0] = z

        # First half step.
        Vel_Evol(x, y, z, vx, vy, vz, dt / 2.0)

        # End of initialisation time.
        exec_time[0][1] = MPI.Wtime()
        # Start of computation time.
        exec_time[1][0] = MPI.Wtime()

        r, v = np.concatenate((x, y, z)), np.concatenate((vx, vy, vz))

    else:
        r = np.empty(3 * n_obj, dtype='d')         
       
        m = np.empty(n_obj, dtype='d')         
        
        v = np.empty(3 * n_obj, dtype='d')         
       
    # Broadcast the masses to the cores.
    comm.Bcast(m, root=root)
    
    # Broadcast the velocities to the cores.
    comm.Bcast(v, root=root)


    # Splitting the velocity arrays and distributing the tasks performed in each array to the cores.
    v_x, v_y, v_z = Array_XYZ_Splitting(v)
    vx_n, vy_n, vz_n = Distribute_between_Tasks(v_x, v_y, v_z, numtasks, taskid)

    # Computing sendcount and offset arrays prior to Allgatherv (Dr. Simon Hanna notes).
    split_m = np.array_split(m, numtasks)
    sendcounts = np.array([3 * len(el) for el in split_m])
    offsets = np.zeros(sendcounts.size, dtype=int)
    offsets[1:] = np.cumsum(sendcounts)[:-1]

    # Broadcast r_full and split into x_full.
    comm.Bcast(r, root=root)


    # Splitting the position arrays and distributing the tasks permormed in each array to the cores. 
    r_x, r_y, r_z = Array_XYZ_Splitting(r)
    x_n, y_n, z_n = Distribute_between_Tasks(r_x, r_y, r_z, numtasks, taskid)

    for i in range(n_t):
        # Each core works on a n_th task which is a subset of: 
        # positions (x_n, y_n, z_n) and velocities (vx_n, vy_n, vz_n).
        
        # Leapfrog method just like in the serial code.
        ax_n, ay_n, az_n = Acc(r_x, r_y, r_z, x_n, y_n, z_n, m, r_min_2)
        
        Vel_Evol(vx_n, vy_n, vz_n, ax_n, ay_n, az_n, dt)
        Vel_Evol(x_n, y_n, z_n, vx_n, vy_n, vz_n, dt / 2.0)

        # Store by grouping x_n=[x_n0…], y_n=[y_n0…], z_n=[z_n0…] to:
        # xyz_n=[x_n0 y_n0 z_n0 x_n1 y_n1 z_n1…].
        # This to reduce communication as much as possible.
        
        xyz_n = np.concatenate(np.stack((x_n, y_n, z_n)).T)

        # Gather xyz_n.
        if taskid == root:
            xyz = np.empty(3 * n_obj, dtype='d')
            
        else:
            xyz = None
        comm.Gatherv(sendbuf=xyz_n, recvbuf=(xyz, sendcounts), root=root)

        if taskid == root:
 
            # Master splits gathered position vector(r) to positions.
            x = xyz[0::3]
            y = xyz[1::3]
            z = xyz[2::3]
            pos_x[:, i + 1] = x
            pos_y[:, i + 1] = y
            pos_z[:, i + 1] = z

        # Next leapfrog.
        Vel_Evol(x_n, y_n, z_n, vx_n, vy_n, vz_n, dt / 2.0)

        # Store by grouping x_n=[x_n0…], y_n=[y_n0…], z_n=[z_n0…] to:
        # xyz_n=[x_n0 y_n0 z_n0 x_n1 y_n1 z_n1…].
        # This is to reduce communication as much as possible.
        
        xyz_n = np.concatenate(np.stack((x_n, y_n, z_n)).T)

        # Gather xyz_n to xyz.
        xyz = np.empty(3 * n_obj, dtype='d')

        comm.Allgatherv([xyz_n, MPI.DOUBLE], [xyz, sendcounts, offsets, MPI.DOUBLE])

    if taskid == root:
        # End of computation time.
        exec_time[1][1] = MPI.Wtime()

        # Calculating initialisation time.
        exec_time[0][2] = exec_time[0][1] - exec_time[0][0]
        # Calculating computation time.
        exec_time[1][2] = exec_time[1][1] - exec_time[1][0]

        # Results saved in folders for post-processing.
        filename = f"output_mpi_{n_obj}_{n_t}_{numtasks}.txt"
        out_text = f"Simulation run on {device} using {numtasks} tasks\n" \
                      f"for {n_obj} objects over {n_t} time steps of {dt} s\n\n" \
                      f"Execution time with MPI.Wtime():\n" \
                      f"    Initialisation time:  {exec_time[0][2]} s\n" \
                      f"    Computation time:     {exec_time[1][2]} s\n\n"
                      
        print(out_text)

        # List of job summary. 
        summary_list = [numtasks, n_obj, n_t, exec_time[0][2], exec_time[1][2]]

        with open(filename, 'a') as f:
            f.write(out_text)

        # Save times and execution times in files to analyse performance.
        save_dict = {'time_vals.npy': time_vals, 'exec_time.npy': exec_time}
 
        # Dictionary methods, as seen form official documentation and tutorials.
        for key, value in save_dict.items():
            with open(key, 'wb') as f:
                np.save(f, value)
 
        # Using pickle extension for the files to post-process later. 
        pickle_filename = f"../summary_{numtasks:02d}_{n_obj:05d}_{n_t:04d}.pkl"

        with open(pickle_filename, 'wb') as f:
            pickle.dump(summary_list, f)

    return None

if int(len(sys.argv)) == 3:
    main(int(sys.argv[1]), int(sys.argv[2]))
else:
    print("Usage: {} <N_OBJECTS> <N_TIME_STEPS>".format(sys.argv[0]))

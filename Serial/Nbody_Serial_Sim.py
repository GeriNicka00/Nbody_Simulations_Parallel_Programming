# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 12:38:37 2021

@author: Geri Nicka
"""

import numpy as np
import time
import os

import Nbody_Initialisation as ninit


def Acc(x, y, z, m, r_min_sq):

    # 2D array storing r_j - r_i = -(r_i - r_j)
    dx = x - np.vstack(x)
    dy = y - np.vstack(y)
    dz = z - np.vstack(z)

    norm3 = (dx ** 2 + dy ** 2 + dz ** 2 + r_min_sq)
    norm3 = np.sqrt(norm3) * norm3

    ax = (dx / norm3) @ m
    ay = (dy / norm3) @ m
    az = (dz / norm3) @ m

    return ax, ay, az


def Acc_En(x, y, z, vx, vy, vz, m, r_min):
    # Kinetic Energy computation.
    ke = 0.5 * np.sum(m * (vx ** 2 + vy ** 2 + vz ** 2))

    # 2D array storing r_j - r_i = -(r_i - r_j)
    dx = x - np.vstack(x)
    dy = y - np.vstack(y)
    dz = z - np.vstack(z)

    norm2 = dx ** 2 + dy ** 2 + dz ** 2
    norm = np.sqrt(norm2)
    norm3 = (norm + r_min) ** 3
    norm[norm > 0] = 1.0 / norm[norm > 0]

    ax = (dx / norm3) @ m
    ay = (dy / norm3) @ m
    az = (dz / norm3) @ m

    pe = np.sum(np.sum(np.triu(-(m * m.T) * norm, 1)))

    return ax, ay, az, ke, pe


def Vel_Evol(v1x, v1y, v1z, v2x, v2y, v2z, dt):
    v1x += v2x * dt
    v1y += v2y * dt
    v1z += v2z * dt


def main():

    exec_time = np.zeros((2, 3))
    
    # Start of initialisation time.
    exec_time[0][0] = time.process_time()

    # Parameters where G=1 for simplicity.

    nb_obj = 50      # Number of objects.
    dt = 0.01        # Size of time step.
    nb_t = 1000      # Number of time steps.
    time_vals = np.linspace(0, nb_t * dt, nb_t)

    r_min = 0.1             # Minimal radius. Avoids collision and 0 denominator.
    r_min_sq = r_min ** 2   # Minimal radius squared.
    m_tot = 20.0            # Total mass.

    device = 'Dell G3'
    compute_energy = True

    x, y, z, vx, vy, vz, m = ninit.Glob(nb_obj, m_tot)

    pos_x = np.zeros((nb_obj, nb_t + 1))
    pos_x[:, 0] = x
    pos_y = np.zeros((nb_obj, nb_t + 1))
    pos_y[:, 0] = y
    pos_z = np.zeros((nb_obj, nb_t + 1))
    pos_z[:, 0] = z
    kin_en = np.zeros(nb_t + 1)
    pot_en = np.zeros(nb_t + 1)

    # End of initialisation time.
    exec_time[0][1] = time.process_time()
    # Start of computation time.
    exec_time[1][0] = time.process_time()

    for i in range(time_vals.size):
        # Leapfrog method.
        Vel_Evol(x, y, z, vx, vy, vz, dt / 2.0)
        
        if compute_energy:
            ax, ay, az, ke, pe = Acc_En(x, y, z, vx, vy, vz, m, r_min)
            kin_en[i + 1] = ke
            pot_en[i + 1] = pe
            
        else:
            ax, ay, az = Acc(x, y, z, m, r_min_sq)
        Vel_Evol(vx, vy, vz, ax, ay, az, dt)
        Vel_Evol(x, y, z, vx, vy, vz, dt / 2.0)

        pos_x[:, i + 1] = x
        pos_y[:, i + 1] = y
        pos_z[:, i + 1] = z

    # End of computation time.
    exec_time[1][1] = time.process_time()
    
    # Calculating total initialisation time.
    exec_time[0][2] = exec_time[0][1] - exec_time[0][0]
    # Calculating total computation time.
    exec_time[1][2] = exec_time[1][1] - exec_time[1][0]

    # Printing results in separate file for post-processing.
    out_file = f"output_serial_{nb_obj}_{nb_t}.txt"
    direct = 'Nbody_Serial_Run'

    # If directory with same name exists then a second is created with '.tmp' extension. 
    if os.path.isdir(direct):
        direct += '_tmp/'
        
    else:
        direct += '/'
    os.mkdir(direct)
    out_file_path = direct + out_file
    out_text = f"Simulation run on {device}\n" \
                  f"    {nb_obj} objects over {nb_t} time steps of {dt} s\n" \
                  f"    Initialisation time:  {exec_time[0][2]} s\n" \
                  f"    Computation time:     {exec_time[1][2]} s\n"
                  
    print(out_text)
    
    with open(out_file_path, 'a') as f:
        f.write(out_text)

    # Save data in files for post-processing.
    save_dict = {'pos_x.npy': pos_x, 'pos_y.npy': pos_y, 'pos_z.npy': pos_z,
                 'time_vals.npy': time_vals, 'exec_time.npy': exec_time}
    
    if compute_energy:
        save_dict['kin_en.npy'] = kin_en
        save_dict['pot_en.npy'] = pot_en
        
    for key, value in save_dict.items():
        with open(direct + key, 'wb') as f:
            np.save(f, value)

if __name__ == "__main__":
    main()

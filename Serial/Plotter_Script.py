# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 11:29:48 2021

@author: Geri Nicka
"""

import os
import numpy as np                      
import matplotlib.pyplot as plt         
import imageio

# Plots masses distribution and energies using data acquired from Nbody_Serial_Sim.py
def Plotter(time_vals, pos_x, pos_y, pos_z,
                    kin_en, pot_en,
                    folder_path, plot_en=False, dim_3d=True, save_gif=True, **kwargs):


    fig = plt.figure(figsize=(8, 10), dpi=80)

    # For 3D visualisation of the distribution of the masses.
    if dim_3d:
        ax1 = fig.add_subplot(projection='3d')
    
    # Plotting the energy of the system of masses.     
    elif plot_en:
        gridsp = plt.GridSpec(2, 1, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gridsp[0])
        ax2 = fig.add_subplot(gridsp[1])
        ylow = 1.1 * np.min(pot_en[:time_vals.size])
        yhigh = 1.1 * np.max(kin_en[:time_vals.size])
        
    else:
        ax1 = plt.subplot(111)

    track = 30
    axes_range = (-5, 5)
    axes_ticks = [_ for _ in range(-5, 6)]

    filenames = []    # Empty list will accept filename which stores the data for plotting.    


    # Looping to plot the time evolution of each mass in real time.
    for i in range(time_vals.size):
        plt.sca(ax1)
        plt.cla()
        xx = pos_x[:, max(i - track, 0):i + 1]
        yy = pos_y[:, max(i - track, 0):i + 1]
        zz = pos_z[:, max(i - track, 0):i + 1]
        plt.title(f"{i+1}/{time_vals.size}")

        # To plot the 3D time evolution of the masses.
        if dim_3d:
            ax1.scatter(xx, yy, zz, s=10, color=[.7, .7, 1])
            ax1.scatter(pos_x[:, i], pos_y[:, i], pos_z[:, i], s=20, color='blue')
            ax1.set(xlim=axes_range, ylim=axes_range, zlim=axes_range)
            ax1.set_aspect('auto', 'box')          
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_xticks(axes_ticks)
            ax1.set_yticks(axes_ticks)
            ax1.set_zticks(axes_ticks)
            
        # To plot the 1D time evolution of the masses.     
        else:
            plt.scatter(xx, yy, s=10, color=[.7, .7, 1])
            ax1.scatter(pos_x[:, i], pos_y[:, i], s=20, color='blue')
            ax1.set(xlim=axes_range, ylim=axes_range)
            ax1.set_aspect('equal', 'box')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_xticks(axes_ticks)
            ax1.set_yticks(axes_ticks)

            # To plot the energy of the system.
            if plot_en:
                t = time_vals[1:i + 1]
                ke = kin_en[1:i + 1]
                pe = pot_en[1:i + 1]
                e_tot = ke + pe
                plt.sca(ax2)
                plt.cla()
                plt.plot(t, ke, color='green', label='KE')
                plt.plot(t, pe, color='blue', label='PE')
                plt.plot(t, e_tot, color='red', label='E')
                ax2.set(xlim=(0, time_vals[-1]), ylim=(ylow, yhigh))
                ax2.legend(loc="lower left")

        # In case the figure needs to be saved.
        if save_gif:
            
            # Creates file and appends filenames.
            filename = folder_path + f'{i}.png'
            filenames.append(filename)

            # Saves figure.
            plt.savefig(filename)
            plt.close()

        plt.pause(0.00001)

    plt.legend()
    plt.show()

    # Gives the option to the save the gif which was created in real time.
    if save_gif:
        
        # Constructing a GIF (Typical processes seen from official documentation and examples).
        with imageio.get_writer(folder_path + 'nbody.gif', mode='I', fps=60) as writer:
            
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Removing files.
        for filename in set(filenames):
            os.remove(filename)

    return None

# Wrapping up all previous functions to create the main function which will create the plots.
def main():
    
    # Choosing settings (Self expanatory).
    dim_3d = False
    save_gif = False
    plot_en = True
    folder_name = 'Nbody_Serial_Run'
    folder_path = folder_name + '/'

    device = 'Dell G3' # This the device of the author of the programme.
    
    # Here the folder which contains the data is being read and # 
    # all positions across the 3 dimensions, time, execution    #
    # time, kinetic energy and potential energy are extracted   #
    # to create the required plots.                             #
    
    with open(folder_path + 'pos_x.npy', 'rb') as f:
        pos_x = np.load(f)
        
    with open(folder_path + 'pos_y.npy', 'rb') as f:
        pos_y = np.load(f)
        
    with open(folder_path + 'pos_z.npy', 'rb') as f:
        pos_z = np.load(f)
        
    with open(folder_path + 'time_vals.npy', 'rb') as f:
        time_vals = np.load(f)
        
    with open(folder_path + 'exec_time.npy', 'rb') as f:
        exec_time_vals = np.load(f)
        
    if plot_en:
        with open(folder_path + 'kin_en.npy', 'rb') as f:
            kin_en = np.load(f)
            
        with open(folder_path + 'pot_en.npy', 'rb') as f:
            pot_en = np.load(f)
            
    else:
        kin_en = np.zeros(pos_x.size)
        pot_en = np.zeros(pos_x.size)

    nb_t = min(200, time_vals.size)
    
    # Printing results. 
    out_text = f"Simulation run on {device}\n"\
                  f"Execution time:\n" \
                  f"    Initialisation time:  {exec_time_vals[0][2]} s\n" \
                  f"    Computation time:     {exec_time_vals[1][2]} s\n\n" \
                  f"Plot over {nb_t} time steps (total available {time_vals.size})"
    print(out_text)
    
    # Calling the plotter. #
    
    Plotter(time_vals[:nb_t], pos_x, pos_y, pos_z,
                    kin_en, pot_en,
                    folder_path=folder_path, plot_en=plot_en, dim_3d=dim_3d, save_gif=save_gif)

if __name__ == "__main__":
    main()
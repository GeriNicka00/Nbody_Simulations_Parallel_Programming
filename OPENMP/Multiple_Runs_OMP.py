# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 15:26:09 2022

@author: Geri Nicka

Adapted from Stephan Cuttelod and Ted Stokes.
"""

import subprocess
import os
import sys
import pickle


def Folder_Dest(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

def main(n_obj, n_t, minutes, sequential_nb):

    folder_name = f"Multiple_omp_{sequential_nb:03d}_n_threads_{n_obj}_obj_{n_t}_t"
    Folder_Dest(folder_name)

    # Settings independent of number of cores
    python_file = 'Nbody_OMP.pyx'                    # OMP Python Simulation.
    python_run_file = "OMP_Runner.py"                # OMP Python Run File.
    partition = 'test'                               # Test partition does not wait in the queue for long.
    nodes = 1                                        # Request a full node.
    ntasks_per_node = 1                              # One task for each node.
    time = f"0:{minutes}:0"                          # Minutes are chosen when the program is called in the terminal.
    mem = '10G'                                      # Requesting a substantial amount of memory to be safe.
    use_symmetry = False                             # Can choose with or without symmetry in the terminal.
    sym = ' -nonsym'                                 # Default.
    
    # Using symmetry.
    if use_symmetry:
        sym = ' -sym'
    save_positions = False
    no_pos = ''
    
    # Not using symmetry.
    if not save_positions:
        no_pos = ' -nopos'

    jobs_summary = []       # Creates a job summary.
    max_cores = 28          # Request all cores in the node.

    # Submit jobs for each core.
    for cpus_per_task in range(2, max_cores + 1):
        
        num_cores = cpus_per_task
        job_name = f"{num_cores:02d}_{n_obj}_{n_t}_nbody_omp"                                # SLURM Job Name.
        sbatch_name = f"nbody_omp_{python_file[:-4]}_{num_cores:02d}_{n_obj}_{n_t}.sh"       # Name of the batch file.
        omp_call = f"python ../../{python_run_file} {n_obj} {n_t} {num_cores}{sym}{no_pos}"  # Call in the terminal.

        subfolder_path = f"{folder_name}/{num_cores:02d}_threads_{n_obj}_obj_{n_t}_t"        # Folder of the batch file.
        
        # Creating the batch file for each job automatically.
        sbatch_file = [
            f'#!/bin/bash\n',
            f'# ======================\n',
            f'# {sbatch_name}\n',
            f'# ======================\n',
            f'\n',
            f'#SBATCH --job-name={job_name}\n',
            f'#SBATCH --partition={partition}\n',
            f'#SBATCH --nodes={nodes}\n',
            f'#SBATCH --ntasks-per-node={ntasks_per_node}\n',
            f'#SBATCH --cpus-per-task={max_cores}\n',
            f'#SBATCH --time={time}\n',
            f'#SBATCH --mem={mem}\n',
            f'#SBATCH --output={subfolder_path}/slurm-%j.out\n',
            f'\n',
            f'module add languages/anaconda3/2020-3.8.5\n',
            f'\n',
            f'cd $SLURM_SUBMIT_DIR/{subfolder_path}\n',
            f'\n',
            f'export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}\n'
            f'\n',
            f'{omp_call}\n',
        ]

        # Saving batch file in the folders.
        Folder_Dest(subfolder_path)
        with open(f"{subfolder_path}/{sbatch_name}", 'w') as f:
            for line in sbatch_file:
                f.write(line)

        # Queuing batch files.
        bashCommand = f"sbatch {subfolder_path}/{sbatch_name}"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        print(f"\n{num_cores} CORES")
        print('OUTPUT:')
        print(str(output))
        print('ERROR:')
        print(error)

        # Receiving JOBID from 'sbatch' command.
        if str(output)[2:22] == 'Submitted batch job ':
            jobs_summary.append(int(str(output)[22:29]))
            
        else:
            raise Exception("Job ID for {num_cores} cores not found")

    pickle_filename = f"{folder_name}/jobs_summary.pkl"
    with open(pickle_filename, 'wb') as f:
        pickle.dump(jobs_summary, f)


if int(len(sys.argv)) == 5:
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
else:
    print("Usage: {} <N_OBJECTS> <N_TIME_STEPS> <MINUTES> <SEQUENTIAL_NB>".format(sys.argv[0]))
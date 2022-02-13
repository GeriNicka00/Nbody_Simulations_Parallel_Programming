# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 15:28:44 2022

@author: Geri Nicka
"""

import sys
from Nbody_OMP import main

if int(len(sys.argv)) in (5,6):
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
else:
    print("Usage: {} <N_OBJECTS> <N_TIME_STEPS> <N_THREADS> <-sym / -nonsym>".format(sys.argv[0]))

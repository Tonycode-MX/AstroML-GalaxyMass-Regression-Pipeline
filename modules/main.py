# Main module for executing stages

# LIBRARIES

# Standard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# External modules

stage = [1]

def stage1():
    print("Executing Stage 1")
    # Add stage 1 specific logic here

def stage2():
    print("Executing Stage 2")
    # Add stage 2 specific logic here

def stage3():
    print("Executing Stage 3")
    # Add stage 3 specific logic here

def stage4():
    print("Executing Stage 4")
    # Add stage 4 specific logic here
    
def stage5():
    print("Executing Stage 5")
    # Add stage 5 specific logic here

# Execute stages
if __name__ == '__main__': 
    if 1 in stage:
        stage1()
    elif 2 in stage:
        stage2()
    elif 3 in stage:
        stage3()
    elif 4 in stage:
        stage4()
    elif 5 in stage:
        stage5()
# %% Run from root
import os
if os.getcwd() == os.path.dirname(os.path.abspath(__file__)):
    os.chdir('..')

# %% Import ML libraries
# pylint: disable=E0401
from config import ROOT_DIR
import numpy as np
import tensorflow as tf
from tensorflow import keras

# %%

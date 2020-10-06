# %% Import libraries
import tensorflow as tf
from tensorflow import keras


# %%
class Decoder(tf.Module):
    def __init__(self) -> None:
        super().__init__()

    @tf.Module.with_name_scope
    def __call__(self) -> None:
        pass

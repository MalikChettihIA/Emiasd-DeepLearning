import os
import numpy as np
import golois
from tensorflow.keras.models import load_model

if __name__ == "__main__":
   model = load_model("jm_fauvel_v0410.h5")
   model.summary()


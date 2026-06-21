import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from go_utils import plot_version, print_validation_results, plot_learning_rate, plot_result
from go_train import train_model
from go_mobilenet import GoMobileNet
from go_mixnet import GoMixNet

if __name__ == "__main__":
    model = GoMobileNet((19, 19, 31), 361)

    #Total params: 101893(398.02 KB)
    #Trainable params: 99459(388.51 KB)
    #Non - trainable params: 2434(9.51 KB)
    #model = model.build(block_num=2, filters=64, factor=4, se=True, drop_out_rate=0.2, activation='swish')


    model = model.build(block_num=8, filters=32, factor=4, se=True, drop_out_rate=0.2, activation='swish')
    #Total params: 93317 (364.52 KB)
    #Trainable params: 89219 (348.51 KB)
    #Non-trainable params: 4098 (16.01 KB)
    model.summary()
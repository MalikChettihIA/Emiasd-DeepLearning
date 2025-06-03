from go_utils import plot_version, print_validation_results, plot_learning_rate, plot_result
from go_train import train_model
from go_mobilenet import GoMobileNet

if __name__ == "__main__":

    plot_version()

    model = GoMobileNet((19, 19, 31), 361)
    model = model.build(block_num=8, filters=32, factor=4, se=True, drop_out_rate=0.2, activation='swish')
    model.summary()
from go_mobilenet import GoMobileNet
from go_mobilenetv2 import GoMobileNetV2
from go_mixnet import GoMixNet
from go_resnet import GoResNet

if __name__ == "__main__":

    model = GoMobileNetV2((19, 19, 31), 361)
    model = model.build(block_num=9, filters=32, factor=4, se=True, drop_out_rate=0.3, activation='swish')
    model.summary()

    #mix_model = GoMixNet((19, 19, 31), 361)
    #mix_model = mix_model.build(block_num=6, filters=32, factor=4, se=True, drop_out_rate=0.3, activation='swish')
    #mix_model.summary()
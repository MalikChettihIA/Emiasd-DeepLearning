from go_mobilenet import GoMobileNet


if __name__ == "__main__":

    model = GoMobileNet((19, 19, 31), 361)
    model = model.build(block_num=2, filters=64, factor=4, se=True, drop_out_rate=0.5, activation='swish')
    model.summary()


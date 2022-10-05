import skeleton

path = "/home/sam/Downloads/IA1_train.csv"


def task0():
    data = skeleton.load_data(path)
    data = skeleton.preprocess_data(data, True, True)
    print(data)


def task1():
    data = skeleton.load_data(path)
    data = skeleton.preprocess_data(data, True, False)
    data = skeleton.gd_train(data, "1", 1)


if __name__ == '__main__':
    task0()
    task1()

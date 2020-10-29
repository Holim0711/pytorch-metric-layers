import numpy as np

def test_confusion_matrix_accumulator():
    from métrics.multiclass import ConfusionMatrixAccumulator, plot_confusion_matrix
    np.random.seed(0)
    accumulator = ConfusionMatrixAccumulator((3, 8))

    for i in range(3):
        true = np.random.randint(3, size=10)
        pred = np.random.randint(5, size=10)
        print(i, 'true', true)
        print(i, 'pred', pred)
        accumulator.update(true, pred)
    print(accumulator.compute())
    print(accumulator.compute(mode='precision'))
    print(accumulator.compute(mode='recall'))
    print(accumulator.compute(mode='f1'))
    print(accumulator.compute(mode='fβ'))
    fig = plot_confusion_matrix(
        accumulator.compute(),
        ylabs=["AAAAA", "BBBBB", "CCCCCCCCCCCCCC"],
        xlabs=["A", "B", "C", "D", "E", "F", "GGGGGGGGGGGG", "헬로"])
    fig.savefig('test.png')

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    test_confusion_matrix_accumulator()

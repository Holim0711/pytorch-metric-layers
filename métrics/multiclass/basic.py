class AccTopK():
    def __init__(self, k=[1]):
        self.k = sorted(k, reverse=True)
        self.__zero__()

    def __zero__(self):
        self.right = [0] * len(self.k)
        self.total = 0

    def update(self, pred, true):
        rank = pred.argsort(descending=True)
        for i, k in enumerate(self.k):
            rank = rank[:, :k]
            self.right[i] += (rank == true.view(-1, 1)).any(dim=1).sum().item()
        self.total += len(true)

    def commit(self):
        acc = {
            ("accuracy@%d" % k) : (x / self.total)
            for k, x in zip(self.k, self.right)
        }
        self.__zero__()
        return acc


if __name__ == "__main__":

    import torch
    from time import time

    pred = [
        torch.tensor([
            [ 0.43, -0.23,  1.16, -0.22,  1.08, -0.78,  0.31,  0.27],
            [-0.55, -1.45,  0.96,  1.28, -0.27, -0.14, -0.38,  0.01],
            [-1.07,  1.03, -0.04, -1.83,  0.33,  0.04,  1.78, -1.22]
        ]),
        torch.tensor([
            [-1.15,  0.09,  1.03,  0.58, -1.18, -0.09, -1.60,  0.19],
            [ 0.08,  0.54,  1.31,  1.39,  1.28, -0.96, -1.63, -0.37],
            [-1.24,  1.79, -0.65, -0.28,  0.48,  0.38, -0.36,  1.82]
        ]),
    ]

    true = [
        torch.tensor([1, 7, 4]),
        torch.tensor([6, 6, 3]),
    ]


    def Test_AccTopK():
        topk = AccTopK([3, 5])

        start = time()
        for p, t in zip(pred, true):
            topk.update(p, t)
        print("Time LRAP:", time() - start)

        result = topk.commit()
        assert(result['accuracy@3'] == (2 / 6))
        assert(result['accuracy@5'] == (3 / 6))

    Test_AccTopK()

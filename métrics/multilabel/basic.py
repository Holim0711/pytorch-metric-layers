from sklearn.metrics import label_ranking_average_precision_score as lrap_score


class PrcRecTopK():
    def __init__(self, k=1):
        self.k = k
        self.right = 0
        self.n_label = 0
        self.n_sample = 0

    def update(self, pred, true):
        topk = pred.argsort()[:, -self.k:]
        for p, q in zip(topk, true):
            self.right += q[p].long().sum().item()
            self.n_label += q.long().sum().item()
        self.n_sample += len(true)

    def commit(self):
        precision = self.right / (self.k * self.n_sample)
        recall = self.right / self.n_label
        self.__init__()
        return precision, recall


class LRAP():
    def __init__(self):
        self.Σ = 0.0
        self.n_sample = 0

    def update(self, pred, true):
        self.Σ += lrap_score(true, pred) * len(true)
        self.n_sample += len(true)

    def commit(self):
        lrap = self.Σ / self.n_sample
        self.__init__()
        return lrap


if __name__ == "__main__":
    k = 3
    topk = PrcRecTopK(k)
    lrap = LRAP()

    import torch
    from time import time

    start = time()
    for _ in range(100):
        pred = torch.randn(32, 1000)
        true = torch.randint(2, (32, 1000))
        topk.update(pred, true)
        lrap.update(pred, true)
    end = time()

    print(end - start)
    print(topk.commit())
    print(lrap.commit())

from sklearn.metrics import label_ranking_average_precision_score as lrap_score


class PrcRecTopK():
    def __init__(self, k=[1]):
        self.k = sorted(k, reverse=True)
        self.__zero__()

    def __zero__(self):
        self.right = [0] * len(self.k)
        self.n_label = 0
        self.n_sample = 0

    def update(self, pred, true):
        rank = pred.argsort(descending=True)
        for i, k in enumerate(self.k):
            rank = rank[:, :k]
            for p, q in zip(rank, true):
                self.right[i] += q[p].long().sum().item()
        self.n_label += true.long().sum().item()
        self.n_sample += len(true)

    def commit(self):
        results = {
            ("precision@%d" % k) : x / (k * self.n_sample)
            for x, k in zip(self.right, self.k)
        }
        results.update({
            ("recall@%d" % k) : x / self.n_label
            for x, k in zip(self.right, self.k)
        })
        self.__zero__()
        return results


class LRAP():
    def __init__(self):
        self.Σ = 0.0
        self.n_sample = 0

    def update(self, pred, true):
        self.Σ += lrap_score(true.cpu(), pred.cpu()) * len(true)
        self.n_sample += len(true)

    def commit(self):
        lrap = self.Σ / self.n_sample
        self.__init__()
        return {"lrap" : lrap}


if __name__ == "__main__":
    import torch
    from time import time

    pred = [
        torch.tensor([
            [-0.4, -0.2, +0.9, -0.5, +0.8, -0.9, -0.1, -1.5],
            [+1.6, -0.5, +0.0, -0.5, +1.4, -0.8, +0.5, -2.2],
            [-1.7, -2.7, +0.3, -0.1, -0.0, +1.1, +0.4, -1.1]
        ]),
        torch.tensor([
            [+0.9, -0.4, -0.5, -0.1, +0.2, +0.4, +1.6, +1.0],
            [+0.2, -0.7, +0.8, -0.8, -0.9, -0.7, -0.0, -0.1],
            [-0.2, -0.4, +1.0, -0.2, -1.9, +2.2, -0.3, +0.4]
        ]),
    ]
    true = [
        torch.tensor([
            [0, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 1, 1]
        ]),
        torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0]
        ]),
    ]

    def Test_PrcRecTopK():
        topk = PrcRecTopK([1, 3, 5])

        start = time()
        for p, t in zip(pred, true):
            topk.update(p, t)
        print("Time PrcRecTopK:", time() - start)

        result = topk.commit()
        assert(result['precision@1'] == (4 / 6))
        assert(result['precision@3'] == (10 / 18))
        assert(result['precision@5'] == (14 / 30))
        assert(result['recall@1'] == (4 / 23))
        assert(result['recall@3'] == (10 / 23))
        assert(result['recall@5'] == (14 / 23))

    def Test_LRAP():
        lrap = LRAP()

        start = time()
        for p, t in zip(pred, true):
            lrap.update(p, t)
        print("Time LRAP:", time() - start)

        result = lrap.commit()

        gold = lrap_score(torch.cat(true), torch.cat(pred))
        assert(result['lrap'] == gold)

    Test_PrcRecTopK()
    Test_LRAP()

class AccTopK():
    def __init__(self, k=1):
        self.k = k
        self.right = 0
        self.total = 0

    def update(self, pred, true):
        topk = pred.argsort()[:, -self.k:]
        self.right += (topk == true.view(-1, 1)).any(dim=1).sum().item()
        self.total += len(true)

    def commit(self):
        acc = self.right / self.total
        self.__init__()
        return acc


if __name__ == "__main__":
    k = 3
    topk = AccTopK(k)

    import torch
    for _ in range(2):
        pred = torch.randn(3, 8)
        true = torch.randint(8, (3,))
        print("pred:")
        print(pred)
        print("top-k pred:")
        print(pred.argsort()[:, -k:])
        print("true:")
        print(true)
        topk.update(pred, true)

    print(topk.commit())

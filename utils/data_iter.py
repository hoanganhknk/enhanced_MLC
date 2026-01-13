class DataIterator:
    """
    Lặp vô hạn qua một DataLoader (giống kiểu iterator dùng trong meta-learning).
    """
    def __init__(self, loader):
        self.loader = loader
        self.it = iter(loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            batch = next(self.it)
        return batch

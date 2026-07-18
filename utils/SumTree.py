import numpy


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.n_expert_data=0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data,expert_data=False):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p,is_add=True)

        self.write += 1
        if expert_data:
            self.n_expert_data+=1
        if self.write >= self.capacity:
            self.write = self.n_expert_data

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p,is_add=False):
        change = p - self.tree[idx]
        if idx>=self.n_expert_data+self.capacity-1 or is_add:
            self.tree[idx] = p
            self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
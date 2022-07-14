import math
from operator import add
from typing import Callable


class SegmentTree:
    def __init__(self, data_len: int, init_value: float, operator: Callable[[float, float], float]):
        """
        Arrays are used to implement segment trees, with each element representing a tree node
        Each leaf node index (i >= capacity - 1) holds an experience priority
        Each non-leaf node index (i < capacity - 1) has:
          - a left child with index i * 2 + 1
          - a right child with index i * 2 + 2
          - a value equal to the sum/min of its left and right child priorities
        The first array element is the sum/min of all experience priorities
        """
        # The capacity is the lowest power of 2 greater than / equal to max_len
        self.capacity = 2 ** math.ceil(math.log2(data_len))
        self.operator = operator
        self.tree = [init_value] * (self.capacity * 2 - 1)
        self.midpoint = self.capacity - 1

    def update(self, index, new_value):
        index += self.midpoint
        self.tree[index] = new_value

        # While the index is not the root node
        while index >= 1:
            index = (index - 1) // 2
            left_child = index * 2 + 1
            right_child = index * 2 + 2
            self.tree[index] = self.operator(self.tree[left_child], self.tree[right_child])

    def get_value(self, index):
        return self.tree[index + self.midpoint]

    def get_root(self):
        return self.tree[0]


class MinTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity, math.inf, min)


class SumTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity, 0., add)

    def get_prefix_sum_index(self, prefix_sum):
        """Find the largest index i, such that the sum of tree[j + midpoint], for 0 <= j <= i, is <= prefix_sum"""
        index = 0

        # While index is a non-leaf node
        while index < self.midpoint:
            left_child = index * 2 + 1
            right_child = index * 2 + 2

            if self.tree[left_child] > prefix_sum:
                index = left_child
            else:
                prefix_sum -= self.tree[left_child]
                index = right_child

        return index - self.midpoint


class RingBuffer:
    def __init__(self, max_len):
        self.max_len = max_len
        self.start = 0
        self.length = 0
        self.data = []

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 0 or index >= self.length:
            raise IndexError('Buffer index out of range')
        return self.data[(self.start + index) % self.max_len]

    def __setitem__(self, index, value):
        if index < 0 or index >= self.length:
            raise IndexError('Buffer index out of range')
        self.data[(self.start + index) % self.max_len] = value

    def __iter__(self):
        if self.length == 0:
            return

        index = self.start
        end_index = self.start

        finished = False
        while not finished:
            yield self.data[index]
            index = (index + 1) % self.length
            finished = index == end_index

    def append(self, value):
        if self.length < self.max_len:
            self.length += 1
            self.data.append(value)
        elif self.length == self.max_len:
            self.start = (self.start + 1) % self.max_len
            self.data[(self.start + self.length - 1) % self.max_len] = value

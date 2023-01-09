# 二叉堆
class BinaryHeap(object):
    def __init__(self, compare_topper=lambda x, y: -1 if x < y else (1 if x > y else 0)):
        """
        二叉堆初始化中采用一个列表来保存堆数据，其中表首下标为0的项无用，
        但为了后面代码可以用到简单的整数乘除法，仍保留它。
        """
        self.heap_list = [None]
        self.compare_topper = compare_topper

    def top(self):
        return self.heap_list[1]

    def empty(self):
        return self.heap_list == [None]

    def size(self):
        return len(self.heap_list) - 1

    def insert(self, key):
        """ 时间复杂度是 O(log n) """
        # 尾追新节点
        self.heap_list.append(key)
        # 上浮
        self.per_up(self.size())

    def pop(self):
        """ 时间复杂度是 O(log n) """
        # 验空
        if self.empty():
            raise ValueError('Heap is empty now!')
        # 移除最小项
        self.heap_list[1], self.heap_list[-1] = self.heap_list[-1], self.heap_list[1]
        temp = self.heap_list.pop()
        # 下沉
        self.per_down(1, self.size())
        return temp

    def build_heap(self, alist):
        self.heap_list = [None] + alist
        size = self.size()
        index = size // 2
        while index > 0:
            self.per_down(index, size)
            index -= 1
        return self.heap_list[1:]

    def per_up(self, index):
        while index // 2 > 0:
            if self.compare_topper(self.heap_list[index // 2], self.heap_list[index]) > 0:
                self.heap_list[index // 2], self.heap_list[index] = self.heap_list[index], self.heap_list[index // 2]
                index = index // 2
            else:
                break

    def per_down(self, index, size):
        while 2 * index <= size:  # 是否有左子树？
            if 2 * index + 1 <= size:  # 是否有右子树？
                if self.compare_topper(self.heap_list[2 * index], self.heap_list[2 * index + 1]) <= 0:  # 左右子树孰小？
                    if self.compare_topper(self.heap_list[index], self.heap_list[2 * index]) <= 0:  # 本节点与最小子节点孰小？
                        break
                    else:
                        self.heap_list[index], self.heap_list[2 * index] = self.heap_list[2 * index], self.heap_list[
                            index]
                        index = index * 2
                else:
                    if self.compare_topper(self.heap_list[index], self.heap_list[2 * index + 1]) <= 0:
                        break
                    else:
                        self.heap_list[index], self.heap_list[2 * index + 1] = self.heap_list[2 * index + 1], \
                                                                               self.heap_list[index]
                        index = index * 2 + 1
            else:
                if self.compare_topper(self.heap_list[index], self.heap_list[2 * index]) <= 0:
                    break
                else:
                    self.heap_list[index], self.heap_list[2 * index] = self.heap_list[2 * index], self.heap_list[index]
                    index = index * 2

class MedianFinder(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.bigheap = []
        self.minheap = []

    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """
        if (len(self.bigheap) + len(self.minheap)) == 0:
            self.heapAdd(num, self.bigheap, 1)
            return
        if (len(self.bigheap) + len(self.minheap)) == 1:
            if self.bigheap[0] < num:
                self.heapAdd(num, self.minheap, 0)
            else:
                self.heapAdd(self.bigheap[0], self.minheap, 0)
                self.bigheap[0] = num
            return
        if num > self.minheap[0]:
            self.heapAdd(num, self.minheap, 0)
            if len(self.bigheap) + 2 == len(self.minheap):
                temp = self.heapPop(0)
                self.heapAdd(temp, self.bigheap, 1)
        else:
            if num > self.bigheap[0]:
                if len(self.bigheap) > len(self.minheap):
                    self.heapAdd(num, self.minheap, 0)
                else:
                    self.heapAdd(num, self.bigheap, 1)
            else:
                self.heapAdd(num, self.bigheap, 1)

                if len(self.bigheap) == len(self.minheap) + 2:
                    temp = self.heapPop(1)
                    self.heapAdd(temp, self.minheap, 0)

    def findMedian(self):
        """
        :rtype: float
        """
        if (len(self.bigheap) + len(self.minheap)) == 0:
            return None
        if (len(self.bigheap) + len(self.minheap)) == 1:
            return self.bigheap[0]
        if (len(self.bigheap) + len(self.minheap)) % 2 == 0:
            return (self.minheap[0] + self.bigheap[0]) / 2.0
        else:
            return float(self.minheap[0] if len(self.minheap) > len(self.bigheap) else self.bigheap[0])

            # Your MedianFinder object will be instantiated and called as such:
            # obj = MedianFinder()
            # obj.addNum(num)
            # param_2 = obj.findMedian()

    def buildHeap(self, nums, flag):
        i = int(len(nums) / 2)
        while i >= 1:
            self.adjust(nums, i, flag)
            i -= 1

    def adjust(self, nums, i, flag):
        if 2 * i - 1 > len(nums) - 1:
            return
        else:
            if flag == 1:
                if nums[i - 1] < nums[2 * i - 1]:
                    temp = nums[2 * i - 1]
                    nums[2 * i - 1] = nums[i - 1]
                    nums[i - 1] = temp
                    self.adjust(nums, 2 * i, flag)

            else:
                if nums[i - 1] > nums[2 * i - 1]:
                    temp = nums[2 * i - 1]
                    nums[2 * i - 1] = nums[i - 1]
                    nums[i - 1] = temp
                    self.adjust(nums, 2 * i, flag)

        if 2 * i > len(nums) - 1:
            return
        else:
            if flag == 1:
                if nums[i - 1] < nums[2 * i]:
                    temp = nums[2 * i]
                    nums[2 * i] = nums[i - 1]
                    nums[i - 1] = temp
                    self.adjust(nums, 2 * i + 1, flag)

            else:
                if nums[i - 1] > nums[2 * i]:
                    temp = nums[2 * i]
                    nums[2 * i] = nums[i - 1]
                    nums[i - 1] = temp
                    self.adjust(nums, 2 * i + 1, flag)

    def heapAdd(self, item, heap, flag):
        heap.append(item)
        j = len(heap)
        while j // 2 >= 1:
            if flag == 1:

                if heap[j // 2 - 1] < heap[j - 1]:
                    temp = heap[j // 2 - 1]
                    heap[j // 2 - 1] = heap[j - 1]
                    heap[j - 1] = temp
            else:
                if heap[j // 2 - 1] > heap[j - 1]:
                    temp = heap[j // 2 - 1]
                    heap[j // 2 - 1] = heap[j - 1]
                    heap[j - 1] = temp
            j = j // 2

    def heapPop(self, flag):
        if flag == 0:
            temp = self.minheap[0]
            self.minheap[0] = self.minheap[-1]
            self.minheap = self.minheap[:-1]
            self.adjust(self.minheap, 1, flag)
        if flag == 1:
            temp = self.bigheap[0]
            self.bigheap[0] = self.bigheap[-1]
            self.bigheap = self.bigheap[:-1]
            self.adjust(self.bigheap, 1, flag)
        return temp


'''
箱线图去除异常点
'''


def BoxDiagram(traceid2xy):
    x_mf = MedianFinder()
    y_mf = MedianFinder()
    for i, x in enumerate(traceid2xy):
        x_mf.addNum(x[0])
        y_mf.addNum(x[1])

    x_1_4_mf = MedianFinder()
    x_3_4_mf = MedianFinder()
    y_1_4_mf = MedianFinder()
    y_3_4_mf = MedianFinder()

    for x in x_mf.bigheap:
        x_1_4_mf.addNum(x)
    for x in x_mf.minheap:
        x_3_4_mf.addNum(x)
    for x in y_mf.bigheap:
        y_1_4_mf.addNum(x)
    for x in y_mf.minheap:
        y_3_4_mf.addNum(x)

    x_Q1 = x_1_4_mf.findMedian()
    x_Q3 = x_3_4_mf.findMedian()
    x_IQR = x_Q3 - x_Q1
    y_Q1 = y_1_4_mf.findMedian()
    y_Q3 = y_3_4_mf.findMedian()
    y_IQR = y_Q3 - y_Q1

    x_down_limitation = x_Q1 - 1.5 * x_IQR
    x_up_limitation = x_Q3 + 1.5 * x_IQR
    y_down_limitation = y_Q1 - 1.5 * y_IQR
    y_up_limitation = y_Q3 + 1.5 * y_IQR

    new_traceid2xy = []
    for i, x in enumerate(traceid2xy):
        if x[0] < x_down_limitation or x[0] > x_up_limitation or x[1] < y_down_limitation or x[1] > y_up_limitation:
            if not (x[2] == 0 and x[3] == 1):
                print("箱线图去除异常点")
                continue
        new_traceid2xy.append([x[0], x[1], x[2], x[3]])
    return new_traceid2xy

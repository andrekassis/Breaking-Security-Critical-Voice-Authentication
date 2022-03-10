class item:
    def __init__(self, Tp):
        self.key, self.val = Tp[0], Tp[1]

    def __lt__(self, other):
        return self.val < other.val

    def __eq__(self, other):
        return self.val == other.val


class SortedDict:
    def __init__(self):
        self.lst = []
        self.map = {}

    def __binary_search(self, low, high, x, prev_mid=-1):
        if high >= low:
            mid = (high + low) // 2
            prev_mid = mid
            if item(self.lst[mid]) == x:
                return mid
            if item(self.lst[mid]) < x:
                return self.__binary_search(mid + 1, high, x, prev_mid)
            return self.__binary_search(low, mid - 1, x, prev_mid)

        if prev_mid < low:
            return prev_mid
        return high

    def _search(self, key, val):
        return self.__binary_search(0, len(self.lst) - 1, item((key, val)))

    def __setitem__(self, key, val):
        if key in self.map.keys():
            self.map[key] = (val, self.map[key][1])
            self.lst[self.map[key][1]] = (key, val)
            return
        idx = self._search(key, val)
        self.lst.insert(idx, (key, val))
        self.map[key] = (val, idx + 1)

    def get(self):
        return self.lst

    def __getitem__(self, key):
        return self.map[key]

    def __delitem__(self, key):
        raise NotImplementedError

    @staticmethod
    def fromlst(lst):
        SD = SortedDict()
        SD.lst = lst
        SD.map = {x[0]: (x[1], i) for i, x in enumerate(lst)}
        return SD

    @staticmethod
    def fromfile(filename, reader):
        with open(filename, "r", encoding="utf8") as f:
            lst = [reader(line.strip()) for line in f]
        return SortedDict.fromlst(lst)

    def tofile(self, filename):
        with open(filename, "w", encoding="utf8") as f:
            for line in self.lst:
                f.write(str(line) + "\n")

    def __str__(self):
        return str(self.get())

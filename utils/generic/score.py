# pylint: disable=W0123

import numpy as np


class Score:
    def __init__(self, cm, asv, num):
        self.cm = cm.tolist() if not isinstance(cm, list) else cm
        self.asv = asv.tolist() if not isinstance(asv, list) else asv
        self.num = num

    def __repr__(self):
        return (
            "Score("
            + repr(self.cm)
            + ", "
            + repr(self.asv)
            + ", "
            + repr(self.num)
            + ")"
        )

    def __str__(self):
        return self.__repr__()

    def __lt__(self, other):
        return np.mean(self.cm) < np.mean(other.cm)

    def __eq__(self, other):
        return np.mean(self.cm) == np.mean(other.cm)

    @staticmethod
    def reader():
        def parse_line(line):
            return eval(line)

        return parse_line

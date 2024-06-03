from enum import Enum
import numpy as np

class Note(Enum):
    E = 0
    F = 1
    Fsharp = 2
    G = 3
    Gsharp = 4
    A = 5
    Asharp = 6
    B = 7
    C = 8
    Csharp = 9
    D = 10
    Dsharp = 11

    def __add__(self, interval):
        return Note(np.mod(self.value + interval, 12))

    def __sub__(self, interval):
        return Note(np.mod(self.value - interval, 12))
from enum import Enum

class Device(Enum):
    CPU  = "cpu"
    CUDA = "cuda"

class SpatialPrecision(Enum):
    PERCENT_50  = 1
    PERCENT_100 = 2
    PERCENT_200 = 3
    PERCENT_400 = 4

class TemporalPrecision(Enum):
    PLANESTATS  = 1
    BUTTERAUGLI = 2
    TOPIQ       = 3

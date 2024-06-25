from enum import Enum

class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"

class SpatialPrecision(Enum):
    PERCENT_50 = 1
    PERCENT_100 = 2
    PERCENT_200 = 3
    PERCENT_400 = 4
    PERCENT_800 = 5

class TemporalPrecision(Enum):
    PlaneStats = 1
    Butteraugli = 2
    TOPIQ = 3
from collections import namedtuple
from app_global import *

map_disk = np.array([], dtype=np.intc)  # signed 32-bit integer
map_word = np.array([], dtype=np.ubyte)
map_boolean = np.array([], dtype=np.ubyte)

# pct_array with predefined size and data type for numerical computations
pct_array = np.zeros(MAX_CATS + 1, dtype=float)  # ARRAY[0..MaxCats] OF double

TLine = namedtuple('TLine', ['p1', 'p2'])

# 'RoadArray' will be a list of floats with a specific length
RoadArray = List[float]

# Array of strings, dynamic sizing
str_vector1 = List[str]

wind_rose = [[0.0 for _ in range(N_WIND_SPEEDS)] for _ in range(N_WIND_DIRECTIONS)]

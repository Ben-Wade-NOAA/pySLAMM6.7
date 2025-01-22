from uncert_defn import TInputDist, TSensParam
from calc_dist import r_normal
from app_global import *


"""
A class to handle uncertainty calculations for the SLAMM model.
Attributes:
    num_dists (int): Number of distributions.
    dist_array (Optional[List[TInputDist]]): Array of input distributions.
    num_sens (int): Number of sensitivity parameters.
    sens_array (Optional[List[TSensParam]]): Array of sensitivity parameters.
    pct_to_vary (int): Percentage to vary.
    seed (int): Seed for random number generation.
    iterations (int): Number of iterations.
    gis_start_num (int): GIS start number.
    use_seed (bool): Flag to use seed.
    use_segt_slope (bool): Flag to use segment slope.
    segt_slope (float): Segment slope value.
    unc_sens_iter (int): Uncertainty sensitivity iteration.
    unc_sens_row (int): Uncertainty sensitivity row.
    n_map_iter (int): Number of map iterations.
    map_deriving (int): Map deriving value.
    q_value (float): Q value.
    z_map (Optional[np.ndarray]): Z map.
    prev_z_map (Optional[np.ndarray]): Previous Z map.
    z_uncert_map (List[np.ndarray]): Final uncertainty maps with appropriate RMSE and correlation.
    map_rows (int): Number of map rows.
    map_cols (int): Number of map columns.
    output_path (str): Output path.
    csv_path (str): CSV path.
    sens_plus_loop (bool): Sensitivity plus loop flag.
"""
#def get_z(self, er: int, ec: int, map_rows: int, map_cols: int) -> float:
"""
Get the Z value at a specific row and column.
Args:
    er (int): Row index.
    ec (int): Column index.
    map_rows (int): Total number of rows in the map.
    map_cols (int): Total number of columns in the map.
Returns:
    float: Z value at the specified location.
"""
#def make_uncert_map(self, map_num: int, nr: int, nc: int, rmse: float, q_val: float):
"""
Create an uncertainty map.
Args:
    map_num (int): Map number.
    nr (int): Number of rows.
    nc (int): Number of columns.
    rmse (float): Root mean square error.
    q_val (float): Q value.
"""
#def thread_uncert_map(self, start_row: int, end_row: int, map_rows: int, map_cols: int, map_deriving: int, q_value: float, r_val: float) -> float:
"""
Execute uncertainty map calculations in a threaded manner.
Args:
    start_row (int): Starting row index.
    end_row (int): Ending row index.
    map_rows (int): Total number of rows in the map.
    map_cols (int): Total number of columns in the map.
    map_deriving (int): Map deriving value.
    q_value (float): Q value.
    r_val (float): R value.
Returns:
    float: Maximum change in the map.
"""
"""
Destructor for TSLAMM_Uncertainty.
"""
#def load_store(self, file, read_version_num: int, is_reading: bool, pss):
"""
Load or store the state of the TSLAMM_Uncertainty object.
Args:
    file: File object for reading or writing.
    read_version_num (int): Version number for reading.
    is_reading (bool): Flag indicating if the operation is reading.
    pss: Additional parameter for loading/storing.
"""

class TSLAMM_Uncertainty:
    def __init__(self):
        self.num_dists = 0
        self.dist_array: Optional[List[TInputDist]] = None
        self.num_sens = 0
        self.sens_array: Optional[List[TSensParam]] = None
        self.pct_to_vary = 15
        self.seed = 20
        self.iterations = 20
        self.gis_start_num = 1
        self.use_seed = True
        self.use_segt_slope = True
        self.segt_slope = 0.6596

        # no load or save below
        self.unc_sens_iter = 0
        self.unc_sens_row = 0
        self.n_map_iter = 0
        self.map_deriving = 0
        self.q_value = 0.0

        self.z_map = None
        self.prev_z_map = None
        self.z_uncert_map = [np.array([]), np.array([])]  # final uncert  maps with appropriate RMSE and correlation
        self.map_rows = 0
        self.map_cols = 0
        self.output_path = ''
        self.csv_path = ''
        self.sens_plus_loop = False

    def get_z(self, er, ec, map_rows, map_cols):
        if er < 0 or ec < 0 or er > map_rows - 1 or ec > map_cols - 1:
            return 0
        else:
            return self.prev_z_map[map_cols * er + ec]

    def make_uncert_map(self, map_num: int, nr: int, nc: int, rmse: float, q_val:float):
        r_val = 0.2
        map_rows, map_cols = nr, nc
        self.q_value = q_val

        if len(self.z_uncert_map) < nr * nc:
            self.z_uncert_map[map_num] = np.zeros(nr * nc)
        self.z_map = np.zeros(nr * nc)
        self.prev_z_map = np.zeros(nr * nc)

        # Initialize with Normal Distribution
        for er in range(nr):
            for ec in range(nc):
                index = nc * er + ec
                self.z_uncert_map[map_num][index] = r_normal(0, rmse)
                self.prev_z_map[index] = r_val * self.z_uncert_map[map_num][index]

        # Simplified parallel execute
        tolerance = rmse / 100
        max_change = float('inf')

        while max_change > tolerance:
            self.n_map_iter += 1
            max_change = self.thread_uncert_map(0, nr - 1, map_rows, map_cols, map_num, self.q_value, r_val)

        # Correct standard deviation
        sum_e2 = np.sum(np.square(self.z_uncert_map[map_num] + self.z_map))
        st_dev_correction = rmse / np.sqrt(sum_e2 / (nr * nc - 1))

        for er in range(nr):
            for ec in range(nc):
                index = nc * er + ec
                self.z_uncert_map[map_num][index] = (self.z_uncert_map[map_num][index] + self.z_map[index]) * st_dev_correction

        # Clean up
        self.z_map = None
        self.prev_z_map = None

    def thread_uncert_map(self, start_row, end_row, map_rows, map_cols, map_deriving, q_value, r_val):
        for er in range(start_row, end_row + 1):
            for ec in range(map_cols):
                self.z_map[map_cols * er + ec] = q_value * (
                        self.get_z(er - 1, ec, map_rows, map_cols) +
                        self.get_z(er + 1, ec, map_rows, map_cols) +
                        self.get_z(er, ec - 1, map_rows, map_cols) +
                        self.get_z(er, ec + 1, map_rows, map_cols)
                ) + self.z_uncert_map[map_deriving][map_cols * er + ec] * r_val

        max_change = -9999
        for er in range(start_row, end_row + 1):
            for ec in range(map_cols):
                index = map_cols * er + ec
                diff = abs(self.z_map[index] - self.prev_z_map[index])
                if diff > max_change:
                    max_change = diff
                self.prev_z_map[index] = self.z_map[index]

        return max_change

    def __del__(self):
        pass

    def load_store(self, file, read_version_num, is_reading, pss):
        self.num_dists = ts_read_write(file, 'NumDists', self.num_dists, int, is_reading)

        # Handling the array of distributions
        if is_reading:
            self.dist_array = [TInputDist(0, pss, False, 0) for _ in range(self.num_dists)]
            for i in range(self.num_dists):
                self.dist_array[i].load_store(file, read_version_num, is_reading)
        else:
            if self.dist_array:
                for dist in self.dist_array:
                    dist.load_store(file, read_version_num, is_reading)

        self.seed = ts_read_write(file, 'Seed', self.seed, int, is_reading)
        self.iterations = ts_read_write(file, 'Iterations', self.iterations, int, is_reading)
        self.gis_start_num = ts_read_write(file, 'GIS_Start_Num', self.gis_start_num, int, is_reading)
        self.use_seed = ts_read_write(file, 'UseSeed', self.use_seed, bool, is_reading)
        self.use_segt_slope = ts_read_write(file, 'UseSEGTSlope', self.use_segt_slope, bool, is_reading)
        self.segt_slope = ts_read_write(file, 'SEGTSlope', self.segt_slope, float, is_reading)

        self.num_sens = ts_read_write(file, 'NumSens', self.num_sens, int, is_reading)
        self.pct_to_vary = ts_read_write(file, 'PctToVary', self.pct_to_vary, int, is_reading)

        # Handling the array of sensitivities
        if is_reading:
            self.sens_array = [None] * self.num_sens
            for i in range(self.num_sens):
                self.sens_array[i] = TSensParam(0, pss, False, 0)
                self.sens_array[i].load_store(file, read_version_num, is_reading)
        else:
            if self.sens_array:
                for sens in self.sens_array:
                    sens.load_store(file, read_version_num, is_reading)

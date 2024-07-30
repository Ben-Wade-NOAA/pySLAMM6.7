from uncert_defn import TInputDist, TSensParam
from calc_dist import r_normal
from app_global import *


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

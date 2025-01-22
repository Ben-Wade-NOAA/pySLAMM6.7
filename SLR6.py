# import numpy as np
# from dataclasses import dataclass, field
# from numba import jit


"""
This module contains classes and functions for managing shared memory arrays, 
shared data structures, and multiprocessing tasks for a SLAMM (Sea Level Affecting Marshes Model) simulation.
Classes:
    SharedMemoryArray: Manages shared memory arrays for multiprocessing.
    SharedData: Manages shared data structures used by threads or the main process.
    SumData: Data class for storing summary output.
    TSLAMM_Simulation: Primary SLAMM simulation object.
Functions:
    jit_get_bit: Get the state of a bit in the b_matrix.
    worker_function: Worker function for multiprocessing tasks.
"""


import sys
import traceback
import pickle
import math
from math import tanh, pi, sinh
from typing import Dict, TextIO, Any, Tuple
import datetime
import time
from datetime import datetime
import app_global
from app_global import *
from categories import TCategories
from SalArray import TSalArray
from infr_data import TRoadInfrastructure, TPointInfrastructure
from raster_files import TSLAMMOutputFile, FileFormat, TSLAMMInputFile
from uncert import TSLAMM_Uncertainty
from utility import cell_width, get_min_elev, get_cell_cat, set_cell_width, float_to_word, set_cat_elev, cat_elev
from multiprocessing import shared_memory, Process, Queue

def jit_get_bit(b_matrix: np.ndarray, cols: int, n: int, row: int, col: int):
    """ Get the state of a bit in the b_matrix. """
    return (b_matrix[row, col] & n) > 0


class SharedMemoryArray:  # this array is passed between multiprocessing instances
    def __init__(self, shape=None, dtype=None, name=None):
        if name:
            self.shm = shared_memory.SharedMemory(name=name)
            self.array = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        elif shape and dtype:
            self.shape = shape
            self.dtype = np.dtype(dtype)  # Ensure dtype is a numpy dtype
            self.size = int(np.prod(shape))
            self.shm = shared_memory.SharedMemory(create=True, size=self.size * self.dtype.itemsize)
            self.array = np.ndarray(shape, dtype=self.dtype, buffer=self.shm.buf)
        else:
            self.shm = None
            self.array = None

    # def __reduce__(self):
    #     print("WARNING:  Pickling SharedMemoryArray  *  *  *  *")    This code ensured shared_memory working properly
    #     return (self.__class__, (self.shape, self.dtype, self.shm.name))

    def close(self):
        if self.shm:
            self.shm.close()

    def unlink(self):
        if self.shm:
            self.shm.unlink()

    def get_array(self):
        return self.array

    @staticmethod
    def from_existing(name, shape, dtype):
        # dtype = np.dtype(dtype)  # Ensure dtype is a numpy dtype
        # shm = shared_memory.SharedMemory(name=name)
        return SharedMemoryArray(shape, dtype, name=name)

    def __getitem__(self, index):
        return self.array[index]

    def __setitem__(self, index, value):
        self.array[index] = value


class SharedData:  # shared data datastructure that is used by threads or main process
    def __init__(self):
        self.map = []
        self.b_matrix = None
        self.erode_matrix = None
        self.map_matrix = None
        self.data_elev = None
        self.max_fetch_arr = None
        self.ros_array = None

    def initialize_map(self, num_blocks, block_size, dtype):
        self.map = [SharedMemoryArray((block_size,), dtype) for _ in range(num_blocks)]

    def initialize_b_matrix(self, shape, dtype=np.uint8):
        self.b_matrix = SharedMemoryArray(shape, dtype)

    def initialize_erode_matrix(self, shape, dtype=np.uint16):
        self.erode_matrix = SharedMemoryArray(shape, dtype)

    def initialize_map_matrix(self, shape, dtype=np.int32):
        self.map_matrix = SharedMemoryArray(shape, dtype)

    def initialize_data_elev(self, shape, dtype=np.uint16):
        self.data_elev = SharedMemoryArray(shape, dtype)

    def initialize_max_fetch_arr(self, shape, dtype=np.uint16):
        self.max_fetch_arr = SharedMemoryArray(shape, dtype)

    def initialize_ros_array(self, shape, dtype=np.uint8):
        self.ros_array = SharedMemoryArray(shape, dtype)

    def close(self):
        # Close shared memory views (detach them from the process)
        for shm_array in self.map:
            shm_array.close()
        if self.b_matrix:
            self.b_matrix.close()
        if self.erode_matrix:
            self.erode_matrix.close()
        if self.map_matrix:
            self.map_matrix.close()
        if self.data_elev:
            self.data_elev.close()
        if self.max_fetch_arr:
            self.max_fetch_arr.close()
        if self.ros_array:
            self.ros_array.close()

    def unlink(self):
        # Unlink (remove) shared memory segments
        for shm_array in self.map:
            shm_array.unlink()
        if self.b_matrix:
            self.b_matrix.unlink()
        if self.erode_matrix:
            self.erode_matrix.unlink()
        if self.map_matrix:
            self.map_matrix.unlink()
        if self.data_elev:
            self.data_elev.unlink()
        if self.max_fetch_arr:
            self.max_fetch_arr.unlink()
        if self.ros_array:
            self.ros_array.unlink()

    def reinitialize_from_shared_memory(self, names, shapes, dtypes):
        try:
            self.map = [SharedMemoryArray.from_existing(name, shape, dtype) for name, shape, dtype in
                        zip(names['map'], shapes['map'], dtypes['map'])]
            if names.get('b_matrix') is not None:
                self.b_matrix = SharedMemoryArray.from_existing(names['b_matrix'], shapes['b_matrix'],
                                                                dtypes['b_matrix'])
            if names.get('erode_matrix') is not None:
                self.erode_matrix = SharedMemoryArray.from_existing(names['erode_matrix'], shapes['erode_matrix'],
                                                                    dtypes['erode_matrix'])
            if names.get('map_matrix') is not None:
                self.map_matrix = SharedMemoryArray.from_existing(names['map_matrix'], shapes['map_matrix'],
                                                                  dtypes['map_matrix'])

            if names.get('data_elev') is not None:
                self.data_elev = SharedMemoryArray.from_existing(names['data_elev'], shapes['data_elev'],
                                                                 dtypes['data_elev'])
            if names.get('max_fetch_arr') is not None:
                self.max_fetch_arr = SharedMemoryArray.from_existing(names['max_fetch_arr'], shapes['max_fetch_arr'],
                                                                     dtypes['max_fetch_arr'])
            if names.get('ros_array') is not None:
                self.ros_array = SharedMemoryArray.from_existing(names['ros_array'], shapes['ros_array'],
                                                                 dtypes['ros_array'])
        except Exception as e:
            print(f"Error in reinitializing shared memory: {e}")
            traceback.print_exc()

    def get_shared_memory_info(self):
        shared_memory_names = {
            'map': [map_array.shm.name for map_array in self.map] if self.map else None,
            'b_matrix': self.b_matrix.shm.name if self.b_matrix else None,
            'erode_matrix': self.erode_matrix.shm.name if self.erode_matrix else None,
            'map_matrix': self.map_matrix.shm.name if self.map_matrix else None,
            'data_elev': self.data_elev.shm.name if self.data_elev else None,
            'max_fetch_arr': self.max_fetch_arr.shm.name if self.max_fetch_arr else None,
            'ros_array': self.ros_array.shm.name if self.ros_array and self.ros_array.size > 0 else None
        }

        shapes = {
            'map': [map_array.shape for map_array in self.map] if self.map else None,
            'b_matrix': self.b_matrix.shape if self.b_matrix else None,
            'erode_matrix': self.erode_matrix.shape if self.erode_matrix else None,
            'map_matrix': self.map_matrix.shape if self.map_matrix else None,
            'data_elev': self.data_elev.shape if self.data_elev else None,
            'max_fetch_arr': self.max_fetch_arr.shape if self.max_fetch_arr else None,
            'ros_array': self.ros_array.shape if self.ros_array and self.ros_array.size > 0 else None
        }

        dtypes = {
            'map': [map_array.dtype for map_array in self.map] if self.map else None,
            'b_matrix': self.b_matrix.dtype if self.b_matrix else None,
            'erode_matrix': self.erode_matrix.dtype if self.erode_matrix else None,
            'map_matrix': self.map_matrix.dtype if self.map_matrix else None,
            'data_elev': self.data_elev.dtype if self.data_elev else None,
            'max_fetch_arr': self.max_fetch_arr.dtype if self.max_fetch_arr else None,
            'ros_array': self.ros_array.dtype if self.ros_array and self.ros_array.size > 0 else None
        }

        return shared_memory_names, shapes, dtypes


#  worker_function for multi processing defined at the module level
def worker_function(start_row, end_row, args, calculate_sums, queue, shared_memory_names, shapes, dtypes,
                    site_n_output_sites, site_max_ros, task_function, *task_args):
    try:
        shared_data = SharedData()
        shared_data.reinitialize_from_shared_memory(shared_memory_names, shapes, dtypes)

        if calculate_sums:
            sumdat = SumData()
            sumdat.cat_sums = np.zeros((site_n_output_sites + 1 + site_max_ros, MAX_CATS))
            sumdat.road_sums = np.zeros((site_n_output_sites + 1 + site_max_ros, MAX_ROAD_ARRAY))

            if args == (None,):
                sumdat = task_function(start_row, end_row, shared_data, sumdat)
            else:
                sumdat = task_function(start_row, end_row, shared_data, *task_args, sumdat)

            queue.put(sumdat)
        else:
            if args == (None,):
                task_function(start_row, end_row, shared_data)
            else:
                task_function(start_row, end_row, shared_data, *task_args)

    except Exception as e:
        print(f"Error in worker_function: {e}")
        traceback.print_exc()
    except BaseException as e:
        print(f"Critical error in worker_function: {e}")
        traceback.print_exc()
    finally:
        if calculate_sums:
            queue.put(None)  # Sentinel value to signal thread completion

    # Close the shared memory views
    shared_data.close()

@dataclass
class SumData:
    cat_sums = None  # will be 2-D numpy array with summary output
    road_sums = None  # will be 2-D numpy array with summary road inundation output


@dataclass
class TSLAMM_Simulation:  # primary slamm simulation object
    file_name: str = ''
    sim_name: str = ''
    description: str = ''
    categories: TCategories = None
    site: TSite = None
    num_fw_flows: int = 0
    sal_rules: TSalinityRules = None
    sal_array: TSalArray = None

    n_road_inf: int = 0
    n_point_inf: int = 0

    n_time_ser_slr: int = 0

    time_step: int = 25
    max_year: int = 2100
    run_specific_years: bool = False
    years_string: str = ''
    run_uncertainty: bool = False
    run_sensitivity: bool = False
    uncert_setup: TSLAMM_Uncertainty = None

    run_custom_slr: bool = False
    n_custom_slr: int = 0
    current_custom_slr: float = 0.0

    display_screen_maps: bool = False
    qa_tools: bool = False
    run_first_year: bool = True
    maps_to_ms_word: bool = False
    maps_to_gif: bool = False
    complete_run_rec: bool = True
    salinity_maps: bool = False
    accretion_maps: bool = False
    ducks_maps: bool = False
    sav_maps: bool = False
    connect_maps: bool = False
    inund_maps: bool = False
    road_inund_maps: bool = False
    save_gis: bool = False
    save_ros_area: bool = False
    save_elevs_gis: bool = False
    save_salinity_gis: bool = False
    save_inund_gis: bool = False
    save_elev_gis_mtl: bool = False
    save_binary_gis: bool = False
    batch_mode: bool = False

    check_connectivity: bool = False
    connect_min_elev: int = 0
    connect_near_eight: int = 0

    cell_size_mult: int = 0

    use_soil_saturation: bool = False
    load_blank_if_no_elev: bool = True

    use_bruun: bool = False
    init_zoom_factor: float = 1.0

    use_flood_forest: bool = False
    use_flood_dev_dry_land: bool = False

    include_dikes: bool = False
    classic_dike: bool = True

    init_elev_stats: bool = False
    n_elev_stats: int = 0
    elev_stats_derived: List[datetime] = field(default_factory=list)

    # elev_grid_colors: Optional['ElevGridColorType'] = None    this is part of the interface

    batch_file_name: str = ''
    elev_file_name: str = ''
    nwi_file_name: str = ''
    slp_file_name: str = ''
    output_file_name: str = ''
    imp_file_name: str = ''
    ros_file_name: str = ''
    dik_file_name: str = ''
    vd_file_name: str = ''
    uplift_file_name: str = ''
    sal_file_name: str = ''
    # old_road_file_name: str = ''
    d2m_file_name: str = ''
    storm_file_name: str = ''

    optimize_level: int = 1
    num_mm_entries: int = 0
    sav_params: Optional['SAVParamsRec'] = SAVParamsRec(0, 0, 0, 0, 0, 0, 0, 0)

    sal_stats_derived: Optional[datetime] = None
    init_sal_stats: bool = False

    tropical: bool = False

    #  No Load or Save below

    changed: bool = True
    tstep_iter: int = 0
    n_time_steps: int = 1
    scen_iter: int = 0
    run_time: str = ""
    start_time: float = None
    memory_load_time: float = None
    cell_ha: float = 0.0
    hectares: float = 0.0
    year: int = 0
    time_zero: bool = False
    run_record_file_name: str = ""
    max_w_erode: float = 0.0
    protect_developed: bool = False
    protect_all: bool = False
    running_fixed: bool = False
    running_custom: bool = False
    running_tsslr: bool = False
    ipcc_sl_rate: IPCCScenarios = IPCCScenarios.Scen_A1B
    ipcc_sl_est: IPCCEstimates = IPCCEstimates.Est_Min
    fixed_scen: int = 0
    tsslr_index: int = 0
    proj_run_string: str = ""
    short_proj_run: str = ""
    max_mtl: float = 0.0
    min_mtl: float = 0.0

    backup_fn: str = ""
    dike_info: List[Any] = field(default_factory=list)  # Define type for DikeInfoRec
    word_app: Any = None
    word_doc: Any = None
    word_initialized: bool = False
    num_rows: int = 0
    summary = None  # will be 3D Numpy Array of floats
    col_label: List[str] = field(default_factory=list)
    file_started: bool = False
    row_label: List[str] = field(default_factory=list)

    cat_sums = None  # will be 2-D numpy array with summary output
    road_sums = None  # will be 2-D numpy array with summary road inundation output

    blank_cell: compressed_cell_dtype = field(default_factory=lambda: compressed_cell_dtype())
    und_dry_cell: compressed_cell_dtype = field(default_factory=lambda: compressed_cell_dtype())
    dev_dry_cell: compressed_cell_dtype = field(default_factory=lambda: compressed_cell_dtype())
    ocean_cell: compressed_cell_dtype = field(default_factory=lambda: compressed_cell_dtype())
    eow_cell: compressed_cell_dtype = field(default_factory=lambda: compressed_cell_dtype())
    user_stop: bool = False
    sav_km: float = -9999
    ros_resize: int = 3  # 100%
    dike_log_init: bool = False
    dike_log: Optional[TextIO] = None
    sav_prob_check: bool = False
    connect_check: bool = False
    connect_arr = None
    inund_arr: np.array([], dtype=np.ubyte) = None
    inund_freq_check: bool = False
    road_inund_check: bool = False

    # def __del__(self):
    #     self.dispose_mem()

    def __init__(self, california: bool = False):
        super().__init__()

        self.fw_flows: List[TFWFlow] = []

        self.use_ss_raster: List[bool] = [False, False]
        self.ss_raster_slr: List[bool] = [False, False]

        self.ss_filen: List[str] = ["", ""]
        self.ss_slr: List[List[float]] = [[0.0, 0.0], [0.0, 0.0]]
        self.ss_rasters: List[List[np.ndarray]] = [
            [np.array([], dtype=np.uint16), np.array([], dtype=np.uint16)],
            [np.array([], dtype=np.uint16), np.array([], dtype=np.uint16)]
        ]

        self.roads_inf: List['TRoadInfrastructure'] = []
        self.point_inf: List['TPointInfrastructure'] = []
        self.time_ser_slrs: List[TTimeSerSLR] = []
        self.ipcc_scenarios: Dict['IPCCScenarios', bool] = {scenario: False for scenario in IPCCScenarios}
        self.ipcc_estimates: Dict['IPCCEstimates', bool] = {scenario: False for scenario in IPCCEstimates}
        self.prot_to_run: Dict['ProtectScenario', bool] = {scenario: False for scenario in ProtectScenario}

        self.shared_data = None  # All shared data for multi-processing  type SharedData(seif.site)
        #  variables below added to shared_data
        # self.shared_data.map: List[SharedMemoryArray] = []  # will be type compressed_cell_dtype
        # self.b_matrix: Optional[np.ndarray] = None  # will be type np.uint8
        # self.erode_matrix: Optional[np.ndarray] = None  # will be type np.uint16
        # self.shared_data.map_matrix: Optional[np.ndarray] = None  # will be type np.int32
        # self.data_elev: Optional[np.ndarray] = None  # will be type np.uint16
        # self.max_fetch_arr: Optional[np.ndarray] = None  # will be type np.uint16
        # self.ros_array: Optional[np.ndarray] = None  # will be type np.uint8

        self.fixed_scenarios: List[bool] = [False] * 11
        self.custom_slr_array: List[float] = []
        
        self.n_time_ser_slr = 2
        while len(self.time_ser_slrs) < self.n_time_ser_slr:
                self.time_ser_slrs.append(TTimeSerSLR(len(self.time_ser_slrs)+1))  # 8/30/2024 add example time series
        
        self.gis_years = "2100"
        self.gis_each_year = True
        self.cpu_count = os.cpu_count() or 1  # Cache the CPU count   1 to go to one CPU  min(os.cpu_count(),4)

        self.wind_rose_initialized = True
        self.wind_rose: List[List[float]] = [[0.0 for _ in range(N_WIND_SPEEDS)] for _ in range(N_WIND_DIRECTIONS)]
        self.wind_rose_initialized = True

        self.sal_rules = TSalinityRules()

        self.categories = TCategories(self)
        if california:
            self.categories.setup_ca_default()
        else:
            self.categories.setup_slamm_default()

        self.site = TSite()
        self.uncert_setup = TSLAMM_Uncertainty()

    def load_store(self, file, filen, read_version_num, is_reading, split_files=False):
        if is_reading: self.file_name = filen
        current_file = file
        line_number = 0

        read_version_num = ts_read_write(current_file, 'ReadVersionNum', read_version_num, float, is_reading)

        # elev_stats in/out
        if is_reading or split_files:
            current_file = split_file(current_file, 'elev_stats', is_reading)
        self.init_elev_stats = ts_read_write(current_file, 'Init_ElevStats', self.init_elev_stats, bool, is_reading)
        self.n_elev_stats = 0 if not self.init_elev_stats else self.n_elev_stats
        self.n_elev_stats = ts_read_write(current_file, 'NElevStats', self.n_elev_stats, int, is_reading)
        current_file = revert_file(current_file, file)

        if is_reading or split_files:
            current_file = split_file(current_file, 'categories', is_reading)
        self.categories.load_store(current_file, read_version_num, is_reading)
        current_file = revert_file(current_file, file)

        self.sim_name = ts_read_write(current_file, 'SimName', self.sim_name, str, is_reading)
        self.description = ts_read_write(current_file, 'Descrip', self.description, str, is_reading)

        if is_reading or split_files:
            current_file = split_file(current_file, 'site_data', is_reading)
        self.site.load_store(current_file, read_version_num, is_reading)
        current_file = revert_file(current_file, file)

        if is_reading or split_files:
            current_file = split_file(current_file, 'wind_rose', is_reading)
        for nd in range(N_WIND_DIRECTIONS):
            for ns in range(N_WIND_SPEEDS):
                key = f'WindRose[{nd + 1},{ns + 1}]'  # Converting to 1-based index for compatibility
                self.wind_rose[nd][ns] = ts_read_write(current_file, key, self.wind_rose[nd][ns], float, is_reading)
        current_file = revert_file(current_file, file)

        if is_reading or split_files:
            current_file = split_file(current_file, 'salinity', is_reading)
        self.num_fw_flows = ts_read_write(current_file, 'NumFwFlows', self.num_fw_flows, int, is_reading)
        if is_reading:
            while len(self.fw_flows) < self.num_fw_flows:
                self.fw_flows.append(TFWFlow())
        for i in range(self.num_fw_flows):
            self.fw_flows[i].load_store(current_file, read_version_num, is_reading)
        self.sal_rules.load_store(current_file, read_version_num, is_reading)
        current_file = revert_file(current_file, file)

        if is_reading or split_files:
            current_file = split_file(current_file, 'infrastructure', is_reading)
        self.n_road_inf = ts_read_write(current_file, 'NRoadInf', self.n_road_inf, int, is_reading)
        if is_reading:
            while len(self.roads_inf) < self.n_road_inf:
                self.roads_inf.append(TRoadInfrastructure(self))
        for i in range(self.n_road_inf):
            self.roads_inf[i].load_store(current_file, read_version_num, is_reading, self)
        self.n_point_inf = ts_read_write(current_file, 'NPointInf', self.n_point_inf, int, is_reading)
        if is_reading:
            while len(self.point_inf) < self.n_point_inf:
                self.point_inf.append(TPointInfrastructure(self))
        for i in range(self.n_point_inf):
            self.point_inf[i].load_store(current_file, read_version_num, is_reading, self)
        current_file = revert_file(current_file, file)

        if is_reading or split_files:
            current_file = split_file(current_file, 'run_options', is_reading)
        self.n_time_ser_slr = ts_read_write(current_file, 'NTimeSerSLR', self.n_time_ser_slr, int, is_reading)
        if is_reading:
            while len(self.time_ser_slrs) < self.n_time_ser_slr:
                self.time_ser_slrs.append(TTimeSerSLR())
        for i in range(self.n_time_ser_slr):
            self.time_ser_slrs[i].load_store(current_file, read_version_num, is_reading)

        self.time_step = ts_read_write(current_file, 'TimeStep', self.time_step, int, is_reading)
        self.max_year = ts_read_write(current_file, 'MaxYear', self.max_year, int, is_reading)
        self.run_specific_years = ts_read_write(current_file, 'RunSpecificYears', self.run_specific_years, bool, is_reading)
        self.years_string = ts_read_write(current_file, 'YearsString', self.years_string, str, is_reading)
        self.run_uncertainty = ts_read_write(current_file, 'RunUncertainty', self.run_uncertainty, bool, is_reading)
        self.run_sensitivity = ts_read_write(current_file, 'RunSensitivity', self.run_sensitivity, bool, is_reading)

        self.ipcc_scenarios[IPCCScenarios.Scen_A1B] = \
            ts_read_write(current_file, 'Scen_A1B', self.ipcc_scenarios.get(IPCCScenarios.Scen_A1B, False), bool, is_reading)
        self.ipcc_scenarios[IPCCScenarios.Scen_A1T] = \
            ts_read_write(current_file, 'Scen_A1T', self.ipcc_scenarios.get(IPCCScenarios.Scen_A1T, False), bool, is_reading)
        self.ipcc_scenarios[IPCCScenarios.Scen_A1F1] = \
            ts_read_write(current_file, 'Scen_A1F1', self.ipcc_scenarios.get(IPCCScenarios.Scen_A1F1, False), bool, is_reading)
        self.ipcc_scenarios[IPCCScenarios.Scen_A2] = \
            ts_read_write(current_file, 'Scen_A2', self.ipcc_scenarios.get(IPCCScenarios.Scen_A2, False), bool, is_reading)
        self.ipcc_scenarios[IPCCScenarios.Scen_B1] = \
            ts_read_write(current_file, 'Scen_B1', self.ipcc_scenarios.get(IPCCScenarios.Scen_B1, False), bool, is_reading)
        self.ipcc_scenarios[IPCCScenarios.Scen_B2] = \
            ts_read_write(current_file, 'Scen_B2', self.ipcc_scenarios.get(IPCCScenarios.Scen_B2, False), bool, is_reading)

        # IPCC Estimates
        self.ipcc_estimates[IPCCEstimates.Est_Min] = \
            ts_read_write(current_file, 'Est_Min', self.ipcc_estimates.get(IPCCEstimates.Est_Min, False), bool, is_reading)
        self.ipcc_estimates[IPCCEstimates.Est_Mean] = \
            ts_read_write(current_file, 'Est_Mean', self.ipcc_estimates.get(IPCCEstimates.Est_Mean, False), bool, is_reading)
        self.ipcc_estimates[IPCCEstimates.Est_Max] = \
            ts_read_write(current_file, 'Est_Max', self.ipcc_estimates.get(IPCCEstimates.Est_Max, False), bool, is_reading)

        self.fixed_scenarios = [
            ts_read_write(current_file, 'Fix1.0M', self.fixed_scenarios[0], bool, is_reading),
            ts_read_write(current_file, 'Fix1.5M', self.fixed_scenarios[1], bool, is_reading),
            ts_read_write(current_file, 'Fix2.0M', self.fixed_scenarios[2], bool, is_reading),
            ts_read_write(current_file, 'NYS_GCM_Max', self.fixed_scenarios[3], bool, is_reading),
            ts_read_write(current_file, 'NYS_1M_2100', self.fixed_scenarios[4], bool, is_reading),
            ts_read_write(current_file, 'NYS_RIM_Min', self.fixed_scenarios[5], bool, is_reading),
            ts_read_write(current_file, 'NYS_RIM_Max', self.fixed_scenarios[6], bool, is_reading),
            ts_read_write(current_file, 'ESVA_Hist', self.fixed_scenarios[7], bool, is_reading),
            ts_read_write(current_file, 'ESVA_Low', self.fixed_scenarios[8], bool, is_reading),
            ts_read_write(current_file, 'ESVA_High', self.fixed_scenarios[9], bool, is_reading),
            ts_read_write(current_file, 'ESVA_Highest', self.fixed_scenarios[10], bool, is_reading),
        ]

        # Protection Scenarios
        self.prot_to_run[ProtectScenario.NoProtect] = ts_read_write(current_file, 'Prot_To_Run[NoProtect]',
                                                                    self.prot_to_run.get(ProtectScenario.NoProtect,
                                                                                         False), bool, is_reading)
        self.prot_to_run[ProtectScenario.ProtDeveloped] = ts_read_write(current_file, 'Prot_To_Run[ProtDeveloped]',
                                                                        self.prot_to_run.get(
                                                                            ProtectScenario.ProtDeveloped, False), bool,
                                                                        is_reading)
        self.prot_to_run[ProtectScenario.ProtAll] = ts_read_write(current_file, 'Prot_To_Run[ProtAll]',
                                                                  self.prot_to_run.get(ProtectScenario.ProtAll, False),
                                                                  bool, is_reading)

        self.run_custom_slr = ts_read_write(current_file, 'RunCustomSLR', self.run_custom_slr, bool, is_reading)
        self.n_custom_slr = ts_read_write(current_file, 'N_CustomSLR', self.n_custom_slr, int, is_reading)
        if is_reading:
            self.custom_slr_array = [int for _ in range(self.n_custom_slr)]
        self.custom_slr_array = [ts_read_write(current_file, 'CustomSLRArray[i]', slr, float, is_reading) for i, slr in
                                 enumerate(self.custom_slr_array)]

        ts_read_write(current_file, 'Make_Data_File', False, bool, is_reading)  # discarded
        self.display_screen_maps = ts_read_write(current_file, 'Display_Screen_Maps', self.display_screen_maps, bool,
                                                 is_reading)
        self.qa_tools = ts_read_write(current_file, 'QA_Tools', self.qa_tools, bool, is_reading)
        self.run_first_year = ts_read_write(current_file, 'RunFirstYear', self.run_first_year, bool, is_reading)
        ts_read_write(current_file, 'Maps_to_MSWord', False, bool, is_reading)  # discarded
        self.maps_to_gif = ts_read_write(current_file, 'Maps_to_GIF', self.maps_to_gif, bool, is_reading)
        self.complete_run_rec = ts_read_write(current_file, 'Complete_RunRec', self.complete_run_rec, bool, is_reading)

        self.salinity_maps = ts_read_write(current_file, 'SalinityMaps', self.salinity_maps, bool, is_reading)
        self.accretion_maps = ts_read_write(current_file, 'AccretionMaps', self.accretion_maps, bool, is_reading)
        self.ducks_maps = ts_read_write(current_file, 'DucksMaps', self.ducks_maps, bool, is_reading)
        self.sav_maps = ts_read_write(current_file, 'SAVMaps', self.sav_maps, bool, is_reading)
        self.connect_maps = ts_read_write(current_file, 'ConnectMaps', self.connect_maps, bool, is_reading)
        self.inund_maps = ts_read_write(current_file, 'InundMaps', self.inund_maps, bool, is_reading)
        self.road_inund_maps = ts_read_write(current_file, 'RoadInundMaps', self.road_inund_maps, bool,
                                             is_reading)  # note same key as inund_maps

        self.save_gis = ts_read_write(current_file, 'SaveGIS', self.save_gis, bool, is_reading)
        self.save_elevs_gis = ts_read_write(current_file, 'SaveElevsGIS', self.save_elevs_gis, bool, is_reading)
        self.save_elev_gis_mtl = ts_read_write(current_file, 'SaveElevGISMTL', self.save_elev_gis_mtl, bool, is_reading)
        self.save_salinity_gis = ts_read_write(current_file, 'SaveSalinityGIS', self.save_salinity_gis, bool, is_reading)
        self.save_inund_gis = ts_read_write(current_file, 'SaveInundGIS', self.save_inund_gis, bool, is_reading)

        self.save_binary_gis = ts_read_write(current_file, 'SaveBinaryGIS', self.save_binary_gis, bool, is_reading)
        self.save_ros_area = ts_read_write(current_file, 'SaveROSArea', self.save_ros_area, bool, is_reading)
        self.batch_mode = ts_read_write(current_file, 'BatchMode', self.batch_mode, bool, is_reading)

        self.check_connectivity = ts_read_write(current_file, 'CheckConnectivity', self.check_connectivity, bool, is_reading)
        self.connect_min_elev = ts_read_write(current_file, 'ConnectMinElev', self.connect_min_elev, float, is_reading)
        self.connect_near_eight = ts_read_write(current_file, 'ConnectNearEight', self.connect_near_eight, bool, is_reading)
        self.use_soil_saturation = ts_read_write(current_file, 'UseSoilSaturation', self.use_soil_saturation, bool, is_reading)
        self.load_blank_if_no_elev = ts_read_write(current_file, 'LoadBlankIfNoElev', self.load_blank_if_no_elev, bool,
                                                   is_reading)

        self.use_bruun = ts_read_write(current_file, 'UseBruun', self.use_bruun, bool, is_reading)

        self.use_flood_forest = ts_read_write(current_file, 'UseFloodForest', self.use_flood_forest, bool, is_reading)
        self.use_flood_dev_dry_land = ts_read_write(current_file, 'UseFloodDevDryLand', self.use_flood_dev_dry_land, bool,
                                                    is_reading)

        self.init_zoom_factor = ts_read_write(current_file, 'InitZoomFactor', self.init_zoom_factor, float, is_reading)

        ts_read_write(current_file, 'SaveMapsToDisk', False, bool, is_reading)  # discarded
        ts_read_write(current_file, 'MakeNewDiskMap', False, bool, is_reading)  # discarded
        self.include_dikes = ts_read_write(current_file, 'IncludeDikes', self.include_dikes, bool, is_reading)
        self.classic_dike = ts_read_write(current_file, 'ClassicDike', self.classic_dike, bool, is_reading)
        current_file = revert_file(current_file, file)

        if is_reading or split_files:
            current_file = split_file(current_file, 'file_setup', is_reading)
        self.batch_file_name = ts_read_write(current_file, 'BatchFileN', self.batch_file_name, str, is_reading)
        self.elev_file_name = ts_read_write(current_file, 'ElevFileN', self.elev_file_name, str, is_reading)
        self.nwi_file_name = ts_read_write(current_file, 'NWIFileN', self.nwi_file_name, str, is_reading)
        self.output_file_name = ts_read_write(current_file, 'OutputFileN', self.output_file_name, str, is_reading)
        self.slp_file_name = ts_read_write(current_file, 'SLPFileN', self.slp_file_name, str, is_reading)
        self.imp_file_name = ts_read_write(current_file, 'IMPFileN', self.imp_file_name, str, is_reading)
        self.ros_file_name = ts_read_write(current_file, 'ROSFileN', self.ros_file_name, str, is_reading)
        self.dik_file_name = ts_read_write(current_file, 'DikFileN', self.dik_file_name, str, is_reading)
        self.storm_file_name = ts_read_write(current_file, 'StormFileN', self.storm_file_name, str, is_reading)
        self.vd_file_name = ts_read_write(current_file, 'VDFileN', self.vd_file_name, str, is_reading)
        self.uplift_file_name = ts_read_write(current_file, 'UpliftFileN', self.uplift_file_name, str, is_reading)
        self.sal_file_name = ts_read_write(current_file, 'SalFileN', self.sal_file_name, str, is_reading)
        ts_read_write(current_file, 'RoadFileN', "", str, is_reading)  # assume discarded for now
        self.d2m_file_name = ts_read_write(current_file, 'D2MFileN', self.d2m_file_name, str, is_reading)
        self.optimize_level = ts_read_write(current_file, 'OptimizeLevel', self.optimize_level, int, is_reading)
        self.num_mm_entries = ts_read_write(current_file, 'NumMMEntries', self.num_mm_entries, int, is_reading)
        current_file = revert_file(current_file, file)

        self.gis_each_year = ts_read_write(current_file, 'GISEachYear', self.gis_each_year, bool, is_reading)
        self.gis_years = ts_read_write(current_file, 'GISYears', self.gis_years, str, is_reading)

        if is_reading or split_files:
            current_file = split_file(current_file, 'uncertainty_sensitivity', is_reading)
        self.uncert_setup.load_store(current_file, read_version_num, is_reading, self)
        current_file = revert_file(current_file, file)

        app_global.drive_externally = ts_read_write(current_file, 'ExecuteImmediately', drive_externally, bool, is_reading)
        self.ros_resize = ts_read_write(current_file, 'ROS_Resize', self.ros_resize, bool, is_reading)
        ts_read_write(current_file, 'RescaleMap', 1, int, is_reading)  # discard
        ts_read_write(current_file, 'CheckSum', VERSION_NUM, float, is_reading)   # useful for deprecated binary read write only

    def make_file_name(self):
        # Check if output file name has a directory specified
        if self.output_file_name:
            fname = self.output_file_name.replace('.', '_')
        else:
            fname = os.path.splitext(self.elev_file_name)[0] + '_OUT'

        # Handling uncertainty analysis filename adjustments
        if self.run_uncertainty:
            uncert_iter = 0 if self.uncert_setup.unc_sens_iter == 0 else (self.uncert_setup.gis_start_num +
                                                                          self.uncert_setup.unc_sens_iter - 1)
            fname = os.path.join(os.path.dirname(
                self.uncert_setup.csv_path), f"{uncert_iter}_{os.path.basename(self.uncert_setup.csv_path)}")

        # Handling sensitivity analysis filename adjustments
        if self.run_sensitivity:
            fname = self.uncert_setup.output_path

        # Append year or initial condition
        if self.year > 0:
            fname += f", {self.year},"
        else:
            fname += ", Initial Condition "

        # Handling specific analysis configurations
        if not self.run_uncertainty and not self.run_sensitivity:
            if self.running_tsslr:
                fname += f" {self.time_ser_slrs[self.tsslr_index - 1].name} "
            elif self.running_fixed:
                fname += f" {LabelFixed[self.fixed_scen]} "
            elif self.running_custom:
                fname += f" custom {self.current_custom_slr}"
            else:
                fname += f" {LabelIPCC[self.ipcc_sl_rate]} {LabelIPCCEst[self.ipcc_sl_est]} "

            if self.protect_all:
                fname += " Protect All Dry Land"
            elif self.protect_developed:
                fname += " Protect Developed Dry Land"

            if not self.include_dikes:
                fname += " No Dikes"

        # Append sensitivity analysis specifics if it's not deterministic
        if self.run_sensitivity and self.uncert_setup.unc_sens_iter > 0:
            fname += f"Sens{self.uncert_setup.unc_sens_iter}_{self.uncert_setup.pct_to_vary}"
            fname += "_Pos" if self.uncert_setup.sens_plus_loop else "_Neg"

        return fname

    def set_a(self, row, col, cell):
        if large_raster_edit:
            self.shared_data.map_matrix[row, col] = ord(cell['cats'][0]) + 1
            return

        mm_val = self.shared_data.map_matrix[row, col]
        if mm_val == -(self.categories.estuarine_water + 1):
            if (cell_width(cell, self.categories.open_ocean) >
                    cell_width(cell, self.categories.estuarine_water)):
                self.shared_data.map_matrix[row, col] = -(self.categories.open_ocean + 1)  # optimized cell, EOW-->OO
            return

        if mm_val < 0:  # optimized cell, not settable
            return

        block_index = mm_val >> BLOCKEXP  # Equivalent to mm_val // 16384
        sub_index = mm_val & BLOCKMASK  # Equivalent to mm_val % 16384

        try:
            self.shared_data.map[block_index].array[sub_index] = cell
        except IndexError as e:
            if self.optimize_level > 0:
                raise ValueError("Range Check Error. Optimized map is the wrong size.".format(e))
            else:
                raise

    def ret_cell_matrix(self, matr_result):
        """ Returns the cell from the matrix map if it is optimized out. """

        if large_raster_edit:
            return self.blank_cell.copy(cats=[matr_result - 1]), True

        if matr_result >= 0:
            return None, False

        if matr_result == BLANK:
            cell = self.blank_cell.copy()
        else:
            mapping = {
                -(self.categories.open_ocean + 1): self.ocean_cell,
                -(self.categories.estuarine_water + 1): self.eow_cell,
                -(self.categories.dev_dry_land + 1): self.dev_dry_cell,
                -(self.categories.und_dry_land + 1): self.und_dry_cell,
            }
            cell = mapping.get(matr_result, None)

        return cell, True

    def ret_a(self, row, col) -> compressed_cell_dtype:
        """ Retrieves a cell from the simulation map or matrix. """
        mm_val = self.shared_data.map_matrix[row, col]

        try:
            cell, handled = self.ret_cell_matrix(mm_val)
            if handled:
                return cell

            block_index = mm_val >> BLOCKEXP  # Equivalent to mm_val // 16384
            sub_index = mm_val & BLOCKMASK  # Equivalent to mm_val % 16384

            return self.shared_data.map[block_index].array[sub_index]
        except IndexError as e:
            raise Exception(
                'SLAMM Memory Read Error. Possibly maps have changed since last time SLAMM "counted" '
                'cells. Try using the "count" button within "File Setup."') from e

    def make_mem(self, cell):
        """Initialize memory for map blocks, filling with default cell values."""
        if self.num_mm_entries < 1:
            self.count_mm_entries()

        num_blocks = (self.num_mm_entries >> BLOCKEXP) + 1  # Equivalent to Num_mm_entries // 16384 )
        if self.shared_data is None:
            self.shared_data = SharedData()
        if self.shared_data.map is None:
            current_length = 0
        else:
            current_length = len(self.shared_data.map)

        if current_length < num_blocks:
            self.dispose_mem()  # Dispose old memory first if resizing
            for _ in range(num_blocks - current_length):
                shared_mem_array = SharedMemoryArray((BLOCKSIZE,), compressed_cell_dtype)
                shared_mem_array[:] = cell  # Initialize the array with default cell values
                self.shared_data.map.append(shared_mem_array)
        # Initialize all cells in all blocks
        for block in self.shared_data.map:
            block.array[:] = cell

    def dispose_mem(self):
        """Explicitly free memory allocated for the map."""
        if self.shared_data:
            self.shared_data.close()
            self.shared_data.unlink()  # Deallocate shared data

    def count_mm_entries(self):
        result = self.make_data_file(True, '', '')

    def set_bit(self, n, row, col, set_true):
        """ Set a bit in the b_matrix. """
        # index = self.site.cols * row + col
        if set_true:
            self.shared_data.b_matrix[row, col] |= n
        else:
            self.shared_data.b_matrix[row, col] &= ~n

    def get_bit(self, n, row, col):
        """ Get the state of a bit in the b_matrix. """
        return jit_get_bit(self.shared_data.b_matrix, self.site.cols, n, row, col)
        # index = self.site.cols * row + col
        # return (self.b_matrix[index] & n) > 0

    def save_raster(self, filename, raster_int):
        if filename == '':
            filename = input("Enter filename to save Raster output: ")

        if os.path.exists(filename):
            if input(f"File {filename} exists. Overwrite? (y/n)").lower() != 'y':
                print("File not overwritten.")
                return

        sof = TSLAMMOutputFile(self.site.cols, self.site.rows, self.site.llx_corner,
                               self.site.lly_corner, self.site.site_scale, self, FileFormat.GEOTIFF)
        header_written, errmsg = sof.write_header(prompt=False)
        if not header_written:
            print(errmsg)
            return False

        cell = None
        for row in range(self.site.rows):
            for col in range(self.site.cols):
                last_number = (row == self.site.rows - 1) and (col == self.site.cols - 1)
                if (raster_int < 4) or (raster_int > 5):
                    cell = self.ret_a(row, col)

                number_to_write = self.get_value_based_on_raster_type(cell, col, row, raster_int)
                sof.write_next_number(number_to_write, last_number)

    def get_value_based_on_raster_type(self, cell: compressed_cell_dtype, col, row, raster_int):
        if raster_int == 1:  # Dikes
            if self.classic_dike:
                if cell_width(cell, BLANK) > 0:
                    return BLANK
                elif cell['prot_dikes']:
                    return 1
                else:
                    return 0
            else:
                if cell_width(cell, BLANK) > 0:
                    return BLANK
                elif not cell['prot_dikes']:
                    return 0
                elif cell['elev_dikes']:
                    return get_min_elev(cell)  # Implement get_min_elev to extract min elev
                else:
                    return -5

        elif raster_int == 2:  # Wetland Raster
            cat_int = get_cell_cat(cell)  # Implement get_cell_cat to get the category
            if cat_int < 0 or cat_int >= self.categories.n_cats:
                return BLANK
            else:
                return self.categories.cats[cat_int].gis_number

        elif raster_int == 3:  # Elevation Raster
            elev = get_min_elev(cell)
            if elev > 998:
                return -9999  # Use appropriate no-data value
            else:
                slope_adjustment = (self.site.site_scale * 0.5) * cell['tan_slope']
                elev += slope_adjustment
                return elev

        elif raster_int == 4:  # Input Subsite Raster
            ss_index = self.site.get_subsite_num(col, row)
            gt = self.site.global_site.gtide_range if ss_index == 0 else self.site.subsites[ss_index - 1].gtide_range
            return gt

        elif raster_int == 5:  # Inundation Raster
            inund_out = translate_inund_num(
                self.inund_arr[self.site.cols * row + col])
            return inund_out

        elif raster_int == 6:  # MLLW
            return cell['d2mllw']

        elif raster_int == 7:  # MHHW
            return cell['d2mhhw']

        return None  # Default case if none match

    def calc_inund_connectivity(self, p_arr: np.array([], dtype=np.ubyte), clear_arr, elev_type):
        """Calculate inundation connectivity based on provided elevation type."""

        #               : Array of byte;       Unchecked = 1
        #                                      Above Inundation Elevation = 2
        #                                      Checked & Connected = 3
        #                                      Checked & Unconnected = 4
        #                                      Currently Being Checked = 6
        #                                      Diked = 7
        #                                      Tidal Water = 8 [navy]
        #                                      Blank = 9
        #                                      Overtopped Dike = 10  Currently Disabled {overtop at H1, writes to log}
        #                                      30,60,90,120,150 = H1,H2,H3,H4,H5 inundation

        def below_inund_elev(cl, c, r):
            """Check if the cell elevation is below the inundation elevation."""
            subsite = self.site.get_subsite(c, r, cl)
            cell_elev = get_min_elev(cl)
            if self.connect_min_elev == 0:
                cell_elev += (self.site.site_scale * 0.5) * cl['tan_slope']

            if elev_type == -1:
                return cell_elev < subsite.salt_elev
            else:
                return cell_elev < subsite.inund_elev[elev_type]

        def connect_check(fx, fy):
            """Check area for connectivity using a stack-based fill algorithm."""
            stack = []
            stack2 = []
            fill_result = 4  # Assume initially unconnected

            stack.append((fx, fy))
            stack2.append((fx, fy))

            while stack:
                tp = stack.pop()
                fx, fy = tp

                # Ensure within bounds
                if not (0 <= fy < self.site.rows and 0 <= fx < self.site.cols):
                    continue  # exit while

                current_num = p_arr[fy * self.site.cols + fx]
                if current_num == 8:
                    fill_result = 3  # Found tidal water
                    continue

                if current_num != 1:
                    continue

                cl = self.ret_a(fy, fx)

                # Skip diked areas
                if (self.classic_dike and cl['prot_dikes']) or \
                        (not self.classic_dike and cl['prot_dikes'] and not cl['elev_dikes']):
                    continue

                if self.is_saline_water(cl):
                    fill_result = 3  # Found open water
                    p_arr[fy * self.site.cols + fx] = 8  # mark this as open water
                    continue

                if not below_inund_elev(cl, fx, fy):
                    p_arr[fy * self.site.cols + fx] = 2
                    if cl['prot_dikes'] and cl['elev_dikes']:
                        p_arr[fy * self.site.cols + fx] = 7
                    continue  # exit if you left the low lying area

                if get_cell_cat(cl) == BLANK:
                    p_arr[fy * self.site.cols + fx] = 9  # mark as blank
                    continue

                stack2.append((fx, fy))
                p_arr[fy * self.site.cols + fx] = 6  # Mark as being checked

                # Add neighbors to the stack
                for nx, ny in [(fx + 1, fy), (fx - 1, fy), (fx, fy + 1), (fx, fy - 1)]:
                    stack.append((nx, ny))

                if self.connect_near_eight:
                    for nx, ny in [(fx + 1, fy + 1), (fx - 1, fy + 1), (fx + 1, fy - 1), (fx - 1, fy - 1)]:
                        stack.append((nx, ny))

            # Mark all cells in stack2 with the result
            while stack2:
                tp = stack2.pop()
                if tp[0] > -99:  # Check valid point
                    if p_arr[tp[1] * self.site.cols + tp[0]] != 8:
                        p_arr[tp[1] * self.site.cols + tp[0]] = fill_result
            # End connect_check

        # Main execution block for calc_inund_connectivity  ------------------------------
        if p_arr is None:
            p_arr = np.zeros(self.site.rows * self.site.cols, dtype=np.ubyte)
        if clear_arr:
            for er in range(self.site.rows):
                for ec in range(self.site.cols):
                    p_arr[er * self.site.cols + ec] = 1

        for er in range(self.site.rows):
            for ec in range(self.site.cols):
                if p_arr[er * self.site.cols + ec] == 1:
                    cell = self.ret_a(er, ec)
                    if (self.classic_dike and cell['prot_dikes']) or \
                            (not self.classic_dike and cell['prot_dikes'] and not cell['elev_dikes']):
                        p_arr[er * self.site.cols + ec] = 7
                    elif get_cell_cat(cell) == BLANK:
                        p_arr[er * self.site.cols + ec] = 9
                    elif self.is_saline_water(cell):
                        p_arr[er * self.site.cols + ec] = 8
                    elif not below_inund_elev(cell, ec, er):
                        p_arr[er * self.site.cols + ec] = 2
                    else:
                        connect_check(ec, er)
        return p_arr

    def is_saline_water(self, cell):
        """Check if the cell contains saline water."""
        if self.is_open_water(cell):
            for i in range(NUM_CAT_COMPRESS):
                if self.categories.get_cat(cell['cats'][i]).is_tidal and cell['widths'][i] > TINY:
                    return True
        return False

    def is_open_water(self, cell):
        """Check if the cell is open water."""
        for i in range(NUM_CAT_COMPRESS):
            if self.categories.get_cat(cell['cats'][i]).is_open_water and cell['widths'][i] > TINY:
                return True
        return False

    def is_dry_land(self, cell):
        """Check if the cell is dry land."""
        for i in range(NUM_CAT_COMPRESS):
            if self.categories.get_cat(cell['cats'][i]).is_dryland and cell['widths'][i] > TINY:
                return True
        return False

    def wave_erosion_category(self, cell):
        """Determine if the cell is subject to wave erosion and calculate relevant metrics."""
        """Returns a tuple of the result, the min elev, and the marsh width"""
        marsh_width = 0
        min_elev = 999
        result = False
        for i in range(NUM_CAT_COMPRESS):
            if self.categories.get_cat(cell['cats'][i]).use_wave_erosion and cell['widths'][i] > TINY:
                result = True
                if cell['min_elevs'][i] < min_elev:
                    min_elev = cell['min_elevs'][i]
                marsh_width += cell['widths'][i]
        return result, min_elev, marsh_width

    def calc_ss_raster(self, ri, pta):
        """ Calculate storm surge heights using raster inputs. """
        tmp_array = np.frombuffer(pta, dtype=np.uint8)  # Assuming pta points to a uint8 buffer
        result = True
        slr = self.site.global_site.newsl - self.site.global_site.t0_slr  # SLR since T0 in meters

        print('Calculating Storm Surge')
        # Assuming you have a progress update mechanism similar to ProgForm in your GUI
        for er in range(self.site.rows):
            for ec in range(self.site.cols):
                cell = self.ret_a(er, ec)  # Assuming ret_a retrieves a compressed cell
                cell_elev = get_min_elev(cell) + (self.site.site_scale * 0.5) * cell['tan_slope'] \
                    if self.connect_min_elev == 0 else get_min_elev(cell)

                if cell_elev < 20 and not self.is_open_water(cell):  # Exclude open water, no data, and high elevations
                    flat_arr_index = er * self.site.cols + ec

                    storm_elev = -10  # Default for no-data cases
                    if slr < 1e-6 or not self.ss_raster_slr[ri]:
                        storm_elev = self.ss_rasters[ri][1][flat_arr_index]  # MTL t0 datum
                    else:
                        val1 = self.ss_rasters[ri][1][flat_arr_index]
                        val2 = self.ss_rasters[ri][2][flat_arr_index]
                        if val2 < -9.99 < val1:
                            storm_elev = val1 - self.ss_slr[ri][1]
                        elif val1 < -9.99 < val2:
                            storm_elev = val2 - self.ss_slr[ri][2]
                        elif val1 > -9.99 and val2 > -9.99:
                            storm_elev = linear_interpolate(val1, val2, self.ss_slr[ri][1],
                                                            self.ss_slr[ri][2], slr) - slr

                    if cell_elev < storm_elev:  # Check elevation against storm level
                        tmp_array[flat_arr_index] = 150 if ri == 2 else 120

        return result

    def calc_inund_freq(self):
        # Initialize the temporary array for holding flood levels
        tmp_array = np.zeros(self.site.rows * self.site.cols, dtype=np.uint8)
        # Properly initialize inund_arr if it's None
        if self.inund_arr is None:
            self.inund_arr = np.zeros_like(tmp_array)

        result = True
        if self.use_ss_raster[1]:
            result = self.calc_ss_raster(2, tmp_array)
        else:
            self.inund_arr = self.calc_inund_connectivity(self.inund_arr, True, 4)
            tmp_array[self.inund_arr == 3] = 150
            self.inund_arr[self.inund_arr == 3] = 1

        if not result:
            return False

        if self.use_ss_raster[0]:
            result = self.calc_ss_raster(1, tmp_array)
        else:  # H4 Inundation
            self.inund_arr = self.calc_inund_connectivity(self.inund_arr, self.inund_arr.all() == 0, 3)
            tmp_array[self.inund_arr == 3] = 120
            tmp_array[(tmp_array < 120) & (self.inund_arr != 3)] = self.inund_arr[
                (tmp_array < 120) & (self.inund_arr != 3)]
            self.inund_arr[self.inund_arr == 3] = 1

        if not result:
            return False

        # H3 Inundation
        self.inund_arr = self.calc_inund_connectivity(self.inund_arr, self.inund_arr.all() == 0, 2)
        tmp_array[self.inund_arr == 3] = 90
        tmp_array[(tmp_array < 90) & (self.inund_arr != 3)] = self.inund_arr[(tmp_array < 90) & (self.inund_arr != 3)]
        self.inund_arr[self.inund_arr == 3] = 1  # Reset for the next level
        if not result:
            return False

        # H2 Inundation
        self.inund_arr = self.calc_inund_connectivity(self.inund_arr, False, 1)
        tmp_array[self.inund_arr == 3] = 60
        self.inund_arr[self.inund_arr == 3] = 1  # Reset for the next level
        if not result:
            return False

        # H1 Inundation
        self.inund_arr = self.calc_inund_connectivity(self.inund_arr, False, 0)
        tmp_array[self.inund_arr == 3] = 30
        self.inund_arr[:] = tmp_array  # Update all elements from temporary to main array

        if not result:
            return False

        return True

    def save_gis_files(self, ff: FileFormat):
        writing_elevs = self.save_elevs_gis
        writing_salin = self.save_salinity_gis
        writing_inund = self.save_inund_gis

        for write_loop in range(1, 5):
            if write_loop == 2 and not writing_elevs:
                continue
            if write_loop == 3 and not writing_salin:
                continue
            if write_loop == 4 and not writing_inund:
                continue

            root_name = self.make_file_name()
            desc_string = '_ELEVS' if write_loop == 2 else '_SALINITY_MLLW' if write_loop == 3 \
                else '_Inund_Freq' if write_loop == 4 else ''
            if write_loop == 2:
                if self.save_elev_gis_mtl:
                    desc_string += '_MTL'
                else:
                    desc_string += '_NAVD88'
            if ff == FileFormat.ASCII:
                file_name = f"{root_name}{desc_string}_GIS.ASC"  # ASCII format
            elif ff == FileFormat.GEOTIFF:
                file_name = f"{root_name}{desc_string}_GIS.TIF"  # GEOTIFF format
            else:
                file_name = f"{root_name}{desc_string}_GIS.SLB"  # SLAMM Binary Format

            sof = TSLAMMOutputFile(self.site.cols, self.site.rows, self.site.llx_corner, self.site.lly_corner,
                                   self.site.site_scale, file_name, ff)
            header_written, errmsg = sof.write_header(False)
            if not header_written:
                print(errmsg)
                return False

            for row in range(self.site.rows):
                for col in range(self.site.cols):
                    current_cell = self.ret_a(row, col)
                    last_number = row == self.site.rows - 1 and col == self.site.cols - 1
                    value_to_write = NO_DATA
                    cell_in = self.in_area_to_save(col, row, not self.save_ros_area)

                    if write_loop == 1:  # Handle dominant category GIS output
                        if cell_in:
                            value_to_write = self.categories.get_cat(get_cell_cat(current_cell)).gis_number
                            if value_to_write == BLANK:
                                value_to_write = NO_DATA

                    elif write_loop == 2:  # Handle elevation GIS output
                        min_elev = get_min_elev(current_cell)
                        if min_elev == 999 or not cell_in:
                            result = value_to_write = NO_DATA
                        else:
                            slope_adjustment = (self.site.site_scale * 0.5) * current_cell['tan_slope']
                            value_to_write = min_elev + slope_adjustment  # Convert to mean elevation

                            # Correct elevation if desired datum is NAVD88
                            if not self.save_elev_gis_mtl:
                                # Calculate SLR from the beginning of the simulation
                                subsite = self.site.get_subsite(col, row, current_cell)

                                # New elevation in NAVD88
                                value_to_write = value_to_write + current_cell['mtl_minus_navd'] + subsite.newsl
                                # current elev MTL + initial MTL-NAVD88 + eustatic SLR from start

                    elif write_loop == 3:  # Handle salinity GIS output
                        if cell_in:
                            value_to_write = current_cell['sal'][0]  # writes salinity at the MLLW inundation level

                    elif write_loop == 4:  # Handle inundation frequency GIS output
                        if cell_in:
                            value_to_write = translate_inund_num(self.inund_arr[row * self.site.cols + col])

                    sof.write_next_number(value_to_write, last_number)
                sof.cr()  # Write new line if needed

        return True

    def in_area_to_save(self, x, y, save_all):
        if save_all:
            return True

        if self.shared_data.ros_array is None:
            return True

        if self.shared_data.ros_array[y][x] > 0:
            return True
        return False

    def save_csv_file(self):
        if self.batch_mode:
            file_name = self.batch_file_name
        else:
            file_name = self.make_file_name() + ".CSV"
        self.csv_file_save(self.summary, self.row_label, self.col_label, self.tstep_iter + 1,
                           self.categories.n_cats + 1, file_name)
        self.uncert_setup.unc_sens_row = self.tstep_iter + 1

    def csv_file_save(self, datary, row_label, col_label, num_row, num_col, file_name):
        def prot_label():
            if self.run_sensitivity and self.uncert_setup.unc_sens_iter > 0:
                return '"' + str(self.uncert_setup.unc_sens_iter)
            else:
                result = '"Protect None'
                if self.protect_all:
                    result = '"Protect All Dry'
                elif self.protect_developed:
                    result = '"Protect Developed Dry'

                if not self.include_dikes:
                    result += ' No Dikes'

                return result + '",'

        def get_sl_label(csv_format=False):
            """Generate scenario label based on current simulation settings."""
            if self.running_tsslr:
                label = self.time_ser_slrs[self.tsslr_index - 1].name
                label = f'"Time Ser. SLR","{label}"' if csv_format else label
            elif self.running_fixed:
                label = LabelFixed[self.fixed_scen]
                label = f'"Fixed","{label}"' if csv_format else label
            elif self.running_custom:
                label = f"{self.current_custom_slr:.4f} meters"
                label = f'"Custom","{label}"' if csv_format else label
            else:
                label = f"{LabelIPCC[self.ipcc_sl_rate]} {LabelIPCCEst[self.ipcc_sl_est]}"
                label = f'"{LabelIPCC[self.ipcc_sl_rate]}","{LabelIPCCEst[self.ipcc_sl_est]}"' if csv_format else label

            return label

        def write_secondary_file():
            write_err = True

            if self.output_file_name:
                secondary_file_name = self.output_file_name + '.CSV'
            else:
                secondary_file_name = self.elev_file_name + '_OUT.CSV'

            while write_err:
                try:
                    write_err = False
                    appending = os.path.exists(secondary_file_name)
                    mode = "a" if appending else "w"
                    with open(secondary_file_name, mode) as f2:
                        if not appending:
                            f2.write('"Site Desc.","Scenario","Parameters","Year","Protection",'
                                     '"GIS Num","Hectares","SLAMMText","SLR (eustatic)","RunDate"\n')

                        run_date = datetime.now()

                        sl_label2 = get_sl_label(True)

                        for i_sites in range(self.site.n_output_sites + self.site.max_ros + 1):
                            for i_rows in range(num_row):
                                for j2 in range(1, num_col + 21):
                                    if self.n_road_inf > 0 or j2 < num_col + 9:  # suppress roads data
                                        if i_sites == 0:
                                            f2.write('"' + self.site.global_site.description + '",')
                                        elif i_sites <= self.site.n_output_sites:
                                            f2.write('"' + self.site.output_sites[i_sites - 1].description + '",')
                                        else:
                                            f2.write('"Raster ' + str(i_sites - self.site.n_output_sites) + '",')

                                        f2.write(sl_label2 + ',')
                                        f2.write('"' + row_label[i_rows] + '",')  # write year
                                        f2.write(prot_label())  # write protection

                                        if j2 < self.categories.n_cats:
                                            f2.write(str(self.categories.get_cat(j2 - 1).gis_number) + ',')  # GIS Num
                                        else:
                                            f2.write('NA,')

                                        f2.write(str(datary[i_sites, i_rows, j2]) + ',')

                                        if j2 < len(col_label):
                                            f2.write('"' + col_label[j2] + '",')  # Slamm Text
                                        else:
                                            f2.write('NA,')

                                        f2.write(str(datary[i_sites, i_rows, 0]) + ',')  # Eustatic SLR
                                        f2.write(run_date.strftime("%Y-%m-%d %H:%M:%S") + '\n')

                    if not (self.run_uncertainty or self.run_sensitivity):
                        try:
                            run_record_file = open(self.run_record_file_name, "a")
                            run_record_file.write(sl_label + " appended to Master CSV: " + file_name + "\n")
                            run_time = round((time.time()-self.start_time)/60, 1)
                            run_record_file.write(f"--  run completed in {run_time} minutes.")

                            run_record_file.close()
                        except Exception as ex:
                            print(f"Error appending to Run-Record File {self.run_record_file_name}: {ex}")

                    write_err = False

                except Exception as ex:
                    choice2 = input(f"Error writing to file {secondary_file_name}. {ex}  Retry? (y/n): ")
                    if choice2.lower() != 'y':
                        write_err = False
                    else:
                        write_err = True

        # main body of CSV_file_save  -----------------------------

        # Try opening or appending to the file
        f = None
        write_error = True
        while write_error:
            write_error = False
            try:
                if self.batch_mode and self.file_started:
                    f = open(file_name, "a")
                else:
                    f = open(file_name, "w")
            except Exception:
                choice = input(f'Error writing to file {file_name}. Retry? (y/n): ')
                if choice.lower() != 'n':
                    write_error = True
                else:
                    return

        sl_label = get_sl_label(False)

        if not self.batch_mode:
            f.write(f'"Sea Level Scenario: {sl_label}')
            f.write(prot_label())
            f.write('\n')
            f.write(f'"Total Ha:",{round(self.hectares):n}\n')
            f.write(' \n')

        if not (self.batch_mode and self.file_started):
            f.write('"Date","Site Desc.","Scenario","Parameters","Protection",')
            for j in range(len(self.col_label)):  # num_col + 20
                f.write(f'"{self.col_label[j]}",')
            f.write('\n')

        self.file_started = True

        for isites in range(self.site.n_output_sites + self.site.max_ros + 1):
            for irows in range(num_row):
                f.write(f'"{self.row_label[irows]}",')
                if isites == 0:
                    f.write(f'"{self.site.global_site.description}",')
                elif isites <= self.site.n_output_sites:
                    f.write(f'"{self.site.output_sites[isites - 1].description}",')
                else:
                    f.write(f'"Raster {isites - self.site.n_output_sites}",')

                if self.running_tsslr:
                    f.write(f'"Time Ser. SLR","{self.time_ser_slrs[self.tsslr_index - 1].name}",')
                elif self.running_fixed:
                    f.write(f'"Fixed","{LabelFixed[self.fixed_scen]}",')
                elif self.running_custom:
                    f.write(f'"Custom","{self.current_custom_slr:.4f} meters ",')
                else:
                    f.write(f'"{LabelIPCC[self.ipcc_sl_rate]}","{LabelIPCCEst[self.ipcc_sl_est]}",')

                f.write(prot_label())

                for j in range(num_col + 20):
                    if (self.n_road_inf > 0) or (j < num_col + 7 + 2):  # suppress roads data
                        f.write(f"{self.summary[isites][irows][j]:10.4f},")

                f.write('\n')

        f.close()

        if not (self.run_uncertainty or self.run_sensitivity):  # separate logs exist for uncertainty & sensitivity
            try:
                run_record_file = open(self.run_record_file_name, "a")
                run_record_file.write(f"{sl_label} Results {file_name}\n")
                run_record_file.close()
            except Exception:
                print(f'Error appending to Run-Record File: {self.run_record_file_name}')

        write_secondary_file()

    def summarize(self, year):
        num_cells = self.site.rows * self.site.cols
        self.hectares = num_cells * self.site.site_scale * self.site.site_scale / 10000  # Convert to hectares

        california = self.categories.are_california()

        if not california:
            self.tropical = (self.cat_sums[0][8] / 10000) / self.hectares > 0.005

        for i in range(self.site.n_output_sites + self.site.max_ros + 1):
            self.row_label[self.tstep_iter] = str(year)
            self.summary[i][self.tstep_iter][0] = self.site.global_site.newsl

            # Initialize aggregated categories
            for j in range(1, 8):
                self.summary[i][self.tstep_iter][self.categories.n_cats + 1 + j] = 0

            # Initialize carbon sequestration
            if self.tstep_iter > 1:
                self.summary[i][self.tstep_iter][self.categories.n_cats + 1 + 7 + 1] = \
                    self.summary[i][self.tstep_iter - 1][self.categories.n_cats + 1 + 7 + 1]
            else:
                self.summary[i][self.tstep_iter][self.categories.n_cats + 1 + 7 + 1] = 0

            for cc in range(self.categories.n_cats):
                self.summary[i][self.tstep_iter][cc + 1] = self.cat_sums[i][cc] / 10000  # Convert to hectares
                agg_c = self.categories.get_cat(cc).agg_cat

                if agg_c != AggCategories.AggBlank:
                    self.summary[i][self.tstep_iter][self.categories.n_cats + 2 + agg_c.value] += \
                        self.cat_sums[i][cc] / 10000

                if self.tstep_iter > 1:
                    k1 = (44 / 12) * 0.47 * self.categories.get_cat(cc).mab
                    k2 = (44 / 12) * self.categories.get_cat(cc).rsc - 21 * self.categories.get_cat(
                        cc).ech4

                    if k1 != 0 or k2 != 0:
                        d_time = year - int(self.row_label[self.tstep_iter - 1])
                        self.summary[i][self.tstep_iter][self.categories.n_cats + 1 + 7 + 1] += \
                            k1 * self.summary[i][self.tstep_iter][cc + 1] + \
                            (k2 * d_time - k1) * self.summary[i][self.tstep_iter - 1][cc + 1]

            if i == 0:
                self.summary[i][self.tstep_iter][self.categories.n_cats + 1] = self.sav_km
            else:
                self.summary[i][self.tstep_iter][self.categories.n_cats + 1] = -9999

            # Roads output
            if self.n_road_inf > 0 and i == 0:
                for j in range(MAX_ROAD_ARRAY + 1):
                    self.summary[i][self.tstep_iter][self.categories.n_cats + 9 + j + 1] = self.road_sums[i][j]

    def check_for_annual_maps(self):
        new_dik_fname, new_sal_fname = "", ""

        def replace_slr(ri, slr_i):
            base_file_name = self.storm_file_name[ri]
            file_ext = os.path.splitext(base_file_name)[1]
            base_file_name = os.path.splitext(base_file_name)[0]
            return_str = base_file_name[-3:]
            slr_str = base_file_name[-5:-3]
            base_file_name = base_file_name[:-6]

            for _ in range(2):
                slr_inch = int(slr_str) + 6
                slr_m = 0.0254 * slr_inch
                slr_str = str(slr_inch).zfill(2)
                new_file_name = f"{base_file_name}{slr_str}_{return_str}{file_ext}"
                if os.path.exists(new_file_name):
                    self.ss_raster_slr[ri] = True
                    self.ss_filen[ri] = new_file_name
                    self.ss_slr[ri][slr_i] = slr_m
                    return self.read_storm_surge(ri, slr_i)

                if slr_i == 2 and self.ss_slr[ri][slr_i] < 0:  # file not found
                    self.ss_raster_slr[ri] = False
                return False

        # start of check for annual maps
        result = True

        if self.include_dikes:
            dik_ext = os.path.splitext(self.dik_file_name)[1]
            new_dik_fname = f"{os.path.splitext(self.dik_file_name)[0]}{self.year}{dik_ext}"
            if not os.path.exists(new_dik_fname):
                new_dik_fname = ""

        if '.xls' not in self.sal_file_name.lower():
            sal_ext = os.path.splitext(self.sal_file_name)[1]
            new_sal_fname = f"{os.path.splitext(self.sal_file_name)[0]}{self.year}{sal_ext}"
            if not os.path.exists(new_sal_fname):
                new_sal_fname = ""
                if self.sal_file_name.strip():
                    response = input(
                        'Are you sure you want to run this SLAMM simulation with a constant salinity raster? (y/n): ').strip().lower()
                    if not response.startswith('y'):
                        sys.exit()

        if new_dik_fname or new_sal_fname:
            result = self.make_data_file(False, new_dik_fname, new_sal_fname)
            if new_sal_fname:
                for fw_flow in self.fw_flows:
                    fw_flow.retention_initialized = False

        if result and self.site.global_site.newsl - self.site.global_site.t0_slr > 0:
            for sl_return in range(0, 2):
                if self.use_ss_raster[sl_return] and self.ss_raster_slr[sl_return]:
                    min_sl = 1 if self.ss_slr[sl_return][1] < self.ss_slr[sl_return][2] else 2
                    max_sl = 2 if min_sl == 1 else 1
                    if self.site.global_site.newsl - self.site.global_site.t0_slr > self.ss_slr[sl_return][max_sl]:
                        result = replace_slr(sl_return, min_sl)  # replace the minimum SLR Scenario
                    if self.site.global_site.newsl - self.site.global_site.t0_slr > self.ss_slr[sl_return][min_sl]:
                        result = replace_slr(sl_return, max_sl)  # Map didn't go far enough. replace max SLR scen too
            return result

        return True

    @staticmethod
    def get_next_number(rf, er, ec):
        try:
            result = rf.get_next_number()
            if result is None:
                raise ESLAMMError(f'Error Reading {rf.file_name} File. Row: {er + 1}; Col: {ec + 1}.')
            return result
        except Exception as e:
            # You might want to handle specific exceptions if get_next_number raises known types
            raise ESLAMMError(f'Error Reading {rf.file_name} File. Row: {er + 1}; Col: {ec + 1}. Exception: {str(e)}')

    @staticmethod
    def prepare_file_for_reading(file_name) -> (bool, TSLAMMInputFile):
        file_exists = file_name != ''
        if file_exists and not os.path.isfile(file_name):
            # ProgForm.hide()  # Uncomment if there's a GUI element to hide
            raise ESLAMMError(f'File Setup Error, Cannot find file "{file_name}"')

        if file_exists:
            t_file = TSLAMMInputFile(file_name)
            if not t_file.prepare_file_for_reading():
                raise ESLAMMError(f'Error Reading Headers for "{file_name}"')
            return True, t_file
        else:
            return False, None

    def read_storm_surge(self, rii, slri):

        # prog_form.setup(
        #     f'Reading Storm File {os.path.basename(self.ss_file_n[rii])}',
        #     prog_form.year_label, prog_form.slr_label, prog_form.protection_label, True
        # )

        sf_exists, storm_file = self.prepare_file_for_reading(self.ss_filen[rii])
        storm_num = -10.0

        if len(self.ss_rasters[rii][slri]) < self.site.rows * self.site.cols:
            self.ss_rasters[rii][slri] = np.empty(self.site.rows * self.site.cols, dtype=np.float32)

        for er in range(self.site.rows):
            for ec in range(self.site.cols):
                # if ec == 0:
                #     result = prog_form.update2gages(int(er / self.site.rows * 100), 0)
                #     if not result:
                #         return False

                if sf_exists:
                    storm_num = self.get_next_number(storm_file, er, ec)

                cl = self.ret_a(er, ec)

                if storm_num > -9.99 and cl['mtl_minus_navd'] > -9998:
                    storm_num -= cl['mtl_minus_navd']  # Convert to MTL basis

                self.ss_rasters[rii][slri][self.site.cols * er + ec] = storm_num

        return True

    def make_data_file(self, count_only, year_dike_fname, year_sal_fname) -> bool:

        elev_file = None
        slp_file = None
        nwi_file = None
        imp_file = None
        vd_file = None
        d2m_file = None
        uplift_file = None
        sal_file = None
        ros_file = None
        storm_file1 = None
        storm_file2 = None
        n_cells = 0

        def read_dike_or_salinity_only(er, ec):
            nonlocal dik_only, read_cell, dik_number
            read_cell = self.ret_a(er, ec)

            if dik_only:
                # Update dike value
                read_cell['prot_dikes'] = (int(dik_number) != NO_DATA) and (dik_number != 0)
                if not self.classic_dike:
                    read_cell['elev_dikes'] = (int(dik_number) != NO_DATA) and (dik_number > 0)

            if sal_only:
                # Update salinity value
                read_cell['sal'][0] = sal_number  # Assuming sal is an array in the cell

            self.set_a(er, ec, read_cell)

        def prepare_storm_files():
            nonlocal storm_file1, storm_file2

            file_ext = os.path.splitext(self.storm_file_name)
            storm_string = os.path.splitext(self.storm_file_name)
            return_str = storm_string[-3:]

            if return_str == '010':
                self.ss_filen[0] = self.storm_file_name
                self.ss_filen[1] = f"{storm_string[:-3]}100{file_ext}"
            else:
                self.ss_filen[1] = self.storm_file_name
                self.ss_filen[0] = f"{storm_string[:-3]}010{file_ext}"

            self.sf_exists1, storm_file1 = self.prepare_file_for_reading(self.ss_filen[0])
            self.sf_exists2, storm_file2 = self.prepare_file_for_reading(self.ss_filen[1])

            self.use_ss_raster[0] = self.sf_exists1
            self.use_ss_raster[1] = self.sf_exists2

            self.ss_slr = [[0, -1], [0, -1]]  # Initializing as per the provided logic

            slr_file_n = f"{storm_string[:-6]}12_010{file_ext}"
            self.ss_raster_slr[0] = os.path.exists(slr_file_n)

            slr_file_n = f"{storm_string[:-6]}12_100{file_ext}"
            self.ss_raster_slr[1] = os.path.exists(slr_file_n)

        # MAIN BODY OF make_data_file
        if count_only:
            print("Calculating Memory Size Required...")
        else:
            print("Loading Data Files...")

        template_cell: compressed_cell_dtype
        gis_lookup = [-99] * 256
        for j in range(self.categories.n_cats):
            gis_lookup[self.categories.get_cat(j).gis_number] = j

        dik_only = year_dike_fname != ''
        sal_only = year_sal_fname != ''

        if count_only and self.optimize_level == 0:
            self.num_mm_entries = self.site.rows * self.site.cols
            return True

        total_width = self.site.site_scale
        self.site.max_ros = 0  # Default to zero raster output subsites
        self.site.ros_bounds = []  # empty list
        self.max_mtl = -9999
        self.min_mtl = 9999

        if self.batch_mode:
            print(f"Simulating {self.site.global_site.description}")

        template_cell = init_cell()

        if not count_only and not dik_only and not sal_only:
            try:
                if not app_global.large_raster_edit:
                    self.make_mem(template_cell)
            except Exception as e:
                if self.shared_data is None:
                    self.shared_data = SharedData()
                if self.shared_data.map is not None:
                    for i in range(len(self.shared_data.map)):
                        self.shared_data.map = None
                self.shared_data.map = []
                # Handling the error and printing the specific exception message
                print(f"Error creating map in memory: {str(e)}")
                raise Exception(f'Error creating map in memory: {str(e)}')

        # Set up reading of the dike file if it exists
        dike_f_exists = False
        if dik_only:
            print(f"Reading Dike File {os.path.basename(year_dike_fname)}")
            dike_f_exists, dik_file = self.prepare_file_for_reading(year_dike_fname)
        else:
            dike_f_exists, dik_file = self.prepare_file_for_reading(self.dik_file_name)

        # Prepare storm files if necessary
        sf_exists1 = sf_exists2 = False
        if self.storm_file_name:
            prepare_storm_files()

        sal_f_exists = False
        # Check if the salinity file is not an Excel file, then prepare it
        if '.xls' not in self.sal_file_name.lower():
            if sal_only:
                print(f"Reading Salinity File {os.path.basename(year_sal_fname)}")
                sal_f_exists, sal_file = self.prepare_file_for_reading(year_sal_fname)
            else:
                sal_f_exists, sal_file = self.prepare_file_for_reading(self.sal_file_name)

        # Further setup and data processing
        if dik_only:
            # Log or print the processing of dike files
            print(f"Processing dike file {os.path.basename(year_dike_fname)}")
            dike_f_exists, dik_file = self.prepare_file_for_reading(year_dike_fname)

        # Prepare files to read when it's not only a dike and not salinity map only
        imp_f_exists = False
        vd_f_exists = False
        d2m_f_exists = False
        uplift_f_exists = False
        ros_f_exists = False
        if not dik_only and not sal_only:
            ros_f_exists, ros_file = self.prepare_file_for_reading(self.ros_file_name)
            if ros_f_exists:
                self.shared_data.ros_array = SharedMemoryArray((self.site.rows, self.site.cols), dtype=np.uint8)
            imp_f_exists, imp_file = self.prepare_file_for_reading(self.imp_file_name)
            nwi_f_exists, nwi_file = self.prepare_file_for_reading(self.nwi_file_name)
            if not nwi_f_exists:
                raise Exception('Must include SLAMM Code (NWI) Data File')
            elev_f_exists, elev_file = self.prepare_file_for_reading(self.elev_file_name)
            if not elev_f_exists:
                raise Exception('Must include Elevation Data File')
            slp_f_exists, slp_file = self.prepare_file_for_reading(self.slp_file_name)
            if not slp_f_exists:
                raise Exception('Must include Slope Data File')
            uplift_f_exists, uplift_file = self.prepare_file_for_reading(self.uplift_file_name)
            d2m_f_exists, d2m_file = self.prepare_file_for_reading(self.d2m_file_name)
            vd_f_exists, vd_file = self.prepare_file_for_reading(self.vd_file_name)

        # If either CountOnly, DikOnly, or SalOnly is false
        if not (count_only or dik_only or sal_only):
            if self.shared_data.map_matrix is None or self.shared_data.map_matrix.array.size < self.site.rows * self.site.cols:
                self.shared_data.initialize_map_matrix((self.site.rows, self.site.cols), dtype=np.int32)

            if (large_raster_edit or (USE_DATAELEV and (self.optimize_level > 1))) and (
                    self.shared_data.data_elev is None or self.shared_data.data_elev.array.size < self.site.rows * self.site.cols):
                self.shared_data.initialize_data_elev((self.site.rows, self.site.cols), dtype=np.uint16)

            if sf_exists1 and (self.ss_rasters[1][1] is None or len(
                    self.ss_rasters[1][1]) < self.site.rows * self.site.cols):
                self.ss_rasters[1][1] = np.zeros(self.site.rows * self.site.cols)
            if sf_exists2 and (len(self.ss_rasters[2][1]) < self.site.rows * self.site.cols):
                self.ss_rasters[2][1] = np.zeros(self.site.rows * self.site.cols)

            if self.optimize_level > 1 and (
                    self.shared_data.max_fetch_arr is None or self.shared_data.max_fetch_arr.array.shape != (
                    self.site.rows, self.site.cols)):
                self.shared_data.initialize_max_fetch_arr((self.site.rows, self.site.cols), dtype=np.uint16)

        progress_interval = self.site.rows // 5
        for ER in range(self.site.rows):
            if ER % progress_interval == 0:
                progress_percentage = (ER // progress_interval) * 20
                print(f"{progress_percentage}...")
                # if progress_percentage == 10 :  return True   used for testing only

            for EC in range(self.site.cols):
                flat_arr_index = ER * self.site.cols + EC

                subsite = None
                read_cell = None
                if not dik_only and not sal_only:
                    read_cell = init_cell()  # Assuming init_cell properly initializes a cell
                    subsite = self.site.get_subsite(EC, ER, read_cell)

                # Read dike data
                dik_number = dik_file.get_next_number() if dike_f_exists else 0

                # Read storm surge data
                storm_num1 = storm_file1.get_next_number() if storm_file1 is not None and sf_exists1 else -10
                storm_num2 = storm_file2.get_next_number() if storm_file2 is not None and sf_exists2 else -10

                # Read salinity data
                sal_number = sal_file.get_next_number() if sal_f_exists else 0

                if dik_only or sal_only:
                    read_dike_or_salinity_only(ER, EC)
                    continue  # Skip to the next iteration

                # Process raster output sites data
                ros_number = ros_file.get_next_number() if self.ros_file_name else 0

                # Get NWI data and date
                elev_number = elev_file.get_next_number()
                slope_number = slp_file.get_next_number()

                nwi_number = int(nwi_file.get_next_number()) if nwi_file else BLANK
                if nwi_number == 0:
                    nwi_number = BLANK
                if nwi_number == NO_DATA:
                    nwi_number = BLANK

                # Handle impervious file data
                if imp_f_exists:
                    pct_imp = int(imp_file.get_next_number())
                    read_cell['imp_coeff'] = pct_imp
                    if pct_imp >= 0 and nwi_number in (1, 2):   # classic categories only, developed & undeveloped
                        nwi_number = 1 if pct_imp > 25 else 2   # 1 is developed dry land

                # MTL Correction and Lagoon adjustment
                mtl_correction = subsite.navd88mtl_correction
                if vd_f_exists:
                    vd_number = vd_file.get_next_number()
                    if int(vd_number) != NO_DATA:
                        mtl_correction = vd_number

                lagoon_correction = 0
                if subsite.lagoon_type != TLagoonType.LtNone:
                    lagoon_correction = subsite.lbeta * subsite.zbeach_crest

                # Distance to Mouth adjustments
                if d2m_f_exists:
                    d2m_number = d2m_file.get_next_number()
                    if int(d2m_number) != NO_DATA:
                        read_cell['d2mouth'] = d2m_number

                # Adjustments for uncertainty
                for i in range(2):
                    if self.run_uncertainty and self.uncert_setup.z_uncert_map[i] is not None and self.uncert_setup.unc_sens_iter > 0:
                        if int(elev_number) != NO_DATA and len(self.uncert_setup.z_uncert_map[i]) > 0:
                            elev_number += self.uncert_setup.z_uncert_map[i][flat_arr_index]

                # Dike attributes
                read_cell['prot_dikes'] = int(dik_number) != NO_DATA and dik_number != 0
                if not self.classic_dike:
                    read_cell['elev_dikes'] = int(dik_number) != NO_DATA and dik_number > 0
                    if (nwi_number in range(15, 20) or nwi_number == NO_DATA) and read_cell['elev_dikes']:  # classic categories only: a cell with a dike in open water is converted to developed dry land
                        nwi_number = 1

                # Assign land cover category
                read_cat = gis_lookup[int(nwi_number)] if 0 <= nwi_number <= 255 else BLANK

                # Salinity assignment
                if sal_f_exists:
                    read_cell['sal'][0] = sal_number

                # MTL to NAVD88 and uplift corrections
                read_cell['mtl_minus_navd'] = mtl_correction
                if mtl_correction > self.max_mtl:
                    self.max_mtl = mtl_correction
                if mtl_correction < self.min_mtl:
                    self.min_mtl = mtl_correction

                read_cell['uplift'] = uplift_file.get_next_number() if uplift_f_exists else 0

                # Initialize erosion losses
                read_cell['erosion_loss'] = 0
                read_cell['btf_erosion_loss'] = 0

                # Handling for no elevation data
                if int(elev_number) == NO_DATA and self.load_blank_if_no_elev:
                    if read_cat not in (self.categories.estuarine_water, self.categories.open_ocean):
                        read_cat = BLANK

                set_cell_width(read_cell, read_cat, total_width)

                # Assign the subsidence rate
                if not uplift_f_exists or int(read_cell['uplift']) == NO_DATA:
                    hist_adj = subsite.historic_trend - subsite.historic_eustatic_trend  # mm/yr
                    # { mm/yr global historic trend, subtracted from
                    # the local historic trend to remove double counting when forecasting
                    read_cell['uplift'] = -hist_adj * 0.1  # Convert mm/yr to cm/yr

                # Set cell slope
                if int(slope_number) != NO_DATA:
                    read_cell['tan_slope'] = abs(
                        math.tan(math.radians(slope_number)))  # Correct handling of negative slopes

                # Dike elevation adjustments
                if not self.classic_dike and read_cell['elev_dikes']:
                    elev_number = dik_number - mtl_correction  # Adjust dike elevation to MTL

                # Process storm data if available and not just counting
                if sf_exists1 and storm_num1 > -9.99 and not count_only:
                    storm_num1 -= mtl_correction  # Convert storm elevation to MTL basis
                    self.ss_rasters[1][1][flat_arr_index] = float_to_word(storm_num1)

                if sf_exists2 and storm_num2 > -9.99 and not count_only:
                    storm_num2 -= mtl_correction  # Convert storm elevation to MTL basis
                    self.ss_rasters[2][1][flat_arr_index] = float_to_word(storm_num2)

                # Assign elevations to the cells
                if int(elev_number) != NO_DATA:
                    # Elevation adjustments for cells that are not dikes
                    if not read_cell['elev_dikes']:
                        # Reference to MTL and adjust for lagoon correction
                        elev_number -= (mtl_correction + lagoon_correction)
                        # Consider uplift for the years between DEM and NWI photo dates
                        land_movement = -(read_cell['uplift'] * 10)  # cm/year to mm/year
                        dem_to_nwi_m = (subsite.nwi_photo_date - subsite.dem_date) * land_movement * 0.001
                        # 0.001 is mm/yr to m/year
                        elev_number -= dem_to_nwi_m
                        # Slope correction
                        slope_adjustment = (self.site.site_scale * 0.5) * read_cell['tan_slope']
                        elev_number -= slope_adjustment

                    # Set the calculated elevation to the cell
                    set_cat_elev(read_cell, read_cat, elev_number)

                    # Store the elevation data if not just counting
                    if not count_only:
                        if large_raster_edit or (USE_DATAELEV and self.optimize_level > 1):
                            self.shared_data.data_elev[ER, EC] = float_to_word(elev_number)

                # Store ROS numbers if the ROS file exists
                ros_number = int(ros_number)
                if ros_f_exists:
                    self.shared_data.ros_array[ER][EC] = ros_number
                    # Assign min and max boundaries for ROS "Raster Output Sites"
                    if self.site.max_ros < ros_number:
                        if ros_number > len(self.site.ros_bounds):
                            self.site.ros_bounds.extend([TRectangle(0,0,0,0)] * (ros_number + 2 - len(self.site.ros_bounds)))

                        for i in range(self.site.max_ros, ros_number):
                            self.site.ros_bounds[i] = TRectangle(99999, 99999, -99999, -99999)
                        self.site.max_ros = ros_number

                    if ros_number > 0:
                        ros_bound = self.site.ros_bounds[ros_number - 1]
                        ros_bound.x1 = min(ros_bound.x1, EC)
                        ros_bound.y1 = min(ros_bound.y1, ER)
                        ros_bound.x2 = max(ros_bound.x2, EC)
                        ros_bound.y2 = max(ros_bound.y2, ER)

                # Determine whether to add this cell to the map
                add_to_map = (self.optimize_level == 0) or (read_cat != BLANK)
                neg_num = -99  # blank
                if self.optimize_level > 1:
                    if (read_cat in [self.categories.open_ocean, self.categories.estuarine_water]) or \
                            ((read_cat in [self.categories.dev_dry_land, self.categories.und_dry_land]) and (
                                    cat_elev(read_cell, read_cat) > ELEV_CUTOFF)):
                        neg_num = -nwi_number
                        add_to_map = False

                # Increment cell counter if the cell is added to the map
                if add_to_map:
                    n_cells += 1

                # Handle the map matrix and cell saving based on the settings
                if not count_only:
                    if add_to_map and (not large_raster_edit):
                        self.shared_data.map_matrix[ER, EC] = n_cells - 1
                    else:
                        self.shared_data.map_matrix[ER, EC] = neg_num

                    if large_raster_edit:
                        self.shared_data.map_matrix[ER, EC] = nwi_number  # use map_matrix to track type only
                    else:
                        self.set_a(ER, EC, read_cell)  # Set the new values of the cell in memory

        # # Ensure at least one timestep is accounted for
        # if self.n_time_steps == 0:
        #     self.n_time_steps = 1

        if self.run_specific_years:
            self.n_time_steps = 3 + self.years_string.count(',')  # TimeIter = InitCond + T0 + 1 + number of commas in comma delimited string
        else:
            t0step = self.site.t0() + self.time_step
            firstyear = max(t0step, 2025)

            if self.time_step < 25 and t0step < 2020:
                firstyear = 2020
            if self.time_step < 15 and t0step < 2010:
                firstyear = 2010

            steps = (self.max_year - firstyear) // self.time_step + 1
            if firstyear + (steps - 1) * self.time_step < self.max_year:  # add max_year, smaller time step if required
                steps += 1
            self.n_time_steps = steps + 2  # other steps + initCond + T0

        # Initialize road variables and potentially overwrite elevations
        for road_info in self.roads_inf:
            road_info.initialize_road_vars()
            road_info.overwrite_elevs()

        # Initialize point information variables
        for point_info in self.point_inf:
            point_info.initialize_point_vars()

        # If only updating dikes or salinity, exit early
        if dik_only or sal_only:
            return True

        # Handle uncertainty settings and adjust the number of memory map entries if needed
        if self.run_uncertainty and self.optimize_level > 1:
            if self.uncert_setup.unc_sens_iter == 0:
                self.num_mm_entries = int(n_cells * 1.10)  # buffer by 10%
                # in case elevation uncertainty push more cells down into the tracked zone.
        else:
            self.num_mm_entries = n_cells

        # Initialize cells with default values
        self.blank_cell = init_cell()
        set_cell_width(self.blank_cell, BLANK, self.site.site_scale)

        self.ocean_cell = init_cell()
        set_cell_width(self.ocean_cell, self.categories.open_ocean, self.site.site_scale)
        set_cat_elev(self.ocean_cell, self.categories.open_ocean, -self.site.global_site.salt_elev)

        self.eow_cell = init_cell()
        set_cell_width(self.eow_cell, self.categories.estuarine_water, self.site.site_scale)
        set_cat_elev(self.eow_cell, self.categories.estuarine_water, -self.site.global_site.salt_elev)

        self.und_dry_cell = init_cell()
        set_cell_width(self.und_dry_cell, self.categories.und_dry_land, self.site.site_scale)
        set_cat_elev(self.und_dry_cell, self.categories.und_dry_land, ELEV_CUTOFF)

        self.dev_dry_cell = init_cell()
        set_cell_width(self.dev_dry_cell, self.categories.dev_dry_land, self.site.site_scale)
        set_cat_elev(self.dev_dry_cell, self.categories.dev_dry_land, ELEV_CUTOFF)

        # Hide progress form if only counting cells
        # if count_only:
        # prog_form.hide()

        return True

    def validate_output_sites(self) -> bool:
        # Ensure all output sites are valid
        valid = True
        for i, site in enumerate(self.site.output_sites):
            if site.use_polygon:
                # Validate polygons
                if site.poly.num_pts > 1:
                    for j in range(site.poly.num_pts):
                        # Ensure points are within bounds
                        point = site.poly.points[j]
                        point.x = max(0, min(point.x, self.site.cols - 1))
                        point.y = max(0, min(point.y, self.site.rows - 1))
                else:
                    # Handle error if polygon is not defined properly
                    print(f"Boundary of polygon output site {i + 1} not properly defined.")
                    valid = False
                    break
            else:
                # Validate rectangular regions
                rect = site.rec
                if not (rect.x1 > 0 or rect.x2 > 0 or rect.y1 > 0 or rect.y2 > 0):
                    print(f"Boundary of output site {i + 1} not properly defined.")
                    valid = False
                    break

                # Adjust rectangle coordinates to be within bounds
                rect.x1 = max(0, min(rect.x1, self.site.cols - 1))
                rect.x2 = max(0, min(rect.x2, self.site.cols - 1))
                rect.y1 = max(0, min(rect.y1, self.site.rows - 1))
                rect.y2 = max(0, min(rect.y2, self.site.rows - 1))

        return valid

    def thread_inundate_erode(self, start_row, end_row, share_dat, erode_loop, sum_data):

        self.shared_data = share_dat
        for er in range(start_row, end_row + 1):
            for ec in range(self.site.cols):
                proc_cell = self.ret_a(er, ec)
                self.transfer(proc_cell, er, ec, erode_loop, sum_data)
                self.set_a(er, ec, proc_cell)

        return sum_data

    def execute_parallel_tasks(self, task_function, *args, calculate_sums=False):
        num_cpus = self.cpu_count
        rows_per_partition = max(1, self.site.rows // num_cpus)

        processes = []
        queue2 = None

        shared_memory_names, shapes, dtypes = self.shared_data.get_shared_memory_info()
        main_thread_shared_data = self.shared_data
        self.shared_data = None

        if calculate_sums:
            queue2 = Queue()

        for i in range(num_cpus):
            start_row = i * rows_per_partition
            end_row = (i + 1) * rows_per_partition - 1 if i != num_cpus - 1 else self.site.rows - 1

            p = Process(target=worker_function, args=(
                start_row, end_row, args, calculate_sums, queue2, shared_memory_names, shapes, dtypes,
                self.site.n_output_sites, self.site.max_ros, task_function, *args))
            processes.append(p)
            p.start()

        # Collect calculate_sums results and wait for all sentinel values that signify thread completion
        active_processes = num_cpus
        if calculate_sums:
            while active_processes > 0:
                sum_data = queue2.get()
                if sum_data is None:
                    active_processes -= 1  # Sentinel received
                else:
                    self.cat_sums += sum_data.cat_sums
                time.sleep(0.01)  # Sleep for 10 milliseconds

        for p in processes:
            p.join()

        self.shared_data = main_thread_shared_data

        return True

    def inundate_erode(self):
        print('     calculating inundation')
        if self.execute_parallel_tasks(self.thread_inundate_erode, False, calculate_sums=True):
            print('     calculating erosion')
            return self.execute_parallel_tasks(self.thread_inundate_erode, True, calculate_sums=True)

    def update_proj_run_string(self):
        if self.year <= 0:
            self.proj_run_string = 'Initial_Inundation'
            self.short_proj_run = 'T0'
        else:
            self.short_proj_run = f'S{self.scen_iter}_{self.year}'
            self.proj_run_string = f'{self.year}_'

            if self.running_tsslr:
                self.proj_run_string += f' {self.time_ser_slrs[self.tsslr_index - 1].name} '
            elif self.running_fixed:
                self.proj_run_string += f'_{LabelFixed[self.fixed_scen]}'
            elif self.running_custom:
                self.proj_run_string += f'_{self.current_custom_slr:.2f}m'
            else:
                self.proj_run_string += f'_{LabelIPCC[self.ipcc_sl_rate]}_{LabelIPCCEst[self.ipcc_sl_est]}'

            if self.protect_all:
                self.proj_run_string += '_PADL'
                self.short_proj_run += '_PA'
            elif self.protect_developed:
                self.proj_run_string += '_PDDL'
                self.short_proj_run += '_PD'

            if not self.include_dikes:
                self.proj_run_string += '_ND'
                self.short_proj_run += '_ND'

    def run_one_year(self, is_t0: bool) -> bool:

        self.eustatic_sl_change(is_t0)  # Updates the "Year" variable and Local SL Variables

        if not is_t0:
            prog_str = f'Running Yearly Scenario : {self.year}'
            # self.prog_form.year_label_visible = True
            self.time_zero = False
        else:
            prog_str = 'Running Time Zero : '
            # self.prog_form.year_label_visible = False
            self.time_zero = True

        if self.cpu_count > 1:
            prog_str += f'({self.cpu_count} CPUs)'

        # Update Projection Run String
        self.update_proj_run_string()

        print(prog_str)
        # self.prog_form.progress_label = prog_str
        # self.prog_form.show()

        # Check for annual update maps if it is not time zero, e.g. yearly salinity or dike updates
        if not is_t0:
            successful = self.check_for_annual_maps()
            if not successful:
                return False

        # self.prog_form.year_label = str(self.year)
        successful = self.update_elevations()
        if not successful:
            return False

        print('     chng_water method')
        successful = self.chng_water(True, not is_t0)
        if not successful:
            return False

        # Inundation
        self.inund_freq_check = False
        if self.inund_maps or (self.save_gis and self.save_inund_gis) or (self.n_point_inf > 0) or (
                self.n_road_inf > 0):
            successful = self.calc_inund_freq()
            self.inund_freq_check = successful
        if not successful:
            return False

        # Check Connectivity
        self.connect_check = False
        if self.check_connectivity or self.connect_maps:
            print('     checking connectivity')
            self.connect_arr = self.calc_inund_connectivity(self.connect_arr, True, -1)
            self.connect_check = True
        else:
            self.connect_arr = None

        if not successful:
            return False

        sav_prob_check = False
        if self.d2m_file_name != '':
            successful = self.calculate_prob_sav()
            sav_prob_check = successful
        if not successful:
            return False

        # Calculate frequency of road inundation if road data exists
        road_inund_check = False
        if self.n_road_inf > 0:
            for j in range(self.n_road_inf):
                successful = self.roads_inf[j].calc_all_roads_inundation()
                road_inund_check = successful

        if self.n_point_inf > 0:
            for j in range(self.n_point_inf):
                successful = self.point_inf[j].calc_all_point_inf_inundation()

        if not successful:
            return False

        if not is_t0 and (self.sal_file_name == '' or '.xls' in self.sal_file_name.lower()):
            successful = self.calc_salinity(False, False)

        if not successful:
            return False

        # Parallel or Sequential Processing based on CPU count
        successful = self.inundate_erode()

        if self.n_road_inf > 0:
            for j in range(self.n_road_inf):
                self.roads_inf[j].update_road_sums(self.tstep_iter, self.road_sums,
                                                   self.site.n_output_sites + self.site.max_ros)

        self.summarize(self.year)  # write array for tabular output

        return successful

    def eustatic_sl_change(self, is_t0):
        """Eustatic SL Change that is later adjusted for local factors when adjusting cell elevations."""

        # Constants that define scenarios and estimations for IPCC Results
        ipcc_results = np.array([
            [[28, 27.5, 30, 26, 27, 28.5],  # MIN, 2025
             [63, 66, 64, 58, 52, 56],  # MIN, 2050
             [100, 125, 94, 103, 76, 85],  # MIN, 2075
             [129, 182, 111, 155, 92, 114]],  # MIN, 2100
            [[76, 81.5, 75.5, 74.5, 75.5, 79],  # MEAN, 2025
             [167, 175, 172, 157, 150, 160],  # MEAN, 2050
             [278.5, 278, 323, 277, 232.5, 255],  # MEAN, 2075
             [387, 367, 491, 424, 310, 358]],  # MEAN, 2100
            [[128, 128.5, 137, 126.5, 128, 134],  # MAX, 2025
             [284, 291, 299, 269, 259, 277],  # MAX, 2050
             [484.5, 553, 491, 478, 412.5, 451],  # MAX, 2075
             [694, 859, 671, 743, 567, 646]]  # MAX, 2100
        ])

        # Fixed results and coefficients for ESVA SLR
        fixed_results = np.array([  # After 3 is NYS Scenarios
            [184.4, 276.7, 368.9, 127, 129.4, 127, 254],  # 2025, Scen
            [409.2, 613.8, 818.4, 304.8, 431.0, 482.6, 736.6],  # 2050, Scen, 2055 for NYS
            [698.1, 1047.2, 1396.3, 584.2, 806.6, 1041.4, 1397],  # 2075, Scen, 2085 for NYS
            [1000, 1500, 2000.0, 717.6, 1000.0, 1327.2, 1720.9]  # 2100, Scen
        ])
        esva_slr_a_coeff = np.array([0, 0.00271, 0.00871, 0.0156])  # Quadratic coefficients for ESVA SLR

        def calc_dt():
            """Calculate Delta T when the model is run beyond T0."""
            if self.run_specific_years:
                return get_next_year_from_string()
            else:
                if self.year < 2025:
                    c_dt = 2025 - self.year
                else:
                    c_dt = self.time_step

                if self.time_step < 25:
                    if self.year < 2020:
                        c_dt = 2020 - self.year
                    else:
                        c_dt = self.time_step

                if self.time_step < 15:
                    if self.year < 2010:
                        c_dt = 2010 - self.year
                    else:
                        c_dt = self.time_step

                self.year += c_dt

                if self.year > 2100:
                    c_dt -= (self.year - 2100)
                    self.year = 2100

                return c_dt

        def get_next_year_from_string():
            min_num = 99999
            max_num = -99999
            input_nums = self.years_string
            eof_found = False

            while not eof_found:
                try:
                    # Simulate the extraction of the next numeric part before the comma
                    holder = input_nums.split(',', 1)[0]
                    number = int(holder.strip())
                except ValueError:
                    raise ValueError(
                        f'Invalid Numeric Input in Specific Years String "'
                        f'{self.years_string}" - Must be comma separated integers.')

                if number > max_num:
                    max_num = number
                if self.year < number < min_num:
                    min_num = number

                # Remove processed part and the comma
                if ',' in input_nums:
                    input_nums = input_nums.split(',', 1)[1]
                else:
                    eof_found = True

                if input_nums.strip() == '':
                    eof_found = True

            if min_num == 99999:
                min_num = self.year

            self.dt = min_num - self.year
            self.year = min_num
            self.max_year = max_num

            if self.year > 2100:
                self.dt -= (self.year - 2100)
                self.year = 2100

            return self.dt

        def norm_data_year(year):
            """Determines the normalized data based on the year."""
            if year > 2100:
                raise Exception('SLAMM Predictions do not extend beyond 2100.')

            # Determine index based on year
            if year in [2050, 2075, 2100]:
                ipcc_year_index = {2050: 1, 2075: 2, 2100: 3}[year]
            else:
                ipcc_year_index = 0

            if self.running_fixed:
                if self.fixed_scen < 8:
                    # NYS SLR Scenarios
                    if self.fixed_scen > 3:
                        ipcc_year_index = {2055: 1, 2085: 2, 2100: 3}.get(year, 0)
                    nrm = fixed_results[ipcc_year_index, self.fixed_scen] * 0.1
                else:
                    # ESVA SLR Scenarios (linear coefficient + subsidence)
                    nrm = (esva_slr_a_coeff[self.fixed_scen - 7] * (year - 1992) + 0.17) * (year - 1992)
            else:
                nrm = ipcc_results[self.ipcc_sl_est.value, ipcc_year_index, self.ipcc_sl_rate.value] * 0.1

            return nrm

        def return_norm(year):
            lyi = -1
            if self.running_tsslr:
                tsslr = self.time_ser_slrs[self.tsslr_index - 1]
                for j in range(tsslr.n_years):
                    if tsslr.slr_arr[j].year < year:
                        lyi = j

                uyi = lyi + 1
                if lyi < 0:
                    lower_norm = 0
                    lower_year = tsslr.base_year
                else:
                    lower_norm = tsslr.slr_arr[lyi].value
                    lower_year = tsslr.slr_arr[lyi].year

                if uyi == tsslr.n_years:
                    raise Exception(f'Running year {year} that is beyond the time series in "{tsslr.name}"')
                else:
                    upper_norm = tsslr.slr_arr[uyi].value
                    upper_year = tsslr.slr_arr[uyi].year

                if upper_year == lower_year:
                    raise Exception(f'Duplicate year {lower_year} in the time series in "{tsslr.name}"')

                return 100 * linear_interpolate(lower_norm, upper_norm, lower_year, upper_year, year, False)

            if self.running_custom:
                self.running_fixed = False
                self.ipcc_sl_rate = IPCCScenarios.Scen_A1B
                self.ipcc_sl_est = IPCCEstimates.Est_Max

            result = None
            if year > 2000:
                if (year % 25 == 0 and not self.running_fixed) or \
                        (year % 25 == 0 and self.running_fixed and self.fixed_scen < 4) or \
                        (self.running_fixed and 3 < self.fixed_scen < 8 and year in (
                                2025, 2055, 2085, 2100)) or \
                        (self.running_fixed and self.fixed_scen > 7 and year > 1992):
                    result = norm_data_year(year)
                else:
                    lower_year = (year // 25) * 25
                    upper_year = lower_year + 25

                    if self.running_fixed and 3 < self.fixed_scen < 8:
                        if lower_year == 2025:
                            upper_year = 2055
                        elif lower_year == 2050:
                            if year < 2055:
                                lower_year = 2025
                                upper_year = 2055
                            else:
                                lower_year = 2055
                                upper_year = 2085
                        elif lower_year == 2075:
                            if year < 2085:
                                lower_year = 2055
                                upper_year = 2085
                            else:
                                lower_year = 2085

                    if lower_year > 2000:
                        lower_norm = norm_data_year(lower_year)
                    else:
                        lower_year = 1990
                        if self.running_fixed and 3 < self.fixed_scen < 8:
                            lower_year = 2002
                        lower_norm = 0

                    if upper_year < 2025:
                        upper_year = 2025
                    upper_norm = norm_data_year(upper_year)

                    result = linear_interpolate(lower_norm, upper_norm, lower_year, upper_year, year, False)

            if self.running_custom:
                if self.current_custom_slr < 0:
                    self.current_custom_slr = 0
                scen_by_2100 = norm_data_year(2100)
                # below code--  if current_custom_slr is 50% scen_by_2100 then all dates are scaled by 50%
                return result * (self.current_custom_slr * 100) / scen_by_2100

            return result

        # start of main function eustatic_sl_change(self, is_t0)  -------------------------------------------
        if is_t0:
            dt = 0
        else:
            dt = calc_dt()

        for i in range(self.site.n_subsites + 1):  # Including global site as well
            tss = self.site.global_site if i == 0 else self.site.subsites[i - 1]

            tss.old_sl = tss.newsl
            tss.delta_t = dt if not is_t0 else self.year - tss.nwi_photo_date

            if self.year == tss.nwi_photo_date:
                tss.norm = 0
            else:
                year_zero = 1990
                if self.running_fixed and self.fixed_scen > 3:
                    year_zero = 2002 if self.fixed_scen < 8 else 1992
                if self.running_tsslr:
                    year_zero = self.time_ser_slrs[self.tsslr_index - 1].base_year

                if self.year > year_zero:
                    tss.norm = return_norm(self.year)
                    if tss.nwi_photo_date < year_zero:
                        tss.norm += (year_zero - tss.nwi_photo_date) * tss.historic_eustatic_trend * 0.1
                    elif tss.nwi_photo_date > year_zero:
                        tss.norm -= return_norm(tss.nwi_photo_date)
                else:
                    tss.norm = (self.year - tss.nwi_photo_date) * tss.historic_eustatic_trend * 0.1

            tss.newsl = tss.norm * 0.01  # NewSL is Eustatic SLR since simulation start
            tss.sl_rise = tss.newsl - tss.old_sl

            if is_t0:
                tss.t0_slr = tss.sl_rise  # Eustatic SLR at T0 if relevant

    @staticmethod
    def third_ord_poly(a, b, c, d, x):
        """
        Return the value of a third order polynomial a*x^3 + b*x^2 + c*x + d
        """
        x2 = x * x
        x3 = x2 * x
        return a * x3 + b * x2 + c * x + d

    def dynamic_accretion(self, cell, cat: int, ss: TSubSite, model_num: int):
        """
        Calculate accretion, mm/year
        """
        if not ss.use_accr_model[model_num]:
            if model_num == 0:
                return ss.fixed_reg_flood_accr
            elif model_num == 1:
                return ss.fixed_irreg_flood_accr
            elif model_num == 2:
                return ss.fixed_tf_beach_sed
            else:
                return ss.fixed_tide_fresh_accr

        if ss.mhhw < TINY:  # Avoid divide by zero for zero tide-range subsites
            return ss.min_accr[model_num]

        # Rescale Accretion rates if not done yet
        if not ss.acc_rescaled[model_num]:
            self.rescale_accretion(ss, cat, model_num)

        # Retrieve the elevation of the cell in meters
        elev_min = self.lower_bound(cat, ss)
        elev_max = self.upper_bound(cat, ss)
        elev = cat_elev(cell, cat)
        elev = max(min(elev, elev_max), elev_min)

        # Convert elevation to HTU (Height Tide Units)
        elev /= ss.mhhw

        # Elevation Effects using a third order polynomial
        a_elev = self.third_ord_poly(ss.raccr_a[model_num], ss.raccr_b[model_num],
                                     ss.raccr_c[model_num], ss.raccr_d[model_num], elev)

        # Minimum accretion adjustments for specific marsh types, uncomment if applicable
        # if model_num == 0 and a_elev < 2.0:  # Reg Flood Marsh minimum
        #     a_elev = 2.0
        # if model_num == 1 and a_elev < 2.0 and lower_half:  # Irreg Flood Marsh lower boundary minimum
        #     a_elev = 2.0

        # No negative elevation changes for now
        if a_elev < 0:
            a_elev = 0

        return a_elev

    def lower_bound(self, cat: int, ss: TSubSite) -> float:
        """
        Calculate the lower bound for a category in meters above Mean Tidal Level (MTL).
        """
        elev_dat = self.categories.get_cat(cat).elev_dat
        if elev_dat.min_unit == ElevUnit.SaltBound:
            return elev_dat.min_elev * ss.salt_elev
        elif elev_dat.min_unit == ElevUnit.HalfTide:
            return elev_dat.min_elev * ss.mhhw
        else:  # ElevUnit.Meters
            return elev_dat.min_elev

    def upper_bound(self, cat: int, ss: TSubSite) -> float:
        """
        Calculate the upper bound for a category in meters above Mean Tidal Level (MTL).
        """
        elev_dat = self.categories.get_cat(cat).elev_dat
        if elev_dat.max_unit == ElevUnit.SaltBound:
            return elev_dat.max_elev * ss.salt_elev
        elif elev_dat.max_unit == ElevUnit.HalfTide:
            return elev_dat.max_elev * ss.mhhw
        else:  # ElevUnit.Meters
            return elev_dat.max_elev

    def rescale_accretion(self, ss, cat, model_num):
        """
        Rescale the accretion coefficients to the Min/Max input accretion parameters.
        """

        def second_ord_poly_roots(a, b, c):
            """
            Return the roots of a second order polynomial a*x^2 + b*x + c = 0.
            """
            if a == 0:
                if b == 0:
                    return [np.nan]  # No solution when a and b both are zero.
                return [-c / b]  # Single solution when "a" is zero.
            delta = b ** 2 - 4 * a * c
            if delta < 0:
                return [np.nan, np.nan]  # No real roots if the discriminant is negative.
            root1 = (-b - np.sqrt(delta)) / (2 * a)
            root2 = (-b + np.sqrt(delta)) / (2 * a)
            return [root1, root2]

        min_elev = self.lower_bound(cat, ss) / ss.mhhw
        max_elev = self.upper_bound(cat, ss) / ss.mhhw
        root_elev = second_ord_poly_roots(3 * ss.accr_a[model_num], 2 * ss.accr_b[model_num], ss.accr_c[model_num])

        # Calculate the accretions at the boundaries
        min_acc = self.third_ord_poly(ss.accr_a[model_num], ss.accr_b[model_num], ss.accr_c[model_num],
                                      ss.accr_d[model_num], min_elev)
        max_acc = self.third_ord_poly(ss.accr_a[model_num], ss.accr_b[model_num], ss.accr_c[model_num],
                                      ss.accr_d[model_num], max_elev)
        if min_acc > max_acc:
            min_acc, max_acc = max_acc, min_acc  # Swap if out of order

        # Check polynomial roots within the interval for extreme values
        for root in root_elev:
            if not np.isnan(root) and min_elev <= root <= max_elev:
                tmp_acc = self.third_ord_poly(ss.accr_a[model_num], ss.accr_b[model_num], ss.accr_c[model_num],
                                              ss.accr_d[model_num], root)
                if tmp_acc > max_acc:
                    max_acc = tmp_acc
                if tmp_acc < min_acc:
                    min_acc = tmp_acc

        # Rescale coefficients a, b, c
        rescal_f = 1 if max_acc != min_acc else 0
        if max_acc != min_acc:
            rescal_f = (ss.max_accr[model_num] - ss.min_accr[model_num]) / (max_acc - min_acc)

        ss.raccr_a[model_num] = rescal_f * ss.accr_a[model_num]
        ss.raccr_b[model_num] = rescal_f * ss.accr_b[model_num]
        ss.raccr_c[model_num] = rescal_f * ss.accr_c[model_num]

        # Rescale coefficient d
        ss.raccr_d[model_num] = ss.accr_d[model_num] if max_acc == min_acc \
            else rescal_f * (ss.max_accr[model_num] - ss.min_accr[model_num]) + ss.min_accr[model_num]

        # Mark the accretion rates as rescaled
        ss.acc_rescaled[model_num] = True

    def chng_water(self, calc_mf, calc_geometry):
        result = self.execute_parallel_tasks(self.thread_chng_water, None, calculate_sums=False)

        if result and calc_mf:
            result = self.execute_parallel_tasks(self.calculate_max_fetch, self.max_w_erode)

        if result and calc_geometry:
            self.calculate_flow_geometry(self.year)

        return result

    def thread_chng_water(self, start_row, end_row, share_dat):

        self.shared_data = share_dat
        for er in range(start_row, end_row + 1):
            for ec in range(self.site.cols):
                # update progress potentially

                self.shared_data.b_matrix[er, ec] = 0
                self.shared_data.erode_matrix[er, ec] = 0

                cwc = self.ret_a(er, ec)

                if cell_width(cwc, self.categories.open_ocean) / self.site.site_scale > 0.5:
                    self.set_bit(EXPOSED_WATER, er, ec, True)
                else:
                    self.set_bit(EXPOSED_WATER, er, ec, False)

                width_open_water = 0
                width_tidal = 0
                width_efsw = 0

                for i in range(NUM_CAT_COMPRESS):
                    if self.categories.get_cat(cwc['cats'][i]).is_open_water:
                        width_open_water += cwc['widths'][i]
                    if self.categories.get_cat(cwc['cats'][i]).is_tidal:
                        width_tidal += cwc['widths'][i]
                    if cwc['cats'][i] == 14:  # hard-wired to CAL EFSW category
                        width_efsw += cwc['widths'][i]

                self.set_bit(MOSTLY_WATER, er, ec, width_open_water / self.site.site_scale > 0.9)
                self.set_bit(HAS_EFSW, er, ec, width_efsw / self.site.site_scale > 0.01)
                self.set_bit(SALT_WATER, er, ec, width_tidal / self.site.site_scale > 0.1)

                # Add accretion to "data elevation" used in salinity calculations
                acc = 0
                if not self.include_dikes or not cwc['prot_dikes']:
                    cat = get_cell_cat(cwc)
                    acc_model = self.categories.get_cat(cat).accr_model
                    subsite = self.site.get_subsite(ec, er, cwc)
                    if acc_model == AccrModels.RegFM:
                        acc = subsite.delta_t * self.dynamic_accretion(cwc, cat, subsite, 0)
                    elif acc_model == AccrModels.IrregFM:
                        acc = subsite.delta_t * self.dynamic_accretion(cwc, cat, subsite, 1)
                    elif acc_model == AccrModels.BeachTF:
                        acc = subsite.delta_t * self.dynamic_accretion(cwc, cat, subsite, 2)
                    elif acc_model == AccrModels.TidalFM:
                        acc = subsite.delta_t * self.dynamic_accretion(cwc, cat, subsite, 3)
                    elif acc_model == AccrModels.InlandM:
                        acc = subsite.delta_t * subsite.inland_fresh_accr
                    elif acc_model == AccrModels.Mangrove:
                        acc = subsite.delta_t * subsite.mangrove_accr
                    elif acc_model == AccrModels.TSwamp:
                        acc = subsite.delta_t * subsite.tswamp_accr
                    elif acc_model == AccrModels.Swamp:
                        acc = subsite.delta_t * subsite.swamp_accr

                if USE_DATAELEV and self.optimize_level > 1 and acc > 0:
                    self.shared_data.data_elev[er, ec] = min(self.shared_data.data_elev[er, ec] + round(acc), 65535)
                    #                     {mm}                       {mm}                {mm}   {max elev (word)}

        return True

    def update_elevations(self) -> bool:
        print('     updating elevations')
        return self.execute_parallel_tasks(self.thread_update_elevs, None)

    def thread_update_elevs(self, start_row: int, end_row: int, shared_dat) -> bool:

        self.shared_data = shared_dat

        # Additional helper functions for accretion and turbidity calculations
        def calculate_accretion(cat, cell):
            accr = 0.0
            if not self.include_dikes or not cell['prot_dikes']:
                a_model = self.categories.get_cat(cat).accr_model
                if a_model == AccrModels.AccrNone:
                    return 0.0
                elif a_model == AccrModels.RegFM:
                    accr = subsite.delta_t * self.dynamic_accretion(cell, cat, subsite, 0) * 0.001
                elif a_model == AccrModels.IrregFM:
                    accr = subsite.delta_t * self.dynamic_accretion(cell, cat, subsite, 1) * 0.001
                elif a_model == AccrModels.BeachTF:
                    accr = subsite.delta_t * self.dynamic_accretion(cell, cat, subsite, 2) * 0.001
                elif a_model == AccrModels.TidalFM:
                    accr = subsite.delta_t * self.dynamic_accretion(cell, cat, subsite, 3) * 0.001
                elif a_model == AccrModels.InlandM:
                    accr = subsite.delta_t * subsite.inland_fresh_accr * 0.001
                elif a_model == AccrModels.Mangrove:
                    accr = subsite.delta_t * subsite.mangrove_accr * 0.001
                elif a_model == AccrModels.TSwamp:
                    accr = subsite.delta_t * subsite.tswamp_accr * 0.001
                elif a_model == AccrModels.Swamp:
                    accr = subsite.delta_t * subsite.swamp_accr * 0.001

            return accr

        def calculate_turbidity_factor(r, c, accr):
            nonlocal self
            turbid_factor = 1.0
            nfw = 0
            if accr > 0:
                for fw in range(self.num_fw_flows):
                    if self.fw_influenced_specific(r, c, fw):
                        if self.fw_flows[fw].use_turbidities:
                            nfw += 1
                            if nfw == 1:
                                turbid_factor = self.fw_flows[fw].turbidity_by_year(self.year)
                            else:
                                turbid_factor = (self.fw_flows[fw].turbidity_by_year(self.year) + (
                                        turbid_factor * (nfw - 1))) / nfw
            return turbid_factor

        for er in range(start_row, end_row + 1):
            for ec in range(self.site.cols):
                # update progress optionally

                ecl = self.ret_a(er, ec)
                subsite = self.site.get_subsite(ec, er, ecl)
                cell_set = False

                for i in range(NUM_CAT_COMPRESS):
                    if ecl['min_elevs'][i] < 998:  # no data
                        this_cat = ecl['cats'][i]
                        if this_cat != BLANK:
                            cell_set = True
                            this_elev = ecl['min_elevs'][i] - subsite.sl_rise  # Adjust for eustatic changes first

                            this_elev += ecl['uplift'] * subsite.delta_t * 0.01  # account for uplift or subsidence
                            accretion = 0
                            if not self.include_dikes or not ecl['prot_dikes']:
                                accretion = calculate_accretion(this_cat, ecl)

                            turb_factor = calculate_turbidity_factor(er, ec, accretion)
                            this_elev += accretion * turb_factor  # account for accretion
                            ecl['min_elevs'][i] = this_elev

                if cell_set:
                    self.set_a(er, ec, ecl)
        return True

    def fw_influenced(self, row: int, col: int) -> bool:
        """Check if the given row and column are influenced by any FW flow."""
        for i in range(self.num_fw_flows):
            if self.fw_influenced_specific(row, col, i):
                return True
        return False

    def fw_influenced_specific(self, row: int, col: int, fw_num: int) -> bool:
        """Check if the given row and column are influenced by a specific FW flow."""
        return self.fw_flows[fw_num].poly.in_poly(row, col)

    def calculate_flow_geometry(self, yr: int) -> bool:
        def setup_geometry():
            for e in range(self.num_fw_flows):
                fw_flow:TFWFlow = self.fw_flows[e]
                if not fw_flow.extent_only:
                    fw_flow.max_rn = 0

                    if (fw_flow.midpts is None) or (len(fw_flow.midpts) < fw_flow.num_segments):
                        fw_flow.midpts = [DPoint(0, 0) for _ in range(fw_flow.num_segments)]
                        fw_flow.origin_line = [DLine(DPoint(0, 0), DPoint(0, 0)) for _ in range(fw_flow.num_segments)]
                        fw_flow.plume_line = [DLine(DPoint(0, 0), DPoint(0, 0)) for _ in range(fw_flow.num_segments)]
                        fw_flow.d2origin = [0.0 for _ in range(fw_flow.num_segments)]

                        fw_flow.estuary_length = 0

                        for i in range(fw_flow.num_segments):
                            # Calculate midpoints
                            fw_flow.midpts[i] = DPoint(
                                (fw_flow.mouth_arr[i].x + fw_flow.origin_arr[i].x) / 2,
                                (fw_flow.mouth_arr[i].y + fw_flow.origin_arr[i].y) / 2,
                            )

                            # Calculate vectors
                            row_vector = fw_flow.mouth_arr[i].y - fw_flow.origin_arr[i].y
                            col_vector = fw_flow.mouth_arr[i].x - fw_flow.origin_arr[i].x

                            # Initialize origin and plume lines
                            fw_flow.origin_line[i] = DLine(DPoint(fw_flow.origin_arr[i].x, fw_flow.origin_arr[i].y),
                                                           DPoint(0, 0))
                            fw_flow.plume_line[i] = DLine(DPoint(fw_flow.mouth_arr[i].x, fw_flow.mouth_arr[i].y),
                                                          DPoint(0, 0))

                            # Update lines based on vectors
                            if col_vector != 0:
                                fw_flow.origin_line[i].p2.y = fw_flow.origin_arr[i].y + col_vector
                                fw_flow.origin_line[i].p2.x = fw_flow.origin_arr[i].x - row_vector
                                fw_flow.plume_line[i].p2.y = fw_flow.mouth_arr[i].y + col_vector
                                fw_flow.plume_line[i].p2.x = fw_flow.mouth_arr[i].x - row_vector
                            else:
                                fw_flow.origin_line[i].p2.y = fw_flow.origin_arr[i].y
                                fw_flow.origin_line[i].p2.x = fw_flow.origin_arr[i].x - 1
                                fw_flow.plume_line[i].p2.y = fw_flow.mouth_arr[i].y
                                fw_flow.plume_line[i].p2.x = fw_flow.mouth_arr[i].x - 1

                            # Update d2origin and estuary_length
                            fw_flow.d2origin[i] = fw_flow.estuary_length
                            fw_flow.estuary_length += distance_2pts_km(
                                DPoint(fw_flow.origin_arr[i].x, fw_flow.origin_arr[i].y),
                                DPoint(fw_flow.mouth_arr[i].x, fw_flow.mouth_arr[i].y),
                                self.site.site_scale
                            )

                        array_len = int(fw_flow.estuary_length / SLICE_INCREMENT) + 50

                        for k in range(NUM_SAL_METRICS):
                            fw_flow.ret_time[k] = np.zeros(array_len)
                            fw_flow.water_z[k] = np.zeros(array_len)
                            fw_flow.xs_salinity[k] = np.zeros(array_len)

                        fw_flow.subsite_arr = [None] * array_len
                        fw_flow.ftables = [[0] * N_FTABLE for _ in range(array_len)]
                        fw_flow.num_cells = [0] * array_len

                    array_len = len(fw_flow.ftables)
                    for i in range(array_len):
                        for j in range(N_FTABLE):
                            fw_flow.ftables[i][j] = 0
                        fw_flow.num_cells[i] = 0
                        for k in range(NUM_SAL_METRICS):
                            fw_flow.water_z[k][i] = 0
                            fw_flow.xs_salinity[k][i] = 0
                        fw_flow.subsite_arr[i] = None

                    fw_flow.river_mouth_index = 999999

        def add_length(max_len):
            nonlocal fw_num
            with self.fw_flows[fw_num] as fw_flow:
                array_len = max_len + 50

                # Extend num_cells
                fw_flow.num_cells.extend([0] * (array_len - len(fw_flow.num_cells)))

                for k in range(NUM_SAL_METRICS):
                    # Extend ret_time, water_z, and xs_salinity
                    if len(fw_flow.ret_time[k]) < array_len:
                        new_ret_time = np.zeros(array_len, dtype=fw_flow.ret_time[k].dtype)
                        new_ret_time[:len(fw_flow.ret_time[k])] = fw_flow.ret_time[k]
                        fw_flow.ret_time[k] = new_ret_time

                    if len(fw_flow.water_z[k]) < array_len:
                        new_water_z = np.zeros(array_len, dtype=fw_flow.water_z[k].dtype)
                        new_water_z[:len(fw_flow.water_z[k])] = fw_flow.water_z[k]
                        fw_flow.water_z[k] = new_water_z

                    if len(fw_flow.xs_salinity[k]) < array_len:
                        new_xs_salinity = np.zeros(array_len, dtype=fw_flow.xs_salinity[k].dtype)
                        new_xs_salinity[:len(fw_flow.xs_salinity[k])] = fw_flow.xs_salinity[k]
                        fw_flow.xs_salinity[k] = new_xs_salinity

                # Extend subsite_arr
                fw_flow.subsite_arr.extend([None] * (array_len - len(fw_flow.subsite_arr)))

                # Extend ftables
                for i in range(len(fw_flow.ftables), array_len):
                    fw_flow.ftables.append([0] * N_FTABLE)

                # Initialize new elements to 0
                for i in range(len(fw_flow.num_cells), array_len):
                    fw_flow.num_cells[i] = 0
                    for k in range(NUM_SAL_METRICS):
                        fw_flow.ret_time[k][i] = 0
                        fw_flow.water_z[k][i] = 0
                        fw_flow.xs_salinity[k][i] = 0
                    fw_flow.subsite_arr[i] = None

        if self.num_fw_flows == 0:
            return True

        print(f"Calculating Flow Geometry: {yr}")

        all_extent_only = all(fw_flow.extent_only for fw_flow in self.fw_flows)
        if all_extent_only:
            return True

        setup_geometry()

        cell_area = self.site.site_scale * self.site.site_scale

        for er in range(self.site.rows):
            for ec in range(self.site.cols):
                for fw_num in range(self.num_fw_flows):  # loop to calculate f tables
                    if self.fw_influenced_specific(er, ec, fw_num) and not self.fw_flows[fw_num].extent_only:
                        rkm, d2c = self.river_km(er, ec, fw_num)
                        rn = int(rkm / SLICE_INCREMENT)

                        with self.fw_flows[fw_num] as fw_flow:
                            fw_flow.test_min = self.site.global_site.mllw - 2
                            fw_flow.test_range = self.site.global_site.mhhw + 2 - fw_flow.test_min

                            if rn + 1 > len(fw_flow.ftables):
                                add_length(rn + 1)
                            if rn > fw_flow.max_rn:
                                fw_flow.max_rn = rn
                                fw_flow.ocean_subsite = self.site.get_subsite(ec, er)
                            fw_flow.subsite_arr[rn] = self.site.get_subsite(ec, er)
                            fw_flow.num_cells[rn] += 1

                        cl = self.ret_a(er, ec)

                        if (self.classic_dike and not cl['prot_dikes']) or (
                                not self.classic_dike and not cl['prot_dikes']) or (
                                not self.classic_dike and cl['elev_dikes']):
                            cell_elev = get_min_elev(cl)

                            for i in range(N_FTABLE):
                                with self.fw_flows[fw_num] as fw_flow:
                                    fw_flow.test_elev = fw_flow.test_min + (fw_flow.test_range / (N_FTABLE - 1)) * i
                                    if cell_elev < fw_flow.test_elev:
                                        fw_flow.ftables[rn][i] += (fw_flow.test_elev - cell_elev) * cell_area

        for fw_num in range(self.num_fw_flows):
            with self.fw_flows[fw_num] as fw_flow:
                if not fw_flow.extent_only and fw_flow.retention_initialized:
                    for rn in range(len(fw_flow.ftables)):
                        for i in range(NUM_SAL_METRICS):
                            if i == 0:
                                sal_z = fw_flow.salt_height(fw_flow.ocean_subsite.mllw, fw_flow.ocean_subsite.newsl,
                                                            rn, yr)
                            elif i == 1:
                                sal_z = fw_flow.salt_height(0.0, fw_flow.ocean_subsite.newsl, rn, yr)
                            elif i == 2:
                                sal_z = fw_flow.salt_height(fw_flow.ocean_subsite.mhhw, fw_flow.ocean_subsite.newsl,
                                                            rn, yr)
                            else:
                                sal_z = fw_flow.salt_height(fw_flow.ocean_subsite.salt_elev,
                                                            fw_flow.ocean_subsite.newsl, rn, yr)

                            fw_flow_val = fw_flow.flow_by_year(yr)
                            vol_fresh = fw_flow_val * fw_flow.ret_time[i][rn]
                            vol_salt = fw_flow.volume_by_elev(sal_z, rn)
                            if vol_salt + vol_fresh > TINY:
                                fw_flow.xs_salinity[i][rn] = ((vol_fresh * fw_flow.fw_ppt) + (
                                            vol_salt * fw_flow.sw_ppt)) / (
                                                                     vol_salt + vol_fresh)
                            else:
                                fw_flow.xs_salinity[i][rn] = 0

                            fw_flow.water_z[i][rn] = fw_flow.elev_by_volume(vol_salt + vol_fresh, rn)

        for fw_num in range(self.num_fw_flows):
            with self.fw_flows[fw_num] as fw_flow:
                if not fw_flow.extent_only and not fw_flow.retention_initialized:
                    for rn in range(fw_flow.max_rn):
                        for i in range(NUM_SAL_METRICS):
                            if i == 0:
                                sal_z = fw_flow.salt_height(fw_flow.ocean_subsite.mllw, fw_flow.ocean_subsite.newsl,
                                                            rn, yr)
                            elif i == 1:
                                sal_z = fw_flow.salt_height(0.0, fw_flow.ocean_subsite.newsl, rn, yr)
                            elif i == 2:
                                sal_z = fw_flow.salt_height(fw_flow.ocean_subsite.mhhw, fw_flow.ocean_subsite.newsl,
                                                            rn, yr)
                            else:
                                sal_z = fw_flow.salt_height(fw_flow.ocean_subsite.salt_elev,
                                                            fw_flow.ocean_subsite.newsl, rn, yr)

                            if i == 0:
                                fw_flow.water_z[i][rn] = fw_flow.subsite_arr[rn].mllw
                            elif i == 1:
                                fw_flow.water_z[i][rn] = 0
                            elif i == 2:
                                fw_flow.water_z[i][rn] = fw_flow.subsite_arr[rn].mhhw
                            else:
                                fw_flow.water_z[i][rn] = fw_flow.subsite_arr[rn].salt_elev

                            vol_salt = fw_flow.volume_by_elev(sal_z, rn)
                            vol_fresh = fw_flow.volume_by_elev(fw_flow.water_z[i][rn], rn) - vol_salt

                            fresh_flow = fw_flow.flow_by_year(yr)
                            if fresh_flow == 0:
                                fw_flow.ret_time[i][rn] = 0
                            else:
                                fw_flow.ret_time[i][rn] = vol_fresh / fresh_flow

                            if vol_salt + vol_fresh > TINY:
                                fw_flow.xs_salinity[i][rn] = ((vol_fresh * fw_flow.fw_ppt) + (
                                            vol_salt * fw_flow.sw_ppt)) / (
                                                                     vol_salt + vol_fresh)
                            else:
                                fw_flow.xs_salinity[i][rn] = 0

                            fw_flow.retention_initialized = True

        self.smooth_xs_salinity()
        return True

    def river_km(self, pr: int, pc: int, fw_num: int) -> Tuple[float, float]:
        River_Influenced = 1  # km, Distance beyond which river reach influence is zero and plume effects dominate
        pt = DPoint(pc, pr)
        found_distance = 1e6
        d2c = 0.0
        river_km_result = 0.0

        fw_flow = self.fw_flows[fw_num]

        if fw_flow.midpts is None:
            return 0.0, d2c  # Avoid crash

        for i in range(fw_flow.num_segments):
            tl = DLine(fw_flow.origin_arr[i], fw_flow.mouth_arr[i])
            plume_cross = cross_line(pt, fw_flow.midpts[i], fw_flow.plume_line[i])

            if not cross_line(pt, fw_flow.midpts[i], fw_flow.origin_line[i]) and not plume_cross:
                this_distance = distance_pt_2_line_km(tl, pt, self.site.site_scale)
                if this_distance < found_distance:
                    river_km_result = distance_pt_2_line_km(fw_flow.origin_line[i], pt, self.site.site_scale) + \
                                      fw_flow.d2origin[i]
                    found_distance = this_distance
                    d2c = found_distance * 1000

            if plume_cross and i < fw_flow.num_segments - 1:
                if cross_line(pt, fw_flow.midpts[i + 1], fw_flow.origin_line[i + 1]):
                    this_distance = distance_2pts_km(fw_flow.origin_arr[i + 1], pt, self.site.site_scale)
                    if this_distance < found_distance:
                        river_km_result = fw_flow.d2origin[i + 1]
                        found_distance = this_distance
                        d2c = found_distance * 1000

        if fw_flow.midpts is None:
            raise ValueError('Cannot run a site with freshwater flows in which the river channel is not defined.')

        if cross_line(pt, fw_flow.midpts[0], fw_flow.origin_line[0]):
            this_distance = distance_2pts_km(pt, fw_flow.origin_arr[0], self.site.site_scale)
            if this_distance < found_distance:
                found_distance = this_distance
                d2c = found_distance * 1000
                river_km_result = 0

        in_plume = cross_line(pt, fw_flow.midpts[fw_flow.num_segments - 1],
                              fw_flow.plume_line[fw_flow.num_segments - 1])
        if not in_plume:
            in_plume = (found_distance > River_Influenced) and (
                    distance_pt_2_line_km(fw_flow.plume_line[fw_flow.num_segments - 1], pt, self.site.site_scale) < 3)

        if in_plume:
            this_distance = distance_2pts_km(pt, fw_flow.mouth_arr[fw_flow.num_segments - 1],
                                             self.site.site_scale)
            if this_distance < found_distance or found_distance > River_Influenced:
                river_km_result = distance_2pts_km(pt, fw_flow.mouth_arr[fw_flow.num_segments - 1],
                                                   self.site.site_scale) + fw_flow.estuary_length
                d2c = (river_km_result - fw_flow.estuary_length) * 1000
                if fw_flow.river_mouth_index > (int(river_km_result / SLICE_INCREMENT) - 1):
                    fw_flow.river_mouth_index = int(river_km_result / SLICE_INCREMENT) - 1

        return river_km_result, d2c

    def smooth_xs_salinity(self):
        for adjustments in range(1, 4):
            for fw_num in range(self.num_fw_flows):
                fw_flow = self.fw_flows[fw_num]
                if not fw_flow.extent_only:
                    for rn in range(fw_flow.max_rn):
                        for i in range(NUM_SAL_METRICS):
                            if (
                                    fw_flow.xs_salinity[i][rn] > fw_flow.xs_salinity[i][rn + 1] > TINY
                            ):
                                # salinity increasing upstream, so average the two
                                mean_xs = (fw_flow.xs_salinity[i][rn] + fw_flow.xs_salinity[i][rn + 1]) * 0.5
                                fw_flow.xs_salinity[i][rn] = mean_xs
                                fw_flow.xs_salinity[i][rn + 1] = mean_xs

        # adjust XS_Salinity so there are no upstream areas more saline than downstream areas
        for fw_num in range(self.num_fw_flows):
            fw_flow = self.fw_flows[fw_num]
            if not fw_flow.extent_only:
                for rn in range(fw_flow.max_rn - 1, -1, -1):
                    for i in range(NUM_SAL_METRICS):
                        if (
                                fw_flow.xs_salinity[i][rn] > fw_flow.xs_salinity[i][rn + 1] > TINY
                        ):
                            # salinity increasing upstream, so set upstream segment to salinity of lower
                            fw_flow.xs_salinity[i][rn] = fw_flow.xs_salinity[i][rn + 1]

    def clear_flow_geometry(self):
        for j in range(self.num_fw_flows):
            fw_flow = self.fw_flows[j]
            fw_flow.midpts = None
            fw_flow.d2origin = None
            for i in range(NUM_SAL_METRICS):
                fw_flow.ret_time[i] = np.array([])
                fw_flow.water_z[i] = np.array([])
                fw_flow.xs_salinity[i] = np.array([])
            fw_flow.origin_line = None
            fw_flow.plume_line = None

    def calculate_max_fetch(self, start_row, end_row, share_dat, l_max_ew):

        self.shared_data = share_dat
        high_fetch = 20.1
        curr_x = 0
        curr_y = 0
        max_fetch_index = 0
        mfc: compressed_cell_dtype
        fetch_index = 0
        this_fetch = 0.0
        over_ocean = False
        marsh_elev = 0.0
        marsh_width = 0.0
        power_array = [0.0] * (N_WIND_DIRECTIONS)
        subs = None

        def valid_cell(c, r):
            return 0 <= r < self.site.rows and 0 <= c < self.site.cols

        def check_cell(c, r):
            nonlocal over_ocean, fetch_index, this_fetch

            if not valid_cell(c, r):
                if self.get_bit(EXPOSED_WATER, curr_y, curr_x):
                    mfc['max_fetch'] = high_fetch
                    this_fetch = high_fetch
                return False
            else:
                if self.get_bit(EXPOSED_WATER, r, c):
                    over_ocean = True
                cc_result = self.get_bit(MOSTLY_WATER, r, c)
                if not cc_result and over_ocean:
                    cell = self.ret_a(r, c)
                    if cell_width(cell, BLANK) > 0:
                        mfc['max_fetch'] = high_fetch
                        this_fetch = high_fetch
                if cc_result:
                    if subs.use_wave_erosion and subs.we_has_bathymetry:
                        fetch_index += 1
                        if fetch_index < max_fetch_index - 1:
                            cell = self.ret_a(r, c)
                            fetch_depths[fetch_index] = get_min_elev(cell)
                return cc_result

        def calculate_wave_power():
            nonlocal this_fetch, fetch_index, mfc, marsh_elev, subs, fetch_depths, power_array, compass_points

            ga = 9.81  # gravity acceleration in m/s^2
            w_dens = 1025  # water density in kg/m^3
            qtu = subs.gtide_range / 4  # quarter tide range

            total_power = 0

            def calc_tot_ip(w_dir):
                nonlocal tot_ip
                deg_45 = 0.7073883
                deg_225 = 0.923955699
                deg_675 = 0.383235147

                tot_ip = 0
                if w_dir in [0, 4, 8, 12]:  # North, South, East, West
                    tot_ip = 1.0
                elif w_dir == 2:  # northeast
                    if check_cell(ec + 1, er):
                        tot_ip += deg_45
                    if check_cell(ec, er - 1):
                        tot_ip += deg_45
                elif w_dir == 6:  # northwest
                    if check_cell(ec - 1, er):
                        tot_ip += deg_45
                    if check_cell(ec, er - 1):
                        tot_ip += deg_45
                elif w_dir == 14:  # southeast
                    if check_cell(ec + 1, er):
                        tot_ip += deg_45
                    if check_cell(ec, er + 1):
                        tot_ip += deg_45
                elif w_dir == 10:  # southwest
                    if check_cell(ec - 1, er):
                        tot_ip += deg_45
                    if check_cell(ec, er + 1):
                        tot_ip += deg_45
                elif w_dir == 1:  # ENE
                    if check_cell(ec + 1, er):
                        tot_ip += deg_225
                    if check_cell(ec, er - 1):
                        tot_ip += deg_675
                elif w_dir == 7:  # WNW
                    if check_cell(ec - 1, er):
                        tot_ip += deg_225
                    if check_cell(ec, er - 1):
                        tot_ip += deg_675
                elif w_dir == 15:  # ESE
                    if check_cell(ec + 1, er):
                        tot_ip += deg_225
                    if check_cell(ec, er + 1):
                        tot_ip += deg_675
                elif w_dir == 9:  # WSW
                    if check_cell(ec - 1, er):
                        tot_ip += deg_225
                    if check_cell(ec, er + 1):
                        tot_ip += deg_675
                elif w_dir == 3:  # NNE
                    if check_cell(ec + 1, er):
                        tot_ip += deg_675
                    if check_cell(ec, er - 1):
                        tot_ip += deg_225
                elif w_dir == 5:  # NNW
                    if check_cell(ec - 1, er):
                        tot_ip += deg_675
                    if check_cell(ec, er - 1):
                        tot_ip += deg_225
                elif w_dir == 13:  # SSE
                    if check_cell(ec + 1, er):
                        tot_ip += deg_675
                    if check_cell(ec, er + 1):
                        tot_ip += deg_225
                elif w_dir == 11:  # SSW
                    if check_cell(ec - 1, er):
                        tot_ip += deg_675
                    if check_cell(ec, er + 1):
                        tot_ip += deg_225
                return tot_ip

            wv_dir = compass_points
            tot_ip = calc_tot_ip(wv_dir)

            avg_depth_mtl = 0
            if subs.we_has_bathymetry:
                if fetch_index > 2:
                    max_search = min(int(fetch_index / 3), (round(7000 / self.site.site_scale) - 1))
                    for i in range(max_search + 1):
                        avg_depth_mtl -= fetch_depths[i]
                    avg_depth_mtl /= max_search
            else:
                avg_depth_mtl = subs.we_avg_shallow_depth

            for ws in range(N_WIND_SPEEDS):
                if self.wind_rose[wv_dir][ws] > 0:
                    for tides in range(1, 6):
                        water_level = (tides - 3) * qtu
                        avg_depth = avg_depth_mtl + water_level

                        if avg_depth > 0:
                            wind_speed = [0.25, 1.25, 3, 5, 7, 9, 11][ws]
                            fetch = this_fetch * 1000
                            nwd = ga * avg_depth / wind_speed ** 2
                            ndf = ga * fetch / wind_speed ** 2
                            a1 = 0.493 * nwd ** 0.75
                            b1 = 3.13e-3 * ndf ** 0.57
                            nwe = 3.63e-3 * (tanh(a1) * tanh(b1 / tanh(a1))) ** 1.74
                            we = w_dens * ga * nwe * wind_speed ** 4 / ga ** 2

                            if water_level <= marsh_elev:
                                a2 = 0.33 * nwd ** 1.01
                                b2 = 5.215e-4 * ndf ** 0.73
                                ndwf = 0.133 * (tanh(a2) * tanh(b2 / tanh(a2))) ** -0.37
                                wf = ndwf * ga / wind_speed
                                wp = 1 / wf
                                g = avg_depth * (2 * pi / wp) ** 2 / ga
                                f = g + 1 / (1 + 0.6522 * g + 0.4622 * g ** 2 + 0.0864 * g ** 4 + 0.0675 * g ** 5)
                                wlen = wp * sqrt(ga * avg_depth / f)
                                wcel = wlen / wp
                                wnum = 2 * pi / wlen
                                wgc = 0.5 * wcel * (1 + 2 * wnum * avg_depth / sinh(2 * wnum * avg_depth))
                                pw = wgc * we

                                pi_w = pw * tot_ip * self.wind_rose[wv_dir][ws] / 500
                                total_power += pi_w
                                power_array[wv_dir] += pi_w

            mfc['pw'] += total_power

        def check_fetch(dx, dy, incr):
            nonlocal over_ocean, this_fetch, curr_x, curr_y, marsh_elev, marsh_width, compass_points, er, ec
            this_fetch = 0
            curr_x = ec
            curr_y = er
            clear_path = True
            over_ocean = False

            while clear_path:
                if abs(dx) > 1:
                    clear_path = check_cell(curr_x + (dx // 2), curr_y) and check_cell(curr_x + dx, curr_y)
                if abs(dy) > 1:
                    clear_path = check_cell(curr_x, curr_y + (dy // 2)) and check_cell(curr_x, curr_y + dy)
                if clear_path:
                    clear_path = check_cell(curr_x + dx, curr_y + dy)
                    if clear_path:
                        this_fetch += incr * self.site.site_scale * 0.001
                        curr_x += dx
                        curr_y += dy

            if this_fetch > 0.1 and subs.use_wave_erosion:
                calc_power, min_elev, marsh_width = self.wave_erosion_category(mfc)
                if calc_power:
                    calculate_wave_power()

            if this_fetch > mfc['max_fetch']:
                mfc['max_fetch'] = this_fetch

        def middle_of_water(c, r):
            if not self.get_bit(MOSTLY_WATER, r, c):
                return False
            if valid_cell(c, r + 1) and valid_cell(c, r - 1) and valid_cell(c + 1, r) and valid_cell(c - 1, r):
                return self.get_bit(MOSTLY_WATER, r - 1, c) and self.get_bit(MOSTLY_WATER, r + 1, c) and \
                    self.get_bit(MOSTLY_WATER, r, c - 1) and self.get_bit(MOSTLY_WATER, r, c + 1)
            return False

        def distribute_wp_erosion(ex, ey, to_distrib):
            nonlocal mfc, power_array

            def distribute(dx, dy, d_frac):
                nonlocal ex, ey, frac_this_wd
                if not valid_cell(ex - dx, ey - dy):
                    return
                d_cell = self.ret_a(ey - dy, ex - dx)
                if self.wave_erosion_category(d_cell)[0]:
                    d_cell['wp_erosion'] += to_distrib * frac_this_wd * d_frac
                    self.set_a(ey - dy, ex - dx, d_cell)

            for cp in range(N_WIND_DIRECTIONS):
                if power_array[cp] > 0:
                    frac_this_wd = power_array[cp] / mfc['pw']
                    if cp == 0:
                        distribute(1, 0, 1)  # E 90
                    elif cp == 1:
                        distribute(0, -1, 0.293)  # E-NE 67.5
                        distribute(1, 0, 0.707)
                    elif cp == 2:
                        distribute(0, -1, 0.5)  # NE 45
                        distribute(1, 0, 0.5)
                    elif cp == 3:
                        distribute(0, -1, 0.707)  # N-NE 22.5
                        distribute(1, 0, 0.293)
                    elif cp == 4:
                        distribute(0, -1, 1)  # N 0
                    elif cp == 5:
                        distribute(0, -1, 0.707)  # N-NW 337.5
                        distribute(-1, 0, 0.293)
                    elif cp == 6:
                        distribute(0, -1, 0.5)  # NW 315
                        distribute(-1, 0, 0.5)
                    elif cp == 7:
                        distribute(0, -1, 0.293)  # W-NW 292.5
                        distribute(-1, 0, 0.707)
                    elif cp == 8:
                        distribute(-1, 0, 1)  # W 270
                    elif cp == 9:
                        distribute(0, 1, 0.293)  # W-SW 247.5
                        distribute(-1, 0, 0.707)
                    elif cp == 10:
                        distribute(0, 1, 0.5)  # SW 225
                        distribute(-1, 0, 0.5)
                    elif cp == 11:
                        distribute(0, 1, 0.707)  # S-SW 202.5
                        distribute(-1, 0, 0.293)
                    elif cp == 12:
                        distribute(0, 1, 1)  # S 180
                    elif cp == 13:
                        distribute(0, 1, 0.707)  # S-SE 157.5
                        distribute(1, 0, 0.293)
                    elif cp == 14:
                        distribute(0, 1, 0.5)  # SE 135
                        distribute(1, 0, 0.5)
                    elif cp == 15:
                        distribute(0, 1, 0.293)  # E-SE 112.5
                        distribute(1, 0, 0.707)

        # ---  main function calculate_max_fetch  ---
        fetch_depths = None
        for er in range(start_row, end_row + 1):
            for ec in range(self.site.cols):
                # potentially update progress
                if not middle_of_water(ec, er):
                    mfc = self.ret_a(er, ec)
                    mfc['max_fetch'] = 0
                    mfc['pw'] = 0
                    mfc['wp_erosion'] = 0
                    subs = self.site.get_subsite(ec, er, mfc)
                    if subs.use_wave_erosion:
                        fetch_index = -1
                        if subs.we_has_bathymetry and fetch_depths is None:
                            fetch_depths = np.zeros(round(7000 / self.site.site_scale))
                        for wd in range(N_WIND_DIRECTIONS):
                            power_array[wd] = 0

                    for compass_points in range(N_WIND_DIRECTIONS):
                        if compass_points == 0:
                            check_fetch(1, 0, 1)
                        elif compass_points == 1:
                            check_fetch(2, -1, 2.236)
                        elif compass_points == 2:
                            check_fetch(1, -1, 1.412)
                        elif compass_points == 3:
                            check_fetch(1, -2, 2.236)
                        elif compass_points == 4:
                            check_fetch(0, -1, 1)
                        elif compass_points == 5:
                            check_fetch(-1, -2, 2.236)
                        elif compass_points == 6:
                            check_fetch(-1, -1, 1.412)
                        elif compass_points == 7:
                            check_fetch(-2, -1, 2.236)
                        elif compass_points == 8:
                            check_fetch(-1, 0, 1)
                        elif compass_points == 9:
                            check_fetch(-2, 1, 2.236)
                        elif compass_points == 10:
                            check_fetch(-1, 1, 1.412)
                        elif compass_points == 11:
                            check_fetch(-1, 2, 2.236)
                        elif compass_points == 12:
                            check_fetch(0, 1, 1)
                        elif compass_points == 13:
                            check_fetch(1, 2, 2.236)
                        elif compass_points == 14:
                            check_fetch(1, 1, 1.412)
                        elif compass_points == 15:
                            check_fetch(2, 1, 2.236)

                    if self.optimize_level > 1:
                        if mfc['max_fetch'] > 65.5:
                            self.shared_data.max_fetch_arr[er, ec] = 65500
                        else:
                            self.shared_data.max_fetch_arr[er, ec] = round(mfc['max_fetch'] * 1000)

                    mfc['wp_erosion'] = mfc['pw'] * subs.we_alpha
                    self.set_a(er, ec, mfc)
                    if mfc['wp_erosion'] > l_max_ew:
                        l_max_ew = mfc['wp_erosion']

                    erosion_beyond_cell = mfc['wp_erosion'] * subs.delta_t - marsh_width
                    if erosion_beyond_cell > 0:
                        distribute_wp_erosion(ec, er, erosion_beyond_cell)
                else:
                    mfc = self.ret_a(er, ec)
                    if mfc['pw'] > 0:
                        mfc['pw'] = 0
                        mfc['wp_erosion'] = 0
                        self.set_a(er, ec, mfc)

        fetch_depths = None
        return True

    def transfer(self, cell, each_row, each_col, erode_step, sum_data):
        # {-------------------------------------------}
        # {      Transfer percentages of Classes      }
        # {         R. A. Park    2/21-29/88          }
        # {        JSC Modified,  1998, 2005-2007     }
        # {-------------------------------------------}

        wetland_type = 0
        water_table = 0.0
        distance_to_water = 0.0
        adj_max_fetch = 0.0
        near_salt = False
        eroding = erode_step
        inund_context: Optional[TInundContext] = None
        frac_lost = [-9999.0 for _ in range(self.categories.n_cats + 1)]

        def adj_cell(adj_row, adj_col, lee):
            # Return adjacent cell to lee if Lee=True
            # AdjRow and AdjCol are inputs and outputs
            # Function False if out-of-bounds
            # True: move opposite of offshore
            nonlocal subsite

            step = -1 if lee else 1
            if subsite.direction_offshore == WaveDirection.Westerly:
                adj_col -= step
            elif subsite.direction_offshore == WaveDirection.Easterly:
                adj_col += step
            elif subsite.direction_offshore == WaveDirection.Northerly:
                adj_row -= step
            else:  # subsite.direction_offshore == WaveDirection.Southerly:
                adj_row += step

            valid_cell = (adj_row >= 0) and (adj_col >= 0) and (adj_row < self.site.rows) and (
                    adj_col < self.site.cols)
            return valid_cell, adj_row, adj_col

        def convert(c_cell, from_cat, to_cat, low_bound):
            # Original version R. A. Park  2/21/88
            # Converts a cell from category to category based on FracLost[FromCategory]

            nonlocal frac_lost, subsite

            if frac_lost[from_cat] == -9999.0:
                raise Exception(f"FRACLOST Initialization Problem, Category {from_cat}")

            if frac_lost[from_cat] > 1.0:
                frac_lost[from_cat] = 1.0
            if frac_lost[from_cat] < 0.0:
                frac_lost[from_cat] = 0.0

            transferred = 0.0 if cell_width(c_cell, from_cat) < 0.001 else frac_lost[from_cat] * cell_width(
                c_cell, from_cat)

            if transferred > 0.0:
                if to_cat == BLANK:
                    raise Exception(f"Convert to Blank Category Error, Category Setup for number {from_cat}")

                from_min_elev = cat_elev(c_cell, from_cat)
                set_cell_width(c_cell, from_cat, cell_width(c_cell, from_cat) - transferred)
                set_cell_width(c_cell, to_cat, cell_width(c_cell, to_cat) + transferred)

                to_cat_elev = cat_elev(c_cell, to_cat)
                if to_cat_elev >= 999 or to_cat_elev > from_min_elev:
                    set_elev = from_min_elev

                    if not self.time_zero:
                        if self.categories.get_cat(from_cat).use_rfm_collapse:
                            set_elev -= subsite.ifm2rfm_collapse
                        if self.categories.get_cat(from_cat).use_ifm_collapse:
                            set_elev -= subsite.ifm2rfm_collapse

                    set_cat_elev(c_cell, to_cat, set_elev)

                if low_bound != -99 and cell_width(c_cell, to_cat) > 0:
                    set_cat_elev(c_cell, from_cat, low_bound)

        def calc_frac_lost_by_slope(c_cell, cat, low_bound):
            # Calculate Frac Lost using LowBound, Minelev and Cell Slope   J Clough  December 2005
            nonlocal frac_lost, eroding

            frac_lost[cat] = 0
            if eroding:
                return

            min_elev = cat_elev(c_cell, cat)
            if min_elev >= low_bound:
                return

            wid = cell_width(c_cell, cat)
            if wid < TINY:
                return

            if c_cell['tan_slope'] == 0:
                frac_lost[cat] = 1
            elif c_cell['tan_slope'] < 0:
                frac_lost[cat] = 0
            else:
                frac_lost[cat] = ((low_bound - min_elev) / c_cell['tan_slope']) / wid

            if frac_lost[cat] < 0:
                frac_lost[cat] = 0

        # Helper function to represent migration of water table near shoreline
        def saturate(s_cell, slot, to_cat):
            nonlocal eroding, water_table

            if eroding:
                return
            calc_frac_lost_by_slope(s_cell, s_cell['cats'][slot], water_table)
            convert(s_cell, s_cell['cats'][slot], to_cat, water_table)

        def convert_cell_by_salinity(s_cell):
            nonlocal frac_lost

            for c in range(NUM_CAT_COMPRESS):
                if s_cell['widths'][c] > 0 and s_cell['cats'][c] != BLANK:
                    cat = s_cell['cats'][c]
                    n_rules = 0
                    if self.categories.get_cat(cat).has_sal_rules:
                        n_rules = self.categories.get_cat(cat).salinity_rules.n_rules

                    for j in range(n_rules):
                        tsr = self.categories.get_cat(cat).salinity_rules.rules[j]

                        if tsr is not None:
                            if s_cell['sal'][NUM_SAL_METRICS-1] < -1:
                                tsr.salinity_tide = 0

                            cell_sal = s_cell['sal'][tsr.salinity_tide-1]

                            for convnum in range(2):
                                if tsr is not None:
                                    if (tsr.greater_than and cell_sal > tsr.salinity_level) or (
                                            not tsr.greater_than and cell_sal < tsr.salinity_level):
                                        frac_lost[cat] = 1.0
                                        convert(s_cell, cat, tsr.to_cat, -99)
                                        cat = tsr.to_cat
                                        tsrs = self.categories.get_cat(cat).salinity_rules
                                        if tsrs is None:
                                            tsr = None
                                        else:
                                            tsr = tsrs.rules[0]

        def soil_saturation(s_cell, slot):
            nonlocal subsite, water_table, distance_to_water, near_salt, wetland_type

            local_sl_rise = subsite.sl_rise - s_cell['uplift'] * subsite.delta_t * 0.01

            if self.year > 2000 and self.use_soil_saturation:
                water_table += (local_sl_rise / 0.91) * math.exp(-0.776 - 0.0012 * distance_to_water)

                if (near_salt and s_cell['min_elevs'][slot] < water_table and wetland_type > 0
                        and s_cell['min_elevs'][slot] < 10):
                    saturate(s_cell, slot, wetland_type)

        def inundate(i_cell):
            nonlocal frac_lost, subsite, each_row, each_col

            for c in range(NUM_CAT_COMPRESS):
                if i_cell['widths'][c] > TINY and i_cell['cats'][c] != BLANK:
                    from_cat = i_cell['cats'][c]
                    low_bound = self.lower_bound(from_cat, subsite)
                    ct_elev = i_cell['min_elevs'][c]

                    if low_bound > ct_elev:
                        calc_frac_lost_by_slope(i_cell, from_cat, low_bound)

                    if frac_lost[from_cat] > 0:
                        inund_context.subs = subsite
                        inund_context.cat_elev = ct_elev
                        inund_context.ew_cat = self.categories.estuarine_water
                        inund_context.cell_row = each_row
                        inund_context.cell_col = each_col

                        to_cat = self.categories.get_cat(from_cat).inund_cat(inund_context)
                        if to_cat != BLANK:
                            convert(i_cell, from_cat, to_cat, low_bound)

                    if self.categories.get_cat(from_cat).is_dryland and not self.categories.get_cat(
                            from_cat).is_developed:
                        soil_saturation(i_cell, c)

        def fetch_threshold():
            nonlocal inund_context, adj_max_fetch
            inund_context.erosion2 = adj_max_fetch

        def frac_to_erode(e_cell, erode_per_year, clss):
            nonlocal eroding, subsite, each_row, each_col

            if not eroding:
                return 0
            if erode_per_year <= 0 or subsite.delta_t <= 0:
                return 0

            tot_erosion = erode_per_year * subsite.delta_t
            width_eroded_os = 0

            adj_row = each_row
            adj_col = each_col
            edge_found = False

            def add_to_width():
                nonlocal width_eroded_os, edge_found
                ac = self.ret_a(adj_row, adj_col)

                if cell_width(ac, clss) > 0.01:  # 1cm
                    width_eroded_os += cell_width(ac, clss)
                else:
                    edge_found = True

                if edge_found:
                    cat = get_cell_cat(ac)
                    if not (self.get_bit(MOSTLY_WATER, adj_row,
                                         adj_col) or cat == self.categories.estuarine_water or cat == self.categories.open_ocean):
                        width_eroded_os = 1e6  # not adjacent to water, no erosion

                width_eroded_os += 0.01 * self.shared_data.erode_matrix[adj_row, adj_col]
                # account for erosion already occurred here this time step

            while not edge_found:
                valid_cell, adj_row, adj_col = adj_cell(adj_row, adj_col, False)
                if valid_cell:
                    add_to_width()
                else:
                    edge_found = True

                if width_eroded_os > tot_erosion:
                    return 0  # No erosion for this cell, enough erosion took place on off-shore side of cell

            cat_width = cell_width(e_cell, clss)
            frac_eroded = (tot_erosion - width_eroded_os) / cat_width

            erosion_loss_meters = cat_width if frac_eroded > 1 else tot_erosion - width_eroded_os

            if self.categories.get_cat(clss).erode_model in [ErosionInputs.ETFlat, ErosionInputs.EOcBeach]:
                e_cell['btf_erosion_loss'] += erosion_loss_meters
            else:
                e_cell['erosion_loss'] += erosion_loss_meters

            if erosion_loss_meters > self.shared_data.erode_matrix[each_row, each_col] * 100:
                self.shared_data.erode_matrix[each_row, each_col] = int(
                    erosion_loss_meters * 100)  # Track cell erosion in cm for more accuracy

            return frac_eroded

        def erode(e_cell):
            #  Original Code   R. A. Park 3/4/88
            #  IF BRUUN USED, Ocean Beach Only & Optional
            #  Recession = 100 * sl_rise (Bruun '62, '86)

            nonlocal subsite, inund_context, frac_lost, each_row, each_col

            def bruun():
                nonlocal e_cell, subsite, frac_lost
                local_sl_rise = subsite.sl_rise - e_cell['uplift'] * subsite.delta_t * 0.01
                # {m}                  {m}         {cm/yr}          {yr}     {cm/m}

                recession = 100.0 * local_sl_rise
                erosion_this_cell = recession - distance_to_water
                # {m}                {m}      {m from front edge to open ocean}

                if erosion_this_cell <= 0:
                    return

                beach_width = cell_width(e_cell, cat)
                if erosion_this_cell > beach_width:
                    erosion_this_cell = beach_width
                # {m}                 {m}

                frac_lost[cat] = erosion_this_cell / beach_width
                convert(e_cell, cat, self.categories.get_cat(cat).erode_to, -99)
                set_cat_elev(e_cell, self.categories.get_cat(cat).erode_to, 0)  # set open water to MTL for now
                e_cell['btf_erosion_loss'] += erosion_this_cell

            def wave_erosion():
                nonlocal e_cell, frac_lost, cat
                w_rate = e_cell['wp_erosion']
                if w_rate <= 0:
                    return

                frac_lost[cat] = frac_to_erode(e_cell, w_rate, cat)
                convert(e_cell, cat, self.categories.get_cat(cat).erode_to, -99)
                set_cat_elev(e_cell, self.categories.get_cat(cat).erode_to, 0)
                # Set the elevation of the category being transferred-to to MTL

            for c in range(NUM_CAT_COMPRESS):
                if e_cell['widths'][c] > TINY and e_cell['cats'][c] != BLANK:
                    cat = e_cell['cats'][c]
                    tc = self.categories.get_cat(cat)

                    if tc.use_wave_erosion and subsite.use_wave_erosion:
                        wave_erosion()
                    else:
                        e_model = self.categories.get_cat(cat).erode_model
                        if e_model == ErosionInputs.EOcBeach and self.use_bruun:
                            bruun()
                        elif e_model in [ErosionInputs.EOcBeach, ErosionInputs.ETFlat]:
                            # non bruun beach and tidal flat model
                            e_rate = subsite.tflat_erosion
                            frac_lost[cat] = frac_to_erode(e_cell, e_rate, cat)
                            convert(e_cell, cat, self.categories.get_cat(cat).erode_to, -99)
                            set_cat_elev(e_cell, self.categories.get_cat(cat).erode_to, 0)
                            # Set the elev of the category being transferred to to MTL
                        elif e_model == ErosionInputs.EMarsh:
                            if inund_context.erosion2 >= subsite.marsh_erode_fetch and (
                                    inund_context.adj_ocean or inund_context.adj_water):
                                # classic marsh erosion, but with editable fetch limit
                                e_rate = subsite.marsh_erosion
                                frac_lost[cat] = frac_to_erode(e_cell, e_rate, cat)
                                convert(e_cell, cat, self.categories.get_cat(cat).erode_to, -99)
                                set_cat_elev(e_cell, self.categories.get_cat(cat).erode_to, 0)
                                # Set the elev of the category being transferred to to MTL
                        elif e_model == ErosionInputs.ESwamp:
                            if inund_context.erosion2 >= 9 and (inund_context.adj_ocean or inund_context.adj_water):
                                # classic swamp erosion
                                e_rate = subsite.swamp_erosion
                                frac_lost[cat] = frac_to_erode(e_cell, e_rate, cat)
                                convert(e_cell, cat, self.categories.get_cat(cat).erode_to, -99)
                                set_cat_elev(e_cell, self.categories.get_cat(cat).erode_to, 0)
                                # Set the elev of the category being transferred to to MTL

        def test_adjacent(a_cell, row, col):
            # is exposed water, water, or saltwater adjacent?  i.e. within 500 m off-shore
            nonlocal inund_context, each_col, adj_cell, adj_max_fetch

            def adj1(iteration, new_row, new_col):
                nonlocal inund_context, adj_max_fetch

                # Implementation of the Adj1 function
                if not inund_context.adj_ocean:
                    inund_context.adj_ocean = self.get_bit(EXPOSED_WATER, new_row, new_col)
                if not inund_context.adj_efsw:
                    inund_context.adj_efsw = self.get_bit(HAS_EFSW, new_row, new_col)
                most_water = self.get_bit(MOSTLY_WATER, new_row, new_col)
                if not inund_context.adj_water:
                    inund_context.adj_water = most_water
                if not inund_context.adj_salt:
                    inund_context.adj_salt = self.get_bit(SALT_WATER, new_row, new_col)
                # ClearPathToOcean = ClearPathToOcean and (MostWater or AdjOcean)

                if iteration < 3:  # make function of cell-size and maybe erosion rate?
                    if self.optimize_level < 2:
                        c2 = self.ret_a(new_row, new_col)
                        if c2['max_fetch'] > adj_max_fetch:
                            adj_max_fetch = c2['max_fetch']  # Two immediately adjacent cells are tested
                    else:
                        if self.shared_data.max_fetch_arr[new_row, new_col] * 0.001 > adj_max_fetch:
                            # {m}                                            {km/m}     {km}
                            adj_max_fetch = self.shared_data.max_fetch_arr[new_row, new_col] * 0.001
                            # {km}                      {m}                                    {km/m}
                # --------   End adj1  ----------------------

            # Variables
            adj_row = row
            adj_col = col

            # Initialize adj_max_fetch
            adj_max_fetch = a_cell['max_fetch']
            if self.optimize_level > 1:
                adj_max_fetch = self.shared_data.max_fetch_arr[row, each_col] * 0.001

            # Initialize InundContext
            inund_context.adj_ocean = False
            inund_context.adj_water = False
            inund_context.adj_salt = False
            inund_context.adj_efsw = False

            # Test 500 m adjacent cells
            j = round(500.0 / self.site.site_scale)
            adj1(0, adj_row, adj_col)  # Test cell itself

            for i in range(1, j + 1):  # Test 500 m adjacent
                valid_cell, adj_row, adj_col = adj_cell(adj_row, adj_col, False)
                if not valid_cell:
                    break
                adj1(i, adj_row, adj_col)

        def test_near(t_cell):
            """
            ************************************************************
            * is shoreline near?                                       *
            * What is the nearest wetland, for saturation?             *
            * What is the estimated distance from cell to the water?   *
            * Clough, December 2005                                    *
            ************************************************************
            """

            nonlocal near_salt, water_table, wetland_type, inund_context, each_row, each_col, adj_cell

            def adj(row, col, idx):
                nonlocal near_salt, wetland_type, water_table, distance_to_water, inund_context, prev_wetland, wetland_width

#                def find_nearest_wetland():
                # Take elevation of nearest 500 m width wetland for water table estimation
                min_width_wetland = 500  # m

                if wetland_type == BLANK:
                    c2 = self.ret_a(row, col)
                    cat = get_cell_cat(c2)

                    if self.categories.get_cat(cat).is_non_tidal_wetland:  # Subject to soil saturation spreading
                        if cat == prev_wetland:
                            wetland_width += self.site.site_scale
                        else:
                            wetland_width = 0

                        prev_wetland = cat

                        if wetland_width > min_width_wetland:
                            if cat_elev(c2, cat) < 10:  # Avoid perched water tables
                                wetland_type = cat
                                water_table = cat_elev(c2, cat)
                # end find nearest wetland

                if not inund_context.near_water:
                    inund_context.near_water = self.get_bit(MOSTLY_WATER, row, col)
                    # Distance to open water cells
                    if inund_context.near_water:
                        inund_context.distance_to_open_salt_water = idx * self.site.site_scale
                        # distance from cell to open salt water, (Cell-width precision)

                # Distance to salt water
                if not near_salt:
                    near_salt = self.get_bit(SALT_WATER, row, col)
                    if near_salt:
                        distance_to_water = idx * self.site.site_scale
                        # distance from cell to salt water, (Cell-width precision)

                return (wetland_type != BLANK) and inund_context.near_water and near_salt  # if true, no need to continue
                #  end of adj

            # Variables
            prev_wetland = None
            wetland_width = 0

            # Initialize variables
            water_table = 0
            wetland_type = BLANK
            if not self.is_dry_land(t_cell):
                wetland_type = -5  # avoid unnecessary calculations of water table above
            inund_context.near_water = False
            near_salt = False

            j = round(6000.0 / self.site.site_scale)
            adj_row = each_row
            adj_col = each_col
            adj(adj_row, adj_col, 0)  # test cell itself

            for i in range(1, j + 1):  # test 6km adjacent
                valid_cell, adj_row, adj_col = adj_cell(adj_row, adj_col, False)
                if not valid_cell:
                    break
                if adj(adj_row, adj_col, i):
                    break

        def test_lee():
            """
            * is shoreline to lee? *
            """
            nonlocal each_row, each_col, adj_cell, inund_context

            def adj(row, col):
                nonlocal inund_context

                if not inund_context.near_water:
                    inund_context.near_water = self.get_bit(MOSTLY_WATER, row, col)
                    return False
                else:
                    return True  #no need to keep checking

            # Variables
            j = round(6000.0 / self.site.site_scale)
            adj_row = each_row
            adj_col = each_col
            adj(adj_row, adj_col)  # test cell itself

            for _ in range(1, j + 1):  # test 6km adjacent to lee
                valid_cell, adj_row, adj_col = adj_cell(adj_row, adj_col, True)
                if not valid_cell:
                    break
                if adj(adj_row, adj_col):
                    break

        # ---------------------------------------
        # ---- Transfer Main Procedure Logic ----
        # ---------------------------------------

        for cc in range(self.categories.n_cats):
            frac_lost[cc] = -9999  # Ensure correct initialization

        eroding = erode_step
        inund_context = TInundContext()  # Initialize the inundation context
        inund_context.cell_fw_influenced = self.fw_influenced(each_row, each_col)
        subsite = self.site.get_subsite(each_col, each_row, cell)  # Set subsite based on cell being evaluated

        if cell_width(cell, BLANK) == 0.0:
            test_adjacent(cell, each_row, each_col)
            fetch_threshold()

            # diked = self.include_dikes and (      # variable not used
            #         (self.classic_dike and cell['prot_dikes']) or
            #         (not self.classic_dike and cell['prot_dikes'] and not cell['elev_dikes'])
            # )

            if ((cell_width(cell, self.categories.open_ocean)+cell_width(cell, self.categories.estuarine_water)) / self.site.site_scale) < 1.0:
                connected_to_water = True  # Assume connected if not checked
                if self.check_connectivity:
                    connected_to_water = self.connect_arr[(self.site.cols * each_row) + each_col] in [3, 8, 10, 30]

                test_near(cell)
                if not inund_context.near_water:
                    test_lee()

                if connected_to_water:
                    if eroding:
                        erode(cell)
                    else:
                        inundate(cell)

                if eroding:
                    if (inund_context.cell_fw_influenced and cell['sal'][NUM_SAL_METRICS-1] > -1) or (
                            self.sal_file_name.strip() and cell['sal'][0] > -1):
                        convert_cell_by_salinity(cell)

                if self.optimize_level > 1:
                    if self.shared_data.data_elev[each_row, each_col] > round(subsite.sl_rise * 0.001):
                        self.shared_data.data_elev[each_row, each_col] -= round(subsite.sl_rise * 0.001)
                    else:
                        self.shared_data.data_elev[each_row, each_col] = 0

        if eroding:
            for i in range(self.site.n_output_sites + 1):
                if ((i == 0) and (self.in_area_to_save(each_col, each_row, not self.save_ros_area))) or \
                        ((i > 0) and (self.site.in_out_site(each_col, each_row, i))):
                    for cc in range(self.categories.n_cats):
                        sum_data.cat_sums[i][cc] += cell_width(cell, cc) * self.site.site_scale

            if self.shared_data.ros_array is not None:
                for i in range(1, self.site.max_ros + 1):
                    if i == self.shared_data.ros_array[each_row, each_col]:
                        for cc in range(self.categories.n_cats):
                            sum_data.cat_sums[i + self.site.n_output_sites][cc] += \
                                cell_width(cell, cc) * self.site.site_scale

    def zero_sums(self):
        for i in range(self.site.n_output_sites + self.site.max_ros + 1):  # global site is +1
            for j in range(MAX_ROAD_ARRAY):
                self.road_sums[i][j] = 0

            for cc in range(self.categories.n_cats):
                self.cat_sums[i][cc] = 0

    def summarize_file_info(self, run_record_file, desc, fn):
        def rel_to_abs(rel_path, base_path):
            return os.path.abspath(os.path.join(base_path, rel_path))

        def get_file_mod_date(filename):
            return datetime.fromtimestamp(os.path.getmtime(filename))

        if not os.path.exists(fn):
            return

        f_date_str = get_file_mod_date(fn).strftime('%Y-%m-%d %H:%M:%S')
        fn2 = fn
        if ':' not in fn2:
            fn2 = rel_to_abs(fn, os.getcwd())

        run_record_file.write(f'"{desc}:" {fn2}     ({f_date_str})\n')

    def write_run_record_file(self):
        if self.run_uncertainty or self.run_sensitivity:
            return  # separate logs exist for uncertainty & sensitivity runs.

        #        try:
        self.run_time = datetime.now().strftime('%Y-%m-%d_(%H.%M.%S)')
        self.run_record_file_name = os.path.splitext(self.file_name)[0]
        self.run_record_file_name = f"{self.run_record_file_name}_{self.run_time}.txt"

        with open(self.run_record_file_name, 'w') as run_record_file:
            run_record_file.write(f'SLAMM {VERS_STR} Run at {datetime.now()}\n')
            run_record_file.write(f'SLAMM File Version {VERSION_NUM} Build {BUILD_STR}\n')
            self.summarize_file_info(run_record_file, 'SLAMM6 File Name', self.file_name)

            run_record_file.write('\n')
            run_record_file.write('EXTERNAL DATA SUMMARY\n')
            self.summarize_file_info(run_record_file, 'DEM File', self.elev_file_name)
            self.summarize_file_info(run_record_file, 'SLAMM Categories', self.nwi_file_name)
            self.summarize_file_info(run_record_file, 'SLOPE File', self.slp_file_name)
            self.summarize_file_info(run_record_file, 'DIKE File', self.dik_file_name)
            self.summarize_file_info(run_record_file, 'Pct. Impervious', self.imp_file_name)
            self.summarize_file_info(run_record_file, 'Raster Output Sites', self.ros_file_name)
            self.summarize_file_info(run_record_file, 'VDATUM', self.vd_file_name)
            self.summarize_file_info(run_record_file, 'Uplift/Subsidence', self.uplift_file_name)
            self.summarize_file_info(run_record_file, 'Salinity', self.sal_file_name)
            self.summarize_file_info(run_record_file, 'Storm Surge', self.storm_file_name)
            self.summarize_file_info(run_record_file, 'Distance to Mouth', self.d2m_file_name)

            run_record_file.write('\n')

            if self.complete_run_rec:
                run_record_file.write(
                    '---------------- ALL SLAMM PARAMETERS SHOWN BELOW.  CLIP BELOW THIS LINE TO READ INTO SLAMM AS TXT FILE  ---------------- \n')
                self.load_store(run_record_file, self.file_name, VERSION_NUM, False)
                run_record_file.write('\n')

            run_record_file.write('---------------- END OF PRE-MODEL-RUN LOG FILE  ---------------- \n')
            run_record_file.close()

        # except Exception as e:
        #     print(f'Error writing Run-Record File {self.run_record_file_name}')
        #     print(str(e))
        #     run_record_file.close()

    def execute_run(self):
        save_stream = bytes

        def init_subsite(pss: TSubSite):
            pss.norm = 0
            pss.newsl = 0.0
            pss.t0_slr = 0.0
            pss.old_sl = 0.0
            pss.sl_rise = 0.0
            pss.delta_t = 0

        def init_year_slam(prot_scen):
            nonlocal self
            result = True
            year = 0

            self.year = 0  # Output as Init Cond
            if self.save_gis:
                if self.save_binary_gis:
                    file_format = FileFormat.GEOTIFF
                else:
                    file_format = FileFormat.ASCII
                if self.output_year(year):
                    result = self.save_gis_files(file_format)
            if not result:
                return False  # user cancel button

            self.year = self.site.t0()  # latest NWI Photo Date for all Subsites

            self.cat_sums = np.zeros((self.site.n_output_sites + 1 + self.site.max_ros, MAX_CATS))
            self.road_sums = np.zeros((self.site.n_output_sites + 1 + self.site.max_ros, MAX_CATS))

            self.summary = []
            if self.run_specific_years:
                self.num_rows = 3 + self.years_string.count(',')
                # num_rows = 3 + number of commas in comma delimited string
            else:
                self.num_rows = (4 + ((self.max_year - 2000) // self.time_step))

            self.summary = np.zeros(  # Initialize the 3D NumPy array with zeros
                (self.site.n_output_sites + 1 + self.site.max_ros, self.num_rows, OUTPUT_MAXC + 1),
                dtype=float
            )

            self.row_label = ['' for _ in range((self.site.n_output_sites + 1 + self.site.max_ros) * self.num_rows)]

            self.cell_ha = self.site.site_scale * self.site.site_scale * 0.0001

            self.protect_developed = (prot_scen != ProtectScenario.NoProtect)
            self.protect_all = (prot_scen == ProtectScenario.ProtAll)

            init_subsite(self.site.global_site)
            for subsite in self.site.subsites:
                init_subsite(subsite)

            self.tstep_iter = 0

            for i in range(self.site.n_output_sites + self.site.max_ros + 1):
                for cc in range(self.categories.n_cats):
                    self.cat_sums[i][cc] = 0

            self.col_label = [''] * OUTPUT_MAXC
            self.col_label[0] = 'SLR (eustatic)'
            for cc in range(1, self.categories.n_cats + 1):
                self.col_label[cc] = self.categories.get_cat(cc - 1).text_name

            self.col_label[self.categories.n_cats + 1] = 'SAV (sq.km)'
            self.col_label[self.categories.n_cats + 2] = 'Aggregated Non Tidal'
            self.col_label[self.categories.n_cats + 3] = 'Freshwater Non-Tidal'
            self.col_label[self.categories.n_cats + 4] = 'Open Water'
            self.col_label[self.categories.n_cats + 5] = 'Low Tidal'
            self.col_label[self.categories.n_cats + 6] = 'Saltmarsh'
            self.col_label[self.categories.n_cats + 7] = 'Transitional'
            self.col_label[self.categories.n_cats + 8] = 'Freshwater Tidal'
            self.col_label[self.categories.n_cats + 9] = 'GHG (10^3 Kg)'

            if self.n_road_inf > 0:
                self.col_label[self.categories.n_cats + 10] = 'total road length (km)'
                self.col_label[self.categories.n_cats + 11] = 'inundated roads elev<H1 (km)'
                self.col_label[self.categories.n_cats + 12] = 'inundated roads elev<H2 (km)'
                self.col_label[self.categories.n_cats + 13] = 'inundated roads elev<H3 (km)'
                self.col_label[self.categories.n_cats + 14] = 'inundated roads elev<H4 (km)'
                self.col_label[self.categories.n_cats + 15] = 'inundated roads elev<H5 (km)'
                self.col_label[self.categories.n_cats + 16] = 'inundated roads elev>H5 (km)'
                self.col_label[self.categories.n_cats + 17] = 'km roads below MTL and connected'
                self.col_label[self.categories.n_cats + 18] = 'km roads open water'
                self.col_label[self.categories.n_cats + 19] = 'km roads blank/diked'
                self.col_label[self.categories.n_cats + 20] = 'km roads irrelevant'

            # result = self.prog_form.update2gages(0, 0)
            # if not result:
            #     return False
            # 
            # self.prog_form.year_label.visible = True
            # self.prog_form.year_label.caption = str(self.year)
            # self.prog_form.show()
            # 
            # self.grid_form.image1.canvas.pen.mode = 'copy'

            for er in range(self.site.rows):
                for ec in range(self.site.cols):
                    proc_cell = self.ret_a(er, ec)
                    for i in range(self.site.n_output_sites + 1):
                        if (i == 0 and self.in_area_to_save(ec, er, not self.save_ros_area)) or (
                                i > 0 and self.site.in_out_site(ec, er, i)):
                            for cc in range(self.categories.n_cats - 1):
                                self.cat_sums[i][cc] += cell_width(proc_cell, cc) * self.site.site_scale

                    if self.shared_data.ros_array is not None:
                        ros_array = self.shared_data.ros_array
                        for i in range(1, self.site.max_ros + 1):
                            if i == ros_array[er, ec]:
                                for cc in range(self.categories.n_cats - 1):
                                    self.cat_sums[i + self.site.n_output_sites][cc] += cell_width(proc_cell,
                                                                                                  cc) * self.site.site_scale

                    # if (ec % 400) == 0:
                    #     self.prog_form.progress_label.caption = 'Collecting Init. Cond Data'
                    #     self.prog_form.progress_label.visible = True
                    #     self.prog_form.halt_button.visible = True
                    #     result = self.prog_form.update2gages(int((er / self.site.rows) * 100), 0)
                    #     if not result:
                    #         return False

            if self.n_road_inf > 0:
                for j in range(self.n_road_inf):
                    self.roads_inf[j].update_road_sums(self.tstep_iter, self.road_sums,
                                                       self.site.n_output_sites + self.site.max_ros + 1)

            # self.prog_form.progress_label.caption = 'Running First Year'

            self.summarize(0)
            self.tstep_iter += 1
            self.zero_sums()

            result = self.run_one_year(True)
            if not result:
                return False

            if self.save_gis:
                if self.save_binary_gis:
                    file_format = FileFormat.GEOTIFF
                else:
                    file_format = FileFormat.ASCII
                if self.output_year(self.year):
                    result = self.save_gis_files(file_format)

            first_step_time = round((time.time()-self.start_time)/60-self.memory_load_time, 1)
            steps_run = self.n_time_steps - 1  # exclude initial condition data slot
            print(f'          Loading data to memory and initialization took {self.memory_load_time} minutes')
            print(f'          First step took {first_step_time} minutes')
            estimated_time = round(first_step_time * steps_run, 1) + self.memory_load_time
            print(f'          This simulation with {steps_run} steps is estimated to take a total of {estimated_time} minutes')
            remaining_time = round(estimated_time - self.memory_load_time - first_step_time, 1)
            print(f'          (an additional {remaining_time} minutes)')

            return result

        def check_valid_slamm():
            result = self.validate_output_sites()
            if not result:
                return False

            for j in range(self.num_fw_flows):
                no_poly = False
                if self.fw_flows[j].poly is None:
                    no_poly = True
                elif self.fw_flows[j].poly.num_pts < 3:
                    no_poly = True

                if no_poly:
                    print('Error, Fresh water flow "' + self.fw_flows[j].name + '" polygon is not defined.')
                    return False

                if not self.fw_flows[j].extent_only and self.fw_flows[j].num_segments == 0:
                    print('Error, Fresh water flow "' + self.fw_flows[j].name + '" origin, mouth is not defined.')
                    return False

            return True

        def save_subsites():
            nonlocal save_stream
            save_stream = pickle.dumps(self.site)

        def restore_subsites():
            nonlocal save_stream
            self.site = pickle.loads(save_stream)
            save_stream = None

        # -----------------------------------------------------------------------------------------------------
        def execute_slamm(prot_scen, ipcc_scen, ipcc_estm, fix_num, tsslri):

            self.start_time = time.time()

            save_subsites()
            self.dike_log_init = False
            self.scen_iter += 1

            result = check_valid_slamm()
            if not result:
                self.user_stop = True
                restore_subsites()
                return False

            if self.sal_rules.n_rules > 0:  # 8/15/22 JSC Copy SalRules to Category as required
                for nc in range(self.categories.n_cats):
                    self.categories.cats[nc].salinity_rules = None
                    self.categories.cats[nc].has_sal_rules = False

                for nr in range(self.sal_rules.n_rules):
                    catg = self.categories.cats[self.sal_rules.rules[nr].from_cat]
                    if catg.salinity_rules is None:
                        catg.salinity_rules = TSalinityRules()
                    if catg.salinity_rules.rules is None:
                        catg.salinity_rules.rules = []

                    catg.has_sal_rules = True
                    catg.salinity_rules.rules.append(self.sal_rules.rules[nr])
                    catg.salinity_rules.n_rules += 1
            try:
                result = self.make_data_file(False, '', '')
            except ValueError as e:
                if self.optimize_level > 0:
                    self.num_mm_entries = 0
                    self.shared_data = None
                    print ('Error loading map to memory.  Optimization size incorrect.  Recounting map size.')
                    result = self.make_data_file(False, '', '')
                else:
                    raise

            if not result:
                self.user_stop = True
                restore_subsites()
                return False

            self.site.init_elev_vars()

            result = self.preprocess_wetlands()
            if not result:
                self.user_stop = True
                restore_subsites()
                return False

            self.year = self.site.global_site.nwi_photo_date
            init_subsite(self.site.global_site)
            for subsite in self.site.subsites:
                init_subsite(subsite)

            if self.shared_data.b_matrix is None or self.shared_data.b_matrix.size < self.site.rows * self.site.cols:
                self.shared_data.b_matrix = SharedMemoryArray((self.site.rows, self.site.cols), dtype=np.uint8)

            if self.shared_data.erode_matrix is None or self.shared_data.erode_matrix.size < self.site.rows * self.site.cols:
                self.shared_data.erode_matrix = SharedMemoryArray((self.site.rows, self.site.cols), dtype=np.uint16)

            if self.sal_file_name.strip() == '' or '.xls' in self.sal_file_name.lower():
                result = self.calc_salinity(True, True)
            else:
                result = True

            if not result:
                self.user_stop = True
                restore_subsites()
                return False

            self.inund_freq_check = False
            if self.inund_maps or (self.save_gis and self.save_inund_gis) or (self.n_road_inf > 0) or (
                    self.n_point_inf > 0):
                result = self.calc_inund_freq()
                self.inund_freq_check = result

            if not result:
                self.user_stop = True
                restore_subsites()
                return False

            self.connect_check = False
            if self.check_connectivity or self.connect_maps:
                self.connect_arr = self.calc_inund_connectivity(self.connect_arr, True, -1)
                self.connect_check = True

            self.sav_km = -9999
            self.sav_prob_check = False
            if (self.d2m_file_name.strip()) != '':
                result = self.calculate_prob_sav()
                self.sav_prob_check = result

            if not result:
                self.user_stop = True
                restore_subsites()
                return False

            self.road_inund_check = False
            if self.n_road_inf > 0:
                for j in range(self.n_road_inf):
                    self.road_inund_check = self.roads_inf[j].calc_all_roads_inundation()
                result = self.road_inund_check

            if not result:
                self.user_stop = True
                restore_subsites()
                return False

            if self.n_point_inf > 0:
                for j in range(self.n_point_inf):
                    result = self.point_inf[j].calc_all_point_inf_inundation()

            if not result:
                self.user_stop = True
                restore_subsites()
                return False

            # self.prog_form.year_label.visible = False
            # self.prog_form.slr_label.visible = True
            # self.prog_form.protection_label.visible = True

            self.running_fixed = (fix_num >= 0) and (fix_num < 11)
            self.running_custom = (fix_num >= 11)
            self.tsslr_index = tsslri
            self.running_tsslr = tsslri > 0
            self.fixed_scen = fix_num
            self.ipcc_sl_rate = ipcc_scen
            self.ipcc_sl_est = ipcc_estm

            self.memory_load_time = round((time.time()-self.start_time)/60,1)

            # if self.running_fixed:
            #     self.prog_form.slr_label.caption = self.label_fixed[fix_num]
            # elif self.running_custom:
            #     self.prog_form.slr_label.caption = f"{self.current_custom_slr:.2f} meters"
            # elif self.running_tsslr:
            #     self.prog_form.slr_label.caption = self.time_ser_slrs[tsslri - 1].name
            # else:
            #     self.prog_form.slr_label.caption = f"{self.label_ipcc[ipcc_scen]} {self.label_ipcc_est[ipcc_est]}"
            #
            # self.prog_form.protection_label.caption = self.label_protect[prot_scen]
            # self.prog_form.show()

            result = init_year_slam(prot_scen)
            if not result:
                self.user_stop = True
                restore_subsites()
                return False

            if self.year < self.max_year and not (
                    self.run_specific_years and (str(self.year) == self.years_string.strip())):
                while self.year < self.max_year:
                    self.tstep_iter += 1
                    self.zero_sums()
                    result = self.run_one_year(False)
                    if not result:
                        self.user_stop = True
                        restore_subsites()
                        return False

                    if self.save_gis:
                        if self.output_year(self.year):
                            if self.save_binary_gis:
                                file_format = FileFormat.GEOTIFF
                            else:
                                file_format = FileFormat.ASCII
                            result = self.save_gis_files(file_format)

                    if not result:
                        self.user_stop = True
                        restore_subsites()
                        return False

            self.save_csv_file()

            for j in range(self.n_road_inf):
                self.roads_inf[j].write_road_dbf(self.scen_iter == 1, self.tstep_iter + 1)
            for j in range(self.n_point_inf):
                self.point_inf[j].write_point_dbf(self.scen_iter == 1, self.tstep_iter + 1)

            # if self.dike_log_init:
            #     self.dike_log.dispose

            restore_subsites()
            return result

        # -----------------------------------------------------------------------------------------------------
        # - execute_run starts here ---------------------------------------------------------------------------

        self.user_stop = True
        self.scen_iter = 0
        self.write_run_record_file()

        # ndbfs = self.n_road_inf + self.n_point_inf
        for i in range(self.n_road_inf):
            if self.roads_inf[i].check_valid():
                self.roads_inf[i].create_output_dbf()
            else:
                return

        #   self.prog_form.update2gages(int(((i + 1) / ndbfs) * 100), 0)

        for i in range(self.n_point_inf):
            if self.point_inf[i].check_valid():
                self.point_inf[i].create_output_dbf()
            else:
                return

        # self.prog_form.update2gages(int(((i + 1 + self.n_road_inf) / ndbfs) * 100), 0)

        self.user_stop = False
        model_ran = False
        # if not self.run_uncertainty and not self.run_sensitivity:
        #     self.word_initialized = False

        for ipcc_scenario in IPCCScenarios:
            for ipcc_est in IPCCEstimates:
                for prot_scenario in ProtectScenario:
                    if self.ipcc_scenarios[ipcc_scenario] and self.prot_to_run[prot_scenario] and self.ipcc_estimates[ipcc_est]:
                        print('Executing SLAMM ' + LabelIPCC[ipcc_scenario] + " " + LabelIPCCEst[ipcc_est] +
                              " " + LabelProtect[prot_scenario])
                        model_ran = True
                        if execute_slamm(prot_scenario, ipcc_scenario, ipcc_est, -1, 0) is False:
                            return

        self.running_custom = False
        ipcc_est = IPCCEstimates.Est_None
        ipcc_scenario = IPCCScenarios.Scen_None
        for prot_scenario in ProtectScenario:
            for fix_loop in range(11):
                if self.fixed_scenarios[fix_loop] and self.prot_to_run[prot_scenario]:
                    print('Executing SLAMM ' + LabelFixed[fix_loop] + " " + LabelProtect[prot_scenario])
                    model_ran = True
                    if execute_slamm(prot_scenario, ipcc_scenario, ipcc_est, fix_loop, 0) is False:
                        return


        if self.run_custom_slr:
            for i in range(self.n_custom_slr):
                self.current_custom_slr = self.custom_slr_array[i]
                self.running_custom = True
                for prot_scenario in ProtectScenario:
                    if self.prot_to_run[prot_scenario]:
                        model_ran = True
                        print('Executing SLAMM ' + f"{self.current_custom_slr:.2f} meters" + " " + LabelProtect[prot_scenario])
                        if execute_slamm(prot_scenario, ipcc_scenario, ipcc_est, 9999, 0) is False:
                            return

        self.running_custom = False
        for i in range(self.n_time_ser_slr):
            if self.time_ser_slrs[i].run_now:
                for prot_scenario in ProtectScenario:
                    if self.prot_to_run[prot_scenario]:
                        model_ran = True
                        print('Executing SLAMM ' + self.time_ser_slrs[i].name + " " + LabelProtect[
                            prot_scenario])
                        if execute_slamm(prot_scenario, ipcc_scenario, ipcc_est, 0, i + 1) is False:
                            return

        if not self.run_uncertainty and not self.run_sensitivity:
            if model_ran:
                print('Completed Simulations')
            else:
                print('You must select at least one SLR scenario and one protection scenario to run')

    def output_year(self, year):
        if self.gis_each_year:
            return True
        gis_year_list = list(map(int, self.gis_years.split(',')))
        return year in gis_year_list

    def calculate_euc_distances(self):
        tolerance = 0.03  # Close enough to MLLW or MHHW to be considered part of the contour
        mllw_arr = []
        mhhw_arr = []
        n_mllw = 0
        n_mhhw = 0

        def ready_for_exit():
            nonlocal mllw_arr, mhhw_arr
            mllw_arr = None
            mhhw_arr = None
            # ProgForm.Hide()

        def add_to_array(arr, counter, er, ec):
            counter += 1
            if counter > len(arr):
                arr.extend([None] * 2000)
            arr[counter - 1] = (ec, er)
            return counter

        def distance_2pts_m(p1, p2):
            return (p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2

        # def distance_to(arr, counter, this_pt):
        #     result2 = 10000000000
        #     result = math.sqrt(result2) * self.site.site_scale
        #     max_dist = 100000
        #
        #     for i in range(counter):
        #         if (
        #                 (this_pt[0] - max_dist) < arr[i][0] < (this_pt[0] + max_dist)
        #                 and (this_pt[1] - max_dist) < arr[i][1] < (this_pt[1] + max_dist)
        #         ):
        #             test_dist = distance_2pts_m(this_pt, arr[i])
        #             if test_dist < result2:
        #                 result2 = test_dist
        #                 result = math.sqrt(test_dist) * self.site.site_scale
        #                 max_dist = round(math.sqrt(result2)) + 1
        #     return result

        def distance_to2(arr, counter, this_pt, p2):
            #  Calculate the minum distance to some line using. Compared to DistanceTo this function
            #  uses the previous cell estimation as initial minimum distance

            result2 = distance_2pts_m(this_pt, p2)
            result = math.sqrt(result2) * self.site.site_scale
            max_dist = round(math.sqrt(result2)) + 1

            for i in range(counter):
                if (
                        (this_pt[0] - max_dist) < arr[i][0] < (this_pt[0] + max_dist)
                        and (this_pt[1] - max_dist) < arr[i][1] < (this_pt[1] + max_dist)
                ):
                    test_dist = distance_2pts_m(this_pt, arr[i])
                    if test_dist < result2:
                        result2 = test_dist
                        result = math.sqrt(test_dist) * self.site.site_scale
                        max_dist = round(math.sqrt(result2)) + 1
                        p2 = arr[i]
            return result, p2

        def thread_calc_euc_distances(start_row, end_row):
            nonlocal mllw_arr, mhhw_arr
            result = True
            p_mllw = (self.site.rows - 1, self.site.cols - 1)
            p_mhhw = (self.site.rows - 1, self.site.cols - 1)

            print("     Calculating Euclidean Distances (SAV module)")

            for er in range(start_row, end_row + 1):
                for ec in range(self.site.cols):
                    # Get cell
                    cl = self.ret_a(er, ec)

                    # Get cell category
                    cat = get_cell_cat(cl)

                    # Get cell elevation
                    cell_elev = get_min_elev(cl)

                    # Initialize distances
                    cl['d2mllw'] = -9999
                    cl['d2mhhw'] = -9999

                    # Calculate the minimum distance to the water lines
                    if (
                            cell_elev < 8
                            and self.categories.get_cat(cat).agg_cat in [AggCategories.SaltMarsh, AggCategories.LowTidal, AggCategories.OpenWater]
                            and -1.9 < cell_elev < 0.5
                    ):
                        this_pt = (ec, er)
                        cl['d2mllw'], p_mllw = distance_to2(mllw_arr, n_mllw, this_pt, p_mllw)
                        cl['d2mhhw'], p_mhhw = distance_to2(mhhw_arr, n_mhhw, this_pt, p_mhhw)

                    # Reset the cell
                    self.set_a(er, ec, cl)
            return result

        # result = True
        # ProgForm.ProgressLabel.Caption = 'Locating MLLW, MHHW'
        # ProgForm.Show()

        print("     Locating MLLW, MHHW (SAV module)")

        mllw_arr = [None] * 1000
        mhhw_arr = [None] * 1000
        n_mllw = 0
        n_mhhw = 0

        # Identify the cells for MHHW and MLLW
        for er in range(self.site.rows):
            for ec in range(self.site.cols):
                # ProgForm.Update2Gages(int(er / self.site.rows * 100), 0)
                # Get cell
                cl = self.ret_a(er, ec)

                # Get cell elevation
                cell_elev = get_min_elev(cl)

                # Get subsite
                subsite = self.site.get_subsite(ec, er, cl)

                if abs(cell_elev - subsite.mhhw) < tolerance:
                    n_mhhw = add_to_array(mhhw_arr, n_mhhw, er, ec)
                if abs(cell_elev - subsite.mllw) < tolerance:
                    n_mllw = add_to_array(mllw_arr, n_mllw, er, ec)

        result = thread_calc_euc_distances(0, self.site.rows - 1)

        ready_for_exit()

        return result

    def calculate_prob_sav(self) -> bool:
        result = self.calculate_euc_distances()
        if not result:
            return False

        print('     Calculating Probability of SAV')
        ev = 0.0

        for er in range(self.site.rows):
            for ec in range(self.site.cols):
                cl = self.ret_a(er, ec)
                # subsite = self.site.get_subsite(ec, er, cl)

                dem = get_min_elev(cl) + cl['mtl_minus_navd']  # DEM expressed as NAVD88

                if cl['d2mllw'] < -9998 or cl['d2mhhw'] < -9998 or dem > 9.0 or cl['d2mouth'] < 0:
                    cl['prob_sav'] = 0
                else:
                    try:
                        logit = (self.sav_params.intcpt + (dem * self.sav_params.c_dem) +
                                 (dem ** 2 * self.sav_params.c_dem2) +
                                 (dem ** 3 * self.sav_params.c_dem3) +
                                 (cl['d2mllw'] * self.sav_params.c_d2mllw) +
                                 (cl['d2mhhw'] * self.sav_params.c_d2mhhw) +
                                 (cl['d2mouth'] * self.sav_params.c_d2m) +
                                 (cl['d2mouth'] ** 2 * self.sav_params.c_d2m2))

                        if logit < -100:
                            cl['prob_sav'] = 0
                        else:
                            cl['prob_sav'] = 1 / (1 + math.exp(-logit))
                    except:
                        cl['prob_sav'] = 0

                    ev += (self.site.site_scale ** 2) * cl['prob_sav'] * 1e-6

                self.set_a(er, ec, cl)

        self.sav_km = ev
        return True

    def fresh_water_height(self, this_cell: compressed_cell_dtype, fr: int, fc: int, tide_int: int) -> tuple:
        cell_elev = get_min_elev(this_cell)

        result = 0.0
        fw_sal = 0.0
        sw_sal = 30.0
        seg_salin = 0.0
        salh = -99.0

        if self.classic_dike and this_cell['prot_dikes']:
            return result, fw_sal, sw_sal, salh, seg_salin
        if not self.classic_dike and this_cell['prot_dikes'] and not this_cell['elev_dikes']:
            return result, fw_sal, sw_sal, salh, seg_salin

        for fw_num in range(self.num_fw_flows):
            if self.fw_influenced_specific(fr, fc, fw_num):
                fw_flow = self.fw_flows[fw_num]
                if not fw_flow.extent_only:
                    rkm, d2c = self.river_km(fr, fc, fw_num)
                    rseg = int(rkm / SLICE_INCREMENT)

                    seg_salin = fw_flow.xs_salinity[tide_int][rseg]

                    if tide_int == 0:
                        sal_z = fw_flow.salt_height(fw_flow.ocean_subsite.mllw, fw_flow.ocean_subsite.newsl, rseg,
                                                    self.year)
                    elif tide_int == 1:
                        sal_z = fw_flow.salt_height(0.0, fw_flow.ocean_subsite.newsl, rseg, self.year)
                    elif tide_int == 2:
                        sal_z = fw_flow.salt_height(fw_flow.ocean_subsite.mhhw, fw_flow.ocean_subsite.newsl, rseg,
                                                    self.year)
                    else:
                        sal_z = fw_flow.salt_height(fw_flow.ocean_subsite.salt_elev, fw_flow.ocean_subsite.newsl, rseg,
                                                    self.year)

                    if sal_z < cell_elev:
                        sal_z = cell_elev

                    if tide_int == 0:
                        this_cell['sal_height_mllw'] = sal_z

                    if sal_z - cell_elev > salh:
                        salh = sal_z - cell_elev

                    fresh_height = fw_flow.water_z[tide_int][rseg] - sal_z

                    if fresh_height > result:
                        fw_sal = ((fw_sal * result) + (fw_flow.fw_ppt * fresh_height)) / (result + fresh_height)
                        sw_sal = ((sw_sal * result) + (fw_flow.sw_ppt * fresh_height)) / (result + fresh_height)

                        result = fresh_height

        return result, fw_sal, sw_sal, salh, seg_salin

    def salinity_from_excel(self):
        print('Interpolating Salinity:' + str(self.year))

        if self.time_zero:
            if self.sal_array is not None:
                self.sal_array.__del__()

        if self.time_zero or self.sal_array is None:
            self.sal_array = TSalArray(self.sal_file_name, self.site)
            self.sal_array.read_sal_array_from_excel(self.sal_file_name, self.site)

        # Get the raster cell of the point closest to the river mouth for calculating the RSLR
        er = int(self.sal_array.s_locs[self.sal_array.n_sal_stations - 1].y)
        ec = int(self.sal_array.s_locs[self.sal_array.n_sal_stations - 1].x)
        sal_cell = self.ret_a(er, ec)

        # Get the subsite
        subsite = self.site.get_subsite(ec, er, sal_cell)

        # Calculation of the RSLR
        slamm_rslr = (subsite.norm * 0.01) - (sal_cell['uplift'] * (self.year - self.site.t0()) * 0.01)
        # {m}             {cm}    {m/cm}                                        {cm/y}       {y}    {m/cm}

        # Extract corresponding salinity values
        slamm_q = self.fw_flows[0].flow_by_year(
            self.year)  # Marco: For now calculating flow from the FwFlows with index=0
        sal_val = self.sal_array.extract_salinity_record(slamm_rslr, slamm_q)

        for er in range(self.site.rows):
            for ec in range(self.site.cols):
                if self.fw_influenced_specific(er, ec, 0):  # Marco: For now Excel salinity only calculated in the FwFlow with index=0
                    sal_cell = self.ret_a(er, ec)

                    # Get cell category
                    cc = get_cell_cat(sal_cell)

                    # Adjustment of the slope
                    slope_adjustment = (self.site.site_scale * 0.5) * sal_cell['tan_slope']

                    # Get minimum elevation
                    ce = cat_elev(sal_cell, cc)

                    # If elevation is 999 (no data) do not process
                    if ce == 999:
                        continue

                    # Then elevation in m is ...
                    if cc == BLANK:
                        elev = 0
                    else:
                        elev = ce + slope_adjustment

                    # Get the cell subsite
                    subsite = self.site.get_subsite(ec, er, sal_cell)

                    # IF elevation <= Salt Boundary then calculate salinity otherwise skip
                    if elev <= subsite.salt_elev:
                        sal_cell['sal'][0] = self.sal_array.get_salinity(ec, er, sal_val)
                        self.set_a(er, ec, sal_cell)

                    # result = prog_form.update_2_gages(int((er / self.site.rows) * 100), 0)

        # if self.time_zero:
        #     result = self.aggr_salinity_stats()

    def calc_salinity(self, run_chng_water: bool, time_zero: bool) -> bool:
        sw_sal, fw_sal, salh = 0.0, 0.0, 0.0

        def calc_salin(sw_height: float, fw_height: float) -> float:
            nonlocal sw_sal, fw_sal, sal_cell
            if sal_cell['prot_dikes'] and (self.classic_dike or not sal_cell['elev_dikes']) :
                return -999
            elif sw_height <= 0 and fw_height <= 0:
                if get_cell_cat(sal_cell) == self.categories.open_ocean:
                    return sw_sal
                else:
                    return -999
            elif sw_height <= 0:
                return fw_sal
            elif fw_height <= 0:
                return sw_sal
            else:
                return ((sw_height * sw_sal) + (fw_height * fw_sal)) / (sw_height + fw_height)

        def cell_salinity():
            nonlocal sal_cell, sw_sal, fw_sal, salh
            MIXINGFACTOR = 0.25
            # cell_elev = get_min_elev(sal_cell)

            for tide_int in range(4):
                (fwh, fw_sal, sw_sal, salh, seg_salin) = self.fresh_water_height(sal_cell, er, ec, tide_int)
                salcalc = ((1 - MIXINGFACTOR) * calc_salin(salh, fwh)) + (MIXINGFACTOR * seg_salin)
                if salcalc < 0:
                    salcalc = -999
                sal_cell['sal'][tide_int] = salcalc


        # start main function calc_salinity
        if ".xls" in self.sal_file_name.lower():
            self.salinity_from_excel()
            return True

        all_extent_only = all(fw_flow.extent_only for fw_flow in self.fw_flows)
        if all_extent_only:
            return True

        if run_chng_water:
            print('     chng_water method')
            result = self.chng_water(False, True)
            if not result:
                return False

        if self.num_fw_flows > 0:
            print('Calculating Salinity:')
            for er in range(self.site.rows):
                for ec in range(self.site.cols):
                    if self.fw_influenced(er, ec):
                        sal_cell = self.ret_a(er, ec)
                        cell_salinity()
                        self.set_a(er, ec, sal_cell)

        # if time_zero:
        #     result = self.aggr_salinity_stats()
        #     if not result:
        #         return False

        return True

    def preprocess_wetlands(self):
        ONLY_PREPROC_NODATA = False  # Make it true if you want to preprocess only the cells with no data.
    
        def process_strip(from_row, from_col, to_row, to_col, sum_adjacent_wetland_cells, end_in_water):
            nonlocal current_subsite_n, cat_min_elev, cat_max_elev, category_loop, dir_offshore

            delta_row, delta_col = 0, 0
            if from_row > to_row:
                delta_row = -1
            elif from_row < to_row:
                delta_row = 1
            if from_col > to_col:
                delta_col = -1
            elif from_col < to_col:
                delta_col = 1
    
            process_row, process_col = from_row, from_col
            cell_subsite_n = self.site.get_subsite_num(from_col, from_row)
            if cell_subsite_n != current_subsite_n:
                return
    
            max_elev_prev_cl = cat_min_elev
            max_elev = cat_max_elev
            if end_in_water:
                max_elev = (cat_max_elev + cat_min_elev) / 2
    
            for _ in range(sum_adjacent_wetland_cells):
                adjacent_cell_min_elev = -9999
                adjacent_row, adjacent_col = process_row, process_col
    
                if dir_offshore in [WaveDirection.Northerly, WaveDirection.Southerly]:
                    adjacent_col -= 1
                else:  # Easterly, Westerly
                    adjacent_row -= 1
    
                if adjacent_row > 0 and adjacent_col > 0:
                    cl = self.ret_a(adjacent_row, adjacent_col)
                    if cell_width(cl, category_loop) > 0:
                        adjacent_cell_min_elev = cat_elev(cl, category_loop)
                    if adjacent_cell_min_elev > 998:
                        adjacent_cell_min_elev = -9999  # no data, must be in another subsite
    
                cl = self.ret_a(process_row, process_col)
                cl['tan_slope'] = abs((max_elev - cat_min_elev) / (sum_adjacent_wetland_cells * self.site.site_scale))
                set_cat_elev(cl, category_loop, max_elev_prev_cl)
                max_elev_prev_cl = cat_elev(cl, category_loop) + cl['tan_slope'] * self.site.site_scale

                if adjacent_cell_min_elev != -9999:
                    set_cat_elev(cl, category_loop, (cat_elev(cl, category_loop) + adjacent_cell_min_elev) * 0.5)
    
                self.set_a(process_row, process_col, cl)
                process_row += delta_row
                process_col += delta_col
    
        def process_map(from_row, from_col, to_row, to_col):
            nonlocal current_subsite_n, current_subsite, category_loop, cat_min_elev, cat_max_elev, dir_offshore

            previous_was_water, previous_was_category = True, False
            sum_adjacent_wetland_cells = 0
            first_wetland_row, first_wetland_col = -1, -1
    
            def process_cell(proc_row, proc_col, last_cell):
                nonlocal current_subsite_n, current_subsite, category_loop, cat_min_elev, cat_max_elev, first_wetland_row
                nonlocal first_wetland_col, previous_was_water, previous_was_category, sum_adjacent_wetland_cells

                cl = self.ret_a(proc_row, proc_col)
                proc_cat = get_cell_cat(cl)
    
                cell_is_wetland_category = False
                cell_is_water = False
                cell_subsite_n = self.site.get_subsite_num(proc_col, proc_row)
                cell_out_of_range = (cell_subsite_n != current_subsite_n)
    
                if self.categories.get_cat(proc_cat).is_open_water:
                    cell_is_water = True
                if proc_cat == category_loop:
                    cell_is_wetland_category = True
                if self.classic_dike and cl['prot_dikes']:
                    cell_is_wetland_category = False
                if not self.classic_dike and cl['prot_dikes'] and not cl['elev_dikes']:
                    cell_is_wetland_category = False

                if ONLY_PREPROC_NODATA and cat_elev(cl, proc_cat) < 998.5:
                    cell_is_wetland_category = False
    
                if not self.categories.get_cat(proc_cat).is_tidal and (
                    cat_elev(cl, proc_cat) > current_subsite.salt_elev
                    or (proc_cat != self.categories.und_dry_land and self.fw_influenced(proc_row, proc_col))
                ):
                    cell_is_wetland_category = False
    
                if cell_is_water and not cell_out_of_range:
                    if previous_was_category and first_wetland_row > -1:
                        process_strip(first_wetland_row, first_wetland_col, proc_row, proc_col, sum_adjacent_wetland_cells, True)
                        first_wetland_row, first_wetland_col = -1, -1
                    previous_was_water, previous_was_category = True, False
                    sum_adjacent_wetland_cells = 0
    
                if cell_is_wetland_category and not cell_out_of_range:
                    sum_adjacent_wetland_cells += 1
                    if not previous_was_category:
                        first_wetland_row, first_wetland_col = proc_row, proc_col
                    previous_was_water, previous_was_category = False, True
    
                if last_cell or cell_out_of_range or (not cell_is_wetland_category and not cell_is_water):
                    if previous_was_category and first_wetland_row > -1:
                        process_strip(first_wetland_row, first_wetland_col, proc_row, proc_col, sum_adjacent_wetland_cells, False)
                        first_wetland_row, first_wetland_col = -1, -1
                    previous_was_water, previous_was_category = False, False
                    sum_adjacent_wetland_cells = 0
    
            if from_row > to_row:
                for loop in range(from_row, to_row - 1, -1):
                    process_cell(loop, from_col, loop == to_row)
            elif to_row > from_row:
                for loop in range(from_row, to_row + 1):
                    process_cell(loop, from_col, loop == to_row)
            if from_col > to_col:
                for loop in range(from_col, to_col - 1, -1):
                    process_cell(from_row, loop, loop == to_col)
            elif to_col > from_col:
                for loop in range(from_col, to_col + 1):
                    process_cell(from_row, loop, loop == to_col)
    
        result = True

        preprocess_noted = False

        for n_subsite in range(self.site.n_subsites + 1):
            num_processed = 0
            current_subsite_n = n_subsite
            if n_subsite == 0:
                if not self.site.global_site.use_preprocessor:
                    continue
                row_min, row_max = 0, self.site.rows - 1
                col_min, col_max = 0, self.site.cols - 1
                dir_offshore = self.site.global_site.direction_offshore
                current_subsite = self.site.global_site
            else:
                subsite = self.site.subsites[n_subsite - 1]
                if not subsite.use_preprocessor:
                    continue
                row_min = max(subsite.poly.min_row(), 0)
                row_max = min(subsite.poly.max_row(), self.site.rows - 1)
                col_min = max(subsite.poly.min_col(), 0)
                col_max = min(subsite.poly.max_col(), self.site.cols - 1)
                dir_offshore = subsite.direction_offshore
                current_subsite = subsite

            if not preprocess_noted: print('Pre-Processing Wetlands Elevations')
            preprocess_noted = True

            for category_loop in range(self.categories.n_cats):
                if not self.categories.get_cat(category_loop).is_open_water:
                    cat_min_elev = self.lower_bound(category_loop, current_subsite)
                    cat_max_elev = self.upper_bound(category_loop, current_subsite)
    
                    if dir_offshore == WaveDirection.Northerly:
                        for cloop in range(col_min, col_max + 1):
                            process_map(row_min, cloop, row_max, cloop)
                    elif dir_offshore == WaveDirection.Southerly:
                        for cloop in range(col_min, col_max + 1):
                            process_map(row_max, cloop, row_min, cloop)
                    elif dir_offshore == WaveDirection.Easterly:
                        for rloop in range(row_min, row_max + 1):
                            process_map(rloop, col_max, rloop, col_min)
                    elif dir_offshore == WaveDirection.Westerly:
                        for rloop in range(row_min, row_max + 1):
                            process_map(rloop, col_min, rloop, col_max)
    
                    num_processed += 1
                    #print(f'     Processing Category: {self.categories.get_cat(category_loop).text_name}')
                    if not result:
                        return False
    
        return True

    def count_runs(self):

        count_runs = 0
        for ipcc_scenario in IPCCScenarios:
            for ipcc_est in IPCCEstimates:
                for prot_scenario in ProtectScenario:
                    if (self.ipcc_scenarios[ipcc_scenario] and self.prot_to_run[prot_scenario] and
                            self.ipcc_estimates[ipcc_est]):
                        count_runs += 1

        for prot_scenario in ProtectScenario:
            for fix_loop in range(11):
                if self.fixed_scenarios[fix_loop] and self.prot_to_run[prot_scenario]:
                    count_runs += 1

        running_custom_slr = False
        if self.run_custom_slr:
            for i in range(self.n_custom_slr):
                self.current_custom_slr = self.custom_slr_array[i]
                for prot_scenario in ProtectScenario:
                    if self.prot_to_run[prot_scenario]:
                        count_runs += 1
                        running_custom_slr = True

        return count_runs, running_custom_slr

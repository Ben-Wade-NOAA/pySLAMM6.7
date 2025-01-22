from app_global import *
from typing import List, Optional
from dataclasses import dataclass, field
import numpy as np
import shutil
from dbf import Table, DbfError, dbf
import os

""" 
This module defines classes and functions for handling infrastructure data, including roads and points, 
within the pySLAMM simulation framework. It includes data structures for representing points, lines, 
and boundaries, as well as classes for managing infrastructure data and calculating inundation frequencies.
Classes:
    DPoint: Represents a point with x and y coordinates.
    DLine: Represents a line with coefficients a, b, and c.
    DBoundary: Represents a boundary with minimum and maximum row and column indices.
    LineRec: Represents a record for a line with various attributes including row, column, shape index, 
             road class, coordinates, elevation, and inundation frequency.
    PointRec: Represents a record for a point with various attributes including row, column, shape index, 
              coordinates, elevation, and inundation frequency.
    TInfrastructure: Base class for infrastructure data management, including methods for calculating 
                     cell inundation, updating inundation frequency, checking validity, loading and storing data, 
                     and creating output DBF files.
    TRoadInfrastructure: Subclass of TInfrastructure for managing road infrastructure data, including methods 
                         for calculating road inundation, loading and saving road data, initializing road variables, 
                         and writing road data to DBF files.
    TPointInfrastructure: Subclass of TInfrastructure for managing point infrastructure data, including methods 
                          for calculating point inundation, initializing point variables, and writing point data 
                          to DBF files.
Functions:
    inund_text(i_f: int) -> str: Returns a string description of the inundation frequency based on the input integer.
"""


@dataclass
class DPoint:
    x: float
    y: float


DPointArray = List[DPoint]


@dataclass
class DLine:
    a: float
    b: float
    c: float

@dataclass
class DBoundary:
    row_min: int
    row_max: int
    col_min: int
    col_max: int


@dataclass
class LineRec:
    row: int = 0
    col: int = 0
    shp_index: int = 0
    road_class: int = 0
    x1: float = 0.0
    x2: float = 0.0
    y1: float = 0.0
    y2: float = 0.0
    elev: float = 0.0
    omit: bool = False
    line_length: float = 0.0

    def __init__(self):
        self.inund_freq: List[int] = []
        self.out_sites: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


TLineData = List[LineRec]


@dataclass
class PointRec:
    row: int = 0
    col: int = 0
    shp_index: int = 0
    x: float = 0.0
    y: float = 0.0
    elev: float = 0.0
    omit: bool = False
    inf_point_class: int = 0
    inund_freq: List[int] = field(default_factory=list)

def inund_text(i_f: int) -> str:
    if i_f == 0:
        return 'not initialized'
    elif i_f == 30:
        return 'every 30 days'
    elif i_f == 60:
        return 'every 60 days'
    elif i_f == 90:
        return 'every 90 days'
    elif i_f == 120:
        return 'storm'
    elif i_f == 150:
        return 'large storm'
    elif i_f == 252:
        return 'irrelevant'
    elif i_f == 253:
        return 'Blank or Diked'
    elif i_f == 254:
        return 'Open Water'
    elif i_f == 255:
        return 'not inundated'
    else:
        return 'unknown'

# Main Classes
class TInfrastructure:
    input_fname: str = ""
    id_name: str = ""
    has_specific_elevs: bool = False
    pss: Optional['TSLAMM_Simulation'] = None
    proj_run_str_arr: List[str] = field(default_factory=lambda: [""] * 1)
    output_fname: str = ""
    fields: List[str] = field(default_factory=list)
    elev_field: Optional[str] = None

    def calculate_cell_inundation(self, row: int, col: int) -> int:
        ss = self.pss
        result = 255
        arr_byte = ss.inund_arr[row * ss.site.cols + col]
        if arr_byte in [0, 30, 60, 90, 120, 150]:
            result = arr_byte
        elif arr_byte == 8:
            result = 30  # Below MTL and connected assigned to 30 d inundation
        elif arr_byte == 7:
            result = 253  # Blank or diked
        elif arr_byte == 5:
            result = 252  # Irrelevant
        return result

    def update_inund_freq(self) -> bool:
        ss = self.pss
        if not ss.inund_freq_check:
            result = ss.calc_inund_freq()
            if not result:
                ss.inund_arr = None
            ss.inund_freq_check = result
        return ss.inund_freq_check

    def check_valid(self) -> bool:
        dbfname = self.input_fname.replace('.shp', '.dbf')
        while not os.path.exists(dbfname):
            dbfname = input(f'{self.id_name} does not exist, please locate this dbf file ({self.input_fname}): ')
            if not dbfname:
                print(f'Error, Data File for Infrastructure Layer "{self.id_name}" does not exist. ({self.input_fname})')
                return False
            self.input_fname = dbfname
        return True

    def load_store(self, file, read_version_num, is_reading, pss):
        self.pss = pss
        self.input_fname = ts_read_write(file, 'InputFName', self.input_fname, str, is_reading)
        self.id_name = ts_read_write(file, 'IDName', self.id_name, str, is_reading)
        self.has_specific_elevs = ts_read_write(file, 'HasElevs', self.has_specific_elevs, bool, is_reading)

        self.proj_run_str_arr = [""] * 1

    def create_output_dbf(self):
        def copy_ext(ext):
            input_with_ext = os.path.splitext(self.input_fname)[0] + ext
            output_with_ext = os.path.splitext(self.output_fname)[0] + ext
            if os.path.exists(input_with_ext):
                shutil.copyfile(input_with_ext, output_with_ext)

        ss = self.pss
        if ss.run_uncertainty or ss.run_sensitivity:
            return  # no infrastructure output for uncertainty / sensitivity at this time

        self.output_fname = os.path.splitext(self.input_fname)[0] + f'_{ss.run_time}.dbf'
        shutil.copyfile(os.path.splitext(self.input_fname)[0] + '.dbf', self.output_fname)

        if not (ss.run_uncertainty or ss.run_sensitivity):
            try:
                run_record_file = open(ss.run_record_file_name, "a")
                run_record_file.write(f'Output DBF Written: {self.output_fname}\n')
                run_record_file.close()
            except Exception as e:
                print(f'Error appending to Run-Record File: {str(e)}')

        for ext in ['.shp', '.shx', '.prj', '.qpj']:
            copy_ext(ext)

    def add_dbf_fields(self, first_write: bool, count_ts: int):
        if not os.path.exists(self.output_fname):
            raise FileNotFoundError(f"Error: DBF File {self.output_fname} does not exist")

        write_elevs = False  # Initially setting write_elevs to False

        # Open the DBF file in read/write mode
        dbf1 = Table(self.output_fname)
        dbf1.open(mode=dbf.READ_WRITE)

        original_field_names = list(dbf1.field_names)

        temp_field_defs = original_field_names[:]
        field_specs = []

        if first_write and self.has_specific_elevs:
            write_elevs = True
            temp_field_defs.append(('Elev MTL', 'N', 6, 3))
            field_specs.append("Elev MTL N(6,3)")

        for k in range(1, count_ts):
            field_name = self.proj_run_str_arr[k]
            temp_field_defs.append((field_name, 'N', 3, 0))
            field_specs.append(f"{field_name} N(3,0)")

        # Add fields to the DBF
        dbf1.add_fields(";".join(field_specs))

        self.elev_field = None
        if write_elevs:
            self.elev_field = 'Elev MTL'

        # Save only the new field names in self.fields
        self.fields = [field[0] if isinstance(field, tuple) else field for field in temp_field_defs]

        return dbf1, write_elevs

@dataclass
class TRoadInfrastructure(TInfrastructure):
    n_road_cl: int = 0
    n_roads: int = 0
    roads_loaded: bool = False
    roads_changed: bool = False

    def __init__(self, ss):
        self.road_data: List['LineRec'] = []
        self.road_classes: List[int] = []
        super().__init__()
        self.id_name = 'New Roads Layer'
        self.roads_loaded = False
        self.roads_changed = False

    def check_valid(self) -> bool:
        if self.n_roads == 0:
            print(f"Error, Roads Layer '{self.id_name}' has no attached roads data.")
            return False
        return super().check_valid()

    def create_output_dbf(self):
        super().create_output_dbf()



    def calc_all_roads_inundation(self) -> bool:
        ss = self.pss
        result = self.update_inund_freq()

        self.proj_run_str_arr[ss.tstep_iter] = ss.short_proj_run
        if result:
            for i in range(self.n_roads):
                inund = self.calculate_cell_inundation(self.road_data[i].row, self.road_data[i].col)
                self.road_data[i].inund_freq[ss.tstep_iter] = inund
                if ss.time_zero and inund == 30:
                    self.road_data[i].omit = True
        return result

    def load_save_data(self, file, is_reading):
        if is_reading:
            print('Loading Roads...')
        else:
            print('Saving Roads...')

        self.n_roads = ts_read_write(file, 'NRoads', self.n_roads, int, is_reading)
        if is_reading:
            while len(self.road_data) < self.n_roads:
                self.road_data.append(LineRec())
        for i in range(self.n_roads):
            self.road_data[i].row = ts_read_write(file, 'Row', self.road_data[i].row, int, is_reading)
            self.road_data[i].col = ts_read_write(file, 'Col', self.road_data[i].col, int, is_reading)
            self.road_data[i].shp_index = ts_read_write(file, 'ShpIndex', self.road_data[i].shp_index, int,
                                                        is_reading)
            self.road_data[i].road_class = ts_read_write(file, 'RoadClass', self.road_data[i].road_class, int,
                                                         is_reading)
            self.road_data[i].elev = ts_read_write(file, 'Elev', self.road_data[i].elev, float, is_reading)
            self.road_data[i].omit = ts_read_write(file, 'Omit', self.road_data[i].omit, bool, is_reading)
            self.road_data[i].line_length = ts_read_write(file, 'LineLength', self.road_data[i].line_length, float,
                                                          is_reading)

        self.roads_changed = False
        if is_reading:
            self.roads_loaded = True
            print('Roads Loaded')
        else:
            print('Roads Saved')

    def load_store(self, file, read_version_num, is_reading, pss):
        ts_read_write(file, 'NRoadCl', self.n_road_cl, int, is_reading)
        for i in range(self.n_road_cl):
            ts_read_write(file, 'RoadClasses[i]', self.road_classes[i], int, is_reading)

        self.load_save_data(file, is_reading)

        super().load_store(file, read_version_num, is_reading, pss)

    def initialize_road_vars(self):
        ss = self.pss

        self.proj_run_str_arr = [""] * ss.n_time_steps
        for i in range(self.n_roads):
            self.road_data[i].inund_freq = [255] * ss.n_time_steps
            self.road_data[i].omit = False
            self.road_data[i].out_sites = [False] * (ss.site.n_output_sites + ss.site.max_ros + 1)

            road_row = self.road_data[i].row
            road_col = self.road_data[i].col
            self.road_data[i].out_sites[0] = ss.in_area_to_save(road_col, road_row, not ss.save_ros_area)
            for ios in range(1, ss.site.n_output_sites):
                self.road_data[i].out_sites[ios] = ss.site.in_out_site(road_col, road_row, ios)

            if ss.shared_data.ros_array:
                for ros in range(1, ss.site.max_ros):
                    self.road_data[i].out_sites[ros + ss.site.n_output_sites] = (ros == ss.shared_data.ros_array[road_row][road_col])

    def load_data(self, file, read_version_num):
        ts_read(file, 'NRoads', self.n_roads)

        print('Reading Roads...')
        self.road_data = [LineRec() for _ in range(self.n_roads)]
        for i in range(self.n_roads):
            ts_read(file, f'Row{i}', self.road_data[i].row)
            if read_version_num < 6.955:
                self.road_data[i].row += 1  # fix horizontal offset problem

            ts_read(file, f'Col{i}', self.road_data[i].col)
            ts_read(file, f'ShpIndex{i}', self.road_data[i].shp_index)
            ts_read(file, f'RoadClass{i}', self.road_data[i].road_class)
            ts_read(file, f'Elev{i}', self.road_data[i].elev)
            if read_version_num > 6.905:
                ts_read(file, f'Omit{i}', self.road_data[i].omit)
            else:
                self.road_data[i].omit = False
            ts_read(file, f'LineLength{i}', self.road_data[i].line_length)

            self.road_data[i].inund_freq = [255]

        self.roads_loaded = True
        self.roads_changed = False
        print('Roads Read')

    def load(self, file, read_version_num, ss):
        ts_read(file, 'NRoadCl', self.n_road_cl)
        self.road_classes = [0] * self.n_road_cl
        for i in range(self.n_road_cl):
            ts_read(file, f'RoadClasses{i}', self.road_classes[i])

        self.load_data(file, read_version_num)
        super().load_store(file, read_version_num, True, ss)

    def overwrite_elevs(self):
        if self.has_specific_elevs:
            print('Overwriting elevations disabled')
            return

    def update_road_sums(self, count_proj, road_out_sum, n_out_sites):
        for oi in range(n_out_sites):
            for rd_index in range(self.n_roads):
                if self.road_data[rd_index].line_length > 0 and self.road_data[rd_index].out_sites[oi]:
                    road_out_sum[oi][0] += self.road_data[rd_index].line_length / 1000

                    if self.road_data[rd_index].inund_freq[count_proj] == 30:
                        road_out_sum[oi][1] += self.road_data[rd_index].line_length / 1000
                    elif self.road_data[rd_index].inund_freq[count_proj] == 60:
                        road_out_sum[oi][2] += self.road_data[rd_index].line_length / 1000
                    elif self.road_data[rd_index].inund_freq[count_proj] == 90:
                        road_out_sum[oi][3] += self.road_data[rd_index].line_length / 1000
                    elif self.road_data[rd_index].inund_freq[count_proj] == 120:
                        road_out_sum[oi][4] += self.road_data[rd_index].line_length / 1000
                    elif self.road_data[rd_index].inund_freq[count_proj] == 150:
                        road_out_sum[oi][5] += self.road_data[rd_index].line_length / 1000
                    elif self.road_data[rd_index].inund_freq[count_proj] == 255:
                        road_out_sum[oi][6] += self.road_data[rd_index].line_length / 1000
                    elif self.road_data[rd_index].inund_freq[count_proj] == 0:
                        road_out_sum[oi][7] += self.road_data[rd_index].line_length / 1000
                    elif self.road_data[rd_index].inund_freq[count_proj] == 254:
                        road_out_sum[oi][8] += self.road_data[rd_index].line_length / 1000
                    elif self.road_data[rd_index].inund_freq[count_proj] == 253:
                        road_out_sum[oi][9] += self.road_data[rd_index].line_length / 1000
                    else:
                        road_out_sum[oi][10] += self.road_data[rd_index].line_length / 1000

    def write_road_dbf(self, first_write: bool, count_ts: int):
        ss = self.pss
        if ss.run_uncertainty or ss.run_sensitivity:
            return  # no infrastructure output for uncertainty / sensitivity at this time

        dbf1, write_elevs = self.add_dbf_fields(first_write, count_ts)

        appending = len(dbf1) == 0

        for i in range(self.n_roads):
            with dbf1.new_record() if appending else dbf1[i] as record:
                if write_elevs:
                    record[self.elev_field] = self.road_data[i].elev
                for k in range(1, count_ts):
                    if self.road_data[i].omit:
                        record[self.fields[k]] = -99
                    else:
                        record[self.fields[k]] = self.road_data[i].inund_freq[k]

        self.fields = []
        self.elev_field = None
        dbf1.close()


class TPointInfrastructure(TInfrastructure):

    def __init__(self, ss):
        super().__init__()
        self.id_name = 'New Infr. Layer'
        self.n_points = 0
        self.has_specific_elevs = False
        self.fields = []
        self.elev_field = None
        self.inf_point_data: List[PointRec] = []

    def check_valid(self):
        if not self.n_points:
            print(f"Error, Points Layer '{self.id_name}' has no attached roads data.")
            return False
        return super().check_valid()

    def initialize_point_vars(self):
        ss = self.pss
        self.proj_run_str_arr = [''] * ss.n_time_steps  # Set length of projection runs string array
        for i in range(self.n_points):
            (self.inf_point_data[i]).inund_freq = [0] * ss.n_time_steps   # Set the length of InundFreq
            (self.inf_point_data[i]).omit = False

    def __del__(self):
        self.inf_point_data = List[PointRec]

    def calc_all_point_inf_inundation(self) -> bool:
        """
        Calculate inundation frequencies for all point elements.
        """
        ss = self.pss
        result = self.update_inund_freq()
        if not result:
            return False

        self.proj_run_str_arr[ss.tstep_iter] = ss.short_proj_run
        for i in range(self.n_points):
            inund = self.calculate_cell_inundation(self.inf_point_data[i].row, self.inf_point_data[i].col)
            self.inf_point_data[i].inund_freq[ss.tstep_iter] = inund
            if ss.time_zero and inund == 30:
                self.inf_point_data[i].omit = True

        return True

    def write_point_dbf(self, first_write: bool, count_ts: int):

        ss = self.pss
        if ss.run_uncertainty or ss.run_sensitivity:
            return  # no infrastructure output for uncertainty / sensitivity at this time

        dbf1, write_elevs = self.add_dbf_fields(first_write, count_ts)

        appending = len(dbf1) == 0

        for i in range(self.n_points):
            with dbf1.new_record() if appending else dbf1[i] as record:
                if write_elevs:
                    record[self.elev_field] = self.inf_point_data[i].elev
                for k in range(1, count_ts):
                    if self.inf_point_data[i].omit:
                        record[self.fields[k]] = -99
                    else:
                        record[self.fields[k]] = self.inf_point_data[i].inund_freq[k]

        self.fields = []
        self.elev_field = None
        dbf1.close()

    def load_store(self, file, read_version_num, is_reading, pss):
        self.n_points = ts_read_write(file, 'NPoints', self.n_points, int, is_reading)

        # Handling the array of point data
        if is_reading:
            self.inf_point_data = [PointRec() for _ in range(self.n_points)]
        for i in range(self.n_points):
            self.inf_point_data[i].row = ts_read_write(file, 'Row', self.inf_point_data[i].row, int, is_reading)
            self.inf_point_data[i].col = ts_read_write(file, 'Col', self.inf_point_data[i].col, int, is_reading)
            self.inf_point_data[i].shp_index = ts_read_write(file, 'ShpIndex', self.inf_point_data[i].shp_index, int, is_reading)
            self.inf_point_data[i].elev = ts_read_write(file, 'Elev', self.inf_point_data[i].elev, float, is_reading)
            self.inf_point_data[i].omit = ts_read_write(file, 'Omit', self.inf_point_data[i].omit, bool, is_reading)
            self.inf_point_data[i].inf_point_class = ts_read_write(file, 'InfPointClass', self.inf_point_data[i].inf_point_class, int, is_reading)
            self.inf_point_data[i].inund_freq = [255]  # minimum length and initial value

        super().load_store(file, read_version_num, is_reading, pss)

from typing import List, Optional
import os
from enum import Enum
from dataclasses import dataclass, field, asdict
from shapely.geometry import Polygon, Point
import numpy as np
from math import sqrt


# Constants
VERSION_NUM = 6.991  # used for file reading and writing, 6.9905 threshold
VERS_STR = '6.7.3 beta'
BUILD_STR = '6.7.0250'

FEET_PER_METER = 3.28084

USE_DATAELEV = True  # Use an array that displays elevations above 8 meters

MTL = 0.0  # use mean tidal level as datum
ELEV_CUTOFF = 8.0  # elev not to model when additional optimize is turned on, in meters above MTL

SMALL = 0.1  # square meters
TINY = 1e-10

OUTPUT_MAXC = 60
MAX_ROAD_ARRAY = 12
MAX_CATS = 30
NUM_ROWS_ACCR = 8  # Number of accretion parameters for each accretion model

SALT_WATER = 1
EXPOSED_WATER = 2
MOSTLY_WATER = 4
HAS_EFSW = 8  # Estuarine forested/shrub wetland, CA categories

BLANK = -99
NO_DATA = -9999

# Variables
drive_externally = False
large_raster_edit = False
iter_count = 0


class ESLAMMError(Exception):
    pass


# Record equivalent in Python using a class
class SAVParamsRec:
    def __init__(self, intcpt, c_dem, c_dem2, c_dem3, c_d2mllw, c_d2mhhw, c_d2m, c_d2m2):
        self.intcpt = intcpt
        self.c_dem = c_dem
        self.c_dem2 = c_dem2
        self.c_dem3 = c_dem3
        self.c_d2mllw = c_d2mllw
        self.c_d2mhhw = c_d2mhhw
        self.c_d2m = c_d2m
        self.c_d2m2 = c_d2m2


# For 'str_vector', we'll define a function that creates a list of strings with a maximum size
def create_str_vector(max_size: int) -> List[str]:
    return [""] * max_size


class WaveDirection(Enum):
    Easterly = 0
    Westerly = 1
    Southerly = 2
    Northerly = 3


class ElevUnit(Enum):
    HalfTide = 0
    SaltBound = 1
    Meters = 2


class ElevCategory(Enum):
    LessThanMLLW = 0
    MLLWtoMLW = 1
    MLWtoMTL = 2
    MTLtoMHW = 3
    MHWtoMHWS = 4
    GreaterThanMHWS = 5


class AccrModels(Enum):
    RegFM = 0
    IrregFM = 1
    BeachTF = 2
    TidalFM = 3
    InlandM = 4
    Mangrove = 5
    TSwamp = 6
    Swamp = 7
    AccrNone = 8


class ErosionInputs(Enum):
    EMarsh = 0
    ESwamp = 1
    ETFlat = 2
    EOcBeach = 3
    ENone = 4


class AggCategories(Enum):
    NonTidal = 0
    FreshNonTidal = 1
    OpenWater = 2
    LowTidal = 3
    SaltMarsh = 4
    Transitional = 5
    FreshWaterTidal = 6
    AggBlank = 7


@dataclass
class ClassElev:
    min_unit = ElevUnit.Meters
    min_elev: float = 0.0
    max_unit = ElevUnit.Meters
    max_elev: float = 0.0

    def load_store(self, file, read_version_num, is_reading):

        def unit_to_str(unit: ElevUnit) -> str:
            if unit == ElevUnit.HalfTide:
                return 'HalfTide'
            elif unit == ElevUnit.SaltBound:
                return 'SaltBound'
            else:
                return 'Meters'

        def str_to_unit(unit_str: str) -> ElevUnit:
            if unit_str == 'HalfTide':
                return ElevUnit.HalfTide
            elif unit_str == 'SaltBound':
                return ElevUnit.SaltBound
            else:
                return ElevUnit.Meters

        # Handle reading or writing for minimum elevation and unit
        self.min_elev = ts_read_write(file, 'MinElev', self.min_elev, float, is_reading)
        min_unit_str = unit_to_str(self.min_unit) if not is_reading else None
        min_unit_str = ts_read_write(file, 'MinUnit', min_unit_str, str, is_reading)
        if is_reading:
            self.min_unit = str_to_unit(min_unit_str)

        # Handle reading or writing for maximum elevation and unit
        self.max_elev = ts_read_write(file, 'MaxElev', self.max_elev, float, is_reading)
        max_unit_str = unit_to_str(self.max_unit) if not is_reading else None
        max_unit_str = ts_read_write(file, 'MaxUnit', max_unit_str, str, is_reading)
        if is_reading:
            self.max_unit = str_to_unit(max_unit_str)


line_num = 0


def check_name(expected_name, file):
    line = file.readline()
    global line_num
    line_num += 1
    if ':' not in line:
        raise ValueError(f"No ':' or blank after ':', Reading Variable '{expected_name}' Line {line_num}")
    name_str, value_str = line.split(':', 1)
    name_str = name_str.strip()
    if name_str != expected_name:
        raise ValueError(f"Text Read Name Mismatch, Line {line_num} Expecting Variable '{expected_name}'; read "
                         f"variable '{name_str}'")
    return value_str.strip()


def ts_write(file, name, value):
    file.write(f"{name}:{value}\n")


def ts_read(file, name, dtype):
    value_str = check_name(name, file)
    if dtype == int:
        return int(value_str)
    elif dtype == float:
        return float(value_str)
    elif dtype == bool:
        return value_str.upper() == 'TRUE'
    elif dtype == str:
        return value_str
    else:
        raise TypeError("Unsupported data type")


def ts_read_write(file, name, value, dtype, is_reading):

    if is_reading:
        value = ts_read(file, name, dtype)
        return value
    else:
        ts_write(file, name, value)
        return value


def split_file(main_file, split_file_name, is_reading):
    base_name, ext = os.path.splitext(main_file.name)
    alt_file_name = f"{base_name}_{split_file_name}{ext}"

    if is_reading:
        if os.path.exists(alt_file_name):
            return open(alt_file_name, 'r')
        else:
            return main_file
    else:  # Writing
        return open(alt_file_name, 'w')

def revert_file(current_file, main_file):
    if current_file != main_file:
        current_file.close()
    return main_file

class DPoint:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class DLine:
    def __init__(self, p1: DPoint, p2: DPoint):
        self.p1 = p1
        self.p2 = p2


def distance_pt_2_line(ln: DLine, p: DPoint) -> float:
    a = ln.p1.x - ln.p2.x
    b = ln.p2.y - ln.p1.y
    c = ln.p1.y * ln.p2.x - ln.p2.y * ln.p1.x
    return (a * p.y + b * p.x + c) / sqrt(b**2 + a**2)


def distance_pt_2_line_km(ln: DLine, p: DPoint, scale: float) -> float:
    return abs(distance_pt_2_line(ln, p)) * scale * 1e-3


def cross_line(p1: DPoint, p2: DPoint, ln: DLine) -> bool:
    """Would a line connecting the two points cross the given line?
    The line is assumed to be of infinite length in this case."""
    return (distance_pt_2_line(ln, p1) * distance_pt_2_line(ln, p2)) <= 0

def distance_2pts_km(p1: DPoint, p2: DPoint, scale: float) -> float:
    """Returns distance between two cells in km."""
    return sqrt((p2.y - p1.y)**2 + (p2.x - p1.x)**2) * scale * 1e-3

class TPolygon:
    def __init__(self, points=None):
        self.polygon = Polygon(points) if points else None
        self.num_pts = 0
        self.points = []
        self.sv_min_row = -9999
        self.sv_max_row = -9999
        self.sv_min_col = -9999
        self.sv_max_col = -9999

    def in_poly(self, row, col):
        point = Point(col, row)
        return self.polygon.contains(point) if self.polygon else False

    def min_row(self):
        if self.sv_min_row != -9999:
            return self.sv_min_row
        else:
            if self.polygon:
                self.sv_min_row = min(point.y for point in self.polygon.exterior.coords)
            return self.sv_min_row

    def max_row(self):
        if self.sv_max_row != -9999:
            return self.sv_max_row
        else:
            if self.polygon:
                self.sv_max_row = max(point.y for point in self.polygon.exterior.coords)
            return self.sv_max_row

    def min_col(self):
        if self.sv_min_col != -9999:
            return self.sv_min_col
        else:
            if self.polygon:
                self.sv_min_col = min(point.x for point in self.polygon.exterior.coords)
            return self.sv_min_col

    def max_col(self):
        if self.sv_max_col != -9999:
            return self.sv_max_col
        else:
            if self.polygon:
                self.sv_max_col = max(point.x for point in self.polygon.exterior.coords)
            return self.sv_max_col

    def copy_polygon(self):
        return TPolygon(points=[(p.x, p.y) for p in self.polygon.exterior.coords])

    def store(self, file):
        ts_write(file, 'Poly.NumPts', self.num_pts)
        for point in self.points:
            ts_write(file, 'TPoints[i-1].x', point.x)
            ts_write(file, 'TPoints[i-1].y', point.y)

    def load(self, file, read_version_num, read_header):
        global line_num
        line_num = 0
        if read_header:
            self.num_pts = ts_read(file, 'Poly.NumPts', int)
            self.points = [Point(0, 0) for _ in range(self.num_pts)]
        for i in range(self.num_pts):
            x = ts_read(file, 'TPoints[i-1].x', float)
            y = ts_read(file, 'TPoints[i-1].y', float)
            self.points[i] = Point(x, y)

        self.polygon = Polygon(self.points) if self.points else None
        # Reset spatial bounds
        self.sv_min_row = -9999
        self.sv_max_row = -9999
        self.sv_min_col = -9999
        self.sv_max_col = -9999


@dataclass
class TRectangle:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class TOutputSite2:
    use_polygon: bool = False
    rec: TRectangle = field(default_factory=lambda: TRectangle(0, 0, 0, 0))
    poly: TPolygon = field(default_factory=TPolygon)
    description: str = ''


N_WIND_DIRECTIONS = 16
N_WIND_SPEEDS = 7

WIND_DIRECTIONS = ('E', 'ENE', 'NE', 'NNE', 'N', 'NNW', 'NW', 'WNW', 'W', 'WSW', 'SW', 'SSW', 'S', 'SSE', 'SE', 'ESE')
WIND_DEGREES = (90, 68, 45, 22, 0, 338, 315, 292, 270, 248, 225, 202, 180, 158, 135, 112)


class TLagoonType(Enum):
    LtNone = 0
    LtOpenTidal = 1
    LtPredOpen = 2
    LtPredClosed = 3
    LtDrainage = 4


N_ACCR_MODELS = 4
ACCR_NAMES = ['Reg Flood', 'Irreg Flood', 'T.Flat', 'Tidal Fresh']

use_accr_var = [False] * N_ACCR_MODELS  # Initializes a list of False
accr_var = [0.0] * N_ACCR_MODELS  # Initializes a list of 0.0
accr_notes = [''] * N_ACCR_MODELS  # Initializes a list of empty strings


@dataclass
class TInundContext:
    distance_to_open_salt_water: float = 0.0
    subs: 'TSubSite' = None
    adj_efsw: bool = False
    adj_ocean: bool = False
    adj_water: bool = False
    near_water: bool = False
    cell_fw_influenced: bool = False
    erosion2: float = 0.0  # Cell Max Fetch in km
    ew_cat: int = 0
    cell_row: int = 0
    cell_col: int = 0
    cat_elev: float = -9999.0


@dataclass
class TSubSite:
    poly: TPolygon = field(default_factory=TPolygon)
    description: str = ""
    nwi_photo_date: int = 0
    dem_date: int = 0
    direction_offshore: WaveDirection = WaveDirection.Northerly  # Default value, adjust as necessary
    historic_trend: float = 0.0
    historic_eustatic_trend: float = 1.7  # Default value
    navd88mtl_correction: float = 0.0
    gtide_range: float = 0.0
    salt_elev: float = 0.0
    inund_elev: List[float] = field(default_factory=lambda: [0.0] * 5)
    marsh_erosion: float = 0.0
    marsh_erode_fetch: float = 0.0
    swamp_erosion: float = 0.0
    tflat_erosion: float = 0.0
    fixed_reg_flood_accr: float = 0.0
    fixed_irreg_flood_accr: float = 0.0
    fixed_tide_fresh_accr: float = 0.0
    inland_fresh_accr: float = 0.0
    mangrove_accr: float = 0.0
    tswamp_accr: float = 0.0
    swamp_accr: float = 0.0
    ifm2rfm_collapse: float = 0.0
    rfm2tf_collapse: float = 0.0
    fixed_tf_beach_sed: float = 0.0
    use_preprocessor: bool = False
    use_wave_erosion: bool = False
    we_alpha: float = 0.0
    we_has_bathymetry: bool = False
    we_avg_shallow_depth: float = 1.0
    lagoon_type: TLagoonType = TLagoonType.LtNone
    zbeach_crest: float = 0.0
    lbeta: float = 0.0
    use_accr_model: List[bool] = field(default_factory=lambda: [False] * N_ACCR_MODELS)
    max_accr: List[float] = field(default_factory=lambda: [0.0] * N_ACCR_MODELS)
    min_accr: List[float] = field(default_factory=lambda: [0.0] * N_ACCR_MODELS)
    accr_a: List[float] = field(default_factory=lambda: [0.0] * N_ACCR_MODELS)
    accr_b: List[float] = field(default_factory=lambda: [0.0] * N_ACCR_MODELS)
    accr_c: List[float] = field(default_factory=lambda: [0.0] * N_ACCR_MODELS)
    accr_d: List[float] = field(default_factory=lambda: [0.0] * N_ACCR_MODELS)
    acc_rescaled: List[bool] = field(default_factory=lambda: [False] * N_ACCR_MODELS)
    raccr_a: List[float] = field(default_factory=lambda: [0.0] * N_ACCR_MODELS)
    raccr_b: List[float] = field(default_factory=lambda: [0.0] * N_ACCR_MODELS)
    raccr_c: List[float] = field(default_factory=lambda: [0.0] * N_ACCR_MODELS)
    raccr_d: List[float] = field(default_factory=lambda: [0.0] * N_ACCR_MODELS)
    accr_notes: List[str] = field(default_factory=lambda: [''] * N_ACCR_MODELS)

    # No load or save below here

    mhhw = 0.0
    mllw = 0.0
    norm = 0.0
    sl_rise = 0.0
    old_sl = 0.0
    newsl = 0.0
    t0_slr = 0.0
    delta_t = 0.0

    def __post_init__(self):
        if self.poly is None:
            self.poly = TPolygon()

    @classmethod
    def create_with(cls, source):
        # Create a new instance, which is a copy of the current one
        new_instance = cls()
        # Copy all attributes from the source instance except the polygons
        for attribute, value in asdict(source).items():
            if attribute not in ['poly']:  # Skip copying polygons
                setattr(new_instance, attribute, value)

        # Always create new polygons for the new instance
        new_instance.poly = TPolygon()
        return new_instance


@dataclass
class TSite:
    llx_corner: float = 0.0  # SW corner longitude
    lly_corner: float = 0.0  # SW corner latitude
    site_scale: float = 1.0  # Scale of the simulation run
    global_site: TSubSite = field(default_factory=TSubSite)
    n_subsites: int = 0
    n_output_sites: int = 0
    max_ros: int = 0  # Max Raster Output Sites
    subsites: List[TSubSite] = field(default_factory=list)
    output_sites: List[TOutputSite2] = field(default_factory=list)
    ros_bounds: List[TRectangle] = field(default_factory=list)
    rows: int = 0
    cols: int = 0
    top_row_height: int = 0
    right_col_width: int = 0

    def load_store(self, file, read_version_num, is_reading):
        self.rows = ts_read_write(file, 'Rows', self.rows, int, is_reading)
        self.cols = ts_read_write(file, 'Cols', self.cols, int, is_reading)
        self.llx_corner = ts_read_write(file, 'LLXCorner', self.llx_corner, float, is_reading)
        self.lly_corner = ts_read_write(file, 'LLYCorner', self.lly_corner, float, is_reading)
        self.site_scale = ts_read_write(file, 'Scale', self.site_scale, float, is_reading)

        if is_reading:
            self.n_subsites = ts_read_write(file, 'NSubSites', None, int, is_reading)
            self.subsites = [TSubSite() for _ in range(self.n_subsites)]
            for ss in self.subsites:
                self.load_store_subsite(file, read_version_num, is_reading, ss)
        else:
            ts_read_write(file, 'NSubSites', len(self.subsites), int, is_reading)
            for ss in self.subsites:
                self.load_store_subsite(file, read_version_num, is_reading, ss)

        # Handle output sites
        if is_reading:
            self.n_output_sites = ts_read_write(file, 'NOutputSites', None, int, is_reading)
            self.output_sites = [TOutputSite2() for _ in range(self.n_output_sites)]
        else:
            ts_read_write(file, 'NOutputSites', len(self.output_sites), int, is_reading)

        for os in self.output_sites:
            os.use_polygon = ts_read_write(file, 'UsePolygon', os.use_polygon, bool, is_reading)
            if os.use_polygon:
                if is_reading:
                    os.poly.load(file, read_version_num, True)
                else:
                    os.poly.store(file)
            os.rec.x1 = ts_read_write(file, 'X1', os.rec.x1, int, is_reading)
            os.rec.y1 = ts_read_write(file, 'Y1', os.rec.y1, int, is_reading)
            os.rec.x2 = ts_read_write(file, 'X2', os.rec.x2, int, is_reading)
            os.rec.y2 = ts_read_write(file, 'Y2', os.rec.y2, int, is_reading)
            os.description = ts_read_write(file, 'Description', os.description, str, is_reading)

        # Handle the global site
        self.load_store_subsite(file, read_version_num, is_reading, self.global_site)

    def load_store_subsite(self, file, read_version_num, is_reading, ss):

        if is_reading:
            ss.poly.load(file, read_version_num, True)
        else:
            ss.poly.store(file)

        ss.description = ts_read_write(file, 'Description', ss.description, str, is_reading)
        ss.nwi_photo_date = ts_read_write(file, 'NWI_Photo_Date', ss.nwi_photo_date, int, is_reading)
        ss.dem_date = ts_read_write(file, 'DEM_date', ss.dem_date, int, is_reading)

        if is_reading:
            offshore_text = ts_read_write(file, 'Direction_Offshore', None, str, is_reading)
            ss.direction_offshore = {'Northerly': WaveDirection.Northerly, 'Southerly': WaveDirection.Southerly,
                                     'Easterly': WaveDirection.Easterly, 'Westerly': WaveDirection.Westerly}[
                offshore_text]
        else:
            offshore_direction = (
                {WaveDirection.Northerly: 'Northerly', WaveDirection.Southerly: 'Southerly',
                 WaveDirection.Easterly: 'Easterly', WaveDirection.Westerly: 'Westerly'}
                [ss.direction_offshore]
            )
            ts_read_write(file, 'Direction_Offshore', offshore_direction, str, is_reading)

        ss.historic_trend = ts_read_write(file, 'Historic_trend', ss.historic_trend, float, is_reading)
        ss.historic_eustatic_trend = ts_read_write(file, 'Historic_Eustatic_trend', ss.historic_eustatic_trend,
                                                   float, is_reading)
        ss.navd88mtl_correction = ts_read_write(file, 'NAVD88MTL_correction', ss.navd88mtl_correction, float,
                                                is_reading)
        ss.gtide_range = ts_read_write(file, 'GTideRange', ss.gtide_range, float, is_reading)
        ss.salt_elev = ts_read_write(file, 'SaltElev', ss.salt_elev, float, is_reading)

        # Inundation data
        ss.inund_elev[0] = ts_read_write(file, '30D Inundation', ss.inund_elev[0], float, is_reading)
        ss.inund_elev[1] = ts_read_write(file, '60D Inundation', ss.inund_elev[1], float, is_reading)
        ss.inund_elev[2] = ts_read_write(file, '90D Inundation', ss.inund_elev[2], float, is_reading)
        ss.inund_elev[3] = ts_read_write(file, 'Storm Inundation 1', ss.inund_elev[3], float, is_reading)
        ss.inund_elev[4] = ts_read_write(file, 'Storm Inundation 2', ss.inund_elev[4], float, is_reading)

        # Erosion and accretion
        ss.marsh_erosion = ts_read_write(file, 'MarshErosion', ss.marsh_erosion, float, is_reading)
        ss.marsh_erode_fetch = ts_read_write(file, 'MarshErodeFetch', ss.marsh_erode_fetch, float, is_reading)
        ss.swamp_erosion = ts_read_write(file, 'SwampErosion', ss.swamp_erosion, float, is_reading)
        ss.tflat_erosion = ts_read_write(file, 'TFlatErosion', ss.tflat_erosion, float, is_reading)
        ss.fixed_reg_flood_accr = ts_read_write(file, 'FixedRegFloodAccr', ss.fixed_reg_flood_accr, float,
                                                is_reading)
        ss.fixed_irreg_flood_accr = ts_read_write(file, 'FixedIrregFloodAccr', ss.fixed_irreg_flood_accr, float,
                                                  is_reading)
        ss.fixed_tide_fresh_accr = ts_read_write(file, 'FixedTideFreshAccr', ss.fixed_tide_fresh_accr, float,
                                                 is_reading)

        # Additional parameters
        ss.inland_fresh_accr = ts_read_write(file, 'InlandFreshAccr', ss.inland_fresh_accr, float, is_reading)
        ss.mangrove_accr = ts_read_write(file, 'MangroveAccr', ss.mangrove_accr, float, is_reading)
        ss.tswamp_accr = ts_read_write(file, 'TSwampAccr', ss.tswamp_accr, float, is_reading)
        ss.swamp_accr = ts_read_write(file, 'SwampAccr', ss.swamp_accr, float, is_reading)
        ss.ifm2rfm_collapse = ts_read_write(file, 'IFM2RFM_Collapse', ss.ifm2rfm_collapse, float, is_reading)
        ss.rfm2tf_collapse = ts_read_write(file, 'RFM2TF_Collapse', ss.rfm2tf_collapse, float, is_reading)
        ss.fixed_tf_beach_sed = ts_read_write(file, 'Fixed_TF_Beach_Sed', ss.fixed_tf_beach_sed, float, is_reading)
        ss.use_preprocessor = ts_read_write(file, 'Use_Preprocessor', ss.use_preprocessor, bool, is_reading)
        ss.use_wave_erosion = ts_read_write(file, 'USE_Wave_Erosion', ss.use_wave_erosion, bool, is_reading)
        ss.we_alpha = ts_read_write(file, 'WE_Alpha', ss.we_alpha, float, is_reading)
        ss.we_has_bathymetry = ts_read_write(file, 'WE_Has_Bathymetry', ss.we_has_bathymetry, bool, is_reading)
        ss.we_avg_shallow_depth = ts_read_write(file, 'WE_Avg_Shallow_Depth', ss.we_avg_shallow_depth, float,
                                                is_reading)
        lagtype = ts_read_write(file, 'LagoonType', ss.lagoon_type.value, int, is_reading)
        if is_reading:
            ss.lagoon_type = TLagoonType(lagtype)
        ss.zbeach_crest = ts_read_write(file, 'ZBeachCrest', ss.zbeach_crest, float, is_reading)
        ss.lbeta = ts_read_write(file, 'LBeta', ss.lbeta, float, is_reading)

        # Loop over accretion models
        for i in range(N_ACCR_MODELS):
            ts_read_write(file, 'UseAccrModel[i]', ss.use_accr_model[i], bool, is_reading)
            ts_read_write(file, 'MaxAccr[i]', ss.max_accr[i], float, is_reading)
            ts_read_write(file, 'MinAccr[i]', ss.min_accr[i], float, is_reading)
            ts_read_write(file, 'AccrA[i]', ss.accr_a[i], float, is_reading)
            ts_read_write(file, 'AccrB[i]', ss.accr_b[i], float, is_reading)
            ts_read_write(file, 'AccrC[i]', ss.accr_c[i], float, is_reading)
            ts_read_write(file, 'AccrD[i]', ss.accr_d[i], float, is_reading)
            ts_read_write(file, 'AccrNotes[i]', ss.accr_notes[i], str, is_reading)

    def __post_init__(self):
        # Initialization of complex dynamic attributes
        if self.global_site is None:
            self.global_site = TSubSite()

    def add_subsite(self):
        new_subsite = TSubSite()
        self.subsites.append(new_subsite)
        self.n_subsites += 1

    def del_subsite(self, index: int):
        if 0 <= index < len(self.subsites):
            del self.subsites[index]
            self.n_subsites -= 1

    def add_output_site(self):
        new_output_site = TOutputSite2()
        self.output_sites.append(new_output_site)
        self.n_output_sites += 1

    def del_output_site(self, index: int):
        if 0 <= index < len(self.output_sites):
            del self.output_sites[index]
            self.n_output_sites -= 1

    def get_subsite_num(self, x: int, y: int, cell=None):
        """
        Retrieve or assign the subsite number based on the coordinates.
        If 'cell' is provided, it checks the subsite_index and updates it if necessary.
        """
        if cell is not None and cell['subsite_index'] != -9999:
            return cell['subsite_index']

        subsite_index = 0  # Default to global site
        for i, subsite in enumerate(reversed(self.subsites)):
            if subsite.poly and subsite.poly.in_poly(y, x):
                subsite_index = len(self.subsites) - i  # high number polygons on top
                break

        # Update the cell's subsite_index directly if cell is provided
        if cell is not None:
            cell['subsite_index'] = subsite_index

        return subsite_index

    def get_subsite(self, x: int, y: int, cell=None) -> TSubSite:
        index = self.get_subsite_num(x, y)
        if index == 0:
            return self.global_site
        else:
            return self.subsites[index-1]

    def in_out_site(self, x: int, y: int, site_num: int) -> bool:
        # Check if the coordinates are within the specified output site
        site = self.output_sites[site_num - 1]  # Adjust for zero-based indexing
        if site.use_polygon:
            return site.poly.in_poly(y, x)
        else:
            rect = site.rec
            return (x >= min(rect.x1, rect.x2)) and (x <= max(rect.x1, rect.x2)) and (y >= min(rect.y1, rect.y2)) and (y <= max(rect.y1, rect.y2))
    def t0(self) -> int:
        # Determine the latest NWI_Photo_Date among all subsites
        latest = self.global_site.nwi_photo_date
        for site in self.subsites:
            if site.nwi_photo_date > latest:
                latest = site.nwi_photo_date
        return latest

    def tide_oc(self, x: int, y: int) -> float:
        # Return the tide occlusion for a given coordinate
        subsite = self.get_subsite(x, y)
        return subsite.gtide_range if subsite else 0

    def init_elev_vars(self):
        def set_subsite_ranges(ss):
            ss.mllw = MTL - ss.gtide_range / 2.0  # Mean Lower Low Water
            ss.mhhw = ss.gtide_range / 2.0  # Mean Higher High Water

        set_subsite_ranges(self.global_site)
        for i in range(self.n_subsites):
            set_subsite_ranges(self.subsites[i])

    def __del__(self):
        self.subsites.clear()
        self.output_sites.clear()


@dataclass
class StatRecord:
    min: float = 0.0
    max: float = 0.0
    sum: float = 0.0
    sum_x2: float = 0.0
    sum_e2: float = 0.0
    mean: float = 0.0
    stdev: float = 0.0
    p05: float = 0.0
    p95: float = 0.0


HIST_WIDTH = 500
SAL_HIST_WIDTH = 100
SLICE_INCREMENT = 0.1


@dataclass
class TElevStats:
    n: int = 0
    stats: List[StatRecord] = None
    p_lower_min: float = 0.0
    p_higher_max: float = 0.0
    histogram: List[List[int]] = None
    values: List[List[float]] = None

    def __post_init__(self):
        if self.stats is None:
            self.stats = [StatRecord() for _ in range(8)]
        if self.histogram is None:
            self.histogram = [[0 for _ in range(HIST_WIDTH)] for _ in range(8)]
        if self.values is None:
            self.values = [[] for _ in range(8)]


@dataclass
class TSalinityRule:
    from_cat: int = BLANK
    to_cat: int = BLANK
    salinity_level: float = 30.0
    greater_than: bool = True
    salinity_tide: int = 3   # 1=MLLW, 2=MTL, 3=MHHW, 4=Monthly High
    description: str = ""

    def load_store(self, file, read_version_num, is_reading):
        self.from_cat = ts_read_write(file, 'FromCat', self.from_cat, int, is_reading)
        self.to_cat = ts_read_write(file, 'ToCat', self.to_cat, int, is_reading)
        self.salinity_level = ts_read_write(file, 'SalinityLevel', self.salinity_level, float, is_reading)
        self.greater_than = ts_read_write(file, 'GreaterThan', self.greater_than, bool, is_reading)
        self.salinity_tide = ts_read_write(file, 'SalinityTide', self.salinity_tide, int, is_reading)
        self.description = ts_read_write(file, 'Descript', self.description, str, is_reading)


@dataclass
class TSalinityRules:
    n_rules: int = 0
    rules: List[TSalinityRule] = field(default_factory=list)

    def load_store(self, file, read_version_num, is_reading):
        self.n_rules = ts_read_write(file, 'NRules', self.n_rules, int, is_reading)
        if is_reading:
            self.rules = [TSalinityRule() for _ in range(self.n_rules)]
            for rule in self.rules:
                rule.load_store(file, read_version_num, is_reading)
        else:
            for rule in self.rules:
                rule.load_store(file, read_version_num, is_reading)


class IPCCScenarios(Enum):
    Scen_None = -1
    Scen_A1B = 0
    Scen_A1T = 1
    Scen_A1F1 = 2
    Scen_A2 = 3
    Scen_B1 = 4
    Scen_B2 = 5


class IPCCEstimates(Enum):
    Est_None = -1
    Est_Min = 0
    Est_Mean = 1
    Est_Max = 2


LabelIPCC = {
    IPCCScenarios.Scen_A1B: "Scenario A1B",
    IPCCScenarios.Scen_A1T: "Scenario A1T",
    IPCCScenarios.Scen_A1F1: "Scenario A1F1",
    IPCCScenarios.Scen_A2: "Scenario A2",
    IPCCScenarios.Scen_B1: "Scenario B1",
    IPCCScenarios.Scen_B2: "Scenario B2"
}

LabelIPCCEst = {
    IPCCEstimates.Est_Min: "Minimum",
    IPCCEstimates.Est_Mean: "Mean",
    IPCCEstimates.Est_Max: "Maximum"
}

LabelFixed = {
    0: '1 meter',
    1: '1.5 meter',
    2: '2 meter',
    3: 'NYS GCM Max',
    4: 'NYS 1M by 2100',
    5: 'NYS RIM Min',
    6: 'NYS RIM Max',
    7: 'ESVA Historic',
    8: 'ESVA Low',
    9: 'ESVA High',
    10: 'ESVA Highest'
}


NUM_SAL_METRICS = 4  # MLLW, MTL, MHW, 30D
N_FTABLE = 60


@dataclass
class TSalStats:
    n: np.ndarray
    min: np.ndarray
    max: np.ndarray
    sum: np.ndarray
    sum_e2: np.ndarray
    mean: np.ndarray
    stdev: np.ndarray
    p05: np.ndarray
    p95: np.ndarray
    histogram: np.ndarray

    def __init__(self):
        self.n = np.zeros(NUM_SAL_METRICS, dtype=int)
        self.min = np.zeros(NUM_SAL_METRICS, dtype=float)
        self.max = np.zeros(NUM_SAL_METRICS, dtype=float)
        self.sum = np.zeros(NUM_SAL_METRICS, dtype=float)
        self.sum_e2 = np.zeros(NUM_SAL_METRICS, dtype=float)
        self.mean = np.zeros(NUM_SAL_METRICS, dtype=float)
        self.stdev = np.zeros(NUM_SAL_METRICS, dtype=float)
        self.p05 = np.zeros(NUM_SAL_METRICS, dtype=float)
        self.p95 = np.zeros(NUM_SAL_METRICS, dtype=float)
        self.histogram = np.zeros((NUM_SAL_METRICS, SAL_HIST_WIDTH), dtype=int)


@dataclass
class TSRecord:
    year: int = 0
    value: float = 0.0

    def load_store(self, file, read_version_num, is_reading):
        self.year = ts_read_write(file, 'Year', self.year, int, is_reading)
        self.value = ts_read_write(file, 'Value', self.value, float, is_reading)


class TSRArray:
    def __init__(self):
        self.records = []

    def add_record(self, year, value):
        self.records.append(TSRecord(year, value))

    def get_records(self):
        return self.records

@dataclass
class TSRecord:
    year: int = 0
    value: float = 0.0


def linear_interpolate(old_val, new_val, old_time, new_time, interp_time, extrapolate=False):
    """
    Interpolates to interp_time between two points defined by (old_val, old_time) and (new_val, new_time).
    """
    if not extrapolate:
        if interp_time > new_time or interp_time < old_time:
            raise ValueError(
                "Interpolation Timestamp Error: interp_time is out of bounds and extrapolation is not allowed.")

    if new_time == old_time:
        raise ValueError(
            "Interpolation Error: old_time and new_time cannot be the same as it results in division by zero.")

    # Calculating the slope (dy/dx)
    slope = (new_val - old_val) / (new_time - old_time)

    # Interpolating the value at interp_time
    return old_val + slope * (interp_time - old_time)


@dataclass
class TFWFlow:
    name: str = 'New Flow'
    extent_only: bool = False
    use_turbidities: bool = False
    sw_ppt: float = 30.0
    fw_ppt: float = 0.0
    origin_km: float = 0.0
    n_turbidities: int = 1
    turbidities: List[TSRecord] = field(default_factory=list)
    n_flows: int = 0
    mean_flow: List[TSRecord] = field(default_factory=list)
    mannings_n: float = 0.0
    fl_width: float = 0.0
    init_salt_wedge_slope: float = 0.0
    ts_elev: float = 0.0

    # Calculations Below, parameters above
    subsite_arr: List[Optional[TSubSite]] = field(default_factory=list)
    ftables: List[np.ndarray] = field(default_factory=list)
    d2origin: List[float] = field(default_factory=list)
    num_cells: List[int] = field(default_factory=list)

    poly: Optional[TPolygon] = None

    sw_z_mouth: float = 0.0
    sw_z_data: float = 0.0
    origin_arr: List[DPoint] = field(default_factory=list)
    mouth_arr: List[DPoint] = field(default_factory=list)
    num_segments: int = 0
    estuary_length: float = 0.0
    river_mouth_index: int = 0

    plume: List[float] = field(default_factory=list)
    vect: List[float] = field(default_factory=list)

    midpts: List[DPoint] = field(default_factory=list)
    max_rn: int = 0
    test_min: float = 0.0
    test_range: float = 0.0

    ocean_subsite: Optional[TSubSite] = None
    retention_initialized: bool = False

    ret_time: List[np.ndarray] = field(default_factory=list)
    water_z: List[np.ndarray] = field(default_factory=list)
    xs_salinity: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        # Initialize Turbidities with default values
        if not self.turbidities:  # If turbidities list is empty
            self.turbidities.append(TSRecord(year=2000, value=0))

        # Initialize the polygon object if needed
        self.poly = TPolygon() if self.poly is None else self.poly

        # Initialize lists or complex types
        self.mean_flow = self.mean_flow if self.mean_flow is not None else []
        self.origin_arr = self.origin_arr if self.origin_arr is not None else []
        self.mouth_arr = self.mouth_arr if self.mouth_arr is not None else []
        self.origin_line = []  # Array of DLine origin is perpendicular to line crossing start point,
        self.plume_line = []  # plume is perpendicular to line crossing end point.
        self.plume = self.plume if self.plume is not None else []
        self.vect = self.vect if self.vect is not None else []
        self.midpts = self.midpts if self.midpts is not None else []

        # Initialize ret_time, water_z, and xs_salinity arrays
        if not self.ret_time:
            self.ret_time = [np.array([]) for _ in range(NUM_SAL_METRICS)]
        if not self.water_z:
            self.water_z = [np.array([]) for _ in range(NUM_SAL_METRICS)]
        if not self.xs_salinity:
            self.xs_salinity = [np.array([]) for _ in range(NUM_SAL_METRICS)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def load_store(self, file, read_version_num, is_reading):
        self.name = ts_read_write(file, 'Name', self.name, str, is_reading)

        self.use_turbidities = ts_read_write(file, 'UseTurbidities', self.use_turbidities, bool, is_reading)
        self.sw_ppt = ts_read_write(file, 'SWPPT', self.sw_ppt, float, is_reading)
        self.fw_ppt = ts_read_write(file, 'FWPPT', self.fw_ppt, float, is_reading)
        self.origin_km = ts_read_write(file, 'Origin_KM', self.origin_km, float, is_reading)

        if is_reading:
            self.n_turbidities = ts_read_write(file, 'NTurbidities', None, int, is_reading)
            self.turbidities = [TSRecord() for _ in range(self.n_turbidities)]
        else:
            ts_read_write(file, 'NTurbidities', len(self.turbidities), int, is_reading)

        for i in range(self.n_turbidities):
            self.turbidities[i].year = ts_read_write(file, 'Turbidities[i].Year', self.turbidities[i].year, int,
                                                     is_reading)
            self.turbidities[i].value = ts_read_write(file, 'Turbidities[i].Value', self.turbidities[i].value, float,
                                                      is_reading)

        if is_reading:
            self.n_flows = ts_read_write(file, 'NFlows', None, int, is_reading)
            self.mean_flow = [TSRecord() for _ in range(self.n_flows)]
        else:
            ts_read_write(file, 'NFlows', len(self.mean_flow), int, is_reading)

        for i in range(self.n_flows):
            self.mean_flow[i].year = ts_read_write(file, 'MeanFlow[i].Year', self.mean_flow[i].year, int, is_reading)
            self.mean_flow[i].value = ts_read_write(file, 'MeanFlow[i].Value', self.mean_flow[i].value, float, is_reading)

        self.mannings_n = ts_read_write(file, 'ManningsN', self.mannings_n, float, is_reading)
        self.fl_width = ts_read_write(file, 'FlWidth', self.fl_width, float, is_reading)
        self.init_salt_wedge_slope = ts_read_write(file, 'SaltWedgeSlope', self.init_salt_wedge_slope, float, is_reading)
        self.extent_only = ts_read_write(file, 'ExtentOnly', self.extent_only, bool, is_reading)
        self.ts_elev = ts_read_write(file, 'TSElev', self.ts_elev, float, is_reading)

        if self.poly:
            if is_reading: self.poly.load(file, read_version_num, True)
            else: self.poly.store(file)

        self.num_segments = ts_read_write(file, 'NumSegments', self.num_segments, int, is_reading)
        if is_reading:
            self.origin_arr = [DPoint(0, 0) for _ in range(self.num_segments)]
            self.mouth_arr = [DPoint(0, 0) for _ in range(self.num_segments)]
        for i in range(self.num_segments):
            x = ts_read_write(file, 'OriginArr[i-1].x', self.origin_arr[i].x, int, is_reading)
            y = ts_read_write(file, 'OriginArr[i-1].y', self.origin_arr[i].y, int, is_reading)
            self.origin_arr[i] = DPoint(x, y)

            x = ts_read_write(file, 'MouthArr[i-1].x', self.mouth_arr[i].x, int, is_reading)
            y = ts_read_write(file, 'MouthArr[i-1].y', self.mouth_arr[i].y, int, is_reading)
            self.mouth_arr[i] = DPoint(x, y)

        self.sw_z_data = ts_read_write(file, 'SW_Z_Data', self.sw_z_data, float, is_reading)
        self.sw_z_mouth = ts_read_write(file, 'SW_Z_Mouth', self.sw_z_mouth, float, is_reading)

    def salt_height(self, tide_height, slr, r_seg, yr):
        or2 = self.origin_km
        if or2 < 0.05:
            or2 = self.max_rn * SLICE_INCREMENT

        or2 -= (tide_height + slr) * 4.82  # Relationship derived from GA data.
        mrn = or2 / SLICE_INCREMENT
        return tide_height - (mrn - r_seg) * SLICE_INCREMENT * 1000 * self.salt_wedge_slope_by_year(yr)

    def salt_wedge_slope_by_year(self, yr):
        delta_flow = self.flow_by_year(yr) - self.flow_by_year(0)
        return self.init_salt_wedge_slope + delta_flow * 2.8e-7

    def _time_series_value_by_year(self, series, n_entries, year):
        # Generic method to retrieve or interpolate time series values by year.
        if n_entries == 0:
            raise Exception("Time-Series Data Missing.")
        if n_entries == 1 or year < series[0].year:
            return series[0].value
        for i in range(n_entries):
            if year == series[i].year:
                return series[i].value
            if year < series[i].year:
                return linear_interpolate(
                    series[i - 1].value, series[i].value,
                    series[i - 1].year, series[i].year, year
                )
        return series[-1].value

    def flow_by_year(self, yr):
        return self._time_series_value_by_year(self.mean_flow, self.n_flows, yr)

    def turbidity_by_year(self, yr):
        if not self.use_turbidities:
            return 1.0
        return self._time_series_value_by_year(self.turbidities, self.n_turbidities, yr)

    def elev_by_volume(self, vol_wat, r_seg):
        # Access the f_table array inside the TFTable object for comparison and operations
        if self.ftables[r_seg][0] > vol_wat:
            return self.test_min
        elif self.ftables[r_seg][-1] < vol_wat:
            return self.test_min + self.test_range
        for i in range(1, N_FTABLE):
            if self.ftables[r_seg][i] > vol_wat:
                min_elev = self.test_min + (self.test_range / (N_FTABLE - 1)) * (i - 1)
                max_elev = self.test_min + (self.test_range / (N_FTABLE - 1)) * i
                min_vol = self.ftables[r_seg][i - 1]
                max_vol = self.ftables[r_seg][i]
                return linear_interpolate(min_elev, max_elev, min_vol, max_vol, vol_wat, False)
        return 0

    def volume_by_elev(self, elev, r_seg):
        if elev < self.test_min:
            return self.ftables[r_seg][0]
        elif elev > self.test_min + self.test_range:
            return self.ftables[r_seg][N_FTABLE - 1]
        for i in range(1, N_FTABLE):
            ft_height = self.test_min + (self.test_range / (N_FTABLE - 1)) * i
            if elev < ft_height:
                min_elev = self.test_min + (self.test_range / (N_FTABLE - 1)) * (i - 1)
                max_elev = ft_height
                min_vol = self.ftables[r_seg][i - 1]
                max_vol = self.ftables[r_seg][i]
                return linear_interpolate(min_vol, max_vol, min_elev, max_elev, elev, False)
        return 0


@dataclass
class TTimeSerSLR:
    name: str = ""
    base_year: int = 2000
    n_years: int = 0
    
    def __init__(self, example_sim: int = 0):
        self.slr_arr = []  # Create an empty list to store TSRecord instances
        self.run_now = (example_sim == 0)
        
        if example_sim == 1:
            self.name = "Example 1M"
            self.n_years = 4
            self.slr_arr = [
                TSRecord(2025, 0.10),
                TSRecord(2050, 0.25),
                TSRecord(2075, 0.45),
                TSRecord(2100, 1.0)
            ]
            
        if example_sim == 2:
            self.name = "Example 1.5M"
            self.n_years = 4
            self.slr_arr = [
                TSRecord(2025, 0.15),
                TSRecord(2050, 0.4),
                TSRecord(2075, 0.75),
                TSRecord(2100, 1.5)
            ]

    def load_store(self, file, read_version_num, is_reading):
        self.name = ts_read_write(file, 'Name', self.name, str, is_reading)
        self.base_year = ts_read_write(file, 'BaseYear', self.base_year, int, is_reading)
        self.n_years = ts_read_write(file, 'NYears', self.n_years, int, is_reading)
        self.run_now = ts_read_write(file, 'RunNow', self.run_now, bool, is_reading)

        if is_reading:
            self.slr_arr = [TSRecord() for _ in range(self.n_years)]
        for i in range(self.n_years):
            self.slr_arr[i].year = ts_read_write(file, 'SLRArr[i].Year', self.slr_arr[i].year, int, is_reading)
            self.slr_arr[i].value = ts_read_write(file, 'SLRArr[i].Value', self.slr_arr[i].value, float, is_reading)


@dataclass
class DikeInfoRec:
    up_row: int
    up_col: int
    dist_origin: float


TDikeInfoArr = List[DikeInfoRec]


class ProtectScenario(Enum):
    NoProtect = 0
    ProtDeveloped = 1
    ProtAll = 2


LabelProtect = {
    ProtectScenario.NoProtect: "No Protect",
    ProtectScenario.ProtDeveloped: "Protect Developed",
    ProtectScenario.ProtAll: "Protect All"
}

NUM_CAT_COMPRESS = 3

# Define the structured dtype for CompressedCell
compressed_cell_dtype = np.dtype([
    # Core Parameters
    ('cats', 'i4', 3),  # Integer categories
    ('min_elevs', 'f4', NUM_CAT_COMPRESS),  # Single precision float
    ('widths', 'f4', NUM_CAT_COMPRESS),  # Single precision float
    ('tan_slope', 'f4'),  # Single float
    ('prot_dikes', 'bool'),  # Boolean for dikes protection
    ('elev_dikes', 'bool'),  # Boolean for elevation dikes
    ('max_fetch', 'f4'),  # Single float
    ('uplift', 'f4'),  # Single float
    ('mtl_minus_navd', 'f4'),  # Single float
    ('subsite_index', 'i2'),  # SmallInt for subsite index
    ('erosion_loss', 'f4'),  # Single float
    ('btf_erosion_loss', 'f4'),  # Single float

    # Only if Salinity is included
    ('sal', 'f4', NUM_SAL_METRICS),  # Array of salinity metrics
    ('sal_height_mllw', 'f4'),  # Single float

    # Only if Erosion Model is included
    ('imp_coeff', 'i4'),  # Impact coefficient
    ('pw', 'f4'),  # Wave power
    ('wp_erosion', 'f4'),  # Wave power erosion

    # Only if SAV Model is included
    ('prob_sav', 'f4'),  # Probability SAV
    ('d2mllw', 'f4'),  # Distance to MLLW
    ('d2mhhw', 'f4'),  # Distance to MHHW
    ('d2mouth', 'f4'),  # Distance to Mouth
])


def init_cell():
    # Create a single compressed cell with default values
    cell = np.empty((), dtype=compressed_cell_dtype)

    # Initialize categorical data and metrics
    cell['cats'] = np.zeros(NUM_CAT_COMPRESS, dtype='i4')
    cell['cats'][:] = BLANK
    cell['min_elevs'][:] = 999
    cell['widths'][:] = 0

    # Initialize single float metrics
    cell['tan_slope'] = 0.0
    cell['max_fetch'] = -1.0
    cell['sal_height_mllw'] = -99
    cell['sal'][:] = -999
    cell['uplift'] = 0
    cell['erosion_loss'] = 0
    cell['btf_erosion_loss'] = 0
    cell['mtl_minus_navd'] = -9999
    cell['d2mllw'] = -9999
    cell['d2mhhw'] = -9999
    cell['d2mouth'] = -9999
    cell['prob_sav'] = -9999
    cell['imp_coeff'] = -9999
    cell['pw'] = -9999
    cell['wp_erosion'] = 0

    # Boolean fields
    cell['prot_dikes'] = False
    cell['elev_dikes'] = False

    # Subsite index
    cell['subsite_index'] = -9999

    return cell


# Define any constants
BLOCKEXP = 14
BLOCKSIZE = 1 << BLOCKEXP  # This is 2**14 or 16384
BLOCKMASK = BLOCKSIZE - 1  # This is 16383
# min_hist = 0.0
max_hist = 35.0


@dataclass
class FractalRecord:
    n_box: List[int] = field(default_factory=lambda: [0, 0])
    min_fd_row: List[int] = field(default_factory=lambda: [0, 0])
    min_fd_col: List[int] = field(default_factory=lambda: [0, 0])
    max_fd_row: List[int] = field(default_factory=lambda: [0, 0])
    max_fd_col: List[int] = field(default_factory=lambda: [0, 0])
    fractal_p: List[float] = field(default_factory=lambda: [0.0, 0.0])
    fractal_d: List[float] = field(default_factory=lambda: [0.0, 0.0])
    shore_protect: float = 0.0


def translate_inund_num(inund_in):
    # Translate into alternative integer output for rasters
    if inund_in == 8:
        return 0    # Open water
    elif inund_in == 30:
        return 1    # H1 Elev 30 d inundation
    elif inund_in == 60:
        return 2    # H2 Elev 60 d inundation
    elif inund_in == 90:
        return 3    # H3 Elev 90 d inundation
    elif inund_in == 120:
        return 4    # H4 Elev Storm inundation 1
    elif inund_in == 150:
        return 5    # H5 Elev Storm inundation 2
    elif inund_in == 2:
        return 6    # Above Storm elevation
    elif inund_in == 4:
        return 7    # Below Storm elevation but not connected
    elif inund_in == 7:
        return 8    # Protected by dikes
    elif inund_in == 10:
        return 10   # Overtopped Dike Location
    elif inund_in == 9:
        return BLANK  # No data/blank
    else:
        return 11

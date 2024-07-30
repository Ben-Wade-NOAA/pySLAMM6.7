from app_global import *
from numba import jit


@jit(nopython=True)
def get_cell_cat(cl: compressed_cell_dtype):
    max_area = 0.0
    cell_cat = -1
    for i in range(NUM_CAT_COMPRESS):
        cat_area = cl['widths'][i]
        if cat_area > max_area:
            cell_cat = cl['cats'][i]
            max_area = cat_area
    return cell_cat


@jit(nopython=True)
def cell_width(cl: compressed_cell_dtype, cat: int):
    for i in range(NUM_CAT_COMPRESS):
        if cl['cats'][i] == cat:
            return cl['widths'][i]
    return 0.0


@jit(nopython=True)
def cat_elev(cl: compressed_cell_dtype, cat: int):
    for i in range(NUM_CAT_COMPRESS):
        if cl['cats'][i] == cat:
            return cl['min_elevs'][i]
    return 999.0


@jit(nopython=True)
def set_cat_elev(cl: compressed_cell_dtype, cat: int, set_val: float):
    for i in range(NUM_CAT_COMPRESS):
        if cl['cats'][i] == cat:
            cl['min_elevs'][i] = set_val
            return

@jit(nopython=True)
def set_cell_width(cl, cat, set_val):
    min_width = 99999.0

    # First loop to find the category and set its width, and also find the minimum width
    for i in range(NUM_CAT_COMPRESS):
        if cl['cats'][i] == cat:
            cl['widths'][i] = set_val
            return
        if cl['widths'][i] < min_width:
            min_width = cl['widths'][i]

    # Second loop to assign the class to the category with the minimum width
    for i in range(NUM_CAT_COMPRESS):
        if cl['widths'][i] == min_width:
            if min_width < set_val:   # keep maximum of two minimum categories
                cl['cats'][i] = cat
                cl['min_elevs'][i] = 999.0
            cl['widths'][i] = set_val + min_width
            return


@jit(nopython=True)
def get_min_elev(cl: compressed_cell_dtype):
    min_elev = 999.0
    for i in range(NUM_CAT_COMPRESS):
        if cl['widths'][i] > 0 and cl['min_elevs'][i] < min_elev:
            min_elev = cl['min_elevs'][i]
    return min_elev


@jit(nopython=True)
def get_avg_elev(cl: compressed_cell_dtype, scale: float):
    min_elev = get_min_elev(cl)
    slope_adjustment = (scale * 0.5) * cl['tan_slope']
    return min_elev + slope_adjustment


def float_to_word(in_var):
    if in_var < -10:
        return 0
    elif in_var > 55.5:
        return 55500
    return round((in_var + 10) * 1000)


def word_to_float(in_var):
    return (in_var / 1000) - 10

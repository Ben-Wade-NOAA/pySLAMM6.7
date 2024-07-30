import pandas as pd
from app_global import *


class SalRec:
    def __init__(self, rslr=0.0, q=0.0):
        self.rslr = rslr
        self.q = q


class TSalArray:
    def __init__(self, file_name, st):
        self.n_sal_recs = 0
        self.n_sal_stations = 0
        self.s_locs = []
        self.origin_lines = []
        self.plume_lines = []
        self.mid_points = []
        self.pts: TSite = st
        self.sal_recs = []
        self.sal_pts = []
        self.read_sal_array_from_excel(file_name, st)

    def read_sal_array_from_excel(self, file_name, st: TSite):
        self.pts = st
        df = pd.read_excel(file_name, sheet_name=0, header=None)

        self.n_sal_stations = int(df.iloc[1, 0])

        self.s_locs = [DPoint(0, 0) for _ in range(self.n_sal_stations)]
        self.origin_lines = [DLine(DPoint(0, 0), DPoint(0, 0)) for _ in range(self.n_sal_stations - 1)]
        self.plume_lines = [DLine(DPoint(0, 0), DPoint(0, 0)) for _ in range(self.n_sal_stations - 1)]
        self.mid_points = [DPoint(0, 0) for _ in range(self.n_sal_stations - 1)]

        for r_row in range(self.n_sal_stations):
            x_loc = float(df.iloc[3 + r_row, 1])
            y_loc = float(df.iloc[3 + r_row, 2])
            self.s_locs[r_row] = self.slam_proj_to_xy(x_loc, y_loc)
            if r_row > 0:
                self.set_origin_plume_midline(r_row - 1)

        df2 = pd.read_excel(file_name, sheet_name=1)
        self.n_sal_recs = len(df2)
        self.sal_recs = [SalRec() for _ in range(self.n_sal_recs)]
        self.sal_pts = np.zeros((self.n_sal_recs, self.n_sal_stations))

        for r_row in range(self.n_sal_recs):
            self.sal_recs[r_row].rslr = float(df2.iloc[r_row, 0])
            self.sal_recs[r_row].q = float(df2.iloc[r_row, 1])
            self.sal_pts[r_row, :] = [float(df2.iloc[r_row, 2 + col]) for col in range(self.n_sal_stations)]

    def set_origin_plume_midline(self, index):
        row_vector = self.s_locs[index + 1].y - self.s_locs[index].y
        col_vector = self.s_locs[index + 1].x - self.s_locs[index].x

        self.origin_lines[index].p1 = self.s_locs[index]
        self.plume_lines[index].p1 = self.s_locs[index + 1]

        if col_vector != 0:
            self.origin_lines[index].p2.y = self.s_locs[index].y + col_vector
            self.origin_lines[index].p2.x = self.s_locs[index].x - row_vector
            self.plume_lines[index].p2.y = self.s_locs[index + 1].y + col_vector
            self.plume_lines[index].p2.x = self.s_locs[index + 1].x - row_vector
        else:
            self.origin_lines[index].p2.y = self.s_locs[index].y
            self.origin_lines[index].p2.x = self.s_locs[index].x - 1
            self.plume_lines[index].p2.y = self.s_locs[index + 1].y
            self.plume_lines[index].p2.x = self.s_locs[index + 1].x - 1

        self.mid_points[index].x = (self.s_locs[index].x + self.s_locs[index + 1].x) / 2
        self.mid_points[index].y = (self.s_locs[index].y + self.s_locs[index + 1].y) / 2

    def slam_proj_to_xy(self, x, y):
        delta_x = (x - self.pts.llx_corner) / self.pts.site_scale
        delta_y = (y - self.pts.lly_corner) / self.pts.site_scale
        result = DPoint(0, 0)
        result.x = delta_x
        result.y = self.pts.rows - delta_y
        return result

    def get_salinity(self, x, y, sal_val):
        result = True
        near_line = -1
        best_distance = float('inf')
        pt = DPoint(x, y)

        for ss in range(self.n_sal_stations - 1):
            if not (self.cross_segment(pt, self.mid_points[ss], self.origin_lines[ss]) or
                    self.cross_segment(pt, self.mid_points[ss], self.plume_lines[ss])):
                dl = DLine(self.s_locs[ss], self.s_locs[ss + 1])
                min_distance = self.abs_distance_pt_to_line(dl, pt)
                if min_distance < best_distance:
                    best_distance = min_distance
                    near_line = ss

        if near_line > -1:
            d2origin_line = self.abs_distance_pt_to_line(self.origin_lines[near_line], pt)
            d2plume_line = self.abs_distance_pt_to_line(self.plume_lines[near_line], pt)
            weight_first = d2plume_line / (d2origin_line + d2plume_line)
            result = (sal_val[near_line] * weight_first) + (sal_val[near_line + 1] * (1 - weight_first))
        else:
            best_distance = float('inf')
            for ss in range(self.n_sal_stations):
                min_distance = self.distance_2_pts(self.s_locs[ss], pt)
                if min_distance < best_distance:
                    best_distance = min_distance
                    result = sal_val[ss]

        return result

    def extract_salinity_record(self, slamm_rslr, slamm_q):
        result = np.zeros(self.n_sal_stations)
        min_rslr = float('-inf')
        max_rslr = float('inf')
        min_row = max_row = -1
        q_compare = False

        for row in range(self.n_sal_recs):
            if np.isclose(slamm_q, self.sal_recs[row].q, atol=0.001):
                q_compare = True
                if slamm_rslr >= self.sal_recs[row].rslr > min_rslr:
                    min_rslr = self.sal_recs[row].rslr
                    min_row = row
                if slamm_rslr <= self.sal_recs[row].rslr < max_rslr:
                    max_rslr = self.sal_recs[row].rslr
                    max_row = row

        if not q_compare:
            raise ValueError('Salinity Excel flows and SLAMM freshwater mean flow do not match.')

        if min_row == -1 or max_row == -1:
            raise ValueError('SLR less than Excel min or exceeds Excel max')

        for col in range(self.n_sal_stations):
            if max_row != min_row:
                result[col] = ((self.sal_pts[max_row][col] - self.sal_pts[min_row][col]) * slamm_rslr +
                               self.sal_pts[min_row][col] * max_rslr - self.sal_pts[max_row][col] * min_rslr) / (
                                      max_rslr - min_rslr)
            else:
                result[col] = self.sal_pts[max_row][col]

        return result

    def distance_pt_to_line(self, ln: DLine, p: DPoint):
        a = ln.p1.x - ln.p2.x
        b = ln.p2.y - ln.p1.y
        c = ln.p1.y * ln.p2.x - ln.p2.y * ln.p1.x
        return (a * p.y + b * p.x + c) / np.sqrt(b ** 2 + a ** 2)

    def abs_distance_pt_to_line(self, ln, p):
        return abs(self.distance_pt_to_line(ln, p))

    def distance_2_pts(self, p1, p2):
        return np.sqrt((p2.y - p1.y) ** 2 + (p2.x - p1.x) ** 2)

    def cross_segment(self, p1, p2, ln):
        return (self.distance_pt_to_line(ln, p1) * self.distance_pt_to_line(ln, p2)) <= 0

    def __del__(self):
        del self.s_locs
        del self.sal_recs
        del self.sal_pts

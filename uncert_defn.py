"""
This module defines classes and functions for handling uncertainty distributions in the pySLAMM application.
Classes:
    DistType(Enum): Enumeration of different distribution types.
    TInputDist: Represents an input distribution with various methods for handling distributions.
    UncertDraw: Represents a single uncertainty draw with value, random draw, and interval number.
    TSensParam(TInputDist): Inherits from TInputDist and adds methods for setting and restoring values for sensitivity parameters.
Constants:
    NUM_UNCERT_PARAMS (int): Number of uncertainty parameters.
    PRE_ACCR (int): Pre-accretion constant.
Functions:
    TInputDist.__init__(self, id_num, p_slamm_sim, all_subs, ss_num): Initializes a TInputDist object.
    TInputDist.load_store(self, file, read_version_num, is_reading): Loads or stores distribution data from/to a file.
    TInputDist.trunc_icdf(self, prob): Calculates the inverse cumulative distribution function for truncated distributions.
    TInputDist.icdf(self, prob): Calculates the inverse cumulative distribution function based on the distribution type.
    TInputDist.trunc_cdf(self, x_val): Calculates the cumulative distribution function for truncated distributions.
    TInputDist.cdf(self, x_val): Calculates the cumulative distribution function based on the distribution type.
    TInputDist.get_name(self, include_ss): Retrieves the name of the distribution, optionally including subsite information.
    TInputDist.name_from_index(indx): Retrieves the name associated with the distribution index.
    TInputDist.z_map_index(self) -> int: Returns the index for elevation data uncertainty.
    TInputDist.get_value_object(self, ssn): Returns a tuple with the object, attribute name, and index for the given subsite number.
    TInputDist.get_value(self): Retrieves the value for the subsite or global subsite if all_subsites is true.
    TInputDist.update_value(self, subsite_num, operation, value=None): Updates the attribute based on the provided operation.
    TInputDist.set_values(self, multiplier): Sets values for the subsite or global subsite.
    TInputDist.restore_values(self): Restores values for the subsite or global subsite.
    TInputDist.summarize_dist(self): Summarizes the distribution parameters as a string.
    TSensParam.set_values(self, multiplier): Sets values for all relevant subsites or globally.
    TSensParam.restore_values(self): Restores values for all relevant subsites or globally.
"""

from app_global import *
from enum import Enum
from calc_dist import *



NUM_UNCERT_PARAMS = 51
PRE_ACCR = 27


class DistType(Enum):
    Triangular = 0
    Uniform = 1
    Normal = 2
    LogNormal = 3
    Elev_Map = 4


class TInputDist:
    def __init__(self, id_num, p_slamm_sim, all_subs, ss_num):
        self.is_subsite_parameter = (id_num > 2) and (id_num != 5)
        self.d_name = self.name_from_index(id_num)
        self.notes = ""
        self.id_num = id_num

        self.all_subsites = all_subs or (id_num == 1)   # SLR (IDNum=1) always pertains to all subsites
        self.subsite_num = max(0, ss_num)

        self.tss = p_slamm_sim
        self.dist_type = DistType.Normal
        self.parm = [1.0, 0.4, 1.3, 0]  # Default values

        if id_num in [2, 5]:  # Custom logic for certain IDs
            self.dist_type = DistType.Elev_Map
            self.parm = [0.12, 0.245, 0, 0]

        self.display_cdf = False
        self.draws = None
        self.point_ests = []
        self.is_subsite_parameter = False

    def load_store(self, file, read_version_num, is_reading):
        self.d_name = ts_read_write(file, 'DistName', self.d_name, str, is_reading)
        self.notes = ts_read_write(file, 'Notes', self.notes, str, is_reading)
        self.id_num = ts_read_write(file, 'IDNum', self.id_num, int, is_reading)

        self.all_subsites = ts_read_write(file, 'AllSubSites', self.all_subsites, bool, is_reading)
        self.subsite_num = ts_read_write(file, 'SubsiteNum', self.subsite_num, int, is_reading)

        dist_type_int = ts_read_write(file, 'DistType', self.dist_type.value if not is_reading else None, int,
                                      is_reading)
        self.dist_type = DistType(dist_type_int) if is_reading else self.dist_type

        self.parm = [
            ts_read_write(file, 'Parm(1)', self.parm[0], float, is_reading),
            ts_read_write(file, 'Parm(2)', self.parm[1], float, is_reading),
            ts_read_write(file, 'Parm(3)', self.parm[2], float, is_reading),
            ts_read_write(file, 'Parm(4)', self.parm[3], float, is_reading)
        ]

        self.display_cdf = ts_read_write(file, 'DisplayCDF', self.display_cdf, bool, is_reading)

    def trunc_icdf(self, prob):
        """ Calculate the icdf for norm and lognorm distributions truncated to zero. """
        if self.dist_type in [DistType.Triangular, DistType.Uniform]:
            new_prob = prob  # already limited to max of zero through interface
        else:
            prob_min = self.cdf(0)
            new_prob = prob_min + prob * (1 - prob_min)
        return self.icdf(new_prob)

    def icdf(self, prob):
        """ Calculate the inverse cumulative distribution function based on the distribution type. """
        res = ERROR_VALUE
        if self.dist_type == DistType.Normal:
            res = icdf_normal(prob, self.parm[0], self.parm[1])
        elif self.dist_type == DistType.Triangular:
            res = icdf_triangular(prob, self.parm[1], self.parm[2], self.parm[0])
        elif self.dist_type == DistType.LogNormal:
            res = icdf_lognormal(prob, math.exp(self.parm[0]), math.exp(self.parm[1]))
        elif self.dist_type == DistType.Uniform:
            res = icdf_uniform(prob, self.parm[0], self.parm[1])

        if res == ERROR_VALUE:
            raise ValueError('Distribution Error! ICDF Called with Invalid Parameters.')
        return res

    def trunc_cdf(self, x_val):
        """ Calculate the cumulative distribution function for norm and lognorm distributions truncated to zero. """
        cdf_x_val = self.cdf(x_val)
        if self.dist_type in [DistType.Triangular, DistType.Uniform]:
            return cdf_x_val
        else:
            cdf_zero = self.cdf(0)
            return (cdf_x_val - cdf_zero) / (1 - cdf_zero)

    def cdf(self, x_val):
        """ Calculate the cumulative distribution function based on the distribution type. """
        res = ERROR_VALUE
        if self.dist_type == DistType.Triangular:
            res = cdf_triangular(x_val, self.parm[1], self.parm[2], self.parm[0])
        elif self.dist_type == DistType.Normal:
            res = cdf_normal(x_val, self.parm[0], self.parm[1])
        elif self.dist_type == DistType.LogNormal:
            res = cdf_lognormal(x_val, math.exp(self.parm[0]), math.exp(self.parm[1]))
        elif self.dist_type == DistType.Uniform:
            res = cdf_uniform(x_val, self.parm[0], self.parm[1])

        if res == ERROR_VALUE:
            raise ValueError('Distribution Error! CDF Called with Invalid Parameters.')
        return res

    def get_name(self, include_ss):
        result = self.d_name
        if include_ss:
            if self.all_subsites:
                result += ' (All Subsites*)'
            elif self.subsite_num == 0:
                result += ' (Global Subsite) '
            else:
                result += f' (Subsite {self.subsite_num})'
        return result

    @staticmethod
    def name_from_index(indx):
        """ Retrieve the name associated with the distribution index. """
        uncertaccr = NUM_ROWS_ACCR - 2  # delete "use" boolean and notes string

        distribution_names = {
            1: 'SLR by 2100 (mult.)',
            2: 'DEM Uncertainty (RMSE in m)',
            3: 'Historic trend (mult.)',
            4: 'NAVD88 - MTL (mult.)',
            5: 'NAVD88 Uncert Map (RMSE in m)',
            6: 'GT Great Diurnal Tide Range (m)',
            7: 'Salt Elev. (mult.)',
            8: 'Marsh Erosion (mult.)',
            9: 'Swamp Erosion (mult.)',
            10: 'T.Flat Erosion (mult.)',
            11: 'Reg. Flood Marsh Accr (mult.)',
            12: 'Irreg. Flood Marsh Accr (mult.)',
            13: 'Tidal Fresh Marsh Accr (mult.)',
            14: 'Inland-Fresh Marsh Accr (mult.)',
            15: 'Mangrove Accr (mult.)',
            16: 'Tidal Swamp Accr (mult.)',
            17: 'Swamp Accretion (mult.)',
            18: 'Beach Sed. Rate (mult.)',
            19: 'Wave Erosion Alpha',
            20: 'Wave Erosion Avg. Shallow Depth (m)',
            21: 'Elev. Beach Crest (m)',
            22: 'Lagoon Beta (frac)',
            23: '<unused superseded>',
            24: '<unused superseded>',
            25: '<unused superseded>',
            26: 'Irreg. Flood Marsh Collapse (mult.)',
            27: 'Reg. Flood Marsh Collapse (mult.)'
        }

        result = distribution_names.get(indx, '')

        if indx > PRE_ACCR:
            accr_num = (indx - PRE_ACCR - 1) // uncertaccr
            an2 = (indx - PRE_ACCR - 1) % uncertaccr
            accr_suffixes = [
                'Max. Accr. (mult.)',
                'Min. Accr. (mult.)',
                'Elev a coeff. (mult.)',
                'Elev b coeff. (mult.)',
                'Elev c coeff. (mult.)',
                'Elev d coeff. (mult.)'
            ]
            result = ACCR_NAMES[accr_num] + accr_suffixes[an2]

        return result

    def z_map_index(self) -> int:
        if self.id_num == 2:
            return 0   # elevation data uncert
        return 1       # NAVD data uncert

    def get_value_object(self, ssn):
        # return a tuple with the object the attribute name and the index

        uncert_accr = NUM_ROWS_ACCR - 2  # delete "use" boolean and notes string

        # Determine the object `p_s` based on `ssn`
        if self.id_num in [1, 2, 5]:
            p_s = None
        else:
            p_s = self.tss.site.global_site if ssn <= 0 else self.tss.site.subsites[ssn - 1]

        result = (None, None, None)  # Default to no valid object, attribute, or index
        if self.id_num == 1:
            if len(self.tss.custom_slr_array) > 0:
                result = (self.tss, 'custom_slr_array', 0)
        elif self.id_num in [2, 5]:
            result = (self, 'parm', 0)
        elif self.id_num == 3:
            result = (p_s, 'historic_trend', None)
        elif self.id_num == 4:
            result = (p_s, 'navd88mtl_correction', None)
        elif self.id_num == 6:
            result = (p_s, 'gtide_range', None)
        elif self.id_num == 7:
            result = (p_s, 'salt_elev', None)
        elif self.id_num == 8:
            result = (p_s, 'marsh_erosion', None)
        elif self.id_num == 9:
            result = (p_s, 'swamp_erosion', None)
        elif self.id_num == 10:
            result = (p_s, 'tflat_erosion', None)
        elif self.id_num == 11:
            result = (p_s, 'fixed_reg_flood_accr', None)
        elif self.id_num == 12:
            result = (p_s, 'fixed_irreg_flood_accr', None)
        elif self.id_num == 13:
            result = (p_s, 'fixed_tide_fresh_accr', None)
        elif self.id_num == 14:
            result = (p_s, 'inland_fresh_accr', None)
        elif self.id_num == 15:
            result = (p_s, 'mangrove_accr', None)
        elif self.id_num == 16:
            result = (p_s, 'tswamp_accr', None)
        elif self.id_num == 17:
            result = (p_s, 'swamp_accr', None)
        elif self.id_num == 18:
            result = (p_s, 'fixed_tf_beach_sed', None)
        elif self.id_num == 19:
            result = (p_s, 'we_alpha', None)
        elif self.id_num == 20:
            result = (p_s, 'we_avg_shallow_depth', None)
        elif self.id_num == 21:
            result = (p_s, 'zbeach_crest', None)
        elif self.id_num == 22:
            result = (p_s, 'lbeta', None)
        elif self.id_num in [23, 24, 25]:
            result = (None, None, None)
        elif self.id_num == 26:
            result = (p_s, 'ifm2rfm_collapse', None)
        elif self.id_num == 27:
            result = (p_s, 'rfm2tf_collapse', None)

        if self.id_num > PRE_ACCR:
            accr_num = (self.id_num - PRE_ACCR - 1) // uncert_accr
            an2 = (self.id_num - PRE_ACCR - 1) % uncert_accr
            accr_attributes = [
                'max_accr', 'min_accr', 'accr_a', 'accr_b', 'accr_c', 'accr_d'
            ]
            if an2 < len(accr_attributes):
                result = (p_s, accr_attributes[an2], accr_num)

        return result

    def get_value(self):
        # retrieves the value for the subsite or the global subsite if all_subsites is true and returns the value
        ssn = 0 if self.all_subsites else self.subsite_num
        obj, attr, idx = self.get_value_object(ssn)
        if obj and attr:
            if idx is not None:
                return getattr(obj, attr)[idx]  # Accessing list-type attributes by index
            else:
                return getattr(obj, attr)  # Accessing direct attributes
        return None  # In case no valid attribute or object found

    def update_value(self, subsite_num, operation, value=None):
        """General method to update the attribute based on the operation provided."""
        obj, attr, idx = self.get_value_object(subsite_num)

        if obj and attr:
            if idx is not None:
                current_value = getattr(obj, attr)[idx]
                if operation == 'set':
                    self.point_ests[subsite_num] = current_value
                    getattr(obj, attr)[idx] = current_value * value
                elif operation == 'restore':
                    getattr(obj, attr)[idx] = self.point_ests[subsite_num]
            else:
                current_value = getattr(obj, attr)
                if operation == 'set':
                    self.point_ests[subsite_num] = current_value
                    setattr(obj, attr, current_value * value)
                elif operation == 'restore':
                    setattr(obj, attr, self.point_ests[subsite_num])

    def set_values(self, multiplier):
        if self.subsite_num < 0 or not self.is_subsite_parameter:
            self.subsite_num = 0
        self.point_ests = [0] * (self.tss.site.n_subsites + 1)
        self.update_value(self.subsite_num, 'set', multiplier)

    def restore_values(self):
        if not self.is_subsite_parameter:
            self.subsite_num = 0
        self.update_value(self.subsite_num, 'restore')

    def summarize_dist(self):
        parm_info = [''] * 5

        if self.dist_type == DistType.Triangular:
            parm_info = ['Triangular', 'Most Likely', 'Minimum', 'Maximum', '<unused>']
        elif self.dist_type == DistType.Normal:
            parm_info = ['Normal', 'Mean', 'Std. Deviation', '<unused>', '<unused>']
        elif self.dist_type == DistType.LogNormal:
            parm_info = ['LogNormal', 'Mean', 'Std. Deviation', '<unused>', '<unused>']
        elif self.dist_type == DistType.Uniform:
            parm_info = ['Uniform', 'Minimum', 'Maximum', '<unused>', '<unused>']
        elif self.dist_type == DistType.Elev_Map:
            parm_info = ['Elev. Map', 'R.M.S.E.', 'P. Val.', '<unused>', '<unused>']
        else:
            parm_info = None

        answer = f"{parm_info[0]} Distribution: "
        if self.dist_type == DistType.Elev_Map:
            answer = "Elev. Map: "

        for i in range(1, 5):
            if parm_info[i] != '<unused>':
                answer += f"{parm_info[i]}={self.parm[i - 1]:.4f}; "

        # Remove the last semicolon and add a period instead
        answer = answer.rstrip('; ') + '.'

        return answer


class UncertDraw:
    def __init__(self, value, random_draw, interval_num):
        self.value = value
        self.random_draw = random_draw
        self.interval_num = interval_num


class TSensParam(TInputDist):  # Inherits from TInputDist

    def set_values(self, multiplier):
        """Set values for all relevant subsites or globally."""
        if not self.is_subsite_parameter:
            top = 0  # If not subsite parameter, only modify global setting
        else:
            top = self.tss.site.n_subsites  # Affect all subsites

        self.point_ests = [0] * (top + 1)  # Initialize storage for original values

        for i in range(top + 1):
            self.update_value(i, 'set', multiplier)  # Use the helper function to set values

    def restore_values(self):
        """Restore values for all relevant subsites or globally."""
        if not self.is_subsite_parameter:
            top = 0  # If not subsite parameter, only restore global setting
        else:
            top = self.tss.site.n_subsites

        for i in range(top + 1):
            self.update_value(i, 'restore')  # Use the helper function to restore values



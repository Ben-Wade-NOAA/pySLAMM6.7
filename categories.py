from app_global import *
from utility import cell_width


@dataclass
class TCategory:
    gis_number: int = -99
    text_name: str = 'Blank'
    is_open_water: bool = False
    is_tidal: bool = False
    is_non_tidal_wetland: bool = False
    is_dryland: bool = False
    is_developed: bool = False
    color: str = 'clWhite'  # Assuming 'clWhite' is defined elsewhere as a constant or in a colors module
    agg_cat: AggCategories = AggCategories.AggBlank
    use_ifm_collapse: bool = False
    use_rfm_collapse: bool = False
    use_wave_erosion: bool = False
    inundate_to: int = -99  # primary inundation target
    n_inund_rules: int = 0  # rules to define alternative inundation results
    inund_rules: List[int] = field(default_factory=list)
    erode_to: int = -99
    elevation_stats: List['TElevStats'] = field(default_factory=list)
    has_sal_rules: bool = False
    salinity_rules: Optional['TSalinityRules'] = None
    has_sal_stats: bool = False
    mab: float = 0.0  # Aboveground biomass per unit area (Dry Matter 10^3 Kg/ha) – [0 - 41]
    rsc: float = 0.0  # Soil carbon storage rate for specified habitat type (10^3 Kg C/ha/year) – Range [0 – 0.35]
    ech4: float = 0.0  # Methane emission rate (10^3 Kg CH4/ha/year) - Range [0 – 0.194]
    cseq_notes: str = ""

    pss: Optional['TSLAMM_Simulation'] = None
    erode_model: ErosionInputs = None
    accr_model: AccrModels = None

    def __init__(self, tss):
        self.pss = tss
        self.elev_dat = ClassElev()
        self.salinity_stats = TSalStats()

    def inund_cat(self, pic: TInundContext):
        tss = self.pss
        inund_res = self.inundate_to

        def adj_cell(adj_row, adj_col, lee):  # Return adjacent cell to lee if Lee=True
            step = -1 if lee else 1
            direction = pic.subs.direction_offshore

            if direction == WaveDirection.Westerly:
                adj_col -= step
            elif direction == WaveDirection.Easterly:
                adj_col += step
            elif direction == WaveDirection.Northerly:
                adj_row -= step
            elif direction == WaveDirection.Southerly:
                adj_row += step

            valid_cell = 0 <= adj_row < tss.site.rows and 0 <= adj_col < tss.site.cols
            return valid_cell, adj_row, adj_col

        def ocean_nearer():
            nonlocal pic, tss
            if pic.ew_cat == BLANK:
                return True
            lee_r, lee_c = pic.cell_row, pic.cell_col
            for i in range(int(pic.distance_to_open_salt_water / tss.site.site_scale) + 2):
                valid_cell, lee_r, lee_c = adj_cell(lee_r, lee_c, False)
                if valid_cell:
                    c2 = tss.ret_a(lee_r, lee_c)
                    if cell_width(c2, pic.ew_cat) > 0:
                        return False
                else:
                    return True  # no more valid cells to check

            return True  # estuarine water not found, ocean nearer

        # ------------------------------------------------------------------------------
        def parse_category_rules(result):

            res = result

            def check_conditions(rule, pss, pic, tss):
                if rule == 1:
                    return -99 if pss.protect_all else None
                elif rule == 2:
                    return -99 if pss.protect_developed else None
                elif rule == 3:
                    return 22 if pss.use_flood_dev_dry_land else None
                elif rule == 4:
                    return 11 if pic.adj_ocean and ocean_nearer() else None
                elif rule == 5:
                    return 9 if pic.adj_water and pic.erosion2 >= 20 else None
                elif rule == 6:
                    return 8 if pss.tropical and pic.near_water else None
                elif rule == 7:
                    return 21 if pic.cell_fw_influenced else None
                elif rule == 8:
                    return 8 if pss.tropical else None
                elif rule == 9:
                    return 18 if pic.adj_ocean else None
                elif rule == 10:
                    return 5 if pic.cell_fw_influenced else None
                elif rule == 11:
                    return 19 if pic.cat_elev >= tss.lower_bound(19, pic.subs) else None
                elif rule == 12:
                    return 7 if pic.cat_elev >= tss.lower_bound(7, pic.subs) else None
                elif rule == 13:
                    return 23 if pss.use_flood_forest else None
                elif rule == 14:
                    return 27 if pss.use_flood_dev_dry_land else None
                elif rule == 15:
                    return 18 if pic.adj_ocean and ocean_nearer() else None
                elif rule == 16:
                    return 12 if pic.cell_fw_influenced else None
                elif rule == 17:
                    return 13 if pic.cell_fw_influenced else None
                elif rule == 18:
                    return 14 if pic.cat_elev >= tss.lower_bound(14, pic.subs) else None
                elif rule == 19:
                    return 19 if pic.cat_elev >= tss.lower_bound(19, pic.subs) else None
                elif rule == 20:
                    return 26 if pic.adj_ocean else None
                elif rule == 21:
                    return 15 if pic.adj_efsw else None
                else:
                    return None

            for i in range(self.n_inund_rules):
                rule = self.inund_rules[i]
                res = check_conditions(rule, self.pss, pic, tss)
                if res is not None:
                    return res

            return result  # end parse_category_rules
        # ------------------------------------------------------------------------------

        if self.n_inund_rules > 0:
            inund_res = parse_category_rules(inund_res)

        return inund_res

    def text_inund_rule(self, nr):
        rule_descriptions = {
            1: 'Do not inundate if Protect All Dry Land is selected.  ',
            2: 'Do not inundate if Protect Developed Dry Land is selected.  ',
            3: 'If "Use Flooded Developed" is selected then inundate to flooded developed dry land.  ',
            4: 'If "AdjOcean" and ocean water is nearer than estuarine water then convert to ocean beach.  ',
            5: 'If "AdjWater" with a fetch > 20 km then inundate to estuarine beach.  ',
            6: 'If site is designated as tropical and cell is "NearWater" then inundate to mangrove.  ',
            7: 'If the cell is "fresh water influenced" then convert to tidal swamp.  ',
            8: 'If site is designated as tropical then inundate to mangrove.  ',
            9: 'If "AdjOcean" then inundate to open ocean.  ',
            10: 'If the cell is "fresh water influenced" then inundate to tidal fresh marsh.  ',
            11: 'If the cell elevation is above the lower bound for irregularly-flooded marsh then convert to '
                'transitional marsh.  ',
            12: 'If the cell elevation is above the lower bound for regularly-flooded marsh then convert to '
                'regularly-flooded marsh.  ',
            13: 'If "Use Flooded Forest" is selected then inundate to "flooded forest."  ',
            14: 'If "Use Flooded Developed" is selected then inundate to flooded developed dry land.  ',
            15: 'If "AdjOcean" and ocean water is nearer than estuarine water then convert to ocean beach.  ',
            16: 'If the cell is "fresh water influenced" then convert to tidal forested/shrub.  ',
            17: 'If the cell is "fresh water influenced" then inundate to tidal fresh marsh.  ',
            18: 'If the cell elevation is above the lower bound for irregularly-flooded marsh then convert to '
                'irregularly-flooded marsh. ',
            19: 'If the cell elevation is above the lower bound for regularly-flooded marsh then convert to '
                'regularly-flooded marsh.  ',
            20: 'If "AdjOcean" then inundate to open ocean.  ',
            21: 'If the cell is "Adjacent to Estuarine forested/shrub wetland," then convert to that category.  '
        }
        return rule_descriptions.get(self.inund_rules[nr - 1], "Invalid rule number")

    def load_store(self, file, read_version_num, is_reading):
        # Direct attribute processing to maintain exact order from Delphi code
        self.gis_number = ts_read_write(file, 'GISNumber', self.gis_number, int, is_reading)
        self.text_name = ts_read_write(file, 'TextName', self.text_name, str, is_reading)
        self.is_open_water = ts_read_write(file, 'IsOpenWater', self.is_open_water, bool, is_reading)
        self.is_tidal = ts_read_write(file, 'IsTidal', self.is_tidal, bool, is_reading)
        self.is_non_tidal_wetland = ts_read_write(file, 'IsNonTidalWetland', self.is_non_tidal_wetland,
                                                  bool, is_reading)
        self.is_dryland = ts_read_write(file, 'IsDryland', self.is_dryland, bool, is_reading)
        self.is_developed = ts_read_write(file, 'IsDeveloped', self.is_developed, bool, is_reading)
        self.color = ts_read_write(file, 'Color', self.color, int, is_reading)

        aggint = self.agg_cat.value
        aggint = ts_read_write(file, 'AggCat', aggint, int, is_reading)
        self.agg_cat = AggCategories(aggint)

        self.use_ifm_collapse = ts_read_write(file, 'UseIFMCollapse', self.use_ifm_collapse, bool, is_reading)
        self.use_rfm_collapse = ts_read_write(file, 'UseRFMCollapse', self.use_rfm_collapse, bool, is_reading)
        self.use_wave_erosion = ts_read_write(file, 'UseWaveErosion', self.use_wave_erosion, bool, is_reading)

        if self.erode_model is None:
            erodeint = 0
        else:
            erodeint = self.erode_model.value
        erodeint = ts_read_write(file, 'ErodeModel', erodeint, int, is_reading)
        self.erode_model = ErosionInputs(erodeint)

        if self.accr_model is None:
            accrint = 0
        else:
            accrint = self.accr_model.value
        accrint = ts_read_write(file, 'AccrModel', accrint, int, is_reading)
        self.accr_model = AccrModels(accrint)

        self.inundate_to = ts_read_write(file, 'InundateTo', self.inundate_to, int, is_reading)
        self.n_inund_rules = ts_read_write(file, 'NInundRules', self.n_inund_rules, int, is_reading)

        # Inundation rules list handling
        if is_reading:
            self.inund_rules = [ts_read_write(file, 'InundRules[i]', 0, int, True
                                              ) for _ in range(self.n_inund_rules)]
        else:
            for i in range(self.n_inund_rules):
                ts_read_write(file, 'InundRules[i]', self.inund_rules[i], int, is_reading)

        self.erode_to = ts_read_write(file, 'ErodeTo', self.erode_to, int, is_reading)

        self.elev_dat.load_store(file, read_version_num, is_reading)

        self.has_sal_rules = ts_read_write(file, 'SalRules', self.has_sal_rules, bool, is_reading)
        self.has_sal_stats = ts_read_write(file, 'HasSalStats', self.has_sal_stats, bool, is_reading)

        self.mab = ts_read_write(file, 'mab', self.mab, float, is_reading)
        self.rsc = ts_read_write(file, 'Rsc', self.rsc, float, is_reading)
        self.ech4 = ts_read_write(file, 'ECH4', self.ech4, float, is_reading)
        self.cseq_notes = ts_read_write(file, 'CSeqNotes', self.cseq_notes, str, is_reading)


# Classic SLAMM Categories =
#  ( DevDryland,  	    {0 1  U   -- cat, gis                 Upland}
#    UndDryland,	    {1 2  U	                              Upland  - default; these two will have to be distinguished using census or land use data (protection)}
#    Swamp,	            {2 3 PFO, PFO1, PFO3-5, PSS           Palustrine forested broad-leaved deciduous}
#    CypressSwamp,	    {3 4 PFO2	                          Ditto, needle-leaved deciduous}
#    InlandFreshMarsh,  {4 5 L2EM,PEM[1&2]["A"-"I"],R2EM      Lacustrine, Palustrine, and Riverine emergent}
#    TidalFreshMarsh,	 {5 6 R1EM,PEM["K"-"U"]               Riverine tidal emergent}
#    ScrubShrub,        {6 7 E2SS1, E2FO                      Estuarine intertidal scrub-shrub broad-leaved deciduous}
#    RegFloodMarsh,	    {7 8 E2EM [no "P"]                    Estuarine intertidal emergent [won't distinguish high and low marsh]}
#    Mangrove,	        {8 9 E2FO3, E2SS3                     Estuarine intertidal forested and scrub-shrub broad-leaved evergreeen}
#    EstuarineBeach,	  {09 10 E2US2 or E2BB (PUS"K")       Estuarine intertidal unconsolidated shore sand or beach-bar}
#    TidalFlat,	        {10 11 E2US3or4, E2FL, M2AB           Estuarine intertidal unconsolidated shore mud/organic or flat, Intertidal Aquatic Bed here for now}
#    OceanBeach,	      {11 12 M2US2, M2BB/UB/USN           Marine intertidal unconsolidated shore sand}
#    OceanFlat,	        {12 13 M2US3or4                       Marine intertidal unconsolidated shore mud or organic}
#    RockyIntertidal,	{13 14  M2RS, E2RS, L2RS              Estuarine & Marine intertidal rocky shore}
#    InlandOpenWater,   {14 15 R3-UB R2-5OW,L1-2OW,POW,PUB,R2UB
#                           (L1-2,UB,AB), PAB, R2AB           Riverine, Lacustrine, and Palustrine open water}
#    RiverineTidal,     {15 16 R1OW	                          Riverine tidal open water}
#    EstuarineWater,    {16 17 E1, (PUB"K" no "h")            Estuarine subtidal}
#    TidalCreek,	    {17 18 E2SB, E2UBN,                 Estuarine intertidal stream bed}
#    OpenOcean,         {18 19 M1                             Marine subtidal  [aquatic beds and reefs can be added later]}
#    IrregFloodMarsh,   {19 20 E2EM[1-5]P                     "P" Irregularly Flooded Estuarine Intertidal Emergent}
#    InlandShore,       {20 22 L2UD,PUS, R[1..4]US/RS         shoreline not pre-processed using Tidal Range Elevations}
#    TidalSwamp,        {21 23 PSS,PFO"K"-"V" /EM1"K"-"V"     Tidally influenced Swamp. }
#    FloodDevDryLand,   {22 25 Flooded Developed Dry Land}
#    FloodForest);      {23 26 Flooded Cypress Swamp }

class TCategories:
    def __init__(self, ss: Optional['TSLAMM_Simulation']):
        self.n_cats: int = 0
        self.cats: List[TCategory] = []
        self.dev_dry_land: int = -1
        self.und_dry_land: int = -1
        self.flood_dev_dry_land: int = -1
        self.open_ocean: int = -1
        self.estuarine_water: int = -1
        self.blank_cat: TCategory = TCategory(ss)
        self.pss = ss  # Pointer to TSLAMM_Simulation

    def clear_cats(self):
        """Clear all categories from the list."""
        self.cats = []

    def destroy(self):
        """Destructor for cleaning up resources."""
        self.clear_cats()
        # Assume Category has a method for cleanup if necessary
        if hasattr(self.blank_cat, 'destroy'):
            self.blank_cat.destroy()

    def get_cat(self, num: int) -> TCategory:
        """Retrieve a category by its number, returning blank_cat if num is out of range."""
        if num < 0 or num >= len(self.cats):
            return self.blank_cat
        return self.cats[num]

    def load_store(self, file, read_version_num, is_reading):
        self.n_cats = ts_read_write(file, 'NCats', self.n_cats, int, is_reading)
        if is_reading:
            self.cats = [TCategory(self.pss) for _ in range(self.n_cats)]
        for cat in self.cats:
            cat.load_store(file, read_version_num, is_reading)

        self.dev_dry_land = ts_read_write(file, 'DevDryLand', self.dev_dry_land, int, is_reading)
        self.und_dry_land = ts_read_write(file, 'UndDryLand', self.und_dry_land, int, is_reading)
        self.flood_dev_dry_land = ts_read_write(file, 'FloodDevDryLand', self.flood_dev_dry_land, int, is_reading)
        self.open_ocean = ts_read_write(file, 'OpenOcean', self.open_ocean, int, is_reading)
        self.estuarine_water = ts_read_write(file, 'EstuarineWater', self.estuarine_water, int, is_reading)

    def setup_slamm_default(self):
        titles = [
            'Developed Dry Land', 'Undeveloped Dry Land', 'Swamp', 'Cypress Swamp', 'Inland-Fresh Marsh',
            'Tidal-Fresh Marsh', 'Trans. Salt Marsh', 'Regularly-Flooded Marsh', 'Mangrove', 'Estuarine Beach',
            'Tidal Flat', 'Ocean Beach', 'Ocean Flat', 'Rocky Intertidal', 'Inland Open Water',
            'Riverine Tidal', 'Estuarine Open Water', 'Tidal Creek', 'Open Ocean', 'Irreg.-Flooded Marsh',
            'Inland Shore', 'Tidal Swamp', 'Flooded Developed Dry Land', 'Flooded Forest'
        ]
        gis_nums = [i + 1 for i in range(24)]
        default_colors = [
            0x002D0398, 0x00002BD5, 0x008000, 0x005500, 0x00FF00,
            0x00A4FFA4, 0x808000, 0x00A6A600, 0x800080, 0x00B3FFFF,
            0xC0C0C0, 0xFFFF00, 0x000059B3, 0xFF80FF, 0xFFA8A8,
            0x0000FF, 0x0000FF, 0x0000FF, 0x000080, 0x004080FF,
            0x00004080, 0x00003E00, 0x00FF64B1, 0x0075D6FF
        ]

        self.clear_cats()
        self.n_cats = 24
        self.cats = [TCategory(self.pss)] * self.n_cats
        for i in range(self.n_cats):
            self.cats[i] = TCategory(self.pss)
            self.cats[i].gis_number = gis_nums[i]
            self.cats[i].text_name = titles[i]
            self.cats[i].color = default_colors[i]

            # Physical Characteristics
            if 14 <= i <= 18:
                self.cats[i].is_open_water = True
            if 5 <= i <= 13 or 15 <= i <= 23:
                self.cats[i].is_tidal = True
            if i in [2, 3, 4]:
                self.cats[i].is_non_tidal_wetland = True
            if i in [0, 1]:
                self.cats[i].is_dryland = True
            if i in [0, 22]:
                self.cats[i].is_developed = True

            # Aggregation Categories
            if i in [0, 1, 22]:
                self.cats[i].agg_cat = AggCategories.NonTidal
            if i in [2, 3, 4, 20]:
                self.cats[i].agg_cat = AggCategories.FreshNonTidal
            if 14 <= i <= 18:
                self.cats[i].agg_cat = AggCategories.OpenWater
            if 9 <= i <= 13:
                self.cats[i].agg_cat = AggCategories.LowTidal
            if i == 7:
                self.cats[i].agg_cat = AggCategories.SaltMarsh
            if i in [6, 8, 19, 23]:
                self.cats[i].agg_cat = AggCategories.Transitional
            if i in [5, 21]:
                self.cats[i].agg_cat = AggCategories.FreshWaterTidal

            # Collapse and Erosion settings
            if i in [6, 19]:
                self.cats[i].use_ifm_collapse = True
            if i == 7:
                self.cats[i].use_rfm_collapse = True
            if i in [5, 6, 7, 19]:
                self.cats[i].use_wave_erosion = True

            # Accretion Models
            if i == 7:
                self.cats[i].accr_model = AccrModels.RegFM
            if i in [6, 19]:
                self.cats[i].accr_model = AccrModels.IrregFM
            if 9 <= i <= 12:
                self.cats[i].accr_model = AccrModels.BeachTF
            if i == 5:
                self.cats[i].accr_model = AccrModels.TidalFM
            if i == 4:
                self.cats[i].accr_model = AccrModels.InlandM
            if i == 8:
                self.cats[i].accr_model = AccrModels.Mangrove
            if i == 21:
                self.cats[i].accr_model = AccrModels.TSwamp
            if i in [2, 3]:
                self.cats[i].accr_model = AccrModels.Swamp

            # Erosion Models
            if i in [5, 6, 7, 19]:
                self.cats[i].erode_model = ErosionInputs.EMarsh
                self.cats[i].erode_to = 16
            if i in [2, 3, 8, 21]:
                self.cats[i].erode_model = ErosionInputs.ESwamp
                self.cats[i].erode_to = 10
            if 9 <= i <= 12:
                self.cats[i].erode_model = ErosionInputs.ETFlat
                self.cats[i].erode_to = 16
            if i == 11:
                self.cats[i].erode_model = ErosionInputs.EOcBeach
                self.cats[i].erode_to = 18
            if i == 3:
                self.cats[i].erode_to = 16  # Cypress swamp erodes to open water

            # inundateTo targets,  note 6=transitional marsh; 16=Estuarine water; 10=Tidal Flat
            #                           7=RegFloodMarsh, 16=Estuarine Water, and 18=OpenOcean, -99 = blank
            inundation_targets = {
                0: 6, 1: 6, 2: 6, 3: 16, 4: 6, 5: 10, 6: 7, 7: 10,
                8: 16, 9: 16, 10: 16, 11: 18, 12: 18, 13: 16, 14: 16,
                15: 16, 16: -99, 17: -99, 18: -99, 19: 7, 20: 16,
                21: 19, 22: -99, 23: -99
            }
            self.cats[i].inundate_to = inundation_targets.get(i)

            # Setting the number of inundation rules for each category
            inund_rules_dict = {
                0: 7, 1: 5, 2: 2, 3: 2, 4: 2,
                5: 3, 6: 1, 7: 1, 13: 1, 19: 1,
                20: 1, 21: 2
            }

            self.cats[i].n_inund_rules = inund_rules_dict.get(i, 0)  # Default to 0 if not specified

            # Setting the inundation rules for each category
            inundation_rules_setup = {
                0: list(range(1, 8)),  # devdryland, 1 to 7
                1: [1, 4, 5, 6, 7],
                2: [7, 6],
                3: [13, 6],
                4: [10, 6],
                5: [6, 11, 12],
                6: [8],
                7: [8],
                13: [9],
                19: [8],
                20: [9],
                21: [10, 6]
            }

            if i in inundation_rules_setup:
                self.cats[i].inund_rules = inundation_rules_setup[i]

            # Setting default elevation data based on category or tidal designation
            cat = self.cats[i]
            if not cat.is_tidal:
                cat.elev_dat.min_unit = ElevUnit.SaltBound
                cat.elev_dat.min_elev = 1.0
                cat.elev_dat.max_unit = ElevUnit.Meters
                cat.elev_dat.max_elev = 3.048  # Assumes 10 foot contour
            if i == 19:  # IrregFloodMarsh
                cat.elev_dat.min_unit = ElevUnit.HalfTide
                cat.elev_dat.min_elev = 0.5
                cat.elev_dat.max_unit = ElevUnit.SaltBound
                cat.elev_dat.max_elev = 1.0
            if i == 6:  # ScrubShrub
                cat.elev_dat.min_unit = ElevUnit.HalfTide
                cat.elev_dat.min_elev = 1.0
                cat.elev_dat.max_unit = ElevUnit.SaltBound
                cat.elev_dat.max_elev = 1.0
            if i == 7:  # Regflood marsh
                cat.elev_dat.min_unit = ElevUnit.Meters
                cat.elev_dat.min_elev = 0
                cat.elev_dat.max_unit = ElevUnit.HalfTide
                cat.elev_dat.max_elev = 1.2
            if i in [20, 11, 9]:  # InlandShore, OceanBeach, EstuarineBeach
                cat.elev_dat.min_unit = ElevUnit.HalfTide
                cat.elev_dat.min_elev = -1.0
                cat.elev_dat.max_unit = ElevUnit.SaltBound
                cat.elev_dat.max_elev = 1.0
            if i in [10, 12]:  # TidalFlat, OceanFlat
                cat.elev_dat.min_unit = ElevUnit.HalfTide
                cat.elev_dat.min_elev = -1.0
                cat.elev_dat.max_unit = ElevUnit.Meters
                cat.elev_dat.max_elev = 0
            if i == 13:  # RockyIntertidal
                cat.elev_dat.min_unit = ElevUnit.HalfTide
                cat.elev_dat.min_elev = -1.0
                cat.elev_dat.max_unit = ElevUnit.SaltBound
                cat.elev_dat.max_elev = 1.0

        # Define special category indices
        self.dev_dry_land = 0
        self.und_dry_land = 1
        self.flood_dev_dry_land = 22
        self.open_ocean = 18
        self.estuarine_water = 16

        self.set_cseq_defaults(False)  # set carbon sequestration defaults

    def are_california(self) -> bool:
        return self.get_cat(10).text_name == 'Dunes'  # For now, use the Dunes designation to identify CA categories

    def setup_ca_default(self):
        titles = [
            'Developed Dry Land', 'Undeveloped Dry Land', 'Agriculture', 'Artificial Pond',
            'Artificial Salt Pond', 'Inland Open Water', 'Inland Shore', 'Freshwater Marsh',
            'Seasonal Freshwater Marsh', 'Seasonally Flooded Agriculture', 'Dunes',
            'Freshwater Forested/Shrub', 'Tidal Freshwater Forested/Shrub', 'Tidal Fresh Marsh',
            'Irreg.-Flooded Marsh', 'Estuarine forested/shrub wetland', 'Artificial reef',
            'Invertebrate reef', 'Ocean Beach', 'Regularly-flooded Marsh', 'Rocky Intertidal',
            'Tidal Flat and Salt Panne', 'Riverine (open water)', 'Riverine Tidal', 'Tidal Channel',
            'Estuarine Open Water', 'Open Ocean', 'Flooded Developed'
        ]
        gis_nums = [
            101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
            111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
            121, 122, 123, 124, 125, 126, 127, 128
        ]
        default_colors = [
            0x002D0398, 0x00002BD5, 0x000000, 0x00FF9797,
            0x00FF8484, 0x00FFA8A8, 0x00004080, 0x00FF00,
            0x00CA56BB, 0x00484848, 0x0000DDDD,
            0x00226a1d, 0x0034e37d, 0x00cde7a5, 0x000D5CD1,
            0x00008888, 0x800080, 0x800080, 0xFFFF00,
            0x00A6A600, 0x00FF80FF, 0xC0C0C0, 0x00ff8913,
            0x000080, 0x000080, 0x0000FF, 0x000080, 0x00FF64B1
        ]

        self.clear_cats()
        self.n_cats = 28
        self.cats = [TCategory(self.pss) for _ in range(self.n_cats)]
        for i, cat in enumerate(self.cats):
            cat.gis_number = gis_nums[i]
            cat.text_name = titles[i]
            cat.color = default_colors[i]

            # Setting attributes based on category index
            if i in range(21, 27):  # Open Water Categories 21 to 26
                cat.is_open_water = True
            if i in list(range(12, 22)) + [23, 24, 25, 26]:
                cat.is_tidal = True
            if i in [7, 8, 11]:  # Non-Tidal Wetland Categories
                cat.is_non_tidal_wetland = True
            if i in [0, 1, 2]:  # Dry Land Categories
                cat.is_dry_land = True
            if i in [0, 27]:  # Developed Land Categories
                cat.is_developed = True

            # Aggregation categories and other settings
            non_tidal = [0, 1, 2, 10]
            fresh_non_tidal = [3, 5, 6, 7, 8, 9, 11]
            open_water = [4, 22, 23, 24, 25, 26]
            low_tidal = [16, 17, 18, 20, 21]
            salt_marsh = [19]
            transitional = [14, 15, 27]
            fresh_water_tidal = [12, 13]
            
            # Aggregation Category
            if i in non_tidal:
                cat.agg_cat = AggCategories.NonTidal
            elif i in fresh_non_tidal:
                cat.agg_cat = AggCategories.FreshNonTidal
            elif i in open_water:
                cat.agg_cat = AggCategories.OpenWater
            elif i in low_tidal:
                cat.agg_cat = AggCategories.LowTidal
            elif i in salt_marsh:
                cat.agg_cat = AggCategories.SaltMarsh
            elif i in transitional:
                cat.agg_cat = AggCategories.Transitional
            elif i in fresh_water_tidal:
                cat.agg_cat = AggCategories.FreshWaterTidal
            else:
                cat.agg_cat = AggCategories.AggBlank

            # Collapse and Erosion Models
            if i in [14, 15]:
                cat.use_ifm_collapse = True
            if i == 19:
                cat.use_rfm_collapse = True
            if i in [13, 19, 14]:
                cat.use_wave_erosion = True

            # Accretion Model
            if i == 19:
                cat.accr_model = AccrModels.RegFM
            elif i in [14, 15]:
                cat.accr_model = AccrModels.IrregFM
            elif i in [18, 21]:
                cat.accr_model = AccrModels.BeachTF
            elif i == 13:
                cat.accr_model = AccrModels.TidalFM
            elif i in [7, 8]:
                cat.accr_model = AccrModels.InlandM
            elif i == 12:
                cat.accr_model = AccrModels.TSwamp
            elif i == 11:
                cat.accr_model = AccrModels.Swamp
            else:
                cat.accr_model = AccrModels.AccrNone

            # Erosion Model
            if i in [13, 14, 19]:
                cat.erode_model = ErosionInputs.EMarsh
                cat.erode_to = 25  # erode to estuarine water
            elif i in [11, 12, 15]:
                cat.erode_model = ErosionInputs.ESwamp 
                cat.erode_to = 21  # erode to tidal flat
            elif i == 21:
                cat.erode_model = ErosionInputs.ETFlat 
                cat.erode_to = 25  # erode to estuarine water
            elif i == 18:
                cat.erode_model = ErosionInputs.EOcBeach 
                cat.erode_to = 26  # erode to open ocean
            else:
                cat.erode_model = ErosionInputs.ENone 

            # Inundation Targets and Rules
            inundation_targets = {
                0: 14, 1: 14, 2: 14, 3: 25, 4: 25, 5: 25, 6: 25,
                7: 14, 8: 14, 9: 14, 10: 18, 11: 15, 12: 14, 13: 21,
                14: 19, 15: 19, 16: 25, 17: 25, 18: 26, 19: 21,
                20: 25, 21: 25, 22: 23, 23: None, 24: None, 25: None,
                26: None, 27: None
            }
            cat.inundate_to = inundation_targets.get(i, None)

            # Inundation Rules setup
            rules_setup = {
                0: [1, 2, 14, 15, 16],
                1: [1, 15, 16],
                2: [1, 15, 16],
                9: [1, 15, 16],
                7: [17],
                8: [17],
                11: [16],
                12: [17],
                13: [18, 19],
                20: [20]
            }
            if i in rules_setup:
                cat.inund_rules = rules_setup[i]
                cat.n_inund_rules = len(rules_setup[i])
            else:
                cat.inund_rules = []
                cat.n_inund_rules = 0

            # Elevation Data setup
            if not cat.is_tidal:
                cat.min_unit = ElevUnit.SaltBound
                cat.min_elev = 1.0
                cat.max_unit = ElevUnit.Meters
                cat.max_elev = 3.048  # assumes 10 foot contour
            if i == 19:
                cat.min_unit = ElevUnit.Meters
                cat.min_elev = 0
                cat.max_unit = ElevUnit.HalfTide
                cat.max_elev = 1.2
            if i == 14:
                cat.min_unit = ElevUnit.HalfTide
                cat.min_elev = 0.5
                cat.max_unit = ElevUnit.SaltBound
                cat.max_elev = 1.0
            if i == 15:
                cat.min_unit = ElevUnit.HalfTide
                cat.min_elev = 1.0
                cat.max_unit = ElevUnit.SaltBound
                cat.max_elev = 1.0
            if i == 21:
                cat.min_unit = ElevUnit.HalfTide
                cat.min_elev = -1.0
                cat.max_unit = ElevUnit.Meters
                cat.max_elev = 0
            if i in [16, 17, 18, 20]:
                cat.min_unit = ElevUnit.HalfTide
                cat.min_elev = -1.0
                cat.max_unit = ElevUnit.SaltBound
                cat.max_elev = 1.0
            if i == 12 or i == 13:
                cat.min_unit = ElevUnit.HalfTide
                cat.min_elev = 0
                cat.max_unit = ElevUnit.SaltBound
                cat.max_elev = 1.0

        # Assigning special category indices
        self.dev_dry_land = 0
        self.und_dry_land = 1
        self.flood_dev_dry_land = 27
        self.open_ocean = 26
        self.estuarine_water = 25

        # Set carbon sequestration defaults specifically for California
        self.set_cseq_defaults(ca=True)

    def set_cseq_defaults(self, ca: bool):

        if ca:
            # California-specific settings
            for i, cat in enumerate(self.cats):
                if i in [0, 4, 6, 10, 16, 17, 18, 20, 21, 24, 25, 26, 27]:
                    cat.mab = 0
                    cat.rsc = 0
                    cat.ech4 = 0
                    cat.cseq_notes = ''
                elif i == 1:
                    cat.mab = 1.6
                    cat.rsc = 0.09
                    cat.ech4 = 0
                    cat.cseq_notes = 'mab: IPCC 2006 V4 Chap 6 - p6.29 & Table 6.4, for Warm Temperate - Dry Regions; Rsc:Kroodsma and Field 2006 value for non-rice annual cropland'
                elif i == 2:
                    cat.mab = 1.6
                    cat.rsc = 0.09
                    cat.ech4 = 0
                    cat.cseq_notes = 'mab: IPCC 2006 V4 Chap 6 - p6.29 & Table 6.4, for Warm Temperate - Dry Regions; Rsc:Kroodsma and Field 2006 value for non-rice annual cropland'
                elif i == 3:
                    cat.mab = 0
                    cat.rsc = 0
                    cat.ech4 = 0.1937
                    cat.cseq_notes = 'Ech4: IPCC 2013 Table 4.14'
                elif i == 5:
                    cat.mab = 0
                    cat.rsc = 0
                    cat.ech4 = 0.1937
                    cat.cseq_notes = 'Ech4: IPCC 2013 Table 4.14'
                elif i == 7:
                    cat.mab = 5.5
                    cat.rsc = 0.35
                    cat.ech4 = 0.1937
                    cat.cseq_notes = 'mab: Onuf 1987, Figure 31: Mean biomass of salt marsh plants in Mugu Lagoon (1977-1981); Rsc: Elgin 2012; Ech4: IPCC 2013 Table 4.14'
                elif i == 8:
                    cat.mab = 5.5
                    cat.rsc = 0.35
                    cat.ech4 = 0.1937
                    cat.cseq_notes = 'mab: Onuf 1987, Figure 31: Mean biomass of salt marsh plants in Mugu Lagoon (1977-1981); Rsc: Elgin 2012; Ech4: IPCC 2013 Table 4.14'
                elif i == 9:
                    cat.mab = 1.6
                    cat.rsc = 0.09
                    cat.ech4 = 0.1937
                    cat.cseq_notes = 'mab: IPCC 2006 V4 Chap 6 - p6.29 & Table 6.4, for Warm Temperate - Dry Regions; Rsc:Kroodsma and Field 2006 value for non-rice annual cropland; Ech4: IPCC 2013 Table 4.14'
                elif i == 11:
                    cat.mab = 5.5
                    cat.rsc = 0.35
                    cat.ech4 = 0.1937
                    cat.cseq_notes = 'mab: Onuf 1987, Figure 31: Mean biomass of salt marsh plants in Mugu Lagoon (1977-1981); Rsc: Elgin 2012; Ech4: IPCC 2013 Table 4.14'
                elif i == 12:
                    cat.mab = 5.5
                    cat.rsc = 0.35
                    cat.ech4 = 0.1937
                    cat.cseq_notes = 'mab: Onuf 1987, Figure 31: Mean biomass of salt marsh plants in Mugu Lagoon (1977-1981); Rsc: Elgin 2012; Ech4: IPCC 2013 Table 4.14'
                elif i == 13:
                    cat.mab = 5.5
                    cat.rsc = 0.35
                    cat.ech4 = 0.1937
                    cat.cseq_notes = 'mab: Onuf 1987, Figure 31: Mean biomass of salt marsh plants in Mugu Lagoon (1977-1981); Rsc: Elgin 2012; Ech4: IPCC 2013 Table 4.14'
                elif i == 14:
                    cat.mab = 3.9
                    cat.rsc = 0.25
                    cat.ech4 = 0
                    cat.cseq_notes = 'Assume 70% cover of regularly flooded salt marsh; Ech4: IPCC 2013 Table 4.14 (0 for saline conditions)'
                elif i == 15:
                    cat.mab = 5.5
                    cat.rsc = 0.35
                    cat.ech4 = 0
                    cat.cseq_notes = 'mab: Onuf 1987, Figure 31: Mean biomass of salt marsh plants in Mugu Lagoon (1977-1981); Rsc: Elgin 2012; Ech4: IPCC 2013 Table 4.14 (0 for saline conditions)'
                elif i == 19:
                    cat.mab = 5.5
                    cat.rsc = 0.35
                    cat.ech4 = 0
                    cat.cseq_notes = 'mab: Onuf 1987, Figure 31: Mean biomass of salt marsh plants in Mugu Lagoon (1977-1981); Rsc: Elgin 2012; Ech4: IPCC 2013 Table 4.14 (0 for saline conditions)'
                elif i == 22:
                    cat.mab = 0
                    cat.rsc = 0
                    cat.ech4 = 0.1937
                    cat.cseq_notes = 'Ech4: IPCC 2013 Table 4.14'
                elif i == 23:
                    cat.mab = 0
                    cat.rsc = 0
                    cat.ech4 = 0.1937
                    cat.cseq_notes = 'Ech4: IPCC 2013 Table 4.14'
        else:
            for i, cat in enumerate(self.cats):
                if i in [0, 3, 8, 9, 10, 11, 12, 13, 16, 17, 18, 20, 22, 23]:
                    cat.mab = 0
                    cat.rsc = 0
                    cat.ech4 = 0
                    cat.cseq_notes = ''
                elif i == 1:
                    cat.mab = 1.6
                    cat.rsc = 0.09
                    cat.ech4 = 0
                    cat.cseq_notes = 'mab: IPCC 2006 V4 Chap 6 - p6.29 & Table 6.4, for Warm Temperate - Dry Regions; Rsc:Kroodsma and Field 2006 value for non-rice annual cropland'
                elif i == 2:
                    cat.mab = 5.5
                    cat.rsc = 0.35
                    cat.ech4 = 0.1937
                    cat.cseq_notes = 'mab: Onuf 1987, Figure 31: Mean biomass of salt marsh plants in Mugu Lagoon (1977-1981); Rsc: Elgin 2012; Ech4: IPCC 2013 Table 4.14'
                elif i == 4:
                    cat.mab = 5.5
                    cat.rsc = 0.35
                    cat.ech4 = 0.1937
                    cat.cseq_notes = 'mab: Onuf 1987, Figure 31: Mean biomass of salt marsh plants in Mugu Lagoon (1977-1981); Rsc: Elgin 2012; Ech4: IPCC 2013 Table 4.14'
                elif i == 5:
                    cat.mab = 5.5
                    cat.rsc = 0.35
                    cat.ech4 = 0.1937
                    cat.cseq_notes = 'mab: Onuf 1987, Figure 31: Mean biomass of salt marsh plants in Mugu Lagoon (1977-1981); Rsc: Elgin 2012; Ech4: IPCC 2013 Table 4.14'
                elif i == 6:
                    cat.mab = 3.9
                    cat.rsc = 0.25
                    cat.ech4 = 0
                    cat.cseq_notes = 'Assume 70% cover of regularly flooded salt marsh; Ech4: IPCC 2013 Table 4.14 (0 for saline conditions)'
                elif i == 7:
                    cat.mab = 5.5
                    cat.rsc = 0.35
                    cat.ech4 = 0
                    cat.cseq_notes = 'mab: Onuf 1987, Figure 31: Mean biomass of salt marsh plants in Mugu Lagoon (1977-1981); Rsc: Elgin 2012; Ech4: IPCC 2013 Table 4.14 (0 for saline conditions)'
                elif i == 14:
                    cat.mab = 0
                    cat.rsc = 0
                    cat.ech4 = 0.1937
                    cat.cseq_notes = 'Ech4: IPCC 2013 Table 4.14'
                elif i == 15:
                    cat.mab = 0
                    cat.rsc = 0
                    cat.ech4 = 0.1937
                    cat.cseq_notes = 'Ech4: IPCC 2013 Table 4.14'
                elif i == 19:
                    cat.mab = 3.9
                    cat.rsc = 0.25
                    cat.ech4 = 0
                    cat.cseq_notes = 'Assume 70% cover of regularly flooded salt marsh; Ech4: IPCC 2013 Table 4.14 (0 for saline conditions)'
                elif i == 21:
                    cat.mab = 5.5
                    cat.rsc = 0.35
                    cat.ech4 = 0.1937
                    cat.cseq_notes = 'mab: Onuf 1987, Figure 31: Mean biomass of salt marsh plants in Mugu Lagoon (1977-1981); Rsc: Elgin 2012; Ech4: IPCC 2013 Table 4.14'

    def write_tech_specs(self):

        file_path = self.pss.file_name+"categories.txt"

        if not file_path:
            return  # User cancelled the dialog

        with open(file_path, 'w') as file:
            file.write(
                'Category Name, GIS Number, OpenWater, Tidal, NonTidal Wetland, Dryland, Developed, Aggregation '
                'Category, IFM Collapse, RFM Collapse, Accretion Model, Erosion Model\n')
            for cat in self.cats:
                properties = [
                    cat.text_name,
                    str(cat.gis_number),
                    'X' if cat.is_open_water else '',
                    'X' if cat.is_tidal else '',
                    'X' if cat.is_non_tidal_wetland else '',
                    'X' if cat.is_dryland else '',
                    'X' if cat.is_developed else '',
                    cat.agg_cat,
                    'X' if cat.use_ifm_collapse else '',
                    'X' if cat.use_rfm_collapse else '',
                    cat.accr_model,
                    cat.erode_model
                ]
                line = ','.join(f'"{p}"' if i == 0 else p for i, p in enumerate(properties)) + '\n'
                file.write(line)

            # Additional category-specific details
            file.write('\n')
            for cat in self.cats:
                file.write(f'[{cat.gis_number}] {cat.text_name}')
                if cat.inundate_to == "Blank":
                    file.write(': inundation model is not relevant for this category.\n')
                else:
                    file.write(
                        f' Inundation Model: When it falls below its lower elevation boundary, this category '
                        f'generally converts to "{self.cats[cat.inundate_to].text_name}."')
                    if cat.n_inund_rules > 0:
                        file.write(f' However, (1) {cat.text_inund_rule(1)}')
                        for ir in range(2, cat.n_inund_rules + 1):
                            file.write(f' Otherwise, ({ir}) {cat.text_inund_rule(ir)}')
                    file.write('\n')

        print ("Success", f"Specs saved to {file_path}")

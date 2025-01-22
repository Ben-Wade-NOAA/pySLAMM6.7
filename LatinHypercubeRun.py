"""
This module contains the implementation of the uncertainty run for the SLAMM model.
Classes:
    UResultsTypes(Enum): Enumeration for different types of uncertainty results.
Functions:
    uncert_run(ss: TSLAMM_Simulation):
        Executes the uncertainty run for the SLAMM model.
        Args:
            ss (TSLAMM_Simulation): The SLAMM simulation object.
        Internal Functions:
            conflicting_distribution(in_dist: int, check_all_subsites: bool) -> bool:
                Checks for conflicting distributions.
            no_commas(in_str: str) -> str:
                Replaces commas with colons in a string.
            random_int(top: int) -> int:
                Generates a random integer up to a specified top value.
            calculate_draw(interval: int) -> float:
                Calculates a draw value based on the distribution type and interval.
            fill_variable_draws(indx: int):
                Fills the list associated with each used distribution with UncertDraw objects.
            accumulate_uncertainty_results(ns_loop):
                Accumulates the uncertainty results for each iteration.
            init_text_results() -> bool:
                Initializes the text results file.
            dist_to_text():
                Writes the distribution summary to the text results file.
            write_dist_summaries():
                Writes the summaries of the distributions to the output files.
            verify_uncert_setup() -> bool:
                Verifies the uncertainty setup.
            cleanup():
                Cleans up the global variables.
            save_gt0_gt1():
                Saves the initial GT and SE values.
            restore_remaining_ses():
                Restores the remaining SE values.
            restore_vals():
                Restores the values of the distributions.
            call_set_values():
                Sets the values for the distributions.
            modify_remaining_ses():
                Modifies the remaining SE values.
        Global Variables:
            text_out: The text output file.
            iterations_done: The number of iterations completed.
            output_files: The list of output files.
            num_output_files: The number of output files.
            delta_se: The delta SE values.
            se0: The initial SE values.
            gt0: The initial GT values.
            gt1: The modified GT values.
"""

import random
from app_global import *
import os
import datetime
from SLR6 import TSLAMM_Simulation
from uncert_defn import UncertDraw, DistType
from enum import Enum



NUM_UNCERT_OUTPUTS = 27


class UResultsTypes(Enum):
    DetRes = 0
    MinRes = 1
    MaxRes = 2
    StdRes = 3
    MeanRes = 4


global text_out, iterations_done
global output_files, num_output_files
global delta_se
global se0, gt0, gt1


def uncert_run(ss: TSLAMM_Simulation):

    def conflicting_distribution(in_dist: int, check_all_subsites: bool) -> bool:
        id_dist = ss.uncert_setup.dist_array[in_dist]
        for d_loop in range(ss.uncert_setup.num_dists):
            if d_loop != in_dist:
                if (ss.uncert_setup.dist_array[d_loop].id_num == id_dist.id_num) and \
                   ((ss.uncert_setup.dist_array[d_loop].all_subsites == id_dist.all_subsites) or not check_all_subsites) and \
                   (ss.uncert_setup.dist_array[d_loop].subsite_num == id_dist.subsite_num):
                    return True
        return False

    def no_commas(in_str: str) -> str:
        return in_str.replace(',', ':')

    def random_int(top: int) -> int:
        return int(random.uniform(0, 1) * top) + 1

    def calculate_draw(interval: int) -> float:
        prob_high = 1.0
        prob_low = 0.0
        if dist.dist_type in [DistType.Normal, DistType.LogNormal]:
            prob_low = dist.cdf(0)
        #   Width is the width of the probability interval, "Where" is the upper
        #   bounds of the probability interval, RandProb is the random value
        #    within the probability interval
        width = (prob_high - prob_low) / ss.uncert_setup.iterations
        where = (width * interval) + prob_low
        rand_prob = where - width * random.uniform(0, 1)
        return dist.icdf(rand_prob)

    def fill_variable_draws(indx: int):

        #     This Procedure fills the List associated with each used
        #     distribution with NumSteps (number of iterations) UncertDraw objects
        #     each which has the numbered interval and the calculated draw from
        #     the Function CalculateDraw}
        #
        #     In order to calculate the rank order of each parameter's draw
        #     (ensures non-repeatability and tells which interval to sample from),
        #     the Procedure calculates a random number for each draw and then ranks
        #     them.  Optimized Feb 10, 97, JSC

        dist.draws = []
        for iteration_loop in range(ss.uncert_setup.iterations):
            new_draw = UncertDraw(0, random.uniform(0, 1), 0)
            dist.draws.append(new_draw)
        for iteration_loop in range(ss.uncert_setup.iterations):
            low_value = 99
            low_val_index = 0
            for find_slot_loop in range(ss.uncert_setup.iterations):
                if dist.draws[find_slot_loop].random_draw <= low_value:
                    low_val_index = find_slot_loop
                    low_value = dist.draws[find_slot_loop].random_draw
            new_draw = dist.draws[low_val_index]
            new_draw.random_draw = 99
            new_draw.interval_num = iteration_loop + 1
            new_draw.value = calculate_draw(iteration_loop + 1)

    def accumulate_uncertainty_results(ns_loop):
        global output_files, num_output_files, iterations_done

        def init_arrays():
            for of_loop in range(num_output_files):
                for res in range(UResultsTypes.MinRes.value, UResultsTypes.MeanRes.value + 1):
                    res_type = UResultsTypes(res)
                    u_results[of_loop][res_type] = np.zeros(NUM_UNCERT_OUTPUTS)
            for of_loop in range(num_output_files):
                for loop in range(NUM_UNCERT_OUTPUTS):
                    for res in range(UResultsTypes.MinRes.value, UResultsTypes.MeanRes.value + 1):
                        res_type = UResultsTypes(res)
                        o_val = ss.summary[of_loop][ss.uncert_setup.unc_sens_row - 1][loop]
                        if res_type == UResultsTypes.MinRes:
                            output_files[of_loop].write(f"{o_val:e}, ")
                        if res_type == UResultsTypes.StdRes:
                            u_results[of_loop][res_type][loop] = o_val ** 2
                        else:
                            u_results[of_loop][res_type][loop] = o_val

        if ns_loop == 0:  # accumulate_uncertainty_results
            output_files = [None] * num_output_files
            for of_loop in range(num_output_files):
                of_string = ''
                if of_loop > 0:
                    of_string = f'_OS{of_loop}'
                if of_loop > ss.site.n_output_sites:
                    of_string = f'_ROS{of_loop - ss.site.n_output_sites}'
                output_file_n = os.path.splitext(all_data_file_n)[0] + of_string + '_alldata.csv'
                output_file_n = output_file_n.replace('\\\\', '\\')
                output_files[of_loop] = open(output_file_n, 'w')
                for i in range(ss.uncert_setup.num_dists):
                    output_files[of_loop].write(f'"Mult. {no_commas(ss.uncert_setup.dist_array[i].get_name(True))}", ')
                for i in range(NUM_UNCERT_OUTPUTS):
                    output_files[of_loop].write(f'"{no_commas(ss.col_label[i])}", ')

        for of_loop in range(num_output_files):
            output_files[of_loop].write("\n")
            for i in range(ss.uncert_setup.num_dists):
                if ss.uncert_setup.dist_array[i].dist_type != DistType.Elev_Map:
                    output_files[of_loop].write(f"{ss.uncert_setup.dist_array[i].draws[ns_loop].value:.12e}, ")
                else:
                    output_files[of_loop].write(f"{ss.uncert_setup.dist_array[i].get_name(True)}, ")

        if ns_loop == 0:
            init_arrays()
        else:
            for of_loop in range(num_output_files):
                for out_loop in range(NUM_UNCERT_OUTPUTS):
                    output_val = ss.summary[of_loop][ss.uncert_setup.unc_sens_row - 1][out_loop]
                    output_files[of_loop].write(f"{output_val:e}, ")
                    for res_lp in UResultsTypes:
                        uncert_val = u_results[of_loop][res_lp][out_loop]
                        if res_lp == UResultsTypes.MeanRes:
                            u_results[of_loop][res_lp][out_loop] += output_val
                        elif res_lp == UResultsTypes.MinRes and output_val < uncert_val:
                            u_results[of_loop][res_lp][out_loop] = output_val
                        elif res_lp == UResultsTypes.MaxRes and output_val > uncert_val:
                            u_results[of_loop][res_lp][out_loop] = output_val
                        elif res_lp == UResultsTypes.StdRes:
                            u_results[of_loop][res_lp][out_loop] += output_val ** 2


        iterations_done = ns_loop + 1
        text_out.write(f'\n Iteration {iterations_done} completed.\n----------------------------\n')


    def init_text_results() -> bool:
        global text_out

        cur_dir = os.getcwd()
        date_holder = datetime.datetime.now().strftime('%m-%d-%y')
        file_prefix = ss.file_name[:4] if len(ss.file_name) >= 4 else ss.file_name
        ss.uncert_setup.csv_path = os.path.join(cur_dir, file_prefix + '_uncert' + date_holder + '.csv')

        date_holder = datetime.datetime.now().strftime('%m-%d-%y %H:%M:%S')

        result = True
        file_n = ''
        file_n = os.path.splitext(ss.uncert_setup.csv_path)[0] + '_uncertlog.txt'
        file_n = file_n.replace('\\\\', '\\')
        text_out = open(file_n, 'w')
        text_out.write('---------------------------------------------------------\n\n')
        text_out.write('        Uncertainty Run for SLAMM                        \n\n')
        text_out.write('---------------------------------------------------------\n')
        text_out.write(f'        Run Starts at {date_holder}\n')
        text_out.write('---------------------------------------------------------\n\n')
        text_out.write('        Technical Details for Uncert Viewer              \n\n')
        text_out.write(f'~FILENAME={os.path.basename(ss.uncert_setup.csv_path)}\n')
        text_out.write(f'~NUM_ITER={ss.uncert_setup.iterations}\n')
        text_out.write(f'~TIME_ZERO={ss.site.t0()}\n\n')
        text_out.write('---------------------------------------------------------\n\n')
        text_out.write('        ** DISTRIBUTIONS SUMMARY **\n')
        text_out.write('     All Distributions are Multipliers\n\n')
        return result

    def dist_to_text():
        global text_out

        text_out.write(f'{dist.get_name(True)}\n')
        text_out.write(f'Initial Point Estimate :{dist.get_value():.4f}\n')
        text_out.write(f'{dist.summarize_dist()}\n\n')

    def write_dist_summaries():
        global output_files, iterations_done

        def post_process_results():
            global iterations_done
            for j in range(num_output_files):
                for i in range(NUM_UNCERT_OUTPUTS):
                    sum_ = u_results[j][UResultsTypes.MeanRes][i]
                    sum_square = u_results[j][UResultsTypes.StdRes][i]
                    n = iterations_done
                    if n > 0:
                        u_results[j][UResultsTypes.MeanRes][i] = sum_ / n
                    if n > 1:
                        in_sqrt = ((n * sum_square) - (sum_ * sum_)) / (n * (n - 1))
                    else:
                        in_sqrt = 0
                    if in_sqrt > 0:
                        u_results[j][UResultsTypes.StdRes][i] = np.sqrt(in_sqrt)
                    else:
                        u_results[j][UResultsTypes.StdRes][i] = 0

        if iterations_done > 0:
            post_process_results()
        for of_loop in range(num_output_files):
            output_files[of_loop].write('\n\nVariable Name, Min, Mean, Max, Std. Dev., Deterministic\n')
            for i in range(NUM_UNCERT_OUTPUTS):
                output_files[of_loop].write(
                    f'{ss.col_label[i]},{u_results[of_loop][UResultsTypes.MinRes][i]},{u_results[of_loop][UResultsTypes.MeanRes][i]},'
                    f'{u_results[of_loop][UResultsTypes.MaxRes][i]},{u_results[of_loop][UResultsTypes.StdRes][i]},{u_results[of_loop][UResultsTypes.DetRes][i]}\n'
                )

    def verify_uncert_setup() -> bool:
        count_runs, running_custom_slr = ss.count_runs()
        err_message = ''

        if count_runs != 1:
            err_message = 'Uncertainty runs must be set up to run one SLR scenario / Protection Scenario only.'
        if not running_custom_slr:
            for i in range(ss.uncert_setup.num_dists):
                dist = ss.uncert_setup.dist_array[i]
                if dist.id_num == 1:
                    err_message = 'You must run SLAMM under "Custom SLR" if you are selecting the "Sea Level Rise by 2100 (multiplier)" as an uncertainty distribution.'
        for d_loop in range(ss.uncert_setup.num_dists):
            if conflicting_distribution(d_loop, True):
                err_message = f'Distribution {ss.uncert_setup.dist_array[d_loop].get_name(True)} conflicts with another distribution.'
        if ss.uncert_setup.use_segt_slope:
            found_se = False
            for d_loop in range(ss.uncert_setup.num_dists):
                dist = ss.uncert_setup.dist_array[d_loop]
                if dist.id_num == 7:  # SE
                    found_se = True
                if dist.id_num == 6 and found_se:  # GT
                    err_message = 'Uncertainty setup error: When using GT to calculate salt elevations, all GT distributions must be above SE distributions in the list so they are calculated first.'
                    break
        if err_message:
            ss.user_stop = True
            print(err_message)
            return False
        return True

    def cleanup():
        global gt0, gt1, se0, delta_se
        gt0, gt1, se0, delta_se = None, None, None, None
        for d_loop in range(ss.uncert_setup.num_dists):
            ss.uncert_setup.dist_array[d_loop].point_ests = None

    def save_gt0_gt1():
        global delta_se, gt0, gt1, se0
        delta_se = [False] * (ss.site.n_subsites + 1)
        gt0 = [0.0] * (ss.site.n_subsites + 1)
        gt1 = [0.0] * (ss.site.n_subsites + 1)
        se0 = [0.0] * (ss.site.n_subsites + 1)
        for i in range(ss.site.n_subsites + 1):
            delta_se[i] = False
            if i == 0:
                gt0[0] = ss.site.global_site.gtide_range
                gt1[i] = ss.site.global_site.gtide_range
                se0[i] = ss.site.global_site.salt_elev
            else:
                gt0[i] = ss.site.subsites[i - 1].gtide_range
                gt1[i] = ss.site.subsites[i - 1].gtide_range
                se0[i] = ss.site.subsites[i - 1].salt_elev

    def restore_remaining_ses():
        global delta_se, se0

        for i in range(ss.site.n_subsites):
            if not delta_se[i]:
                if i == 0:
                    ss.site.global_site.salt_elev = se0[i]
                else:
                    ss.site.subsites[i - 1].salt_elev = se0[i]

    def restore_vals():
        for d_loop in range(ss.uncert_setup.num_dists):
            dist = ss.uncert_setup.dist_array[d_loop]
            if dist.dist_type != DistType.Elev_Map:
                if dist.all_subsites and dist.is_subsite_parameter:
                    for i in range(ss.site.n_subsites):
                        dist.subsite_num = i
                        if not conflicting_distribution(d_loop, False):
                            dist.restore_values()
                else:
                    dist.restore_values()
        restore_remaining_ses()

    def call_set_values():
        global text_out, gt1, gt0, delta_se
        ssn = dist.subsite_num
        result = dist.get_value_object(ssn)
        obj, attr, idx = result  # Extract object, attribute, and index from the result

        if obj is None or attr is None:
            raise ValueError("Uncertainty Setup Error -- Invalid object or attribute: obj or attr is None.")

        if dist.id_num == 7 and ss.uncert_setup.use_segt_slope:
            dist.point_ests = [0] * (ss.site.n_subsites + 1)
            pd = getattr(obj, attr)  # Get the attribute value from the object
            dist.point_ests[ssn] = pd
            se1 = pd + ss.uncert_setup.segt_slope * (gt1[ssn] - gt0[ssn])
            text_out.write(f'{dist.get_name(True)} calculated as a function of GT as {se1:.5f}\n')
            new_value = se1 + (draw_value - 1) * pd
            setattr(obj, attr, new_value)  # Set the new value back to the attribute
            delta_se[ssn] = True
        else:
            dist.set_values(draw_value)
            if dist.id_num == 6:
                setattr(obj, attr, draw_value)
                gt1[ssn] = getattr(obj, attr)

    def modify_remaining_ses():
        global text_out, se0, gt0, gt1, delta_se
        for i in range(ss.site.n_subsites):
            if not delta_se[i]:
                ss_name = 'Global' if i == 0 else str(i)
                se1 = se0[i] + ss.uncert_setup.segt_slope * (gt1[i] - gt0[i])
                text_out.write(f'SaltElev for subsite[{ss_name}] calculated as a function of GT as {se1:.5f}\n')
                if i == 0:
                    ss.site.global_site.salt_elev = se1
                else:
                    ss.site.subsites[i - 1].salt_elev = se1

    global text_out, num_output_files, output_files, iterations_done
    gt0, gt1, se0, delta_se = None, None, None, None
    output_files, text_out = None, None
    iterations_done = 0
    unc_analysis = True
    ss.uncert_setup.unc_sens_iter = 0
    user_interrupt = False
    ss.user_stop = False

    if not verify_uncert_setup():
        return

    # try:
    if not init_text_results():
        return

    all_data_file_n = ss.uncert_setup.csv_path
    test_file_n = os.path.splitext(all_data_file_n)[0] + '_alldata.csv'
    test_file_n = test_file_n.replace('\\\\', '\\')

    # if os.path.exists(test_file_n):
    #     if input(f'Overwrite {test_file_n}? (y/n): ').lower() != 'y':
    #         all_data_file_n = input('Select AllData Output File: ')
    #         if not all_data_file_n:
    #             user_interrupt = True
    #             ss.user_stop = True
    #             return

    if ss.uncert_setup.use_seed:
        random.seed(ss.uncert_setup.seed)
    else:
        random.seed()

    print('Deterministic Run...')
    ss.execute_run()

    if ss.user_stop:
        return

    num_output_files = 1 + ss.site.n_output_sites + ss.site.max_ros
    u_results = [{res: np.zeros(NUM_UNCERT_OUTPUTS) for res in UResultsTypes} for _ in range(num_output_files)]

    for of_loop in range(num_output_files):
        for loop in range(NUM_UNCERT_OUTPUTS):  # for each datapoint
            o_val = ss.summary[of_loop][ss.uncert_setup.unc_sens_row - 1][loop]
            u_results[of_loop][UResultsTypes.DetRes][loop] = o_val  # save deterministic results

    print('Calculating Latin Hypercube Draws...')

    for loop in range(ss.uncert_setup.num_dists):
        dist = ss.uncert_setup.dist_array[loop]
        if dist.dist_type != DistType.Elev_Map:
            fill_variable_draws(loop)
        dist_to_text()

    for ns_loop in range(ss.uncert_setup.iterations):
        it_str = f'Iteration {ns_loop+1} of {ss.uncert_setup.iterations}'
        print(it_str)

        if ss.uncert_setup.use_segt_slope:
            save_gt0_gt1()

        for d_loop in range(ss.uncert_setup.num_dists):
            dist = ss.uncert_setup.dist_array[d_loop]
            if dist.dist_type != DistType.Elev_Map:
                draw_value = dist.draws[ns_loop].value
                if dist.all_subsites and dist.is_subsite_parameter:
                    for i in range(ss.site.n_subsites):
                        dist.subsite_num = i
                        if not conflicting_distribution(d_loop, False):
                            call_set_values()
                else:
                    call_set_values()
                text_out.write(f'{dist.get_name(True)} (Multiplied by) {draw_value:.5f}\n')
            else:
                ss.uncert_setup.make_uncert_map(dist.z_map_index(), ss.site.rows, ss.site.cols, dist.parm[1], dist.parm[2])
                text_out.write(f'{dist.get_name(True)}  RMSE {dist.parm[1]:.5f}\n')

        if ss.uncert_setup.use_segt_slope:
            modify_remaining_ses()

        ss.uncert_setup.unc_sens_iter += 1

        ss.execute_run()

        # try:
        #     ss.execute_run()
        # except Exception as e:
        #     restore_vals()
        #     user_interrupt = True
        #     ss.user_stop = True
        #     date_holder = datetime.datetime.now().strftime('%m-%d-%y %H:%M:%S')
        #     text_out.write(f'Execute Model Error at {date_holder}\n')
        #     print('Model Execution Error.')
        #     ss.uncert_setup.z_uncert_map[0] = None
        #     ss.uncert_setup.z_uncert_map[1] = None
        #     break

        restore_vals()

        # prog = round((ns_loop / ss.uncert_setup.iterations) * 100)
        # print(f'Progress: {prog}%')

        if ss.user_stop:
            user_interrupt = True
            ss.user_stop = True
            date_holder = datetime.datetime.now().strftime('%m-%d-%y %H:%M:%S')
            text_out.write(f'Run Tferminated at {date_holder}\n')
            break
        else:
            accumulate_uncertainty_results(ns_loop)

    if not user_interrupt and not ss.user_stop:
        date_holder = datetime.datetime.now().strftime('%m-%d-%y %H:%M:%S')
        text_out.write(f'Run Successfully Completed At {date_holder}\n')
        text_out.write('---------------------------------------------------------\n')

#    except Exception as e:
    unc_analysis = False
    # ss.uncert_setup.z_uncert_map[0] = None
    # ss.uncert_setup.z_uncert_map[1] = None
    # cleanup()
    # print('Run-Time Error During Uncertainty Iteration')
    # print(str(e))
    # date_holder = datetime.datetime.now().strftime('%m-%d-%y %H:%M:%S')
    # try:
    #     text_out.write(f'Run Terminated at {date_holder}\n')
    #     text_out.write(f'    Due to {str(e)}\n')
    # except Exception as e:
    #     print('No Data Written')

    unc_analysis = False
    ss.uncert_setup.z_uncert_map[0] = None
    ss.uncert_setup.z_uncert_map[1] = None

    write_dist_summaries()
    text_out.close()
    for of_loop in range(num_output_files):
        output_files[of_loop].close()

    # try:
    #     if iterations_done > 0:
    #         write_dist_summaries()
    #
    #     text_out.close()
    #     for of_loop in range(num_output_files):
    #         output_files[of_loop].close()

    # except Exception as e:
    #     print('Run-Time Error Writing to File After Uncertainty Run')
    #     print(str(e))
        # text_out.close()
        # for of_loop in range(num_output_files):
        #     output_files[of_loop].close()

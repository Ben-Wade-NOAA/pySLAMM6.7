import datetime
from SLR6 import TSLAMM_Simulation
from app_global import *
from uncert_defn import TSensParam, DistType

NUM_SENS_OUTPUTS = 27

neg_test = False
user_interrupt = False
step_num = 0
num_tests = 0
write_row = 0
text_out = None
neg_str = ""
date_holder = ""
test_val = 0.0
sens_param: Optional[TSensParam] = None
tex = []
file_backed = []
p_max = 0.0
p_min = 0.0
num_output_files = 0
param_names = []
file_names = []

def sens_run(ss: TSLAMM_Simulation):

    def write_outputs(out_f: int, is_determ: bool):
        global text_out, tex
        write_col = 0
        for i in range(NUM_SENS_OUTPUTS):
            write_col += 1
            out_val = ss.summary[out_f][ss.uncert_setup.unc_sens_row - 1][i]
            tex[out_f].write(f'{out_val:.4e}, ')
            if out_f == 0:
                text_out.write(f', {out_val:.4e}')
        tex[out_f].write('\n')
        if is_determ:
            tex[out_f].write('Test Parameter\n')

    def create_csv_files():
        global num_output_files, tex, file_names
        nonlocal ss
        tex = [None] * num_output_files
        file_names = [None] * num_output_files
        for of_loop in range(num_output_files):
            of_string = ''
            if of_loop > 0:
                of_string = f'_OS{of_loop}'
            if of_loop > ss.site.n_output_sites:
                of_string = f'_ROS{of_loop - ss.site.n_output_sites}'
            file_n = os.path.splitext(ss.uncert_setup.output_path)[0] + of_string + '.csv'
            file_n = file_n.replace('\\\\', '\\')
            file_names[of_loop] = file_n
            tex[of_loop] = open(file_n, 'w')

            # Write headers
            headers = [
                          f'{ss.uncert_setup.pct_to_vary}% Sensitivity Test', 'Parameter Value'
                      ] + [ss.col_label[i] for i in range(NUM_SENS_OUTPUTS)]

            tex[of_loop].write(','.join(headers) + '\nBase Case,N A,')

    def save_param_names():
        global num_output_files, write_row, tex, text_out, param_names

        text_out.write('Output Columns --->')

        col_n = 0
        for of_loop in range(num_output_files):

            for i in range(NUM_SENS_OUTPUTS):
                col_n += 1
                param_names.append(ss.col_label[i])
                if of_loop == 0:
                    text_out.write(f', "{ss.col_label[i]}"')
            if of_loop == 0:
                text_out.write('\n\n')
            write_row = 2

    def init_text_results() -> bool:
        global text_out, date_holder, user_interrupt

        cur_dir = os.getcwd()
        date_holder = datetime.datetime.now().strftime('%m-%d-%y')
        base_name = os.path.basename(ss.file_name)
        output_file_name = base_name[:4] + '_sens' + date_holder + '.csv'
        ss.uncert_setup.output_path = os.path.join(cur_dir, output_file_name)

        file_n = os.path.splitext(ss.uncert_setup.output_path)[0] + '_sensitivitylog.txt'
        file_n = file_n.replace('\\\\', '\\')
        text_out = open(file_n, 'w')
        text_out.write('---------------------------------------------------------\n\n')
        text_out.write('            Sensitivity Test for SLAMM 6 Model \n\n')
        text_out.write('---------------------------------------------------------\n')
        date_holder = datetime.datetime.now().strftime('%m-%d-%y %H:%M:%S')
        text_out.write(f'        Run Starts at {date_holder}\n')
        text_out.write('---------------------------------------------------------\n\n')
        return True

    def count_num_tests():
        global num_tests
        num_tests = ss.uncert_setup.num_sens * 2

    def write_sensitivity_results():
        global step_num, num_output_files, write_row, text_out, sens_param, test_val, neg_str, tex, file_names
        for of_loop in range(num_output_files):
            if tex[of_loop].closed:
                tex[of_loop] = open(file_names[of_loop], 'a', newline='')
            if of_loop == 0:
                write_row += 1
            tex[of_loop].write(f'{sens_param.d_name} {neg_str}, {test_val}, ')
            if of_loop == 0:
                text_out.write(f'"{sens_param.d_name} {neg_str}"')
                text_out.write(f' Multiplied by {test_val:.4e}       > ')
            write_outputs(of_loop, False)
            tex[of_loop].close()  # Close the file after each iteration to ensure data is not lost

        text_out.write('\n')
        text_out.write(f'Was Testing {sens_param.d_name} At Value {test_val:.4e}\n')
        text_out.write(f'Iteration {step_num} Completed.  CSV File Updated.\n')
        text_out.write('---------------------------------------------------------\n')

    def calc_sens_parameters():
        global p_max, p_min
        vary_amt = ss.uncert_setup.pct_to_vary / 100
        p_max = 1 + vary_amt
        p_min = 1 - vary_amt

    def restore_vals():
        global sens_param
        if sens_param.dist_type != DistType.Elev_Map:
            sens_param.restore_values()

    def close_csv_files():
        global tex
        try:
            if tex:
                for file_obj in tex:
                    file_obj.close()
        finally:
            tex = []

    global tex, text_out, num_tests, p_max, p_min, step_num, user_interrupt, date_holder, num_output_files, sens_param, neg_str, test_val

    if not verify_setup(ss):
        return
#    try:
    if not init_text_results():
        return

    count_num_tests()

    print('Deterministic Run...')

    for i in range(2):
        ss.uncert_setup.z_uncert_map[i] = None

#        try:
    ss.execute_run()
    num_output_files = 1 + ss.site.n_output_sites + ss.site.max_ros
    create_csv_files()
    save_param_names()

    for of_loop in range(num_output_files):
        write_outputs(of_loop, True)

    text_out.write('\n')
    text_out.write('Deterministic Iteration Completed.  CSV File Updated.\n')
    text_out.write('---------------------------------------------------------\n')

    # except Exception as e:
    #     print(f'Model Execution Error: {str(e)}')
    #     close_csv_files()
    #     text_out.close()
    #     # prog_form.hide()
    #     return

    file_backed = None

    for i in range(2):
        ss.uncert_setup.z_uncert_map[i] = None

    for i in range(ss.uncert_setup.num_sens):
        for neg_test in [False, True]:
            sens_param = ss.uncert_setup.sens_array[i]
            calc_sens_parameters()
            step_num += 1
            ss.uncert_setup.unc_sens_iter += 1

            it_str = f'Iteration {step_num} of {num_tests}'
            # prog_form.uncert_status_label.Caption = it_str

            date_holder = datetime.datetime.now().strftime('%m-%d-%y %H:%M:%S')
            it_str = f'Iteration {step_num} of {num_tests} Starts at {date_holder}'
            text_out.write(it_str + '\n\n')
            print('------------' + it_str)

            if neg_test:
                neg_str = '-'
                test_val = p_min
            else:
                neg_str = '+'
                test_val = p_max
            ss.sens_plus_loop = not neg_test

            sens_param.set_values(test_val)

            try:
                user_interrupt = False
                ss.execute_run()
            except Exception as e:
                restore_vals()
                user_interrupt = True
                date_holder = datetime.datetime.now().strftime('%m-%d-%y %H:%M:%S')
                text_out.write(f'Execute Model Error at {date_holder}\n')
                print(f'Model Execution Error: {str(e)}')
                for j in range(2):
                    ss.uncert_setup.z_uncert_map[j] = None
                break

            sens_param = ss.uncert_setup.sens_array[i]
            restore_vals()

            # application.process_messages()
            # print(f'Sensitivity Run - {step_num} of {num_tests}')

            if ss.user_stop:
                user_interrupt = True

            if user_interrupt:  # prog_form.modal_result or user_interrupt:
                user_interrupt = True
                date_holder = datetime.datetime.now().strftime('%m-%d-%y %H:%M:%S')
                text_out.write(f'Run Terminated at {date_holder}\n')
                break
            else:
                write_sensitivity_results()

    if not user_interrupt:
        date_holder = datetime.datetime.now().strftime('%m-%d-%y %H:%M:%S')
        text_out.write(f'Run Successfully Completed At {date_holder}\n')
        text_out.write('---------------------------------------------------------\n')

    # except Exception as e:
    #     sens_analysis = False
    #     for i in range(2):
    #         ss.uncert_setup.z_uncert_map[i] = None
    #     error_string = str(e)
    #     # prog_form.modal_result = 1
    #     print(f'Run-Time Error During Sensitivity Iteration: {error_string}')
    #     text_out.write(f'Run Terminated at {date_holder}\n')
    #     text_out.write(f'    Due to {error_string}')
    #     close_csv_files()

    # finally:
    # ss.sens_analysis = False

    for i in range(2):
        ss.uncert_setup.z_uncert_map[i] = None
#        try:
#            text_out.close()
#        except Exception as e:
#            error_string = str(e)
#            print(f'Run-Time Error Writing to File After Sensitivity Run: {error_string}')
            # prog_form.modal_result = 1
#            close_csv_files()
        # prog_form.hide()
    close_csv_files()

# Function to verify the setup for sensitivity runs
def verify_setup(ss: TSLAMM_Simulation) -> bool:

    count_runs, running_custom_slr = ss.count_runs()

    if count_runs != 1:
        err_message = 'Sensitivity runs must be set up to run one SLR scenario / Protection Scenario only.'

    if not running_custom_slr:
        for i in range(ss.uncert_setup.num_dists):
            dist = ss.uncert_setup.dist_array[i]
            if dist.id_num == 1:
                err_message = 'You must run SLAMM under "Custom SLR" if you are selecting the "Sea Level Rise by 2100 (multiplier)" as an sensitivity parameter.'
    if err_message:
        ss.user_stop = True
        print(err_message)
        return False
    return True


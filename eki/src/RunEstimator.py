from function_tracker import track_function
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'text.usetex': False, 'text.latex.preamble':
    '\\usepackage{gensymb}'})
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import sys
import time
import argparse
import yaml
import pickle
import Optimizer_EKI_np
import datetime
import pandas as pd
import shutil
import glob


@track_function
def _parse():
    parser = argparse.ArgumentParser(description='Run EKI')
    parser.add_argument('input_config', help='Check input_config')
    parser.add_argument('input_data', help='Name of input_data')
    return parser.parse_args()


@track_function
def _read_file(input_config, input_data):
    with open(input_config, 'r') as config:
        input_config = yaml.load(config, yaml.SafeLoader)
    with open(input_data, 'r') as data:
        input_data = yaml.load(data, yaml.SafeLoader)
    return input_config, input_data


@track_function
def progressbar(it, prefix='', size=60, file=sys.stdout):
    count = len(it)

    @track_function
    def show(j):
        x = int(size * j / count)
        file.write('%s[%s%s] %i/%i\r' % (prefix, '#' * x, '.' * (size - x),
            j, count))
        file.flush()
        file.write('\n')
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
        file.write('\n')
    file.flush()


@track_function
def save_results(dir_out, results):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    file_out = dir_out + '/' + dir_out + '.p'
    pickle.dump(results, open(file_out, 'wb'))


if __name__ == '__main__':
    args = _parse()
    input_config, input_data = _read_file(args.input_config, args.input_data)
Checktime_List = []
Optimization_list = input_config['Optimizer_order']
for op in range(len(Optimization_list)):
    opt = Optimization_list[op]
    Checktime_list1 = []
    Checktime_list2 = []
    print('Start:' + opt)
    for e in range(input_config['nrepeat']):
        input_config, input_data = _read_file(args.input_config, args.
            input_data)
        input_config['Optimizer'] = opt
        input_config['sample'] = input_config['sample_ctrl'] * (e + 1)
        print('Sample:' + str(input_config['sample']))
        sample = input_config['sample']
        ave = []
        err = []
        Best_List = []
        Misfits_List = []
        Discrepancy_bools_List = []
        Residual_bools_List = []
        Residuals_List = []
        Misfit_List = []
        Discrepancy_bool_List = []
        Residual_bool_List = []
        Residual_List = []
        Noise_List = []
        EnsXiter_List = []
        Diff_List = []
        Time_List = []
        Info_list = []
        receptor_range = input_data['nreceptor']
        t1 = time.time()
        for i in progressbar(range(1, receptor_range + 1), 'Computing: ', 40):
            input_config, input_data = _read_file(args.input_config, args.
                input_data)
            input_config['Optimizer'] = opt
            input_config['sample'] = input_config['sample_ctrl'] * (e + 1)
            posterior0 = None
            posterior_iter0 = None
            info_list = []
            misfit_list = []
            discrepancy_bool_list = []
            residual_bool_list = []
            residual_list = []
            noise_list = []
            ensXiter_list = []
            diff_list = []
            misfits_list = []
            discrepancy_bools_list = []
            residual_bools_list = []
            residuals_list = []
            t2i = time.time()
            if input_config['Receptor_Increment'] == 'Off':
                input_data['nreceptor'] = receptor_range
                print(f'receptor:', input_data['nreceptor'])
            elif input_config['Receptor_Increment'] == 'On':
                input_data['nreceptor'] = i
                print(f'receptor:', input_data['nreceptor'])
            else:
                print('Check the number of receptor')
                break
            (posterior0, posterior_iter0, info_list, misfit_list,
                discrepancy_bool_list, residual_bool_list, residual_list,
                noise_list, ensXiter_list, diff_list, misfits_list,
                discrepancy_bools_list, residual_bools_list, residuals_list
                ) = Optimizer_EKI_np.Run(input_config, input_data)
            Info_list = info_list
            posterior = posterior0.copy()
            Best_List.append(posterior_iter0.copy())
            Misfits_List.append(misfits_list.copy())
            Discrepancy_bools_List.append(discrepancy_bools_list.copy())
            Residual_bools_List.append(residual_bools_list.copy())
            Residuals_List.append(residuals_list.copy())
            Misfit_List.append(misfit_list.copy())
            Discrepancy_bool_List.append(discrepancy_bool_list.copy())
            Residual_bool_List.append(residual_bool_list.copy())
            Residual_List.append(residual_list.copy())
            Noise_List.append(noise_list.copy())
            EnsXiter_list = np.array(ensXiter_list.copy())
            EnsXiter = 0 if np.nonzero(EnsXiter_list)[0
                ].size == 0 else EnsXiter_list[np.nonzero(EnsXiter_list)[0][0]]
            EnsXiter_List.append(EnsXiter)
            Diff_List.append(diff_list.copy())
            ave.append(np.mean(posterior, 1))
            err.append(np.std(posterior, 1))
            if input_config['Receptor_Increment'] == 'Off':
                Best_Iter0 = np.mean(np.array(Best_List[0]), axis=2)
                Best_Iter0_std = np.std(np.array(Best_List[0]), axis=2)
            elif input_config['Receptor_Increment'] == 'On':
                Best_Iter0 = None
                Best_Iter0_std = None
                Best_Iter0 = np.mean(np.array(Best_List[-1]), axis=2)
                Best_Iter0_std = np.std(np.array(Best_List[-1]), axis=2)
            else:
                print('Check the number of receptor_increment')
                break
            if input_data['Source_location'] == 'Fixed':
                Best_Iter_reshape = None
                Best_Iter_std_reshape = None
                print(f'Debug: Best_Iter0 shape: {Best_Iter0.shape}')
                print(f'Debug: Best_Iter0[-1] shape: {Best_Iter0[-1].shape}')
                print(f'Debug: Best_Iter0[-1] size: {Best_Iter0[-1].size}')
                print(f"Debug: nsource: {input_data['nsource']}")
                print(
                    f"Debug: time_intervals: {int(input_data['time'] * 24 / input_data['inverse_time_interval'])}"
                    )
                expected_size = input_data['nsource'] * int(input_data[
                    'time'] * 24 / input_data['inverse_time_interval'])
                print(f'Debug: expected size: {expected_size}')
                if Best_Iter0[-1].size == expected_size:
                    Best_Iter_reshape = Best_Iter0[-1].reshape([input_data[
                        'nsource'], int(input_data['time'] * 24 /
                        input_data['inverse_time_interval'])])
                    Best_Iter_std_reshape = Best_Iter0_std[-1].reshape([
                        input_data['nsource'], int(input_data['time'] * 24 /
                        input_data['inverse_time_interval'])])
                else:
                    print(
                        f'Warning: Size mismatch. Using data as is for plotting.'
                        )
                    Best_Iter_reshape = Best_Iter0[-1] if Best_Iter0[-1
                        ].ndim > 0 else Best_Iter0[-1].reshape(1, -1)
                    Best_Iter_std_reshape = Best_Iter0_std[-1
                        ] if Best_Iter0_std[-1].ndim > 0 else Best_Iter0_std[-1
                        ].reshape(1, -1)
            elif input_data['Source_location'] == 'Single':
                Best_Iter_reshape = None
                Best_Iter_std_reshape = None
                Best_Iter_reshape_position = None
                Best_Iter_std_reshape_position = None
                Best_Iter_reshape = Best_Iter0[-1][3:].reshape([input_data[
                    'nsource'], int(input_data['time'] * 24 / input_data[
                    'inverse_time_interval'])])
                Best_Iter_std_reshape = Best_Iter0_std[-1][3:].reshape([
                    input_data['nsource'], int(input_data['time'] * 24 /
                    input_data['inverse_time_interval'])])
                Best_Iter_reshape_position = Best_Iter0[-1][:3]
                Best_Iter_std_reshape_position = Best_Iter0_std[-1][:3]
            elif input_data['Source_location'] == 'Multiple':
                Best_Iter_reshape = None
                Best_Iter_std_reshape = None
                Best_Iter_reshape = Best_Iter0[-1][:].reshape([input_data[
                    'nsource'], int(input_data['time'] * 24 / input_data[
                    'inverse_time_interval']) + 3])
                Best_Iter_std_reshape = Best_Iter0_std[-1][:].reshape([
                    input_data['nsource'], int(input_data['time'] * 24 /
                    input_data['inverse_time_interval']) + 3])

            @track_function
            def plot_iteration_graph(iterations, means, stds, file_path,
                csv_file_path):
                plt.figure(figsize=(10, 6))
                plt.plot(iterations, means, label='Mean', color='blue')
                plt.errorbar(iterations, means, yerr=stds, fmt='o', color=
                    'red', label='Standard Deviation')
                plt.xlabel('Iterations')
                plt.ylabel('Values')
                plt.title('Mean and Standard Deviation over Iterations')
                plt.legend()
                plt.grid(True)
                plt.savefig(file_path)
                plt.close()
                data = {'Iterations': iterations, 'Mean': means,
                    'Standard Deviation': stds}
                df = pd.DataFrame(data)
                df.to_csv(csv_file_path, index=False)

            @track_function
            def plot_time_range_graph(time_range, means_list, stds_list,
                labels, file_path, csv_file_path):
                plt.figure(figsize=(10, 6))
                for i, (means, stds, label) in enumerate(zip(means_list,
                    stds_list, labels)):
                    if 'True value' in label:
                        plt.plot(time_range, means, label=label, color=
                            'green', linewidth=2)
                    else:
                        plt.plot(time_range, means, label=label, color='blue')
                        if stds is not None and any(stds):
                            plt.errorbar(time_range, means, yerr=stds, fmt=
                                'o', color='orange', label=f'Std Dev {label}')
                plt.xlabel('Time (hours)')
                plt.ylabel('Concentration')
                plt.title('Emission Source Estimation over Time')
                plt.legend()
                plt.grid(True)
                plt.savefig(file_path)
                plt.close()
                data = {'Time_range': time_range}
                for means, stds, label in zip(means_list, stds_list, labels):
                    data[f'Mean {label}'] = means
                    data[f'Std Dev {label}'] = stds
                df = pd.DataFrame(data)
                df.to_csv(csv_file_path, index=False)

            @track_function
            def plot_position_graph(means, stds, file_path, csv_file_path):
                plt.figure(figsize=(10, 6))
                plt.plot(['x', 'y', 'z'], means, label='Mean', color='blue')
                plt.errorbar(['x', 'y', 'z'], means, yerr=stds, fmt='o',
                    color='red', label='Standard Deviation')
                plt.xlabel('Position')
                plt.ylabel('Values')
                plt.title('Mean and Standard Deviation over (x,y,z)')
                plt.legend()
                plt.grid(True)
                plt.savefig(file_path)
                plt.close()
                data = {'Mean': means, 'Standard Deviation': stds}
                df = pd.DataFrame(data)
                df.to_csv(csv_file_path, index=False)
            output_path = f'./results_ens{sample}'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            current_nreceptor = input_data['nreceptor']
            num_intervals = int(input_data['time'] * 24 / input_data[
                'inverse_time_interval'])
            time_range = [(input_data['inverse_time_interval'] * i) for i in
                range(num_intervals)]
            for s in range(np.array(Best_Iter0).shape[1]):
                iterations = [(iter + 1) for iter in range(len(Best_Iter0[:,
                    s].tolist()))]
                plot_iteration_graph(iterations, Best_Iter0[:, s], 
                    Best_Iter0_std[:, s] * 2.58 / len(Best_Iter0_std[:, s]) **
                    0.5,
                    f'./results_ens{sample}/plot_receptor_{current_nreceptor}_all_source_{s}.png'
                    ,
                    f'./results_ens{sample}/plot_receptor_{current_nreceptor}_all_source_{s}.csv'
                    )
            try:
                if 'Source_1' in input_data and len(input_data['Source_1']
                    ) > 3:
                    source_concentrations = input_data['Source_1'][3]
                    if isinstance(source_concentrations, list) and len(
                        source_concentrations) == num_intervals:
                        source_data = [source_concentrations]
                        print(
                            f"Loaded true values from input_data: {len(source_concentrations)} concentration values for {input_data['time'] * 24} hours"
                            )
                    else:
                        raise ValueError(
                            f'Source_1 concentrations must be a list of {num_intervals} values'
                            )
                else:
                    raise KeyError(
                        'Source_1 not found or incomplete in input_data')
            except Exception as e:
                print(
                    f'Warning: Could not load source data from input_data: {e}'
                    )
                print('Using fallback hardcoded values')
                source_data = [[100000.0] * num_intervals]
            source_std = [([0] * len(source)) for source in source_data]
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            for s in range(input_data['nsource']):
                if input_data['Source_location'] == 'Fixed':
                    if Best_Iter_reshape.ndim == 2:
                        est_data = Best_Iter_reshape[s, :]
                        est_std = Best_Iter_std_reshape[s, :] * 2.58 / len(
                            Best_Iter_std_reshape[s, :]) ** 0.5
                    else:
                        est_data = [Best_Iter_reshape[0]] * len(source_data[s])
                        est_std = [Best_Iter_std_reshape[0]] * len(source_data
                            [s])
                    means_list = [est_data, source_data[s]]
                    stds_list = [est_std, source_std[s]]
                    labels = ['Estimated', f'True value Source {s + 1}']
                    plot_time_range_graph(time_range, means_list, stds_list,
                        labels,
                        f'./results_ens{sample}/plot_receptor_{current_nreceptor}_time_range_{s}.png'
                        ,
                        f'./results_ens{sample}/plot_receptor_{current_nreceptor}_time_range_{s}.csv'
                        )
                    ldm_results_dir = (
                        '/home/jrpark/EKI-LDM5-dev/ldm-20241030/eki_results')
                    os.makedirs(ldm_results_dir, exist_ok=True)
                    existing_files = glob.glob(os.path.join(ldm_results_dir,
                        '*.png'))
                    existing_nums = []
                    for f in existing_files:
                        basename = os.path.basename(f)
                        name_without_ext = os.path.splitext(basename)[0]
                        if name_without_ext.isdigit():
                            existing_nums.append(int(name_without_ext))
                    next_num = 0
                    while next_num in existing_nums:
                        next_num += 1
                    src_png = (
                        f'./results_ens{sample}/plot_receptor_{current_nreceptor}_time_range_{s}.png'
                        )
                    dst_png = os.path.join(ldm_results_dir, f'{next_num}.png')
                    if os.path.exists(src_png):
                        shutil.copy2(src_png, dst_png)
                        print(f'Saved plot as: {dst_png}')
                elif input_data['Source_location'] == 'Single':
                    plot_time_range_graph(time_range, Best_Iter_reshape[s,
                        :], Best_Iter_std_reshape[s, :] * 2.58 / len(
                        Best_Iter_std_reshape[s, :]) ** 0.5,
                        f'./results_ens{sample}/plot_receptor_{current_nreceptor}_time_range_{s}.png'
                        ,
                        f'./results_ens{sample}/plot_receptor_{current_nreceptor}_time_range_{s}.csv'
                        )
                    plot_position_graph(Best_Iter_reshape_position, 
                        Best_Iter_std_reshape_position * 2.58 / len(
                        Best_Iter_std_reshape_position) ** 0.5,
                        f'./results_ens{sample}/plot_receptor_{current_nreceptor}_position.png'
                        ,
                        f'./results_ens{sample}/plot_receptor_{current_nreceptor}_position.csv'
                        )
                    ldm_results_dir = (
                        '/home/jrpark/EKI-LDM5-dev/ldm-20241030/eki_results')
                    os.makedirs(ldm_results_dir, exist_ok=True)
                    existing_files = glob.glob(os.path.join(ldm_results_dir,
                        '*.png'))
                    existing_nums = []
                    for f in existing_files:
                        basename = os.path.basename(f)
                        name_without_ext = os.path.splitext(basename)[0]
                        if name_without_ext.isdigit():
                            existing_nums.append(int(name_without_ext))
                    next_num = 0
                    while next_num in existing_nums:
                        next_num += 1
                    src_png = (
                        f'./results_ens{sample}/plot_receptor_{current_nreceptor}_time_range_{s}.png'
                        )
                    dst_png = os.path.join(ldm_results_dir, f'{next_num}.png')
                    if os.path.exists(src_png):
                        shutil.copy2(src_png, dst_png)
                        print(f'Saved plot as: {dst_png}')
                elif input_data['Source_location'] == 'Multiple':
                    plot_time_range_graph(time_range, Best_Iter_reshape[s, 
                        3:], Best_Iter_std_reshape[s, 3:] * 2.58 / len(
                        Best_Iter_std_reshape[s, 3:]) ** 0.5,
                        f'./results_ens{sample}/plot_receptor_{current_nreceptor}_time_range_{s}.png'
                        ,
                        f'./results_ens{sample}/plot_receptor_{current_nreceptor}_time_range_{s}.csv'
                        )
                    plot_position_graph(Best_Iter_reshape[s, :3], 
                        Best_Iter_std_reshape[s, :3] * 2.58 / len(
                        Best_Iter_std_reshape[s, :3]) ** 0.5,
                        f'./results_ens{sample}/plot_receptor_{current_nreceptor}_position_{s}.png'
                        ,
                        f'./results_ens{sample}/plot_receptor_{current_nreceptor}_position_{s}.csv'
                        )
                    ldm_results_dir = (
                        '/home/jrpark/EKI-LDM5-dev/ldm-20241030/eki_results')
                    os.makedirs(ldm_results_dir, exist_ok=True)
                    existing_files = glob.glob(os.path.join(ldm_results_dir,
                        '*.png'))
                    existing_nums = []
                    for f in existing_files:
                        basename = os.path.basename(f)
                        name_without_ext = os.path.splitext(basename)[0]
                        if name_without_ext.isdigit():
                            existing_nums.append(int(name_without_ext))
                    next_num = 0
                    while next_num in existing_nums:
                        next_num += 1
                    src_png = (
                        f'./results_ens{sample}/plot_receptor_{current_nreceptor}_time_range_{s}.png'
                        )
                    dst_png = os.path.join(ldm_results_dir, f'{next_num}.png')
                    if os.path.exists(src_png):
                        shutil.copy2(src_png, dst_png)
                        print(f'Saved plot as: {dst_png}')
            print(i / receptor_range * 100)
            t2 = time.time()
            Time_List.append(t2 - t2i)
            print(f'Time:', t2 - t1)
            if input_config['Receptor_Increment'] == 'Off':
                sys.exit()
        t3 = time.time()
        print(f'Time:', t3 - t1)
        dir_str = input_config['Optimizer'] + '_nsource' + str(input_data[
            'nsource']) + '_nreceptor' + str(input_data['nreceptor']
            ) + '_sample' + str(input_config['sample']) + '_iteration' + str(
            input_config['iteration']) + '_err' + str(input_data[
            'nreceptor_err'])
        dir_out = 'results_{}'.format(dir_str)
        os.makedirs(dir_out, exist_ok=True)
        results = (ave, err, Best_List, Misfits_List,
            Discrepancy_bools_List, Residual_bools_List, Residuals_List,
            Misfit_List, Discrepancy_bool_List, Residual_bool_List,
            Residual_List, Noise_List, EnsXiter_List, Diff_List, Time_List,
            Info_list)
        save_results(dir_out, results)
        nreceptor = []
        seepoint = receptor_range
        for i in range(0, seepoint):
            nreceptor.append(i + 1)
        receptorPoint = receptor_range
        iterations = [(i + 1) for i in range(input_config['iteration'])]
        if input_config['Optimizer'] == 'EKI':
            Best_Iter = np.mean(np.array(Best_List[receptorPoint - 1]), axis=2)
            Best_Iter_std = np.std(np.array(Best_List[receptorPoint - 1]),
                axis=2)
            Residuals_Iter = np.array(Residuals_List[receptorPoint - 1][1:])
        else:
            Best_Iter = np.array(Best_List[receptorPoint - 1])
            Residuals_Iter = np.array(Residuals_List[receptorPoint - 1][1:])
        Checktime_list1.append([nreceptor, Time_List, EnsXiter_List])

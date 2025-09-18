from function_tracker import track_function
import numpy as np
import concurrent.futures
import multiprocessing
from multiprocessing import Process, Queue
from copy import deepcopy
import socket
import struct
import time
desired_gpu_index_cupy = 0


@track_function
def receive_gamma_dose_matrix(expected_rows, expected_cols):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 8080))
    expected_total_bytes = expected_rows * expected_cols * 4
    data = b''
    while len(data) < expected_total_bytes:
        packet = client_socket.recv(expected_total_bytes - len(data))
        if not packet:
            break
        data += packet
    client_socket.close()
    actual_elements = len(data) // 4
    print(
        f'Expected {expected_rows}x{expected_cols} = {expected_rows * expected_cols} elements'
        )
    print(f'Received {len(data)} bytes = {actual_elements} elements')
    gamma_dose_matrix = struct.unpack(f'<{actual_elements}f', data)
    if actual_elements == expected_rows * expected_cols:
        reshaped_data = np.expand_dims(np.array(gamma_dose_matrix).reshape(
            (expected_cols, expected_rows)).transpose(), axis=0)
    else:
        padded_matrix = np.zeros(expected_rows * expected_cols)
        copy_size = min(actual_elements, expected_rows * expected_cols)
        padded_matrix[:copy_size] = gamma_dose_matrix[:copy_size]
        reshaped_data = np.expand_dims(padded_matrix.reshape((expected_cols,
            expected_rows)).transpose(), axis=0)
    return reshaped_data


@track_function
def send_tmp_states(tmp_states):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 8080))
    print(f'=== EKI TRANSMISSION DEBUG ===')
    print(f'Input tmp_states shape: {tmp_states.shape}')
    print(f'Input tmp_states ndim: {tmp_states.ndim}')
    if tmp_states.ndim == 1:
        tmp_states = tmp_states.reshape(-1, 1)
        print(f'Reshaped 1D array to: {tmp_states.shape}')
    elif tmp_states.shape[0] != 24:
        if tmp_states.shape[1] == 24:
            tmp_states = tmp_states.T
            print(f'Transposed to correct format: {tmp_states.shape}')
        else:
            raise ValueError(
                f'Invalid state shape after processing: {tmp_states.shape}. Expected (24, ensemble_size)'
                )
    if tmp_states.shape[0] != 24:
        raise ValueError(
            f'Final state must have 24 time intervals, got shape: {tmp_states.shape}'
            )
    rows, cols = tmp_states.shape
    print(f'Final tmp_states shape being sent: {rows}x{cols}')
    print(f'Format: (time_intervals={rows}, ensemble_size={cols})')
    print(
        f'State min/max/mean: {np.min(tmp_states):.2e}/{np.max(tmp_states):.2e}/{np.mean(tmp_states):.2e}'
        )
    print(f'First few values: {tmp_states.flatten()[:5]}')
    if cols > 1:
        print(f'First time interval for all ensembles: {tmp_states[0, :]}')
    print(f'Non-zero values: {np.sum(tmp_states != 0)}/{tmp_states.size}')
    dimensions = struct.pack('ii', rows, cols)
    client_socket.sendall(dimensions)
    data = tmp_states.astype(np.float32).tobytes()
    client_socket.sendall(struct.pack('I', len(data)))
    client_socket.sendall(data)
    print(f'tmp_states data sent successfully! Shape: {tmp_states.shape}')
    print(f'Data size: {len(data)} bytes ({len(data) // 4} floats)')
    print(f'Expected size: {24 * cols * 4} bytes ({24 * cols} floats)')
    with open('/home/jrpark/EKI-LDM5-dev/eki_ldm_debug_log.txt', 'a'
        ) as debug_log:
        debug_log.write('\n=== EKI STATE TRANSMISSION VERIFICATION ===\n')
        debug_log.write(f'EKI -> LDM Communication (Port 8080):\n')
        debug_log.write(f'Sent state matrix dimensions: {rows} x {cols}\n')
        debug_log.write(f'Expected: 24 time intervals x 100 ensemble members\n'
            )
        debug_log.write(f'Data size: {len(data)} bytes\n')
        debug_log.write(f'Transmission status: SUCCESS\n')
        debug_log.write(f'\nState Data Statistics:\n')
        debug_log.write(
            f'Value range: {np.min(tmp_states):.2e} to {np.max(tmp_states):.2e} Bq/s\n'
            )
        debug_log.write(f'Mean: {np.mean(tmp_states):.2e} Bq/s\n')
        zero_count = np.sum(tmp_states == 0)
        total_count = tmp_states.size
        debug_log.write(f'Zero count: {zero_count}/{total_count}\n')
        debug_log.write(f'First 5 values: {list(tmp_states.flatten()[:5])}\n')
        debug_log.write(f'\nSample Time Series (Ensemble 0):\n')
        if cols > 0:
            for t in [0, 6, 12, 18, 23]:
                if t < rows:
                    debug_log.write(f'  t={t}: {tmp_states[t, 0]:.2e} Bq/s\n')
    client_socket.close()


@track_function
def receive_gamma_dose_matrix_ens():
    """Receive gamma dose matrix from LDM via binary socket communication"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    port = 8080
    max_retries = 5
    for attempt in range(max_retries):
        try:
            server_socket.bind(('127.0.0.1', port))
            break
        except OSError as e:
            if e.errno == 98 and attempt < max_retries - 1:
                print(
                    f'Port {port} in use, retrying in 1 second... (attempt {attempt + 1}/{max_retries})'
                    )
                time.sleep(1)
            else:
                raise
    server_socket.listen(1)
    print('Waiting for LDM connection on port 8080...')
    conn, addr = server_socket.accept()
    print(f'Connection from {addr} established.')
    try:
        dimensions_data = conn.recv(12)
        if len(dimensions_data) != 12:
            raise RuntimeError('Failed to receive complete dimension data')
        ensemble_size, num_receptors, time_intervals = struct.unpack('iii',
            dimensions_data)
        print(
            f'Received matrix dimensions: {ensemble_size} ensembles × {num_receptors} receptors × {time_intervals} time intervals'
            )
        expected_size = ensemble_size * num_receptors * time_intervals * 4
        print(f'Expecting {expected_size} bytes of gamma dose data')
        data = b''
        while len(data) < expected_size:
            packet = conn.recv(min(4096, expected_size - len(data)))
            if not packet:
                break
            data += packet
        if len(data) != expected_size:
            raise RuntimeError(
                f'Received {len(data)} bytes, expected {expected_size} bytes')
        gamma_dose_flat = np.frombuffer(data, dtype=np.float32)
        h_gamma_dose_3d = gamma_dose_flat.reshape((ensemble_size,
            num_receptors, time_intervals))
        print(
            f'Successfully received gamma dose matrix: {h_gamma_dose_3d.shape}'
            )
        print(
            f'Data range: [{np.min(h_gamma_dose_3d):.2e}, {np.max(h_gamma_dose_3d):.2e}]'
            )
        with open('/home/jrpark/EKI-LDM5-dev/eki_reception_log.txt', 'w'
            ) as reception_log:
            reception_log.write(
                """===============================================================================
"""
                )
            reception_log.write('EKI BINARY RECEPTION LOG\n')
            reception_log.write(
                """===============================================================================
"""
                )
            reception_log.write(f"Timestamp: {np.datetime64('now')}\n")
            reception_log.write(
                f"""Dimensions received: [{ensemble_size}, {num_receptors}, {time_intervals}]
"""
                )
            reception_log.write(
                f"""Binary data size: {expected_size} bytes ({len(gamma_dose_flat)} floats)
"""
                )
            zero_count = np.sum(h_gamma_dose_3d == 0)
            total_count = h_gamma_dose_3d.size
            reception_log.write(
                f"""Data statistics: min={np.min(h_gamma_dose_3d):.6e}, max={np.max(h_gamma_dose_3d):.6e}, mean={np.mean(h_gamma_dose_3d):.6e}
"""
                )
            reception_log.write(
                f"""Zero percentage: {100 * zero_count / total_count:.1f}% ({zero_count}/{total_count})

"""
                )
            reception_log.write('=== SAMPLE DATA FOR VERIFICATION ===\n')
            for ens in range(min(2, ensemble_size)):
                for r in range(num_receptors):
                    reception_log.write(
                        f'Ensemble {ens}, Receptor {r} time series:\n[')
                    for t in range(time_intervals):
                        if t > 0:
                            reception_log.write(', ')
                        reception_log.write(f'{h_gamma_dose_3d[ens, r, t]:.6e}'
                            )
                    reception_log.write(']\n')
                reception_log.write('\n')
            reception_log.write('=== RAW BINARY DATA INTEGRITY CHECK ===\n')
            reception_log.write('First 10 values (linear indices 0-9):\n')
            for i in range(min(10, len(gamma_dose_flat))):
                reception_log.write(f'[{i}] = {gamma_dose_flat[i]:.6e}\n')
            reception_log.write(
                f"""
Last 10 values (linear indices {len(gamma_dose_flat) - 10}-{len(gamma_dose_flat) - 1}):
"""
                )
            for i in range(max(0, len(gamma_dose_flat) - 10), len(
                gamma_dose_flat)):
                reception_log.write(f'[{i}] = {gamma_dose_flat[i]:.6e}\n')
            reception_log.write('\nReception status: COMPLETED\n')
        with open('/home/jrpark/EKI-LDM5-dev/eki_ldm_debug_log.txt', 'a'
            ) as debug_log:
            debug_log.write('\n=== EKI DATA RECEPTION VERIFICATION ===\n')
            debug_log.write(f'LDM -> EKI Communication (Port 8080):\n')
            debug_log.write(
                f"""Received gamma dose dimensions: {ensemble_size} x {num_receptors} x {time_intervals}
"""
                )
            debug_log.write(f'Data size: {expected_size} bytes\n')
            debug_log.write(f'Reception status: SUCCESS\n')
            debug_log.write(f'\nGamma Dose Data Statistics:\n')
            debug_log.write(
                f"""Value range: {np.min(h_gamma_dose_3d):.2e} to {np.max(h_gamma_dose_3d):.2e} Sv/h
"""
                )
            debug_log.write(f'Mean: {np.mean(h_gamma_dose_3d):.2e} Sv/h\n')
            zero_count = np.sum(h_gamma_dose_3d == 0)
            total_count = h_gamma_dose_3d.size
            debug_log.write(
                f"""Zero count: {zero_count}/{total_count} ({100 * zero_count / total_count:.1f}%)
"""
                )
            debug_log.write(
                f'\nSample Gamma Dose Data (Ensemble 0, Receptor 0):\n')
            debug_log.write('Time series for ens=0, receptor=0:\n')
            for t in [0, 6, 12, 18, 23]:
                if t < time_intervals:
                    dose_val = h_gamma_dose_3d[0, 0, t
                        ] if t < h_gamma_dose_3d.shape[2] else 0
                    debug_log.write(f'  t={t}: {dose_val:.2e} Sv/h\n')
            debug_log.write(f'\nTotal transmission statistics:\n')
            nonzero_count = np.sum(h_gamma_dose_3d != 0)
            debug_log.write(
                f"""Non-zero observations: {nonzero_count}/{total_count} ({100 * nonzero_count / total_count:.1f}%)
"""
                )
            debug_log.write(f'Data integrity check: PASSED\n')
        with open('/home/jrpark/EKI-LDM5-dev/eki_obs_debug_log.txt', 'w'
            ) as obs_debug:
            obs_debug.write(
                """===============================================================================
"""
                )
            obs_debug.write(
                'EKI OBSERVATION DEBUG LOG - GAMMA DOSE ARRAY ANALYSIS\n')
            obs_debug.write(
                """===============================================================================
"""
                )
            obs_debug.write(f"Generated: {np.datetime64('now')}\n")
            obs_debug.write(
                """Purpose: Verify gamma dose array layout, reshape consistency, and observation quality

"""
                )
            obs_debug.write(
                '=== 1) RAW GAMMA DOSE ARRAY LAYOUT VERIFICATION ===\n')
            obs_debug.write(
                f"""Raw array shape after receive_gamma_dose_matrix_ens: {h_gamma_dose_3d.shape}
"""
                )
            obs_debug.write(
                f"""Array layout: [ensemble_size={ensemble_size}, num_receptors={num_receptors}, time_intervals={time_intervals}]
"""
                )
            obs_debug.write(f'Total elements: {h_gamma_dose_3d.size}\n')
            obs_debug.write(
                f'Memory layout: C-contiguous = {h_gamma_dose_3d.flags.c_contiguous}\n'
                )
            obs_debug.write(
                f'\nEnsemble 0, Receptor 0 - Complete 24 time series:\n')
            ens0_rec0_series = h_gamma_dose_3d[0, 0, :].tolist()
            obs_debug.write(f'{ens0_rec0_series}\n')
            obs_debug.write(
                f'\nEnsemble 0, Receptor 1 - Complete 24 time series:\n')
            ens0_rec1_series = h_gamma_dose_3d[0, 1, :].tolist()
            obs_debug.write(f'{ens0_rec1_series}\n')
            if num_receptors >= 3:
                obs_debug.write(
                    f'\nEnsemble 0, Receptor 2 - Complete 24 time series:\n')
                ens0_rec2_series = h_gamma_dose_3d[0, 2, :].tolist()
                obs_debug.write(f'{ens0_rec2_series}\n')
            obs_debug.write(
                f'\nTime series stored for reshape consistency verification...\n'
                )
        return h_gamma_dose_3d
    finally:
        try:
            conn.close()
        except:
            pass
        try:
            server_socket.close()
        except:
            pass


class Model(object):

    @track_function
    def __init__(self, input_config, input_data):
        self.name = 'gaussian_puff_model'
        self.nGPU = input_config['nGPU']
        self.input_data = input_data
        self.sample = input_config['sample']
        self.nsource = input_data['nsource']
        self.nreceptor = input_data['nreceptor']
        self.nreceptor_err = input_data['nreceptor_err']
        self.nreceptor_MDA = input_data['nreceptor_MDA']
        if input_data['receptor_position'] == []:
            input_data['receptor_position'] = [list(np.random.randint(low=[
                input_data['xreceptor_min'], input_data['yreceptor_min'],
                input_data['zreceptor_min']], high=[input_data[
                'xreceptor_max'] + 1, input_data['yreceptor_max'] + 1, 
                input_data['zreceptor_max'] + 1]).astype(float)) for _ in
                range(self.nreceptor)]
        self.real_state_init_list = []
        self.real_decay_list = []
        self.real_source_location_list = []
        self.real_dosecoeff_list = []
        self.total_real_state_list = []
        for s in range(self.nsource):
            actual_source = 'Source_{0}'.format(s + 1)
            self.real_decay_list.append(self.input_data[actual_source][0])
            self.real_dosecoeff_list.append(self.input_data[actual_source][1])
            self.real_source_location_list.append(self.input_data[
                actual_source][2])
            self.real_state_init_list.append(self.input_data[actual_source][3])
            self.total_real_state_list.append(input_data[actual_source][2] +
                input_data[actual_source][3])
        self.real_state_init = np.array(self.real_state_init_list).reshape(-1)
        self.real_decay = np.array(self.real_decay_list)
        self.real_source_location = np.array(self.real_source_location_list).T
        self.real_dosecoeff = np.array(self.real_dosecoeff_list)
        self.state_init_list = []
        self.source_location_list = []
        self.state_std_list = []
        self.source_location_std_list = []
        self.decay_list = []
        self.total_state_list = []
        self.total_state_std_list = []
        for s in range(self.nsource):
            source = 'Prior_Source_{0}'.format(s + 1)
            self.source_location_list.append(input_data[source][2][0])
            self.state_init_list.append(input_data[source][3][0])
            self.source_location_std_list.append((np.array(input_data[
                source][2][0]) * input_data[source][2][1]).tolist())
            self.state_std_list.append((np.array(input_data[source][3][0]) *
                input_data[source][3][1]).tolist())
            self.decay_list.append(input_data[source][0])
            self.total_state_list.append(input_data[source][2][0] +
                input_data[source][3][0])
            self.total_state_std_list.append((np.array(input_data[source][2
                ][0]) * input_data[source][2][1]).tolist() + (np.array(
                input_data[source][3][0]) * input_data[source][3][1]).tolist())
        if input_data['Source_location'] == 'Fixed':
            self.real_state_init = np.hstack(self.real_state_init_list)
            self.state_init = np.hstack(self.state_init_list)
            self.state_std = np.hstack(self.state_std_list)
            self.nstate_partial = np.array(self.state_init_list).shape[1]
            self.source_location_case = 0
        elif input_data['Source_location'] == 'Single':
            self.real_state_init = np.hstack(self.real_source_location_list
                [0] + self.real_state_init_list)
            self.state_init = np.hstack(self.source_location_list[0] + self
                .state_init_list)
            self.state_std = np.hstack(self.source_location_std_list[0] +
                self.state_std_list)
            self.nstate_partial = np.array(self.state_init_list).shape[1]
            self.source_location_case = 1
        elif input_data['Source_location'] == 'Multiple':
            self.total_real_state = np.array(self.total_real_state_list
                ).reshape(-1)
            self.total_state = np.array(self.total_state_list).reshape(-1)
            self.total_state_std = np.array(self.total_state_std_list).reshape(
                -1)
            self.nstate_partial = np.array(self.total_state_list).shape[1]
            self.real_state_init = self.total_real_state
            self.state_init = self.total_state
            self.state_std = self.total_state_std
            self.nstate_partial = self.nstate_partial
            self.source_location_case = 2
        else:
            print('Check the Source_location_info')
        self.state_init = np.array(self.state_init).reshape(-1)
        self.state_std = np.array(self.state_std).reshape(-1)
        self.decay = self.real_decay
        self.nstate = len(self.state_init)
        self.nstate_partial = np.array(self.state_init_list).shape[1]
        print('=== USING LDM INITIAL RUN AS TRUE OBSERVATION ===')
        time_intervals = int(self.input_data['time'] * 24 / self.input_data
            ['inverse_time_interval'])
        gamma_dose_data = receive_gamma_dose_matrix(self.nreceptor,
            time_intervals)
        print(
            f'Received observation data shape: {np.array(gamma_dose_data).shape}'
            )
        print(f'Observation data: {gamma_dose_data}')
        self.obs = np.array(gamma_dose_data[0]).reshape(-1)
        np.savetxt('/home/jrpark/EKI-LDM5-dev/true_observation_debug.txt',
            self.obs, fmt='%.6e')
        print(f'=== TRUE OBSERVATION VALUES ===')
        print(f'Shape: {self.obs.shape}')
        print(f'First 24 values: {self.obs[:24]}')
        print(
            f'Min: {np.min(self.obs):.6e}, Max: {np.max(self.obs):.6e}, Mean: {np.mean(self.obs):.6e}'
            )
        print(f'Non-zero count: {np.count_nonzero(self.obs)}/{len(self.obs)}')
        true_source_key = f'Source_{1}'
        if true_source_key in self.input_data:
            true_source = self.input_data[true_source_key]
            print(f'True source parameters (for reference): {true_source}')
            true_emission_rates = np.array(true_source[3])
            print(f'True emission rates (first 5): {true_emission_rates[:5]}')
            print(f'True emission rates (peak): {np.max(true_emission_rates)}')
        print(
            f'True observation range: [{np.min(self.obs):.2e}, {np.max(self.obs):.2e}]'
            )
        print(f'True observation values (first 10): {self.obs[:10]}')
        self.obs_err = np.diag(np.floor(self.obs * 0) + np.ones([len(self.
            obs)]) * self.nreceptor_err)
        self.obs_MDA = np.diag(np.floor(self.obs * 0) + np.ones([len(self.
            obs)]) * self.nreceptor_MDA)
        self.lowerbounds_list = []
        self.upperbounds_list = []
        for s in range(self.nsource):
            real_source = 'real_source{0}_boundary'.format(s + 1)
            for r in range(12):
                self.lowerbounds_list.append(input_data[real_source][0])
                self.upperbounds_list.append(input_data[real_source][1])
        self.bounds = np.array([self.lowerbounds_list, self.upperbounds_list]
            ).T

    @track_function
    def __str__(self):
        return self.name

    @track_function
    def make_ensemble(self):
        state = np.empty([self.nstate, self.sample])
        for i in range(self.nstate):
            state[i, :] = np.abs(np.random.normal(self.state_init[i], self.
                state_std[i], self.sample))
        return state

    @track_function
    def state_to_ob(self, state):
        model_obs_list = []
        print('=== EKI SENDING DATA ===')
        print(f'Original state shape: {state.shape}')
        if state.shape[0] == 1:
            ensemble_size = state.shape[1]
            tmp_states = np.tile(state, (24, 1))
            print(
                f'Expanded scalar state {state.shape} to time-series {tmp_states.shape}'
                )
        elif state.shape[0] == 24:
            tmp_states = state.copy()
            print(f'State already in time-series format: {tmp_states.shape}')
        else:
            raise ValueError(
                f'Unsupported state shape: {state.shape}. Expected (1, ensemble_size) or (24, ensemble_size)'
                )
        print(f'tmp_states shape: {tmp_states.shape}')
        print(
            f'tmp_states min/max/mean: {np.min(tmp_states):.2e}/{np.max(tmp_states):.2e}/{np.mean(tmp_states):.2e}'
            )
        print(f'tmp_states first 5 values: {tmp_states.flatten()[:5]}')
        if tmp_states.shape[1] > 1:
            print(
                f'First ensemble values for time intervals 0-4: {tmp_states[:5, 0]}'
                )
            print(
                f'Second ensemble values for time intervals 0-4: {tmp_states[:5, 1]}'
                )
        print(
            f'Number of zeros in tmp_states: {np.sum(tmp_states == 0)}/{tmp_states.size}'
            )
        send_tmp_states(tmp_states)
        tmp_results = receive_gamma_dose_matrix_ens()
        print('=== EKI RECEIVED DATA ===')
        print(f'tmp_results shape: {tmp_results.shape}')
        print(
            f'tmp_results min/max/mean: {np.min(tmp_results):.2e}/{np.max(tmp_results):.2e}/{np.mean(tmp_results):.2e}'
            )
        print(f'tmp_results first 5 values: {tmp_results.flatten()[:5]}')
        print(f'tmp_results last 5 values: {tmp_results.flatten()[-5:]}')
        print(
            f'Number of zeros in tmp_results: {np.sum(tmp_results == 0)}/{tmp_results.size}'
            )
        for ens in range(tmp_states.shape[1]):
            model_obs_list.append(np.asarray(tmp_results[ens]).reshape(-1))
        print(np.array(tmp_results).shape)
        with open('/home/jrpark/EKI-LDM5-dev/eki_obs_debug_log.txt', 'a'
            ) as obs_debug:
            obs_debug.write(
                f'\n=== 2) OBSERVATION RESHAPE CONSISTENCY VERIFICATION ===\n')
            obs_debug.write(f'tmp_results shape: {tmp_results.shape}\n')
            obs_debug.write(f'After reshape to model_obs_list:\n')
            first_ens_reshaped = np.asarray(tmp_results[0]).reshape(-1)
            obs_debug.write(
                f'First ensemble reshaped length: {len(first_ens_reshaped)}\n')
            obs_debug.write(
                f"""Expected: {tmp_results.shape[1] * tmp_results.shape[2]} = {tmp_results.shape[1]} receptors × {tmp_results.shape[2]} time intervals
"""
                )
            if tmp_results.shape[1] >= 2:
                receptor_0_from_reshape = first_ens_reshaped[0:24].tolist()
                receptor_1_from_reshape = first_ens_reshaped[24:48].tolist()
                ens0_rec0_original = tmp_results[0, 0, :].tolist()
                ens0_rec1_original = tmp_results[0, 1, :].tolist()
                obs_debug.write(f'\nReshape Consistency Check:\n')
                obs_debug.write(
                    f"""Receptor 0 time series from reshape (elements 0-23): {receptor_0_from_reshape}
"""
                    )
                obs_debug.write(
                    f'Receptor 0 original time series: {ens0_rec0_original}\n')
                receptor_0_match = (receptor_0_from_reshape ==
                    ens0_rec0_original)
                obs_debug.write(f'Receptor 0 match: {receptor_0_match}\n')
                obs_debug.write(
                    f"""
Receptor 1 time series from reshape (elements 24-47): {receptor_1_from_reshape}
"""
                    )
                obs_debug.write(
                    f'Receptor 1 original time series: {ens0_rec1_original}\n')
                receptor_1_match = (receptor_1_from_reshape ==
                    ens0_rec1_original)
                obs_debug.write(f'Receptor 1 match: {receptor_1_match}\n')
                if tmp_results.shape[1] >= 3:
                    receptor_2_from_reshape = first_ens_reshaped[48:72].tolist(
                        )
                    ens0_rec2_original = tmp_results[0, 2, :].tolist()
                    obs_debug.write(
                        f"""
Receptor 2 time series from reshape (elements 48-71): {receptor_2_from_reshape}
"""
                        )
                    obs_debug.write(
                        f'Receptor 2 original time series: {ens0_rec2_original}\n'
                        )
                    receptor_2_match = (receptor_2_from_reshape ==
                        ens0_rec2_original)
                    obs_debug.write(f'Receptor 2 match: {receptor_2_match}\n')
        del tmp_results
        model_obs = np.asarray(model_obs_list).T
        with open('/home/jrpark/EKI-LDM5-dev/eki_obs_debug_log.txt', 'a'
            ) as obs_debug:
            obs_debug.write(
                f'\nFinal model_obs shape after transpose: {model_obs.shape}\n'
                )
            obs_debug.write(
                f"""Expected format: [72, sample] where 72 = 3 receptors × 24 time intervals
"""
                )
            first_ensemble_obs = model_obs[:, 0]
            obs_debug.write(
                f'\nFirst ensemble observations (length {len(first_ensemble_obs)}): \n'
                )
            obs_debug.write(
                f"""Elements 0-23 (should be receptor 0): {first_ensemble_obs[0:24].tolist()}
"""
                )
            obs_debug.write(
                f"""Elements 24-47 (should be receptor 1): {first_ensemble_obs[24:48].tolist()}
"""
                )
            if len(first_ensemble_obs) >= 72:
                obs_debug.write(
                    f"""Elements 48-71 (should be receptor 2): {first_ensemble_obs[48:72].tolist()}
"""
                    )
        return model_obs

    @track_function
    def get_ob(self, time):
        self.obs_err = (self.obs * self.obs_err + self.obs_MDA) ** 2
        return self.obs, self.obs_err

    @track_function
    def predict(self, state, time):
        return state

from function_tracker import track_function
import importlib
import os
import sys
import numpy as np
from typing import Optional, Union
import Model_Connection_np_Ensemble as Model_np
desired_gpu_index_cupy = 0


@track_function
def Run(input_config={}, input_data={}):
    np.random.seed(0)
    inverse = Inverse(input_config)
    model = Model_np.Model(input_config, input_data)
    time = np.arange(input_config['time'], dtype=int)
    iteration = np.arange(input_config['iteration'], dtype=int)
    state_update_list = []
    state_update_iteration_list = []
    misfits_list = []
    misfits_list_tmp = []
    noises_list = []
    residuals_list = []
    discrepancy_bools_list = []
    residual_bools_list = []
    misfit_list = []
    misfit_list_tmp = []
    noise_list = []
    residual_list = []
    discrepancy_bool_list = []
    residual_bool_list = []
    ensXiter_list = []
    diff_list = []
    info_list = []
    for t in time:
        if t == 0:
            state_predict = model.make_ensemble()
        else:
            state_predict = model.predict(state_update, t)
        ob, ob_err = model.get_ob(t)
        obs = np.tile(ob, (input_config['sample'], 1)).T
        if input_config['perturb_option'] == 'On_time':
            obs = _perturb(ob, ob_err, input_config['sample'])
        alpha_inv_history = []
        T_n = 0.0
        for i in iteration:
            print(f'iteration: {i + 1}')
            if input_config['perturb_option'] == 'On_iter':
                obs = _perturb(ob, ob_err, input_config['sample'])
            state_in_ob = model.state_to_ob(state_predict)
            if i == 0:
                with open('/home/jrpark/EKI-LDM5-dev/eki_obs_debug_log.txt',
                    'a') as obs_debug:
                    obs_debug.write(
                        f"""
=== 3) OBSERVATION QUALITY STATISTICS (Iteration {i + 1}) ===
"""
                        )
                    obs_debug.write(
                        f"""state_in_ob shape after model.state_to_ob: {state_in_ob.shape}
"""
                        )
                    obs_debug.write(
                        f"""Expected format: [72, sample] where 72 = 3 receptors × 24 time intervals
"""
                        )
                    first_ensemble_observations = state_in_ob[:, 0
                        ] if state_in_ob.shape[1] > 0 else state_in_ob.flatten(
                        )
                    total_obs = len(first_ensemble_observations)
                    nonzero_count = np.sum(first_ensemble_observations != 0)
                    nonzero_ratio = (nonzero_count / total_obs if total_obs >
                        0 else 0)
                    obs_debug.write(
                        f'\nObservation Quality Statistics (First Ensemble):\n'
                        )
                    obs_debug.write(f'Total observations: {total_obs}\n')
                    obs_debug.write(f'Non-zero count: {nonzero_count}\n')
                    obs_debug.write(
                        f"""Non-zero ratio: {nonzero_ratio:.4f} ({nonzero_ratio * 100:.2f}%)
"""
                        )
                    if nonzero_count > 0:
                        nonzero_values = first_ensemble_observations[
                            first_ensemble_observations != 0]
                        obs_debug.write(
                            f'Min value: {np.min(nonzero_values):.2e}\n')
                        obs_debug.write(
                            f'Max value: {np.max(nonzero_values):.2e}\n')
                        obs_debug.write(
                            f'Mean value: {np.mean(nonzero_values):.2e}\n')
                        obs_debug.write(
                            f"""95th percentile: {np.percentile(nonzero_values, 95):.2e}
"""
                            )
                    else:
                        obs_debug.write(f'No non-zero values found!\n')
                    obs_debug.write(f'\nTime Index Non-zero Histogram:\n')
                    if total_obs >= 72:
                        time_nonzero_counts = []
                        for t in range(24):
                            receptor_0_val = first_ensemble_observations[t
                                ] if t < 24 else 0
                            receptor_1_val = first_ensemble_observations[24 + t
                                ] if 24 + t < total_obs else 0
                            receptor_2_val = first_ensemble_observations[48 + t
                                ] if 48 + t < total_obs else 0
                            time_nonzeros = sum([(1) for val in [
                                receptor_0_val, receptor_1_val,
                                receptor_2_val] if val != 0])
                            time_nonzero_counts.append(time_nonzeros)
                            if t < 10 or time_nonzeros > 0:
                                obs_debug.write(
                                    f"""  Time {t:2d}: {time_nonzeros}/3 receptors have non-zero values
"""
                                    )
                        obs_debug.write(
                            f'\nTime histogram summary: {time_nonzero_counts}\n'
                            )
                        obs_debug.write(
                            f"""Time indices with any non-zero: {[i for i, count in enumerate(time_nonzero_counts) if count > 0]}
"""
                            )
                    else:
                        obs_debug.write(
                            f'Insufficient observations for time histogram analysis\n'
                            )
                    obs_debug.write(f'\n=== obs Array After Replication ===\n')
                    obs_debug.write(f'obs shape: {obs.shape}\n')
                    obs_debug.write(
                        f"""obs first column (should match first_ensemble_observations): {obs[:, 0].tolist()[:24]}...
"""
                        )
                    if obs.shape[0] >= 72 and len(first_ensemble_observations
                        ) >= 72:
                        obs_first_72 = obs[:72, 0]
                        state_in_ob_first_72 = first_ensemble_observations[:72]
                        obs_debug.write(
                            f'\n=== OBS RESHAPE CONSISTENCY CHECK ===\n')
                        obs_debug.write(
                            f"""obs elements 0-23 (receptor 0): {obs_first_72[0:24].tolist()}
"""
                            )
                        obs_debug.write(
                            f"""obs elements 24-47 (receptor 1): {obs_first_72[24:48].tolist()}
"""
                            )
                        obs_debug.write(
                            f"""obs elements 48-71 (receptor 2): {obs_first_72[48:72].tolist()}
"""
                            )
                        consistency_match = np.allclose(obs_first_72,
                            state_in_ob_first_72, rtol=1e-10)
                        obs_debug.write(
                            f'\nobs vs state_in_ob consistency: {consistency_match}\n'
                            )
                        if not consistency_match:
                            obs_debug.write(
                                f'WARNING: obs and state_in_ob arrays do not match!\n'
                                )
                            diff_indices = np.where(~np.isclose(
                                obs_first_72, state_in_ob_first_72, rtol=1e-10)
                                )[0]
                            obs_debug.write(
                                f'Differences at indices: {diff_indices[:10]}\n'
                                )
                    obs_debug.write(f'\n' + '=' * 50 + '\n')
                print(f'\n=== OBSERVATION QUALITY SUMMARY ===')
                print(
                    f'Non-zero observation ratio: {nonzero_ratio:.4f} ({nonzero_ratio * 100:.2f}%)'
                    )
                print(
                    f'Total observations: {total_obs}, Non-zero: {nonzero_count}'
                    )
                if nonzero_count > 0:
                    nonzero_values = first_ensemble_observations[
                        first_ensemble_observations != 0]
                    print(
                        f'Value range: [{np.min(nonzero_values):.2e}, {np.max(nonzero_values):.2e}]'
                        )
                print(
                    f'Log saved to: /home/jrpark/EKI-LDM5-dev/eki_obs_debug_log.txt'
                    )
                print(f'=' * 40)
            if input_config['Adaptive_EKI'] == 'On':
                Phi_n = compute_Phi_n(obs, state_in_ob, ob_err)
                alpha_inv = compute_alpha_inv(len(state_in_ob), Phi_n,
                    alpha_inv_history, i)
                alpha_inv_history.append(alpha_inv)
                print(alpha_inv >= 1.0 - T_n)
                if len(alpha_inv_history) > 1 and (alpha_inv >= 1.0 - T_n or
                    alpha_inv < 0.0):
                    break
                T_n += alpha_inv
                print(T_n)
            if input_config.get('Localized_EKI') == 'On':
                if input_config.get('Adaptive_EKI') == 'On':
                    state_update = inverse.Adaptive_EnKF_with_Localizer(i,
                        state_predict, state_in_ob, obs, ob_err, ob, alpha_inv)
                else:
                    state_update = inverse.EnKF_with_Localizer(i,
                        state_predict, state_in_ob, obs, ob_err, ob)
            elif input_config.get('Adaptive_EKI') == 'On':
                state_update = inverse.Adaptive_EnKF(i, state_predict,
                    state_in_ob, obs, ob_err, ob, alpha_inv)
            elif input_config.get('Regularization') == 'On':
                state_update = inverse.REnKF(i, state_predict, state_in_ob,
                    obs, ob_err, ob)
            elif input_config.get('Non_negative_Regularization') == 'On':
                print('inverse_model: Non_negative_EKI')
                state_update = (inverse.
                    EnKF_Nonnegative_Integrated_new8_logTransform2(i,
                    state_predict, state_in_ob, obs, ob_err, pos_transform=
                    'off', barrier_mode='off', tau=40.0, projection_mode=
                    'gaussian', rho_schedule='none', lm_schedule='off',
                    sigma_target=0.05, infl_cap_max=1.01, use_prior_reg=False))
            else:
                state_update = inverse.EnKF(i, state_predict, state_in_ob,
                    obs, ob_err, ob)
                print('inverse_model: Standard_EKI')
            misfits = np.mean(state_update, 1) - np.mean(state_predict, 1)
            state_predict = state_update.copy()
            state_update_iteration_list.append(state_update.copy())
            misfits_list_tmp.append(misfits)
            misfits_list.append(misfits.copy())
            misfits_err = np.zeros(misfits.shape)
            discrepancy_bools, residual_bools, residuals, noises = (
                _convergence(misfits_list_tmp, misfits_err))
            discrepancy_bools_list.append(discrepancy_bools)
            residual_bools_list.append(residual_bools)
            residuals_list.append(np.asarray(residuals).copy())
            noises_list.append(noises.copy())
            misfit = np.linalg.norm(np.mean(obs - state_in_ob, axis=1))
            misfit_list_tmp.append(misfit)
            misfit_list.append(misfit.copy())
            discrepancy_bool, residual_bool, residual, noise = _convergence(
                misfit_list_tmp, ob_err)
            discrepancy_bool_list.append(discrepancy_bool)
            residual_bool_list.append(residual_bool)
            residual_list.append(np.asarray(residual).copy())
            noise_list.append(noise.copy())
            ensXiter = 0
            if residual_bool:
                ensXiter = input_config['sample'] * (i + 1)
                ensXiter_list.append(ensXiter)
            else:
                ensXiter_list.append(ensXiter)
        diff = np.abs(np.mean(state_update.copy(), 1) - model.real_state_init
            ) / model.real_state_init
        diff_list.append(diff)
        state_update_list.append(state_update.copy())
    if input_config['time'] == 1:
        state_update_list = state_update_list[0]
        diff_list = diff_list[0]
    return (state_update_list, state_update_iteration_list, info_list,
        misfit_list, discrepancy_bool_list, residual_bool_list,
        residual_list, noise_list, ensXiter_list, diff_list, misfits_list,
        discrepancy_bools_list, residual_bools_list, residuals_list)


class Inverse(object):

    @track_function
    def __init__(self, input_config):
        self.sample = input_config['sample']
        self.input_config = input_config
        self.time = 0
        self.alpha = input_config['EnKF_MDA_steps']
        self.beta = input_config['EnRML_step_length']
        self.lambda_value = input_config['REnKF_lambda']
        self.weighting_factor = input_config['Localization_weighting_factor']

    @track_function
    def EnKF(self, iteration, state_predict, state_in_ob, obs, ob_err, ob):
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0 / (self.sample - 1.0) * np.dot(x, hx.T)
        pzz = 1.0 / (self.sample - 1.0) * np.dot(hx, hx.T)
        k = np.dot(pxz, np.linalg.pinv(pzz + ob_err))
        dx = np.dot(k, obs - state_in_ob)
        state_update = np.array(state_predict) + dx
        return state_update

    @track_function
    def Adaptive_EnKF(self, iteration, state_predict, state_in_ob, obs,
        ob_err, ob, alpha_inv):
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0 / (self.sample - 1.0) * np.dot(x, hx.T)
        pzz = 1.0 / (self.sample - 1.0) * np.dot(hx, hx.T)
        alpha = 1.0 / alpha_inv
        k_modified = np.dot(pxz, np.linalg.pinv(pzz + alpha * ob_err))
        xi = _perturb(np.zeros(ob.shape), ob_err, self.sample)
        perturbed_diff = obs + np.sqrt(alpha) * xi - state_in_ob
        dx = np.dot(k_modified, perturbed_diff)
        state_update = np.array(state_predict) + dx
        return state_update

    @track_function
    def centralized_localizer(matrix, L):
        distances1 = compute_distances(matrix.shape[0])
        distances2 = compute_distances(matrix.shape[1])
        Psi1 = np.vectorize(lambda d: np.exp(-d ** 2 / (2 * L ** 2)))(
            distances1)
        Psi2 = np.vectorize(lambda d: np.exp(-d ** 2 / (2 * L ** 2)))(
            distances2)
        localized_matrix = matrix * Psi1 * Psi2
        return localized_matrix

    @track_function
    def EnKF_with_Localizer(self, iteration, state_predict, state_in_ob,
        obs, ob_err, ob, localizer_func=centralized_localizer):
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0 / (self.sample - 1.0) * np.dot(x, hx.T)
        pzz = 1.0 / (self.sample - 1.0) * np.dot(hx, hx.T)
        if localizer_func:
            pxz = localizer_func(pxz, self.weighting_factor)
            pzz = localizer_func(pzz, self.weighting_factor)
        k_modified = np.dot(pxz, np.linalg.pinv(pzz + ob_err))
        dx = np.dot(k_modified, obs - state_in_ob)
        state_update = state_predict + dx
        return state_update

    @track_function
    def Adaptive_EnKF_with_Localizer(self, iteration, state_predict,
        state_in_ob, obs, ob_err, ob, alpha_inv, localizer_func=
        centralized_localizer):
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0 / (self.sample - 1.0) * np.dot(x, hx.T)
        pzz = 1.0 / (self.sample - 1.0) * np.dot(hx, hx.T)
        if localizer_func:
            pxz = localizer_func(pxz, self.weighting_factor)
            pzz = localizer_func(pzz, self.weighting_factor)
        alpha = 1.0 / alpha_inv
        k_modified = np.dot(pxz, np.linalg.pinv(pzz + alpha * ob_err))
        xi = _perturb(np.zeros(ob.shape), ob_err, self.sample)
        perturbed_diff = obs + np.sqrt(alpha) * xi - state_in_ob
        dx = np.dot(k_modified, perturbed_diff)
        state_update = state_predict + dx
        return state_update

    @track_function
    def EnRML(self, iteration, state_predict, state_in_ob, obs, ob_err, ob):
        if iteration == 0:
            self.state0 = state_predict
        x0 = _ave_substracted(self.state0)
        p0 = 1.0 / (self.sample - 1.0) * np.dot(x0, x0.T)
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        sen = np.dot(hx, np.linalg.pinv(x))
        p0_sen = np.dot(p0, sen.T)
        gn_sub = np.dot(np.dot(sen, p0), sen.T) + ob_err
        gn = np.dot(p0_sen, np.linalg.inv(gn_sub))
        dx = np.dot(gn, obs - state_in_ob) + np.dot(gn, np.dot(sen, 
            state_predict - self.state0))
        state_update = self.beta * self.state0 + (1.0 - self.beta
            ) * state_predict + self.beta * dx
        return state_update

    @track_function
    def EnKF_MDA(self, iteration, state_predict, state_in_ob, obs, ob_err, ob):
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0 / (self.sample - 1.0) * np.dot(x, hx.T)
        pzz = 1.0 / (self.sample - 1.0) * np.dot(hx, hx.T)
        k_modified = np.dot(pxz, np.linalg.inv(pzz + self.alpha * ob_err))
        obs_origin = np.tile(ob, (obs.shape[1], 1)).T
        ob_perturb = obs - obs_origin
        obs_mda = obs_origin + np.sqrt(self.alpha) * ob_perturb
        dx = np.dot(k_modified, obs_mda - state_in_ob)
        state_update = state_predict + dx
        return state_update

    @track_function
    def REnKF(self, iteration, state_predict, state_in_ob, obs, ob_err, ob):

        @track_function
        def tanh_penalty(x):
            return np.tanh(x)

        @track_function
        def tanh_penalty_derivative(x):
            return 1 - np.tanh(x) ** 2

        @track_function
        def constraint_func(x):
            penalty = np.zeros_like(x)
            penalty[x < 0.0] = tanh_penalty(x[x < 0.0])
            penalty[x > 1000000000000000.0] = tanh_penalty(x[x > 
                1000000000000000.0] - 1000000000000000.0)
            return penalty

        @track_function
        def constraint_derivative(x):
            derivative = np.zeros_like(x)
            derivative[x < 0] = tanh_penalty_derivative(x[x < 0])
            derivative[x > 1000000000000000.0] = tanh_penalty_derivative(x[
                x > 1000000000000000.0] - 1000000000000000.0)
            return derivative

        @track_function
        def lambda_function(iteration):
            lambda_value = self.lambda_value
            return lambda_value
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0 / (self.sample - 1.0) * np.dot(x, hx.T)
        pzz = 1.0 / (self.sample - 1.0) * np.dot(hx, hx.T)
        k = np.dot(pxz, np.linalg.pinv(pzz + ob_err))
        dx = np.dot(k, obs - state_in_ob)
        pxx = 1.0 / (self.sample - 1.0) * np.dot(x, x.T)
        pzx = 1.0 / (self.sample - 1.0) * np.dot(hx, x.T)
        k_constraints = -pxx + np.dot(k, pzx)
        constraint_mat = np.zeros([len(state_predict), self.sample])
        w = np.identity(state_predict.shape[0])
        lamda = lambda_function(iteration)
        for j in range(self.sample):
            jstate = state_predict[:, j]
            gw = np.dot(constraint_derivative(jstate).T, w)
            gwg = np.dot(gw, constraint_func(jstate))
            constraint_mat[:, j] += lamda * gwg
        dx_constraints = np.dot(k_constraints, constraint_mat)
        state_update = state_predict + dx + dx_constraints
        return state_update

    @track_function
    def EnKF_Nonnegative_Integrated_new8_logTransform2(self, i: int,
        state_predict: np.ndarray, state_in_ob: np.ndarray, obs: np.ndarray,
        ob_err: np.ndarray, *, pos_transform: str='off', eps_pos: float=
        1e-08, tau: float=50.0, lower: float=1e-12, upper: float=1e+30,
        barrier_mode: str='mean', projection_mode: str='off', rcond: float=
        0.0001, gain_clip: Optional[float]=None, rho_schedule: str='none',
        rho_max: float=0.5, kappa: float=0.005, sigma_target: float=0.05,
        alpha_rho: float=0.5, infl_cap_max: float=1.05, beta_rho: bool=
        False, shrink: float=0.02, use_prior_reg: bool=False, lm_schedule:
        str='off', gamma_gain: float=0.01) ->np.ndarray:
        """
        One-step EKI update with optional positivity transform,
        adaptive ρ-inflation, barrier, projection, prior-Tikhonov,
        and rmse logging (CSV + PNG).

        Options for pos_transform:
        "off", "sqrt", "x_log", "xy_log", "x_asinh", "xy_asinh",
        "x_softplus","xy_softplus","x_sinh","xy_sinh"
        """
        if not hasattr(self, 'rmse_history'):
            self.rmse_history = []
        d, N = state_predict.shape
        scale = 1000000000000000.0
        to_s_x_map = {'off': (lambda u: u, lambda s: s), 'sqrt': (lambda u:
            np.sqrt(np.maximum(u, 0.0)), lambda s: s ** 2), 'x_log': (lambda
            u: np.log(np.maximum(u, eps_pos)), np.exp), 'x_asinh': (np.
            arcsinh, np.sinh), 'x_sinh': (lambda u: scale * np.sinh(u), lambda
            s: np.arcsinh(s / scale)), 'x_softplus': (lambda u: np.log(np.
            expm1(np.maximum(u, eps_pos))), lambda s: np.log1p(np.exp(s)))}
        for suffix in ('log', 'asinh', 'sinh', 'softplus'):
            to_s_x_map[f'xy_{suffix}'] = to_s_x_map[f'x_{suffix}']
        to_s_y_map = {k.replace('x_', 'xy_'): v for k, v in to_s_x_map.items()}
        to_s_x, from_s_x = to_s_x_map.get(pos_transform, to_s_x_map['off'])
        to_s_y, from_s_y = to_s_y_map.get(pos_transform, to_s_y_map.get(
            'off', (lambda v: v, lambda s: s)))
        s_pred = to_s_x(state_predict)
        s_in_ob = to_s_y(state_in_ob)
        s_obs = to_s_y(obs)
        jacobian_map = {'xy_log': lambda y: 1.0 / np.maximum(np.mean(y,
            axis=1), eps_pos), 'xy_asinh': lambda y: 1.0 / np.sqrt(1.0 + np
            .mean(y, axis=1) ** 2), 'xy_sinh': lambda y: scale * np.cosh(np
            .mean(y, axis=1)), 'xy_softplus': lambda y: 1.0 / (1.0 - np.exp
            (-np.maximum(np.mean(y, axis=1), eps_pos)))}
        if pos_transform in jacobian_map:
            invJ = jacobian_map[pos_transform](obs)
            ob_err_s = invJ[:, None] * ob_err * invJ[None, :]
        else:
            ob_err_s = ob_err
        if i == 0:
            X0 = s_pred - np.mean(s_pred, axis=1, keepdims=True)
            self.C0 = X0 @ X0.T / (N - 1)
        X = s_pred - np.mean(s_pred, axis=1, keepdims=True)
        HX = s_in_ob - np.mean(s_in_ob, axis=1, keepdims=True)
        Cxu = X @ HX.T / (N - 1)
        Cuu = X @ X.T / (N - 1)
        Cuu = (1 - shrink) * Cuu + shrink * np.eye(d)
        Cpp = HX @ HX.T / (N - 1)
        try:
            eigs = np.linalg.eigvalsh(Cpp + ob_err_s)
            lam_max = eigs[-1]
        except np.linalg.LinAlgError:
            print(
                'Warning: Eigenvalue computation failed, using fallback method'
                )
            lam_max = np.trace(Cpp + ob_err_s) / Cpp.shape[0]
            lam_max = max(lam_max, 1e-08)
        if rho_schedule == 'none':
            damp = 0.0
        else:
            damp = max(0.01 * lam_max, 1e-06 * lam_max, 0.001)
        if lm_schedule == 'off':
            gamma_gain = 0.0
        else:
            gamma_gain = gamma_gain
        try:
            inv_B = np.linalg.pinv(Cpp + ob_err_s + damp * np.eye(Cpp.shape
                [0]) + gamma_gain * np.eye(Cpp.shape[0]), rcond=rcond)
        except np.linalg.LinAlgError:
            print('Warning: numpy SVD did not converge, trying scipy')
            try:
                from scipy.linalg import pinv as scipy_pinv
                inv_B = scipy_pinv(Cpp + ob_err_s + damp * np.eye(Cpp.shape
                    [0]) + gamma_gain * np.eye(Cpp.shape[0]), rcond=rcond)
            except:
                print(
                    'Warning: Using heavily regularized inverse as final fallback'
                    )
                reg_strength = max(0.01, np.trace(Cpp + ob_err_s + 1e-08 *
                    np.eye(Cpp.shape[0])) * 0.01 / max(1, Cpp.shape[0]))
                B_reg = Cpp + ob_err_s + reg_strength * np.eye(Cpp.shape[0])
                try:
                    inv_B = np.linalg.pinv(B_reg, rcond=1e-06)
                except:
                    print('Warning: Using identity matrix as ultimate fallback'
                        )
                    inv_B = np.eye(Cpp.shape[0]) / reg_strength
        K = Cxu @ inv_B
        if gain_clip is not None:
            K = np.clip(K, -gain_clip, gain_clip)
        dx = K @ (s_obs - s_in_ob)
        if rho_schedule == 'adaptive':
            spread_obs = float(np.mean(np.std(s_in_ob, axis=1)))
            sigma_eff = (sigma_target * spread_obs if sigma_target < 1.0 and
                i == 0 else max(sigma_target, 1e-12))
            rho_t = max(0.0, min(1.0 - spread_obs / sigma_eff, rho_max))
            self.rho = rho_t if i == 0 else (1 - alpha_rho
                ) * self.rho + alpha_rho * rho_t
            rho = self.rho
        elif rho_schedule == 'fixed':
            rho = rho_max
        elif rho_schedule == 'exp':
            rho = rho_max * (1 - np.exp(-kappa * i))
        else:
            rho = 0.0
        if rho > 0:
            Gbar = np.mean(s_in_ob, axis=1, keepdims=True)
            dx += rho * (K @ (s_in_ob - Gbar))
        if beta_rho and rho > 0:
            reg = s_pred - np.mean(s_pred, axis=1, keepdims=True)
            dx += rho * (Cuu @ np.linalg.pinv(self.C0) @ reg)
        s_update = s_pred + dx
        s_update = np.nan_to_num(s_update, nan=0.0, posinf=upper, neginf=-upper
            )
        if use_prior_reg:
            s_update -= Cuu @ np.linalg.pinv(self.C0) @ s_pred
        if barrier_mode != 'off':
            eps = 1e-12
            lb = np.full((d, 1), lower)
            ub = np.full((d, 1), upper)
            if barrier_mode == 'mean':
                u_ref = from_s_x(np.mean(s_update, axis=1, keepdims=True))
                grad = 1 / (ub - u_ref + eps) - 1 / (u_ref - lb + eps)
                grad = np.tile(grad, (1, N))
            else:
                u_now = from_s_x(s_update)
                grad = 1 / (ub - u_now + eps) - 1 / (u_now - lb + eps)
            s_update += 1.0 / tau * (Cuu @ grad)
        if rho_schedule != 'none':
            infl = min(infl_cap_max, 1.0 / max(1e-12, 1.0 - rho))
            mean_s = np.mean(s_update, axis=1, keepdims=True)
            s_update = mean_s + infl * (s_update - mean_s)
        else:
            infl = 1.0
        state_update = from_s_x(s_update)
        if i == 0 or i % 5 == 0:
            print(f'\n=== ITERATION {i} ESTIMATION RESULTS ===')
            ensemble_mean = np.mean(state_update, axis=1)
            ensemble_std = np.std(state_update, axis=1)
            print(f'Estimated mean (first 5): {ensemble_mean[:5]}')
            print(f'Estimated std (first 5): {ensemble_std[:5]}')
            print(
                f'Estimated range: [{np.min(ensemble_mean):.2e}, {np.max(ensemble_mean):.2e}]'
                )
            true_emission = [100000.0, 120000.0, 150000.0, 200000.0, 
                250000.0, 300000.0, 400000.0, 500000.0, 700000.0, 900000.0,
                1200000.0, 1500000.0, 2000000.0, 1800000.0, 1500000.0, 
                1200000.0, 900000.0, 700000.0, 500000.0, 400000.0, 300000.0,
                250000.0, 200000.0, 150000.0]
            print(f'True source (first 5): {true_emission[:5]}')
            print(f'True source (peak): {max(true_emission)}')
            if len(ensemble_mean) == len(true_emission):
                rmse = np.sqrt(np.mean((ensemble_mean - np.array(
                    true_emission)) ** 2))
                rel_error = rmse / np.mean(true_emission) * 100
                print(f'RMSE: {rmse:.2e}, Relative Error: {rel_error:.1f}%')
        if projection_mode != 'off':
            lbv = np.full(d, lower)
            ubv = np.full(d, upper)
            if projection_mode == 'clip':
                state_update = np.clip(state_update, lbv[:, None], ubv[:, None]
                    )
            elif projection_mode == 'uniform':
                state_update = stochastic_projection(state_update, lbv, ubv)
            elif projection_mode == 'gaussian':
                state_update = stochastic_gaussian(state_update, lbv, ubv)
            elif projection_mode == 'adaptive_uniform':
                state_update = adaptive_stochastic_projection(state_update,
                    lbv, ubv)
            elif projection_mode == 'adaptive_gaussian':
                state_update = adaptive_gaussian_projection(state_update,
                    lbv, ubv)
        rmse = float(np.linalg.norm(obs - state_in_ob) / np.sqrt(obs.size))
        smax = float(np.max(np.abs(state_update)))
        print(
            f'iter {i:2d} | rho={rho:.2f} infl={infl:.2f} max|state|={smax:.3e} RMSE={rmse:.3e}'
            )
        self.rmse_history.append(rmse)
        return state_update


@track_function
def stochastic_projection(state: np.ndarray, lb: np.ndarray, ub: np.ndarray,
    jitter: float=1e-06) ->np.ndarray:
    noise = jitter * np.random.random(state.shape)
    state = np.where(state < lb[:, None], lb[:, None] + noise, state)
    state = np.where(state > ub[:, None], ub[:, None] - noise, state)
    return state


@track_function
def stochastic_gaussian(state: np.ndarray, lb: np.ndarray, ub: np.ndarray,
    jitter: float=10000.0) ->np.ndarray:
    noise = abs(jitter * np.random.normal(size=state.shape))
    state = np.where(state < lb[:, None], lb[:, None] + noise, state)
    state = np.where(state > ub[:, None], ub[:, None] - noise, state)
    return state


@track_function
def adaptive_stochastic_projection(state: np.ndarray, lb: np.ndarray, ub:
    np.ndarray, small_jitter: float=1e-06, large_jitter: float=0.01,
    collapse_threshold: float=1e-30) ->np.ndarray:
    spread = np.max(state, 1) - np.min(state, 1) + 1e-12
    jitter = np.where(spread[:, None] < collapse_threshold, large_jitter,
        small_jitter)
    noise = jitter * np.random.random(state.shape)
    state = np.where(state < lb[:, None], lb[:, None] + noise, state)
    state = np.where(state > ub[:, None], ub[:, None] - noise, state)
    return state


@track_function
def adaptive_gaussian_projection(state: np.ndarray, lb: np.ndarray, ub: np.
    ndarray, small_sigma: float=1e-06, large_sigma: float=0.01,
    collapse_threshold: float=1e-30) ->np.ndarray:
    spread = np.max(state, 1) - np.min(state, 1) + 1e-12
    sigma = np.where(spread[:, None] < collapse_threshold, large_sigma,
        small_sigma)
    noise = sigma * np.random.normal(size=state.shape)
    state = np.where(state < lb[:, None], lb[:, None] + noise, state)
    state = np.where(state > ub[:, None], ub[:, None] - noise, state)
    return state


@track_function
def _perturb(ave, cov, samp, perturb=1e-30):
    dim = len(ave)
    cov += np.eye(dim) * perturb
    chol_decomp = np.linalg.cholesky(cov)
    corr_perturb = np.random.normal(loc=0.0, scale=1.0, size=(dim, samp))
    perturbation = np.matmul(chol_decomp, corr_perturb)
    get_perturb = np.tile(ave, (samp, 1)).T + perturbation
    return get_perturb


@track_function
def _ave_substracted(m):
    samp = m.shape[1]
    ave = np.array([np.mean(m, 1)])
    ave = np.tile(ave.T, (1, samp))
    ave_substracted = np.array(m) - ave
    return ave_substracted


@track_function
def _convergence(misfit_list, ob_err, noise_factor=1.0, min_residual=1e-06):
    if np.any(ob_err) == True:
        noise_level = np.sqrt(np.trace(ob_err))
        noise = noise_factor * noise_level
    else:
        noise = noise_factor * ob_err
    discrepancy_bool = misfit_list[-1] < noise
    mIter = len(misfit_list) - 1
    if mIter == 0:
        residual = np.nan
        residual_bool = False
    else:
        residual = abs(misfit_list[mIter] - misfit_list[mIter - 1])
        residual /= abs(misfit_list[0])
        residual_bool = residual < min_residual
    return discrepancy_bool, residual_bool, residual, noise


@track_function
def sec_fisher(cov, N_ens):
    rows, cols = cov.shape
    if rows == cols:
        v = np.sqrt(np.diag(cov))
        V = np.diag(v)
        V_inv = np.linalg.inv(V)
        R = V_inv @ cov @ V_inv
    else:
        row_scales = np.sqrt(np.diag(np.dot(cov, cov.T)))
        col_scales = np.sqrt(np.diag(np.dot(cov.T, cov)))
        R = cov / row_scales[:, None] / col_scales[None, :]
    R_sec = np.zeros_like(R)
    for i in range(rows):
        for j in range(min(i + 1, cols)):
            r = R[i, j]
            if (i == j) | (r >= 1):
                R_sec[i, j] = 1
            else:
                s = np.arctanh(r)
                σ_s = 1 / np.sqrt(N_ens - 3)
                σ_r = (np.tanh(s + σ_s) - np.tanh(s - σ_s)) / 2
                Q = r / σ_r
                alpha = Q ** 2 / (1 + Q ** 2)
                R_sec[i, j] = alpha * r
    if rows == cols:
        return V @ R_sec @ V
    else:
        return R_sec * row_scales[:, None] * col_scales[None, :]


@track_function
def compute_Phi_n(y, state_in_ob, Gamma):
    results_matrix = np.dot(np.linalg.pinv(Gamma) ** 0.5, y - state_in_ob)
    phi_n = np.linalg.norm(results_matrix, ord=2, axis=0)
    return phi_n


@track_function
def compute_alpha_inv(M, Phi_n, alpha_inv_history, n):
    Phi_mean = np.mean(Phi_n)
    Phi_var = np.var(Phi_n)
    t_n = sum(alpha_inv_history[:n]) if n != 1 else 0
    alpha_inv = min(max(M / (2.0 * Phi_mean), np.sqrt(M / (2.0 * Phi_var))),
        1.0 - t_n)
    return alpha_inv


@track_function
def compute_distances(size):
    return np.abs(np.arange(size) - np.arange(size)[:, np.newaxis])

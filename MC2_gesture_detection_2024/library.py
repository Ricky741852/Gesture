import os
import io
import shutil

import numpy as np
from contextlib import redirect_stdout


def scale_range(img, min_value, max_value):
    """SSIM pre-process for numpy img"""
    img = img.astype('float32')
    img += -(np.min(img))
    img /= np.max(img) / (max_value - min_value)
    img += min_value
    return img


def checkpath(path, do_func=None, *args):
    try:
        if not os.path.exists(path):
            print(Color.WARN + f'{path} is not exist.' + Color.RESET)
            if do_func:
                print(Color.OK + f'Created {path} successfully!' + Color.RESET)
                do_func(*args)
            return False
        return True
    except PermissionError:
        print(Color.FAIL + 'Permission deny.' + Color.RESET)


def check_file(file):
    if os.path.exists(file):
        print(Color.OK + f'Created {file} successfully!' + Color.RESET)
    else:
        print(Color.FAIL + f'Failed to create {file}.' + Color.RESET)


def check_directories(*directories):
    for directory in directories:
        checkpath(directory, os.makedirs, directory)


def copy_hfile_to_gemmini(hfile_name, hfile_path, gemmini_path):
    if os.path.isfile(hfile_path):
        print(f'{hfile_path} is exist.')
    hfile_path = os.path.join(hfile_path, hfile_name)
    gemmini_path = os.path.join(gemmini_path, 'include')
    shutil.copy2(hfile_path, gemmini_path)
    print(Color.MSG + f'Successfully copy {hfile_path} to {gemmini_path}' + Color.RESET)


def copy_hdir_to_gemmini(hfile_path, gemmini_path, folder=''):
    if os.path.isfile(hfile_path):
        print(f'{hfile_path} is exist.')
    gemmini_path = os.path.join(gemmini_path, 'include', folder)
    check_directories(gemmini_path)
    shutil.copytree(hfile_path, gemmini_path, dirs_exist_ok=True)
    print(Color.MSG + f'Successfully copy {hfile_path} to {gemmini_path}' + Color.RESET)


def copy_cfile_for_git(cfile_name, gemmini_path, target_path):
    gemmini_path = os.path.join(gemmini_path, 'bareMetalC', cfile_name)
    shutil.copy2(gemmini_path, target_path)
    # print(f'Successfully copy {gemmini_path} to {target_path}/')


def copy_hfile_for_git(hfile_name, gemmini_path, target_path):
    gemmini_path = os.path.join(gemmini_path, 'include', hfile_name)
    shutil.copy2(gemmini_path, target_path)
    # print(f'Successfully copy {gemmini_path} to {target_path}/')


def Folding_Conv_BN(conv1d_weights: np.ndarray, BN: np.ndarray, eps=1e-3):
    """Folding BN weights into Conv1d-weights
     *  conv1d_weights : (1, kernel_size, in_channel, out_channel)
     *  Y_bn = gamma * (y - moving_means)/(sqrt(moving_var + eps)) + beta
     *  r_hat = gamma / sqrt(moving_var + eps)
     *  W_hat = r_hat * W
     *  bias_hat = r_hat * (bias - moving_means) + beta
     """
    conv1d_bias = np.zeros((conv1d_weights.shape[3],))
    gamma = BN[0].reshape((1, 1, 1, BN[0].shape[0]))
    beta = BN[1]
    mean = BN[2]
    variance = BN[3].reshape((1, 1, 1, BN[3].shape[0]))
    # sqrt_func = np.vectorize(math.sqrt)
    new_weights = conv1d_weights * gamma / np.sqrt(variance + eps)
    new_bias = beta + (conv1d_bias - mean) * gamma / np.sqrt(variance + eps)
    return new_weights, new_bias


def Folding_Conv_BN_2D(conv2d_weights: np.ndarray, BN: np.ndarray, eps=1e-3):
    """Folding BN weights into Conv1d-weights
     *  conv2d_weights : (1, kernel_size, kernel_size, in_channel, out_channel)
     *  Y_bn = gamma * (y - moving_means)/(sqrt(moving_var + eps)) + beta
     *  r_hat = gamma / sqrt(moving_var + eps)
     *  W_hat = r_hat * W
     *  bias_hat = r_hat * (bias - moving_means) + beta
     """
    conv1d_bias = np.zeros((conv2d_weights.shape[4],))
    gamma = BN[0].reshape((1, 1, 1, 1, BN[0].shape[0]))
    beta = BN[1]
    mean = BN[2]
    variance = BN[3].reshape((1, 1, 1, 1, BN[3].shape[0]))
    new_weights = conv2d_weights * gamma / np.sqrt(variance + eps)
    new_bias = beta + (conv1d_bias - mean) * gamma / np.sqrt(variance + eps)
    return new_weights, new_bias


def cal_scaleZeroPoint(r_max, r_min, q_max=127, q_min=-128):
    buf = io.StringIO()

    with redirect_stdout(buf):
        print(Color.WARN + f'{cal_scaleZeroPoint.__name__}' + '=' * 40 + Color.RESET)
        print(f'q_max: {q_max}; q_min: {q_min}')
        print(f'r_max: {r_max}; r_min: {r_min}')
        scale = (r_max - r_min) / (q_max - q_min)
        print(f'scale: {scale}')
        zeroPoint = q_max - (r_max / scale)
        print(f'zeroPoint: {zeroPoint}')
        zeroPoint = np.clip(zeroPoint, q_min, q_max)
        print(f'zeroPoint (clipped): {zeroPoint}')
        zeroPoint = int(zeroPoint)

    # Uncomment the following line to debug if needed
    # output = buf.getvalue()
    # print(output)

    return scale, zeroPoint


# def cal_scaleZeroPoint(r_max, r_min, q_max=127, q_min=-128):
#     if q_max - q_min != 0:
#         scale = (r_max - r_min) / (q_max - q_min)
#     else:
#         # 如果 q_max 和 q_min 相等，则将 scale 设为 0
#         scale = 0
#     # scale = (r_max - r_min) / (q_max - q_min)
#
#     # zeroPoint = q_max - (r_max / scale)
#     if scale != 0 and not np.isnan(scale):
#         zeroPoint = q_max - (r_max / scale)
#     else:
#         # 如果 scale 是0或NaN，则将 zeroPoint 设为 NaN
#         zeroPoint = 0
#
#     zeroPoint = np.clip(zeroPoint, q_min, q_max)
#     if np.isnan(zeroPoint):
#         zeroPoint = 0
#         # ValueError: cannot convert float NaN to integer
#     else:
#         zeroPoint = int(zeroPoint)
#     return scale, zeroPoint


def Quantization(r: np.ndarray, scale, zeroPoint):
    q = np.array(np.clip(np.round(r / scale + zeroPoint), -128, 127), dtype=np.int8)
    return q


def Dequantization(q: np.ndarray, scale, zeroPoint):
    r = np.array(scale * (q - zeroPoint), dtype=np.float32)
    return r


def cosineSimilarity(P: np.ndarray, Q: np.ndarray):
    dot = np.sum(p * q for p, q in zip(P, Q))
    norm_p = np.sum(p * p for p in P) ** 0.5
    norm_q = np.sum(q * q for q in Q) ** 0.5
    cos_sim = dot / ((norm_p * norm_q) + 1e-5)
    # cosine_similarity([P], [Q])
    return cos_sim


def statics_data(x: np.ndarray, isfloat=True):
    x = x.flatten()
    statics = {}
    for i in range(len(x)):
        tmp = x[i]
        if isfloat:
            tmp = np.round(x[i], 3)
        statics[tmp] = statics.get(tmp, 0) + 1

    indices = sorted(statics.keys())
    counts = [statics[k] for k in indices]
    return indices, counts


def bias_Correction(bias: np.ndarray):
    """optional"""
    best_minn, best_maxx = optimal_min_max(bias)

    scale_bias, zeroPoint_bias = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
    Qbias = Quantization(bias, scale_bias, zeroPoint_bias)
    DEQbias = Dequantization(Qbias, scale_bias, zeroPoint_bias)
    error_bias = bias - DEQbias
    bias_Corr = bias - error_bias
    return bias_Corr


def optimal_min_max(x: np.ndarray):
    # print(f'x: {x}\n')
    best_cosine_sim = -1
    best_minn, best_maxx = float('inf'), float('-inf')
    # print(f'best_minn: {best_minn}\n')
    # print(f'best_maxx: {best_maxx}\n')
    means, std = np.mean(x), np.std(x)
    region = np.arange(2, 3.1, 0.1)
    for i in range(len(region)):
        minn, maxx = means - region[i] * std, means + region[i] * std
        # print(f'minn: {minn}\n')
        # print(f'maxx: {maxx}\n')
        scale, zero_point = cal_scaleZeroPoint(r_min=minn, r_max=maxx)
        Q_x = Quantization(x, scale, zero_point)
        DE_Q_x = Dequantization(Q_x, scale, zero_point)
        cosine_sim = cosineSimilarity(x, DE_Q_x)
        if np.mean(cosine_sim) > best_cosine_sim:
            best_minn, best_maxx = minn, maxx
            best_cosine_sim = np.mean(cosine_sim)
            # print(best_cosine_sim)
    return best_minn, best_maxx


# def approximate_M(M: float):
#     """M = S1 * S2 / S4 , could be approximated to a fixed-point-number(m0) with bit shift(n) """
#     m0, n = math.frexp(M)
#     return m0, n


# def round_near_even(x: float):
#     float_x = x
#     int_x = int(float_x)
#
#     if float_x < 0:
#         next_x = int(float_x) - 1
#     else:
#         next_x = int(float_x) + 1
#     remain = abs(float_x - int_x)
#
#     if remain < 0.5:
#         result = int_x
#     else:
#         if remain > 0.5:
#             result = next_x
#         else:
#             if int_x % 2 == 0:
#                 result = int_x
#             else:
#                 result = next_x
#     return result


# def QRelu_clip(x: np.ndarray, use_relu: bool, z3=None, z4=None):
#     if use_relu:
#         x = np.where(x < z3, z4, x)
#         x = np.clip(x, z4, 127)
#     else:
#         x = np.clip(x, -128, 127)
#     return x


def reshape_kernel(kernel: np.ndarray):
    kernel_size = kernel.shape[1]
    in_channels = kernel.shape[2]
    out_channels = kernel.shape[3]
    return np.reshape(kernel, (kernel_size * in_channels, out_channels))


# def reshape_feature(input_feature: np.ndarray, kernel_size, stride_size, out_width):
#     batch_size = input_feature.shape[0]
#     in_channels = input_feature.shape[3]
#     reshape_featre = []
#     for idx in range(batch_size):
#         reshape_f = np.zeros((out_width, kernel_size * in_channels))
#         flatten_f = input_feature[idx][0].flatten()
#         start = 0
#         for i in range(out_width):
#             for j in range(kernel_size * in_channels):
#                 reshape_f[i][j] = flatten_f[(start + j)]
#             start += stride_size * in_channels
#         reshape_featre.append([reshape_f])
#     return np.array(reshape_featre)


# def Qconv1d(reshaped_feature: np.ndarray, reshaped_kernel: np.ndarray, kernel_bias: np.ndarray, out_width, KS_inChannel,
#             out_channels, down_scalar, z3, z4, is_conv=True):
#     if is_conv:
#         batch_size = reshaped_feature.shape[0]
#         out_feature = np.zeros((batch_size, 1, out_width, out_channels))
#         for idx in range(batch_size):
#             for i in range(out_width):
#                 tmp_res = 0
#                 for j in range(out_channels):
#                     for k in range(KS_inChannel):
#                         tmp_res += reshaped_feature[idx][0][i][k] * reshaped_kernel[k][j]
#                     out_feature[idx][0][i][j] = round_near_even((down_scalar * (tmp_res + kernel_bias[idx][0][i][j])))
#                     tmp_res = 0
#         out_feature = QRelu_clip(out_feature, z3=z3, z4=z4, use_relu=True)
#     else:
#         out_feature = np.zeros((out_width, out_channels))
#         for i in range(out_width):
#             tmp_res = 0
#             for j in range(out_channels):
#                 for k in range(KS_inChannel):
#                     tmp_res += reshaped_feature[i][k] * reshaped_kernel[k][j]
#                 out_feature[i][j] = round_near_even((down_scalar * (tmp_res + kernel_bias[i][j])))
#                 tmp_res = 0
#         out_feature = QRelu_clip(out_feature, use_relu=False)
#     return out_feature


# def pre_compute_bias(input_feature: np.ndarray, kernel: np.ndarray, bias: np.ndarray, KS_inChannel,
#                      s1, z1, s2, z2, s2_b, z2_b, s3, z3, relu_s4, relu_z4, is_conv=True):
#     batch_size = input_feature.shape[0]
#     if is_conv:
#         out_width = bias.shape[0]
#         out_channels = bias.shape[1]
#         total_bias = np.zeros((batch_size, 1, out_width, out_channels))
#         for idx in range(batch_size):
#             for i in range(out_width):
#                 tmp_bias = 0
#                 for j in range(out_channels):
#                     for k in range(KS_inChannel):
#                         tmp_bias += ((z1 * z2) - (kernel[k][j] * z1) - (input_feature[idx][0][i][k] * z2))
#                     total_bias[idx][0][i][j] = round(
#                         tmp_bias + ((s2_b / (s1 * s2)) * (bias[i][j] - z2_b)) + (z3 / ((s1 * s2) / relu_s4)))
#                     tmp_bias = 0
#         total_bias = total_bias.astype(np.int32)
#         return total_bias
#     else:
#         out_width = batch_size
#         out_channels = bias.shape[0]
#         res_bias = np.zeros((out_width, out_channels))
#         for i in range(out_width):
#             tmp_bias = 0
#             for j in range(out_channels):
#                 for k in range(KS_inChannel):
#                     tmp_bias += ((z1 * z2) - (kernel[k][j] * z1) - (input_feature[i][k] * z2))
#                 res_bias[i][j] = np.round(tmp_bias + ((s2_b / (s1 * s2)) * (bias[j] - z2_b)) + (z3 / ((s1 * s2) / s3)))
#                 tmp_bias = 0
#         res_bias = res_bias.astype(np.int32)
#         return res_bias


class Color:
    FAIL = '\033[31m'  # RED
    OK = '\033[32m'  # GREEN
    WARN = '\033[33m'  # YELLOW
    INFO = '\033[34m'  # BLUE
    NOTE = '\033[35m'  # PURPLE
    MSG = '\033[36m'  # CYAN
    RED = '\033[41m'
    GREEN = '\033[42m'
    YELLOW = '\033[43m'
    BLUE = '\033[44m'
    PURPLE = '\033[45m'
    CYAN = '\033[46m'
    H_FAIL = '\033[91m'  # RED
    H_OK = '\033[92m'  # GREEN
    H_WARN = '\033[93m'  # YELLOW
    H_INFO = '\033[94m'  # BLUE
    H_NOTE = '\033[95m'  # PURPLE
    H_MSG = '\033[96m'  # CYAN
    RESET = '\033[0m'  # RESET COLOR

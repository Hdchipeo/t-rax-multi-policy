#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

def decompose_scale_to_multiplier_and_shift(scale: float):
    """
    Phân tách hệ số scale thực thành (multiplier_int32, right_shift) 
    theo chuẩn Fixed-point của ESP-NN / TFLite-Micro:
    effective_scale = multiplier * (2 ** -shift)
    """
    if scale == 0:
        return 0, 0
    # Chuẩn hóa scale về khoảng [0.5, 1.0)
    shift = 0
    while scale < 0.5:
        scale *= 2.0
        shift += 1
    while scale >= 1.0:
        scale /= 2.0
        shift -= 1
    
    # Nhân với 2^31 để đưa vào số nguyên 32-bit có dấu
    multiplier = int(round(scale * (1 << 31)))
    multiplier = min(multiplier, (1 << 31) - 1)
    # Trong esp-nn (esp_nn_multiply_by_quantized_mult):
    # right_shift = shift > 0 ? 0 : -shift;
    # do đó shift truyền vào phải là số âm (-shift)
    return multiplier, -shift

def quantize_tensor_symmetric_int8(weight: np.ndarray):
    """
    Lượng tử hóa đối xứng signed int8 [-127 .. 127] cho Weights
    Zero-point = 0
    """
    max_val = np.max(np.abs(weight))
    scale = max_val / 127.0 if max_val > 0 else 1.0
    q_weight = np.clip(np.round(weight / scale), -128, 127).astype(np.int8)
    return q_weight, scale

def quantize_bias_int32(bias: np.ndarray, input_scale: float, weight_scale: float):
    """
    Lượng tử hóa Bias thành signed int32 với scale = input_scale * weight_scale
    """
    bias_scale = input_scale * weight_scale
    q_bias = np.round(bias / bias_scale).astype(np.int32)
    return q_bias, bias_scale

def generate_calibration_data(num_samples: int = 1000, obs_dim: int = 39) -> np.ndarray:
    """
    Tạo dữ liệu hiệu chuẩn đại diện (Representative Calibration Dataset)
    Mô phỏng phân phối trạng thái thực tế của WorldState
    """
    data = []
    if obs_dim == 6:
        for _ in range(num_samples):
            vis = 1.0 if np.random.uniform() > 0.3 else 0.0
            x_norm = np.random.uniform(-1.0, 1.0) if vis > 0.5 else 0.0
            motion = np.random.uniform(0.0, 0.8)
            pan = np.random.uniform(-0.8, 0.8)
            t_event = np.random.uniform(0.0, 1.0)
            action = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
            data.append([vis, x_norm, motion, pan, t_event, action])
    else:
        for _ in range(num_samples):
            # Kinematics (8)
            pan = np.random.uniform(-0.8, 0.8)
            tilt = np.random.uniform(-0.6, 0.6)
            lin_v = np.random.uniform(-0.5, 0.5)
            ang_v = np.random.uniform(-0.8, 0.8)
            l_mot = np.random.uniform(-0.5, 0.5)
            r_mot = np.random.uniform(-0.5, 0.5)
            front_dist = np.random.uniform(0.1, 1.0)
            battery = np.random.uniform(0.5, 1.0)
            # Vision (16)
            f_vis = 1.0 if np.random.uniform() > 0.4 else 0.0
            f_x = np.random.uniform(-0.8, 0.8) if f_vis > 0.5 else 0.0
            f_y = np.random.uniform(-0.5, 0.5) if f_vis > 0.5 else 0.0
            f_sz = np.random.uniform(0.1, 0.8) if f_vis > 0.5 else 0.0
            f_conf = np.random.uniform(0.7, 0.95) if f_vis > 0.5 else 0.0
            o_vis = 1.0 if np.random.uniform() > 0.5 else 0.0
            o_x = np.random.uniform(-0.8, 0.8) if o_vis > 0.5 else 0.0
            o_y = np.random.uniform(-0.5, 0.5) if o_vis > 0.5 else 0.0
            o_sz = np.random.uniform(0.1, 0.5) if o_vis > 0.5 else 0.0
            o_conf = np.random.uniform(0.6, 0.9) if o_vis > 0.5 else 0.0
            motion = np.random.uniform(0.0, 0.8)
            scene_chg = np.random.uniform(0.0, 0.5)
            emb = np.random.uniform(-0.5, 0.5, size=4)
            # Temporal (6)
            t_obj = np.random.uniform(0.0, 1.0)
            t_hum = np.random.uniform(0.0, 1.0)
            t_resp = np.random.uniform(0.0, 1.0)
            t_mot = np.random.uniform(0.0, 1.0)
            t_attn = np.random.uniform(0.0, 1.0)
            t_base = np.random.uniform(0.0, 1.0)
            # Internal (5)
            curiosity = np.random.uniform(0.0, 1.0)
            boredom = np.random.uniform(0.0, 1.0)
            social = np.random.uniform(0.0, 1.0)
            energy = np.random.uniform(0.5, 1.0)
            conf = np.random.uniform(0.7, 1.0)
            # Exploration (4)
            v_nov = np.random.uniform(0.0, 1.0)
            r_nov = np.random.uniform(0.0, 1.0)
            o_nov = np.random.uniform(0.0, 1.0)
            act_div = np.random.uniform(0.0, 1.0)

            vec = [
                pan, tilt, lin_v, ang_v, l_mot, r_mot, front_dist, battery,
                f_vis, f_x, f_y, f_sz, f_conf, o_vis, o_x, o_y, o_sz, o_conf, motion, scene_chg,
                emb[0], emb[1], emb[2], emb[3],
                t_obj, t_hum, t_resp, t_mot, t_attn, t_base,
                curiosity, boredom, social, energy, conf,
                v_nov, r_nov, o_nov, act_div
            ]
            data.append(vec[:obs_dim])
    return np.array(data, dtype=np.float32)

def generate_c_header(layers_info, output_header_path, policy_name="autonomous_idle"):
    """
    Tự động sinh mã nguồn C Header (`idle_actor_weights_espnn.h`)
    chứa trọng số int8, biases int32 và cấu hình QuantizedLayerConfig
    sẵn sàng nạp trực tiếp vào PolicyRuntime trên ESP32-S3!
    """
    os.makedirs(os.path.dirname(output_header_path), exist_ok=True)
    with open(output_header_path, "w") as f:
        f.write("/*\n")
        f.write(f" * Auto-generated by ESP-PPQ Quantizer for T-RAX Policy: {policy_name}\n")
        f.write(" * Optimized for ESP32-S3 PIE (Processor Instruction Extension) via esp-nn\n")
        f.write(" */\n\n")
        f.write("#pragma once\n")
        f.write('#include "policy_runtime.h"\n\n')

        for idx, layer in enumerate(layers_info):
            l_num = idx + 1
            w = layer["weights"]
            b = layer["bias"]
            
            # Khai báo Weights int8
            f.write(f"// Layer {l_num}: [{layer['in_features']} -> {layer['out_features']}] Weights (int8)\n")
            f.write(f"static const int8_t s_layer{l_num}_weights[{len(w)}] = {{\n    ")
            for i, val in enumerate(w):
                f.write(f"{int(val):4d},")
                if (i + 1) % 16 == 0 and i < len(w) - 1:
                    f.write("\n    ")
            f.write("\n};\n\n")

            # Khai báo Bias int32
            f.write(f"// Layer {l_num}: Bias (int32)\n")
            f.write(f"static const int32_t s_layer{l_num}_bias[{len(b)}] = {{\n    ")
            for i, val in enumerate(b):
                f.write(f"{int(val):8d}, ")
            f.write("\n};\n\n")

            # Khai báo Struct Config
            f.write(f"static const QuantizedLayerConfig s_layer{l_num}_config = {{\n")
            f.write(f"    .in_features = {layer['in_features']},\n")
            f.write(f"    .out_features = {layer['out_features']},\n")
            f.write(f"    .weights = s_layer{l_num}_weights,\n")
            f.write(f"    .bias = s_layer{l_num}_bias,\n")
            f.write(f"    .output_multiplier = {layer['multiplier']},\n")
            f.write(f"    .output_shift = {layer['shift']},\n")
            f.write(f"    .input_zero_point = {layer['input_zp']},\n")
            f.write(f"    .output_zero_point = {layer['output_zp']},\n")
            f.write(f"    .use_relu = {'true' if layer['use_relu'] else 'false'}\n")
            f.write("};\n\n")

        f.write("// Model Quantization Constants\n")
        f.write(f"#define IDLE_POLICY_INPUT_SCALE {layers_info[0]['input_scale']:.8f}f\n")
        f.write(f"#define IDLE_POLICY_INPUT_ZERO_POINT {layers_info[0]['input_zp']}\n")
        f.write(f"#define IDLE_POLICY_OBS_DIM {layers_info[0]['in_features']}\n")
        f.write(f"#define IDLE_POLICY_ACTION_DIM {layers_info[-1]['out_features']}\n")
    print(f"Generated ESP32-S3 C weights header: {output_header_path}")

def quantize_onnx_model(onnx_path: str, output_dir: str):
    print(f"Reading ONNX model from: {onnx_path} ...")
    model = onnx.load(onnx_path)
    obs_dim = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
    metadata_path = Path(onnx_path).with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if obs_dim not in (6, 39) or metadata.get("requires_goal_runtime"):
        raise ValueError("Goal-residual v2 requires a matching firmware goal controller and calibrated int8 validation; refusing to emit a v1 manifest")

    # Trích xuất weights & biases từ ONNX Initializers
    weights_dict = {}
    for init in model.graph.initializer:
        weights_dict[init.name] = numpy_helper.to_array(init)

    # Lấy các tham số tuyến tính theo đúng thứ tự node Gemm trong đồ thị ONNX
    linear_weights = []
    linear_biases = []

    for node in model.graph.node:
        if node.op_type == "Gemm":
            w_name = node.input[1]
            if w_name in weights_dict:
                linear_weights.append((w_name, weights_dict[w_name]))
            if len(node.input) > 2:
                b_name = node.input[2]
                if b_name in weights_dict:
                    linear_biases.append((b_name, weights_dict[b_name]))

    print(f"Found {len(linear_weights)} Linear layers in Actor model:")
    for w_name, w_arr in linear_weights:
        print(f"  - {w_name}: shape {w_arr.shape}")

    # Tạo calibration data để đo dải động (dynamic range) của activations
    calib_data = generate_calibration_data(num_samples=1000, obs_dim=linear_weights[0][1].shape[1])
    input_scale = 1.0 / 127.0  # Observation đã chuẩn hóa [-1.0 .. 1.0]

    layers_info = []
    curr_input_scale = input_scale
    curr_input_zp = 0

    for i in range(len(linear_weights)):
        w_name, w_arr = linear_weights[i]
        b_name, b_arr = linear_biases[i]
        is_last_layer = (i == len(linear_weights) - 1)

        out_features, in_features = w_arr.shape

        # Quantize weights int8
        q_w, w_scale = quantize_tensor_symmetric_int8(w_arr.flatten())
        # Quantize bias int32
        q_b, _ = quantize_bias_int32(b_arr, curr_input_scale, w_scale)

        # Ước tính output scale
        if is_last_layer:
            if out_features == 5:
                # Continuous 5D action bounded [-1.0 .. 1.0]
                out_scale = 1.0 / 127.0
            else:
                # Output logits không ReLU, phạm vi khoảng [-10 .. 10]
                out_scale = 10.0 / 127.0
            out_zp = 0
            use_relu = False
        else:
            # ReLU output >= 0
            out_scale = 6.0 / 127.0
            out_zp = -128
            use_relu = True

        effective_scale = (curr_input_scale * w_scale) / out_scale
        mult, shift = decompose_scale_to_multiplier_and_shift(effective_scale)

        layers_info.append({
            "layer_name": f"layer_{i+1}",
            "in_features": in_features,
            "out_features": out_features,
            "weights": q_w,
            "bias": q_b,
            "multiplier": mult,
            "shift": shift,
            "input_scale": curr_input_scale,
            "input_zp": curr_input_zp,
            "output_scale": out_scale,
            "output_zp": out_zp,
            "use_relu": use_relu
        })

        curr_input_scale = out_scale
        curr_input_zp = out_zp

    input_dim = linear_weights[0][1].shape[1]
    output_dim = linear_weights[-1][1].shape[0]
    is_behavioral = (input_dim >= 30)

    policy_default_name = "autonomous_behavioral_dynamics" if is_behavioral else "autonomous_idle"
    header_default_name = "behavioral_actor_weights_espnn.h" if is_behavioral else "idle_actor_weights_espnn.h"

    # 1. Sinh C header nạp vào ESP32-S3 firmware
    c_header_path = os.path.join(output_dir, header_default_name)
    generate_c_header(layers_info, c_header_path, policy_name=policy_default_name)

    # 2. Sinh manifest.json
    manifest = {
        "name": policy_default_name,
        "version": "1.0.0",
        "algorithm": "PPO",
        "target_hardware": "ESP32-S3",
        "vector_acceleration": "Xtensa PIE via esp-nn",
        "quantization_type": "int8_symmetric_weights_asymmetric_acts",
        "control_hz": 10,
        "observation": {
            "schema": "behavioral_obs_v1" if is_behavioral else "idle_obs_v1",
            "dim": input_dim,
            "scale": input_scale,
            "zero_point": 0
        },
        "action": {
            "type": "continuous" if is_behavioral else "discrete",
            "dim": output_dim,
            "mapping": ["linear_velocity", "angular_velocity", "pan_velocity", "tilt_velocity", "attention_intensity"] if is_behavioral else ["REST", "LOOK_LEFT", "LOOK_RIGHT", "LOOK_CENTER", "TRACK_TARGET"]
        },
        "layers": [
            {
                "id": idx + 1,
                "in": l["in_features"],
                "out": l["out_features"],
                "multiplier": l["multiplier"],
                "shift": l["shift"],
                "use_relu": l["use_relu"]
            } for idx, l in enumerate(layers_info)
        ]
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated Policy Manifest JSON: {manifest_path}")

def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX Actor model for ESP-DL / ESP-NN")
    parser.add_argument("--onnx", type=str, default="artifacts/autonomous_behavioral_dynamics/1.0.0/actor.onnx", help="Input ONNX path")
    parser.add_argument("--outdir", type=str, default="artifacts/autonomous_behavioral_dynamics/1.0.0", help="Output directory")
    args = parser.parse_args()

    quantize_onnx_model(args.onnx, args.outdir)

if __name__ == "__main__":
    main()

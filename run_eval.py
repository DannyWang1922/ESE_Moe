import subprocess
import sys

nv_cmd = "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0"
moe_layers_list = ["[0]", "[1]", "[2]", "[8]", "[9]", "[10]"]


cmd_list = []
for moe_layer in moe_layers_list:
    layer_num = moe_layer.strip("[]")
    output_dir = f"./results/bert_moe_sst2_{layer_num}"
    cmd = f"python bert_moe.py --moe_layers '{moe_layer}' --output_dir {output_dir}"
    cmd_list.append(cmd)

# 执行所有命令
for cmd in cmd_list:
    full_cmd = nv_cmd + " " + cmd
    print(f"\nRunning cmd: {full_cmd}\n")
    try:
        # subprocess.run(full_cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print(f"Command Done: {full_cmd}")
        print("=" * 100)
    except subprocess.CalledProcessError as e:
        print(f"Command execution failure: {e}")
        print("Error message:")
        print(e.stderr)
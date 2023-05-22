import os
import sys
import pathlib
import subprocess
import argparse

FLOAT_ERR = 1e-6


def collect_output(path: pathlib.Path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines


def compare_float(float1, float2) -> bool:
    return abs(float1 - float2) < FLOAT_ERR


def compare_line(base_line, version_line) -> int:
    """
    return 0: same
    return -1: version has less
    return 1: version has more
    return 2: different number
    """
    base_line = base_line.split(" ")
    version_line = version_line.split(" ")
    if len(base_line) != len(version_line):
        print(f"[valid error: hit/miss] base len: {len(base_line)}, version: {len(version_line)}")
        if len(base_line) < len(version_line):
            return 1
        else:
            return -1
    elif len(base_line) == 2:
        return 0
    else:
        base_f1 = float(base_line[2])
        base_f2 = float(base_line[3])
        base_f3 = float(base_line[4])
        version_f1 = float(version_line[2])
        version_f2 = float(version_line[3])
        version_f3 = float(version_line[4])
        if compare_float(base_f1, version_f1) and compare_float(base_f2, version_f2) and compare_float(base_f3, version_f3):
            return 0
        else:
            print(f"[valid error: float error] base: {base_line}, version: {version_line}")
            return 2


def validation(version='quick'):
    # first test only one
    output_tmp_dir = pathlib.Path('./tmp')
    if not output_tmp_dir.exists():
        print(f"creating tmp dir: {output_tmp_dir}")
        os.mkdir(output_tmp_dir)
    base_output = output_tmp_dir / pathlib.Path('dragon-base.vis')
    output =  output_tmp_dir / pathlib.Path(f'dragon-{version}.vis')
    
    # delete the old output
    if base_output.exists():
        print(f"deleting old base output: {base_output}")
        base_output.unlink()
    if output.exists():
        print(f"deleting old output: {output}")
        output.unlink()
    
    args_base = ['./bin/quick', '-f', './assets/dragon.tri', '-s', str(base_output)]
    args = [f'./bin/{version}', '-f', './assets/dragon.tri', '-s', str(output)]
    
    print(f"running base: {args_base}")
    print(f"running {version}: {args}")
    subprocess.run(args_base)
    subprocess.run(args)
    
    base_output_lines = collect_output(base_output)
    output_lines = collect_output(output)
    
    assert len(base_output_lines) == len(output_lines)
    float_err_cnt = 0
    mis_err_cnt = 0
    hit_cnt = 0
    correct_cnt = 0
    for (base_line, version_line) in zip(base_output_lines, output_lines):
        comp_res = compare_line(base_line, version_line)
        if comp_res == 2:
            float_err_cnt += 1
        if comp_res == -1:
            mis_err_cnt += 1
        if comp_res == 1:
            hit_cnt += 1
        else:
            correct_cnt += 1
    if correct_cnt == len(base_output_lines):
        print(f"====================VALIDATION PASS====================")
    else:
        print(f"====================VALIDATION FAIL====================")
    print(f"float error: {float_err_cnt}, miss error: {mis_err_cnt}, hit error: {hit_cnt}, correct: {correct_cnt}, total: {len(base_output_lines)}")

    
if __name__ == "__main__":
    # read argv[1] as version
    version = 'quick'
    if len(sys.argv) > 1:
        version = sys.argv[1]
    
    print(f"====================validation: {version}====================")
    validation(version)
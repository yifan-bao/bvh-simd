import os
import sys
import time
import json
import pathlib
import subprocess

STAT_ITEMS = ['build_cycles', 'traverse_cycles', 'total_flops', 'build_nodes']


def parse_result(result):
    res_dict = {k:-1 for k in STAT_ITEMS}
    
    lines = result.split('\n')
    for line in lines:
        for item in STAT_ITEMS:
            if line.startswith(item):
                res_dict[item] = int(line.split()[1])
    
    return res_dict
    

def main():
    version = 'quick_count'
    if len(sys.argv) > 1:
        version = sys.argv[1]
  
    output_dir = pathlib.Path('./output')
    if not output_dir.exists():
        print(f"creating output dir: {output_dir}")
        os.mkdir(output_dir)
    # get time stamp number, not str
    time_stamp = int(time.time())
    log_file = output_dir / pathlib.Path(f'{time_stamp}-{version}.log')
    logf = open(log_file, 'w')
    print(f"logging to: {log_file}")
    
    for i in range(1, 20):
        tri_num = i * 100000
        print(f">>>> running {version} with {tri_num} triangles", end=' ')
        start = time.time()
        args = [f'./bin/{version}', '-t', str(tri_num)]
        result = subprocess.run(args, capture_output=True, text=True)
        delta = time.time() - start
        print(f"took {delta:.2f} seconds")
        res_dict = parse_result(result.stdout)
        res_dict['tri_num'] = tri_num
        res_dict['run_time'] = delta
        # write a json line
        logf.write(json.dumps(res_dict) + '\n')
    
    logf.close()


if __name__ == '__main__':
    main()
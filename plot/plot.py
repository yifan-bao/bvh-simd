import re
import matplotlib.pyplot as plt

filenames = []
nodes = []
build_times = []
cycles = []
times = []
flops = []
filename = './result/quickbuild_output.txt'

with open('./result/quickbuild_output.txt', 'r') as f:
    lines = f.readlines()
    
    i = 0
    filename = lines[i].strip().split('.')[0]
    
    for i in range(0, len(lines), 5):
        if (i == 50): break
        filename = lines[i].strip().split('.')[0]
        nodes_match = re.search(r'BVH \((\d+) nodes\)', lines[i+1].strip())
        nodes.append(int(nodes_match.group(1)))
        build_time = float(lines[i+1].strip().split(' ')[-1].split('ms')[0])
        build_times.append(build_time)
        tracing_cycles = float(lines[i+2].strip().split(': ')[1])
        tracing_time_str = lines[i+3].strip().split(': ')[1]
        tracing_time = float(tracing_time_str.split('ms')[0])
        flop_count = int(lines[i+4].strip().split(': ')[1].split()[0])
        filenames.append(filename)
        cycles.append(tracing_cycles)
        times.append(tracing_time)
        flops.append(flop_count)

plt.subplots(figsize=(12, 6))
plt.bar(filenames, cycles)
plt.xlabel('models')
plt.ylabel('Tracing cycles')
plt.savefig('tracing_cycles.png')  

plt.subplots(figsize=(12, 6))
plt.bar(filenames, times)
plt.xlabel('models')
plt.ylabel('Tracing time')
plt.savefig('tracing_time.png')  

plt.subplots(figsize=(12, 6))
plt.bar(filenames, flops)
plt.xlabel('models')
plt.ylabel('flops')
plt.savefig('tracing_flops.png')  

plt.subplots(figsize=(12, 6))
plt.bar(filenames, build_times)
plt.xlabel('models')
plt.ylabel('build_times')
plt.savefig('build_times.png')  

plt.subplots(figsize=(12, 6))
plt.bar(filenames, nodes)
plt.xlabel('models')
plt.ylabel('nodes')
plt.savefig('build_nodes.png')  

performance = [f/c for f, c in zip(flops, cycles)]
plt.subplots(figsize=(12, 6))
plt.bar(filenames, performance)
plt.xlabel('models')
plt.ylabel('performance (flops/cycle))')
plt.savefig('tracing.png')  

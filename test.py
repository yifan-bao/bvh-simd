import os
import subprocess
directory = './assets'
for filename in os.listdir(directory):
    if filename.endswith('.tri'):
        with open(os.path.join(directory, filename), 'r') as f:
            lines = f.readlines()
        # print(filename)
        # print(len(lines))
        args = ['./quickbuild.out', os.path.join(directory, filename), str(len(lines)-1)]
        # # args = ['./quickbuild.out', './assets/dragon.tri', "50000"]
        result = subprocess.run(args, capture_output=True, text=True)
        with open('./result/quickbuild_output.txt', 'a') as f:
            f.write(result.stdout)
        # print(result.stdout)

        
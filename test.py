import os
import subprocess
directory = './assets'

def test_random():
    for triCount in range(100, 50001, 200):
        args = ['./quickbuild.out', str(triCount)]
        result = subprocess.run(args, capture_output=True, text=True)
        if not os.path.exists('./result'):
            os.mkdir('./ressult')
        with open('./result/quickbuild_N.txt', 'a') as f:
            f.write(str(triCount) + '\n')
            f.write(result.stdout)
            # print(result.stdout)

def test_file():
    for filename in os.listdir(directory):
        if filename.endswith('.tri'):
            with open(os.path.join(directory, filename), 'r') as f:
                lines = f.readlines()
            # print(filename)
            # print(len(lines))
            args = ['./quickbuild.out', os.path.join(directory, filename), str(len(lines)-1)]
            # # args = ['./quickbuild.out', './assets/dragon.tri', "50000"]
            result = subprocess.run(args, capture_output=True, text=True)
            if not os.path.exists('./result'):
                os.mkdir('./ressult')
            with open('./result/quickbuild_output.txt', 'a') as f:
                f.write(filename + '\n')
                f.write(result.stdout)
            # print(result.stdout)

if __name__ == '__main__':
    # test_file()
    test_random()
# ASL Team Project

Hang Hu, Xiaoyuan Jin, Yifan Bao, Yiqun Liu

## Topic Introduction

...

Let's work hard!

quickbuild.cpp is our baseline model
profiling -- Intel VTune. Linux Perf
## Build and Run

Build:

- `make count` for counting version
- `make non-count` or simply `make` for non-counting version

Run:

- `./bin/quick` for non-counting version
- `./bin/quick_count` for counting version

for both `./bin/quick` and `./bin/quick_count`, you can
- specify the input file by adding `-f` after the command followed by the file name, e.g. `./bin/quick -f assets/dragon.tri`
- generate a random input file by adding `-t` after the command followed by tri number, e.g. `./bin/quick -t <trinumber>`

```bash
$ ./bin/quick -h

Usage: quick [options]
ASL Team09 Project: BVH.

    -h, --help                show this help message and exit
    -t, --trinumber=<int>     random trinumber
    -v, --valid               validate the result
    -f, --file=<str>          read from tri file
    -s, --save=<str>          save result to file
```

## Validation and Test

### Validation

```bash
python valid.py [<version_name(default=quick)>]

# e.g.
python valid.py quick
```
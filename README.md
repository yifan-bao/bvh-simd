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
- specify the input file by adding the file name after the command, e.g. `./bin/quick assets/dragon.tri`
- generate a random input file by adding `random` after the command followed by tri number, e.g. `./bin/quick random <trinumber>`


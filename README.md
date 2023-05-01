# ASL Team Project

Hang Hu, Xiaoyuan Jin, Yifan Bao, Yiqun Liu

## Topic Introduction

...

Let's work hard!

quickbuild.cpp is our baseline model
profiling -- Intel VTune. Linux Perf
## Build
g++ basics.cpp -Itemplate -I./ -I./lib -o output
g++ quickbuild.cpp -Itemplate -I./ -I./lib -fdeclspec -o quickbuild.out -DCOUNTFLOPS -std=c++17

#pragma once

#include "common.h"
#include "bvh.h"

// TODO(xiaoyuan): this file actually seems redundant, all the things here can be put in common.h

// max min
int max_int(int a, int b) {
    return a > b ? a : b;
}

int min_int(int a, int b) {
    return a < b ? a : b;
}

uint max_uint(uint a, uint b) {
    return a > b ? a : b;
}

uint min_uint(uint a, uint b) {
    return a < b ? a : b;
}


// swap
void swap_float(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

void swap_uint(uint *a, uint *b) {
    uint temp = *a;
    *a = *b;
    *b = temp;
}

void swap_bvhnode(BVHNode *a, BVHNode *b) {
    BVHNode temp = *a;
    *a = *b;
    *b = temp;
}
#define _GNU_SOURCE
#include <time.h>
#include <sched.h>
#include <stdio.h>
#include "utils.h"

void log_allowed_cpus() {
    cpu_set_t mask;
    CPU_ZERO(&mask);

    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_getaffinity");
        fflush(stdout);
        return;
    }

    printf("Allowed CPU cores: ");
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &mask)) {
            printf("%d ", i);
            fflush(stdout);
        }
    }
    printf("\n");
    fflush(stdout);
}


int get_allowed_cpu_count() {
    cpu_set_t mask;
    CPU_ZERO(&mask);

    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_getaffinity");
        return -1;
    }

    int count = 0;
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &mask)) {
            count++;
        }
    }

    return count;
}

void print_usage(const char* prog_name) {
    printf("Usage:\n");
    printf("  %s master <port>\n", prog_name);
    printf("  %s slaver <port> <master_ip>\n", prog_name);
}

double time_in_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
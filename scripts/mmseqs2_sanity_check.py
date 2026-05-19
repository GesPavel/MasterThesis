#!/usr/bin/env python3

import sys

def count_same_cluster_and_genome(tsv_path):
    total = 0
    same = 0

    with open(tsv_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 2:
                continue

            cluster_name = parts[0]
            genome_name = parts[1]

            total += 1

            if cluster_name == genome_name:
                same += 1

    print(f"Total lines: {total}")
    print(f"Same cluster/genome: {same}")
    print(f"Different cluster/genome: {total - same}")

    if total > 0:
        print(f"Percentage same: {100 * same / total:.2f}%")

if __name__ == "__main__":

    count_same_cluster_and_genome("raw_data/result02_cluster.tsv")
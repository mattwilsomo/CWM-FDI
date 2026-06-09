#!/usr/bin/env python3
"""A simple matrix multiplication program for Linux measurement lab.

It computes a simple O(n^3) multiplication with a checksum at the end
so the computation has an observable result.

"""

import sys
from typing import List
import time 

Matrix = List[List[float]]

#The matrix entries are initialized deterministically so that every run uses
#the same data. The exact constants are not important; they only create 
#non-uniform values that vary by row and column.
def init_matrix(n: int, seed: float) -> Matrix:
    return [
        [seed + ((i * 131 + j * 17) % 100) / 100.0 for j in range(n)]
        for i in range(n)
    ]

#Zero out output entries.
def zero_matrix(n: int) -> Matrix:
    return [[0.0 for _ in range(n)] for _ in range(n)]


# Intentionally simple O(n^3) matrix multiplication.
# This loop order is correct but cache-unfriendly for matrix B.
def matmul_slow(a: Matrix, b: Matrix, c: Matrix, n: int) -> None:
    start_func = time.time()
	for i in range(n):
        	for j in range(n):
		    start_elem = time.time()
	            total = 0.0
	            for k in range(n):
	                total += a[i][k] * b[k][j]
	            c[i][j] = total
		    end_elem = time.time()
		    elapsed_elem = end_elem - start_elem
	end_func = time.time()
	elapsed_func = end_func - start_func


def matmul_transpose(a: Matrix, b: Matrix, c: Matrix, n: int) -> None:
	bt = [[0 for _ in range( n)] for _ in range(n)]
	# transposes matrix b and saves it to matrix bt
	for i in range(n):
		for  j in range(n):
			bt[i][j] = bt[j][i]
	#print(b[:3][:3], "\n", bt[:3][:3])
	for i in range(n):
		for j in range(n):
			total = 0.0
			for k in range(n):
				total += a[i][k] * b[j][k]
			c[i][j] = total 


def checksum(m: Matrix, n: int) -> float:
    total = 0.0
    step = (n // 16) + 1
    for i in range(0, n, step):
        for j in range(0, n, step):
            total += m[i][j]
    return total


def usage(prog: str) -> None:
    print(
        f"Usage: {prog} [matrix_size] [repetitions]\n"
        "  matrix_size  : matrix dimension N for an N x N multiply (default: 192)\n"
        "  repetitions  : number of repeated multiplies (default: 1)",
        file=sys.stderr,
    )


def parse_args(argv: list[str]) -> tuple[int, int]:
    #Default values to use when none provided.
    n = 128
    reps = 2

    if len(argv) > 1:
        n = int(argv[1])
    if len(argv) > 2:
        reps = int(argv[2])
    if len(argv) > 3 or n <= 0 or reps <= 0:
        usage(argv[0])
        raise SystemExit(1)

    return n, reps


def main(argv: list[str]) -> int:
    n, reps = parse_args(argv)

    #Initialise around a seed.
    a = init_matrix(n, 1.0)
    b = init_matrix(n, 2.0)

    c = zero_matrix(n)

    for _ in range(reps):
        matmul_slow(a, b, c, n)

    print(f"n={n} reps={reps} checksum={checksum(c, n):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


import numpy as np
import sys

def get_stats(filename):
	

	data = np.loadtxt(filename)

	sorted_data = np.sort(data)
	max_val = np.max(data)
	min_val = np.sort(data)
	mean_val = np.mean(data)
	median_val = np.median(sorted_data)
	percentile_90th = np.percentile(sorted_data, 90)
	percentile_99th = np.percentile(sorted_data, 99)	
	
	return (max_val, min_val, mean_val, median_val, percentile_90th, percentile_99th)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <filename>")
        sys.exit(1)

    filename = sys.argv[1]

    stats = get_stats(filename)

    print(f"Max: {stats[0]}")
    print(f"Min: {stats[1]}")
    print(f"Mean: {stats[2]}")
    print(f"Median: {stats[3]}")
    print(f"90th percentile: {stats[4]}")
    print(f"99th percentile: {stats[5]}")

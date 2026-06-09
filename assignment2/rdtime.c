#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include <time.h>

/*
 * Read the CPU Time Stamp Counter (TSC).
 *
 * LFENCE is used before and after RDTSC to serialize execution
 * and reduce out-of-order execution effects.
 *
 * Returns:
 *     Current TSC value in CPU cycles.
 */
static inline uint64_t read_tsc(void) {
   // _mm_lfence();
    uint64_t t = __rdtsc();
    //_mm_lfence();
    return t;
}

/*
 * Read the CPU frequency from /proc/cpuinfo.
 *
 * Searches for the "cpu MHz" entry and converts the value
 * from MHz to GHz.
 *
 * Returns:
 *     CPU frequency in GHz.
 *
 * Exits:
 *     Terminates the program if the frequency cannot be read.
 */
double read_cpu_freq_ghz(void) {
    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (!fp) {
        perror("fopen");
        exit(1);
    }

    char line[256];
    double mhz = 0.0;

    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "cpu MHz", 7) == 0) {
            sscanf(line, "cpu MHz\t: %lf", &mhz);
            break;
        }
    }

    fclose(fp);

    if (mhz <= 0.0) {
        fprintf(stderr, "Could not determine CPU frequency\n");
        exit(1);
    }

    return mhz / 1000.0;
}

/*
 * Convert CPU cycles to nanoseconds.
 *
 * Formula:
 *     time_ns = cycles / frequency_GHz
 *
 * Since:
 *     1 GHz = 10^9 cycles/sec
 *     1 ns  = 10^-9 sec
 *
 * Parameters:
 *     cycles         Number of CPU cycles.
 *     cpu_freq_ghz   CPU frequency in GHz.
 *
 * Returns:
 *     Approximate elapsed time in nanoseconds.
 */
double cycles_to_ns(uint64_t cycles, double cpu_freq_ghz) {
    return (double)cycles / cpu_freq_ghz;
}

/*
 * Write a list of values to a file, one per line.
 *
 * Mirrors write_list_to_file() in rdtime.py so the C and Python
 * tools produce comparable output files.
 *
 * Parameters:
 *     values     Array of values to write.
 *     count      Number of values in the array.
 *     filename   Destination file (overwritten if it exists).
 *
 * Exits:
 *     Terminates the program if the file cannot be opened.
 */
void write_list_to_file(const uint64_t *values, size_t count, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("fopen");
        exit(1);
    }

    for (size_t i = 0; i < count; i++) {
        fprintf(fp, "%llu\n", (unsigned long long)values[i]);
    }

    fclose(fp);
}

/*
 * Measure the minimum difference between consecutive TSC reads.
 *
 * The function:
 *   1. Allocates an array for timestamps.
 *   2. Collects TSC values as quickly as possible.
 *   3. Computes differences between adjacent timestamps.
 *   4. Optionally writes all differences to a file.
 *   5. Returns the minimum observed difference.
 *
 * Parameters:
 *     num        Number of TSC samples to collect.
 *     filename   File to write the consecutive differences to, or
 *                NULL to skip writing.
 *
 * Returns:
 *     Minimum consecutive TSC difference in CPU cycles.
 *
 * Exits:
 *     Terminates the program on invalid input or allocation failure.
 */
uint64_t min_consecutive_diff(size_t num, const char *filename) {
    if (num < 2) {
        fprintf(stderr, "num must be at least 2\n");
        exit(1);
    }

    uint64_t *timestamps = malloc(num * sizeof(uint64_t));
    if (!timestamps) {
        perror("malloc");
        exit(1);
    }

    uint64_t *diffs = malloc((num - 1) * sizeof(uint64_t));
    if (!diffs) {
        perror("malloc");
        exit(1);
    }

    /*
     * Collect TSC readings.
     */
    for (size_t i = 0; i < num; i++) {
        timestamps[i] = read_tsc();
	
	//clock_t start_time = clock();
	
	//while (clock() < start_time + 10);

    }

    /*
     * Initialize minimum difference using first pair.
     */
    uint64_t min_diff = timestamps[1] - timestamps[0];

    /*
     * Compute all consecutive differences and track minimum.
     */
    for (size_t i = 1; i < num; i++) {
        uint64_t diff = timestamps[i] - timestamps[i - 1];
        diffs[i - 1] = diff;

        if (diff < min_diff) {
            min_diff = diff;
        }
    }

    /*
     * Write the differences to file, similar to rdtime.py.
     */
    if (filename) {
        write_list_to_file(diffs, num - 1, filename);
    }

    free(diffs);
    free(timestamps);
    return min_diff;
}

/*
 * Program entry point.
 *
 * Usage:
 *     ./program [num_samples]
 *
 * Example:
 *     ./program 1000000
 *
 * The program:
 *   1. Reads CPU frequency from /proc/cpuinfo.
 *   2. Measures minimum consecutive TSC delta.
 *   3. Converts cycles to approximate nanoseconds.
 *   4. When run with an argument, writes all consecutive
 *      differences to "time_c.txt", similar to rdtime.py.
 *   5. Prints the results.
 *
 * Parameters:
 *     argc   Argument count.
 *     argv   Argument vector.
 *
 * Returns:
 *     0 on success.
 */
int main(int argc, char **argv) {
    size_t num = (argc > 1)
        ? (size_t)strtoull(argv[1], NULL, 10)
        : 2;

    const char *filename = (argc > 1) ? "time_c.txt" : NULL;

    double cpu_freq_ghz = read_cpu_freq_ghz();

    uint64_t cycles = min_consecutive_diff(num, filename);

    printf("CPU frequency: %.3f GHz\n", cpu_freq_ghz);

    printf("Minimum consecutive TSC diff: %llu cycles\n",
           (unsigned long long)cycles);

    printf("Approximate time: %.2f ns\n",
           cycles_to_ns(cycles, cpu_freq_ghz));

    return 0;
}

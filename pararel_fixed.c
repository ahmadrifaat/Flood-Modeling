// flood_simulation_parallel_v3_fixed.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <time.h>

#define NX 1000
#define NY 1000
#define DX 0.000078055556f
#define DY 0.000081111111f
#define DT 0.1f
#define NSTEPS 1000
#define G 9.81f
#define C 1.0f
#define THRESHOLD 0.001f

float z[NX][NY];
float eta[NX][NY];
float eta_new[NX][NY];
float H[NX][NY];
float p[NX][NY];
float q[NX][NY];
float rain_source[NX][NY];

int read_dem(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening DEM file");
        return 0;
    }
    int ncols, nrows;
    float xllcorner, yllcorner, dx, dy, NODATA_value;
    fscanf(file, "ncols %d\n", &ncols);
    fscanf(file, "nrows %d\n", &nrows);
    fscanf(file, "xllcorner %f\n", &xllcorner);
    fscanf(file, "yllcorner %f\n", &yllcorner);
    fscanf(file, "dx %f\n", &dx);
    fscanf(file, "dy %f\n", &dy);
    fscanf(file, "NODATA_value %f\n", &NODATA_value);

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            if (fscanf(file, "%f", &z[i][j]) != 1)
                z[i][j] = 0.0f;
            eta[i][j] = z[i][j];
            rain_source[i][j] = 0.0f;
        }
    }
    fclose(file);
    return 1;
}

float rand_float(unsigned int *seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return ((float)(*seed) / 0x7fffffff);
}

float rain_intensity_random(float time, unsigned int *seed) {
    return 0.001f * rand_float(seed);
}

float rain_intensity_moving(float time) {
    int center_x = (int)(NX / 2 + 100 * sinf(time));
    int center_y = NY / 2;
    float intensity = 0.005f;
    float radius = 100.0f;
    float rain = 0.0f;

    #pragma omp parallel for reduction(+:rain)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            float dx = i - center_x;
            float dy = j - center_y;
            float distance = sqrtf(dx * dx + dy * dy);
            if (distance < radius) {
                rain_source[i][j] = intensity * (1.0f - distance / radius);
                rain += rain_source[i][j];
            } else {
                rain_source[i][j] = 0.0f;
            }
        }
    }
    return rain;
}

float rain_intensity_chicago(float time) {
    float peak = 0.01f;
    float duration = 50.0f;
    float t = fmodf(time, duration);
    return peak * expf(-0.1f * t);
}

float compute_rainfall(float time, int pattern) {
    float total_rain = 0.0f;
    unsigned int seed = (unsigned int)(time * 1000);

    switch (pattern) {
        case 0:
            #pragma omp parallel for reduction(+:total_rain)
            for (int i = 0; i < NX; i++) {
                unsigned int local_seed = seed + i;
                for (int j = 0; j < NY; j++) {
                    float rain = rain_intensity_random(time, &local_seed);
                    rain_source[i][j] = rain;
                    total_rain += rain;
                }
            }
            break;
        case 1:
            total_rain = rain_intensity_moving(time);
            break;
        case 2:
            {
                float rain = rain_intensity_chicago(time);
                #pragma omp parallel for reduction(+:total_rain)
                for (int i = 0; i < NX; i++) {
                    for (int j = 0; j < NY; j++) {
                        rain_source[i][j] = rain;
                        total_rain += rain;
                    }
                }
            }
            break;
    }
    return total_rain * DX * DY;
}

float sign(float x) {
    return (x > 0.0f) - (x < 0.0f);
}

void simulate_step() {
    #pragma omp parallel for
    for (int i = 1; i < NX - 1; i++) {
        for (int j = 1; j < NY - 1; j++) {
            float h = eta[i][j] - z[i][j];
            float deta_dx = (eta[i + 1][j] - eta[i - 1][j]) / (2.0f * DX);
            float deta_dy = (eta[i][j + 1] - eta[i][j - 1]) / (2.0f * DY);

            if (h > THRESHOLD && isfinite(h) && isfinite(deta_dx) && isfinite(deta_dy)) {
                float sqrt_dx = sqrtf(fmaxf(0.0f, C * C * fabsf(deta_dx)));
                float sqrt_dy = sqrtf(fmaxf(0.0f, C * C * fabsf(deta_dy)));
                p[i][j] = -h * sqrt_dx * sign(deta_dx);
                q[i][j] = -h * sqrt_dy * sign(deta_dy);

                if (!isfinite(p[i][j]) || fabsf(p[i][j]) > 1e5f) p[i][j] = 0.0f;
                if (!isfinite(q[i][j]) || fabsf(q[i][j]) > 1e5f) q[i][j] = 0.0f;
            } else {
                p[i][j] = q[i][j] = 0.0f;
            }
        }
    }

    #pragma omp parallel for
    for (int i = 1; i < NX - 1; i++) {
        for (int j = 1; j < NY - 1; j++) {
            float dpdx = (p[i + 1][j] - p[i - 1][j]) / (2.0f * DX);
            float dqdy = (q[i][j + 1] - q[i][j - 1]) / (2.0f * DY);
            eta_new[i][j] = eta[i][j] - DT * (dpdx + dqdy) + DT * rain_source[i][j];
            if (!isfinite(eta_new[i][j]) || fabsf(eta_new[i][j]) > 1e5f)
                eta_new[i][j] = 0.0f;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < NX; i++)
        for (int j = 0; j < NY; j++)
            eta[i][j] = eta_new[i][j];
}

void write_log(FILE *logfile, int step, float time, float rain_volume) {
    float eta_volume = 0.0f, max_eta = -INFINITY, min_eta = INFINITY;
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            float h = eta[i][j] - z[i][j];
            if (h > 0.0f) eta_volume += h * DX * DY;
            if (eta[i][j] > max_eta) max_eta = eta[i][j];
            if (eta[i][j] < min_eta) min_eta = eta[i][j];
        }
    }
    fprintf(logfile, "%d,%.2f,%f,%f,%f,%f\n", step, time, rain_volume, eta_volume, max_eta, min_eta);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <dem_file.asc> <rain_pattern (0=rand,1=mov,2=flash)>\n", argv[0]);
        return 1;
    }
    if (!read_dem(argv[1])) return 1;
    int rain_pattern = atoi(argv[2]);

    FILE *logfile = fopen("log_parallel_v3.csv", "w");
    fprintf(logfile, "step,time,rain_volume,eta_volume,max_eta,min_eta\n");

    double start_time = omp_get_wtime();
    for (int step = 0; step < NSTEPS; step++) {
        float sim_time = step * DT;
        float rain = compute_rainfall(sim_time, rain_pattern);
        simulate_step();
        write_log(logfile, step, sim_time, rain);
    }
    double end_time = omp_get_wtime();
    fclose(logfile);

    printf("Simulation completed in %.2f seconds.\n", end_time - start_time);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#define N 1000
#define DX 0.000078055556f
#define DY 0.000081111111f
#define DT 0.1f
#define C 50.0f
#define TOTAL_TIME 10.0f
#define THRESHOLD 1e-6f

// Parameter IDF
#define IDF_A 1000.0f
#define IDF_B 10.0f
#define IDF_N 0.75f

float elevation[N][N], water[N][N], flow_x[N][N], flow_y[N][N];

void read_dem(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Gagal membuka file: %s\n", filename);
        exit(1);
    }

    char header[256];
    for (int i = 0; i < 6; ++i)
        fgets(header, sizeof(header), file); // Lewati header

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            if (fscanf(file, "%f", &elevation[y][x]) != 1) {
                printf("Gagal membaca data elevasi.\n");
                exit(1);
            }
            if (elevation[y][x] < -1000) elevation[y][x] = 0.0f; // NODATA jadi 0
            water[y][x] = 0.0f;
        }
    }

    fclose(file);
}

float compute_rainfall(float time_seconds) {
    float time_minutes = time_seconds / 60.0f;
    float intensity_mm_per_hour = IDF_A / powf(time_minutes + IDF_B, IDF_N);
    float rain_height = intensity_mm_per_hour * (DT / 3600.0f); // Konversi ke meter
    return rain_height;
}

void apply_rainfall(float rain) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            water[y][x] += rain;
        }
    }
}

void update_flow() {
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < N - 1; y++) {
        for (int x = 1; x < N - 1; x++) {
            float dhdx = ((elevation[y][x] + water[y][x]) - (elevation[y][x + 1] + water[y][x + 1])) / DX;
            float dhdy = ((elevation[y][x] + water[y][x]) - (elevation[y + 1][x] + water[y + 1][x])) / DY;
            flow_x[y][x] = -C * dhdx;
            flow_y[y][x] = -C * dhdy;
        }
    }
}

void update_water() {
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < N - 1; y++) {
        for (int x = 1; x < N - 1; x++) {
            float dqx = (flow_x[y][x - 1] - flow_x[y][x]) / DX;
            float dqy = (flow_y[y - 1][x] - flow_y[y][x]) / DY;
            float dV = DT * (dqx + dqy);
            water[y][x] += dV;
            if (water[y][x] < 0.0f) water[y][x] = 0.0f;
        }
    }
}

float compute_total_volume() {
    float total = 0.0f;
    #pragma omp parallel for reduction(+:total) collapse(2)
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            total += water[y][x] * DX * DY;
        }
    }
    return total;
}

int main() {
    read_dem("dem_reproject.asc");

    FILE *log_file = fopen("simulation_log.csv", "w");
    fprintf(log_file, "Time(s),Rain(m),Total_Volume(m3)\n");

    int steps = (int)(TOTAL_TIME / DT);
    for (int step = 0; step < steps; step++) {
        float t = step * DT;
        float rain = compute_rainfall(t);
        apply_rainfall(rain);
        update_flow();
        update_water();
        float volume = compute_total_volume();
        fprintf(log_file, "%.2f,%.6f,%.6f\n", t, rain, volume);
    }

    fclose(log_file);
    printf("Simulasi selesai. Hasil disimpan di simulation_log.csv\n");
    return 0;
}

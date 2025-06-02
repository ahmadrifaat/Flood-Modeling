#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define N 1000
#define DX 0.000078055556f
#define DY 0.000081111111f
#define DT 0.1f
#define C 50.0f
#define THRESHOLD 1e-6f
#define TOTAL_TIME 100.0f

#define IDF_A 1000.0f
#define IDF_B 10.0f
#define IDF_N 0.75f
#define MINUTES_TO_SECONDS 60.0f
#define MM_PER_HOUR_TO_M_PER_SEC (1.0f / 3600000.0f)

#define RAIN_PATTERN 1  // 1: Random, 2: Moving, 3: Flash

float sign(float x) {
    return (x > 0) - (x < 0);
}

float rain_intensity_idf(float t_seconds) {
    float t_min = t_seconds / MINUTES_TO_SECONDS;
    float intensity_mmhr = IDF_A / powf(t_min + IDF_B, IDF_N);
    return intensity_mmhr * MM_PER_HOUR_TO_M_PER_SEC;
}

float rain_intensity_chicago(float t, float T) {
    float r = 0.3f;
    float tp = r * T;
    float Imax = rain_intensity_idf(tp);

    if (t <= tp)
        return (Imax / tp) * t;
    else if (t <= T)
        return (Imax / (T - tp)) * (T - t);
    else
        return 0.0f;
}

void generate_random_rain(float (*rain_source)[N], float intensity) {
    #pragma omp parallel
    {
        // RNG lokal per thread (xorshift32 sederhana)
        unsigned int seed = (unsigned int)(time(NULL) + omp_get_thread_num() * 1234567);
        #pragma omp for collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                // xorshift32
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                float rand01 = (float)(seed % 10000) / 10000.0f;
                rain_source[i][j] = intensity * rand01;
            }
        }
    }
}

void generate_moving_rain(float (*rain_source)[N], float intensity, int step) {
    int radius = 30;
    int center_i = (step / 2) % N;
    int center_j = N / 2;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dist = sqrtf((i - center_i)*(i - center_i) + (j - center_j)*(j - center_j));
            rain_source[i][j] = (dist < radius) ? intensity * (1.0f - dist / radius) : 0.0f;
        }
    }
}

void generate_flash_rain(float (*rain_source)[N], float intensity, float t) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            rain_source[i][j] = (t >= 3.0f && t <= 4.0f && i >= 300 && i < 700 && j >= 300 && j < 700)
                                ? intensity * 5.0f : 0.0f;
}

int read_dem_asc(const char* filename, float (*z)[N]) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Gagal membuka file DEM: %s\n", filename);
        return 0;
    }

    int ncols, nrows;
    float xllcorner, yllcorner, dx, dy, nodata;
    fscanf(file, "ncols %d\n", &ncols);
    fscanf(file, "nrows %d\n", &nrows);
    fscanf(file, "xllcorner %f\n", &xllcorner);
    fscanf(file, "yllcorner %f\n", &yllcorner);
    fscanf(file, "dx %f\n", &dx);
    fscanf(file, "dy %f\n", &dy);
    fscanf(file, "NODATA_value %f\n", &nodata);

    if (ncols != N || nrows != N) {
        printf("Ukuran DEM tidak sesuai dengan N=%d. DEM: %dx%d\n", N, ncols, nrows);
        fclose(file);
        return 0;
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float val;
            if (fscanf(file, "%f", &val) != 1) {
                printf("Gagal membaca data elevasi pada [%d][%d]\n", i, j);
                fclose(file);
                return 0;
            }
            z[i][j] = (val == nodata) ? 0.0f : val;
        }

    fclose(file);
    return 1;
}

void simulate_step(float (*eta)[N], float (*z)[N], float (*H)[N],
                   float (*p)[N], float (*q)[N],
                   float (*eta_new)[N], float (*rain_source)[N]) {
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++) {
            H[i][j] = eta[i][j] - z[i][j];
            float deta_dx = (eta[i+1][j] - eta[i-1][j]) / (2.0f * DX);
            float deta_dy = (eta[i][j+1] - eta[i][j-1]) / (2.0f * DY);
            float h = H[i][j];
            if (h > THRESHOLD) {
                p[i][j] = -h * sqrtf(C * C * fabsf(deta_dx)) * sign(deta_dx);
                q[i][j] = -h * sqrtf(C * C * fabsf(deta_dy)) * sign(deta_dy);
            } else {
                p[i][j] = q[i][j] = 0.0f;
            }
        }

    float damping = 0.98f;
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++) {
            p[i][j] *= damping;
            q[i][j] *= damping;
        }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++) {
            float dpdx = (p[i+1][j] - p[i-1][j]) / (2.0f * DX);
            float dqdy = (q[i][j+1] - q[i][j-1]) / (2.0f * DY);
            eta_new[i][j] = eta[i][j] - DT * (dpdx + dqdy) + DT * rain_source[i][j];
        }

    for (int i = 0; i < N; i++) {
        eta_new[i][0] = eta_new[i][1];
        eta_new[i][N-1] = eta_new[i][N-2];
        eta_new[0][i] = eta_new[1][i];
        eta_new[N-1][i] = eta_new[N-2][i];
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Gunakan: %s <nama_file_dem.asc>\n", argv[0]);
        return 1;
    }

    float (*eta)[N]         = malloc(sizeof(float) * N * N);
    float (*eta_new)[N]     = malloc(sizeof(float) * N * N);
    float (*z)[N]           = malloc(sizeof(float) * N * N);
    float (*H)[N]           = malloc(sizeof(float) * N * N);
    float (*p)[N]           = malloc(sizeof(float) * N * N);
    float (*q)[N]           = malloc(sizeof(float) * N * N);
    float (*rain_source)[N] = malloc(sizeof(float) * N * N);

    if (!eta || !eta_new || !z || !H || !p || !q || !rain_source) {
        fprintf(stderr, "Gagal mengalokasikan memori.\n");
        return 1;
    }

    memset(eta[0], 0, sizeof(float) * N * N);
    memset(eta_new[0], 0, sizeof(float) * N * N);
    memset(z[0], 0, sizeof(float) * N * N);
    memset(H[0], 0, sizeof(float) * N * N);
    memset(p[0], 0, sizeof(float) * N * N);
    memset(q[0], 0, sizeof(float) * N * N);
    memset(rain_source[0], 0, sizeof(float) * N * N);

    if (!read_dem_asc(argv[1], z)) {
        return 1;
    }

    printf("DEM berhasil dibaca.\n");
    FILE* log_fp = fopen("log_parallel_v3.csv", "w");
    fprintf(log_fp, "step,time,rain_volume,eta_volume,max_eta,min_eta\n");

    int total_steps = (int)(TOTAL_TIME / DT);
    double start_time = omp_get_wtime();

    for (int step = 0; step < total_steps; step++) {
        float t = step * DT;
        float intensity = rain_intensity_chicago(t, TOTAL_TIME);

        #if RAIN_PATTERN == 1
            generate_random_rain(rain_source, intensity);
        #elif RAIN_PATTERN == 2
            generate_moving_rain(rain_source, intensity, step);
        #elif RAIN_PATTERN == 3
            generate_flash_rain(rain_source, intensity, t);
        #endif

        simulate_step(eta, z, H, p, q, eta_new, rain_source);

        float rain_vol = 0.0f, eta_vol = 0.0f;
        float max_eta = -1e9f, min_eta = 1e9f;

        #pragma omp parallel for collapse(2) reduction(+:rain_vol,eta_vol) reduction(max:max_eta) reduction(min:min_eta)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                rain_vol += rain_source[i][j] * DT * DX * DY;
                eta_vol  += eta_new[i][j] * DX * DY;
                if (eta_new[i][j] > max_eta) max_eta = eta_new[i][j];
                if (eta_new[i][j] < min_eta) min_eta = eta_new[i][j];
            }

        fprintf(log_fp, "%d,%.2f,%.6f,%.6f,%.4f,%.4f\n", step, t, rain_vol, eta_vol, max_eta, min_eta);

        if (step % 10 == 0) {
            // Mencetak jumlah thread aktif dari dalam blok paralel
            #pragma omp parallel
            {
                #pragma omp single
                printf("Langkah %d, waktu %.2fs, thread aktif: %d\n", step, t, omp_get_num_threads());
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                eta[i][j] = eta_new[i][j];
    }

    double end_time = omp_get_wtime();
    printf("Simulasi selesai dalam %.3f detik.\n", end_time - start_time);
    fclose(log_fp);

    free(eta); free(eta_new); free(z); free(H); free(p); free(q); free(rain_source);
    return 0;
}

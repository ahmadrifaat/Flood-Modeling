// program_pararel_omp.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#define MAX_ROWS 1000
#define MAX_COLS 1000
#define DT 0.1f
#define DX 1.0f
#define DY 1.0f
#define TOTAL_TIME 10.0f
#define RAIN_INTENSITY 0.00001f
#define THRESHOLD 1e-6f
#define C 50.0f

int nrows, ncols;
float z[MAX_ROWS][MAX_COLS];
float eta[MAX_ROWS][MAX_COLS];
float eta_new[MAX_ROWS][MAX_COLS];
float H[MAX_ROWS][MAX_COLS];
float p[MAX_ROWS][MAX_COLS];
float q[MAX_ROWS][MAX_COLS];

int read_dem_asc(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Gagal membuka file: %s\n", filename);
        return 0;
    }

    char line[1024];
    int header_count = 0;
    while (header_count < 6 && fgets(line, sizeof(line), fp)) {
        if (strstr(line, "ncols")) sscanf(line, "ncols %d", &ncols);
        else if (strstr(line, "nrows")) sscanf(line, "nrows %d", &nrows);
        header_count++;
    }

    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            if (fscanf(fp, "%f", &z[i][j]) != 1) {
                printf("Kesalahan membaca nilai elevasi di baris %d kolom %d\n", i, j);
                fclose(fp);
                return 0;
            }
        }
    }
    fclose(fp);
    return 1;
}

int main() {
    if (!read_dem_asc("dem_tamalanrea.asc")) {
        printf("Gagal membaca file elevasi.\n");
        return 1;
    }
    printf("File elevasi berhasil dibaca: %d baris x %d kolom.\n", nrows, ncols);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nrows; i++)
        for (int j = 0; j < ncols; j++) {
            eta[i][j] = z[i][j] + 1.0f;
        }

    FILE* log_fp = fopen("log.csv", "w");
    fprintf(log_fp, "step,time,total_eta\n");

    int total_steps = (int)(TOTAL_TIME / DT);
    for (int step = 0; step < total_steps; step++) {
        float t = step * DT;

        // Stage 1: Hitung debit p dan q
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < nrows - 1; i++) {
            for (int j = 1; j < ncols - 1; j++) {
                H[i][j] = eta[i][j] - z[i][j];
                float deta_dx = (eta[i+1][j] - eta[i-1][j]) / (2.0f * DX);
                float deta_dy = (eta[i][j+1] - eta[i][j-1]) / (2.0f * DY);
                float h = H[i][j];

                if (h > THRESHOLD) {
                    p[i][j] = -h * sqrtf(C * C * fabsf(deta_dx)) * ((deta_dx > 0) - (deta_dx < 0));
                    q[i][j] = -h * sqrtf(C * C * fabsf(deta_dy)) * ((deta_dy > 0) - (deta_dy < 0));
                } else {
                    p[i][j] = 0.0f;
                    q[i][j] = 0.0f;
                }
            }
        }

        // Stage 2: Update eta
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < nrows - 1; i++) {
            for (int j = 1; j < ncols - 1; j++) {
                float dpdx = (p[i+1][j] - p[i-1][j]) / (2.0f * DX);
                float dqdy = (q[i][j+1] - q[i][j-1]) / (2.0f * DY);
                eta_new[i][j] = eta[i][j] - DT * (dpdx + dqdy) + DT * RAIN_INTENSITY;
            }
        }

        // Boundary condition
        #pragma omp parallel for
        for (int i = 0; i < ncols; i++) {
            eta_new[0][i] = eta_new[1][i];
            eta_new[nrows-1][i] = eta_new[nrows-2][i];
        }
        #pragma omp parallel for
        for (int i = 0; i < nrows; i++) {
            eta_new[i][0] = eta_new[i][1];
            eta_new[i][ncols-1] = eta_new[i][ncols-2];
        }

        // Swap eta
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nrows; i++)
            for (int j = 0; j < ncols; j++)
                eta[i][j] = eta_new[i][j];

        // Logging
        if (step % 10 == 0) {
            float total_eta = 0.0f;
            #pragma omp parallel for reduction(+:total_eta) collapse(2)
            for (int i = 0; i < nrows; i++)
                for (int j = 0; j < ncols; j++)
                    total_eta += eta[i][j];

            fprintf(log_fp, "%d,%.2f,%f\n", step, t, total_eta);
            printf("Step %d selesai.\n", step);
        }
    }

    fclose(log_fp);
    printf("Simulasi selesai.\n");
    return 0;
}

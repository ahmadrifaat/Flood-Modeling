#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 10
#define DX 1.0f
#define DY 1.0f
#define DT 0.1f
#define C 50.0f
#define THRESHOLD 1e-6f
#define TOTAL_TIME 10.0f

#define IDF_A 1000.0f
#define IDF_B 10.0f
#define IDF_N 0.75f
#define MINUTES_TO_SECONDS 60.0f
#define MM_PER_HOUR_TO_M_PER_SEC (1.0f / 3600000.0f)

#define RAIN_PATTERN 2  // 1: Random, 2: Moving, 3: Flash

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

// Pola hujan modular
void generate_random_rain(float rain_source[N][N], float intensity) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            rain_source[i][j] = intensity * ((float)rand() / RAND_MAX);
}

void generate_moving_rain(float rain_source[N][N], float intensity, int step) {
    int radius = 2;
    int center_i = (step / 2) % N;
    int center_j = N / 2;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dist = sqrtf((i - center_i)*(i - center_i) + (j - center_j)*(j - center_j));
            if (dist < radius) {
                rain_source[i][j] = intensity * (1.0f - dist / radius);
            } else {
                rain_source[i][j] = 0.0f;
            }
        }
    }
}

void generate_flash_rain(float rain_source[N][N], float intensity, float t) {
    if (t >= 3.0f && t <= 4.0f) {
        for (int i = 3; i < 7; i++)
            for (int j = 3; j < 7; j++)
                rain_source[i][j] = intensity * 5.0f;
    } else {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                rain_source[i][j] = 0.0f;
    }
}

void log_data(FILE* fp, int step, float time, float rain_vol, float eta_vol, float max_eta, float min_eta) {
    fprintf(fp, "%d,%.2f,%.6f,%.6f,%.4f,%.4f\n", step, time, rain_vol, eta_vol, max_eta, min_eta);
}

void simulate_step_multistage(
    float eta[N][N],
    float z[N][N],
    float H[N][N],
    float p[N][N],
    float q[N][N],
    float eta_new[N][N],
    float rain_source[N][N]
) {
    // Stage 1: Debit
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            H[i][j] = eta[i][j] - z[i][j];
            float deta_dx = (eta[i+1][j] - eta[i-1][j]) / (2.0f * DX);
            float deta_dy = (eta[i][j+1] - eta[i][j-1]) / (2.0f * DY);
            float h = H[i][j];

            if (h > THRESHOLD) {
                p[i][j] = -h * sqrtf(C * C * fabsf(deta_dx)) * sign(deta_dx);
                q[i][j] = -h * sqrtf(C * C * fabsf(deta_dy)) * sign(deta_dy);
            } else {
                p[i][j] = 0.0f;
                q[i][j] = 0.0f;
            }
        }
    }

    // Stage 2: Damping
    float damping = 0.98f;
    for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++) {
            p[i][j] *= damping;
            q[i][j] *= damping;
        }

    // Stage 3: Update eta
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            float dpdx = (p[i+1][j] - p[i-1][j]) / (2.0f * DX);
            float dqdy = (q[i][j+1] - q[i][j-1]) / (2.0f * DY);
            float d_eta_dt = - (dpdx + dqdy);
            eta_new[i][j] = eta[i][j] + DT * d_eta_dt;
        }
    }

    // Stage 4: Tambahkan hujan
    for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++)
            eta_new[i][j] += DT * rain_source[i][j];

    // Stage 5: Boundary
    for (int i = 0; i < N; i++) {
        eta_new[i][0]       = eta_new[i][1];
        eta_new[i][N-1]     = eta_new[i][N-2];
        eta_new[0][i]       = eta_new[1][i];
        eta_new[N-1][i]     = eta_new[N-2][i];
    }
}

void print_eta(const float eta[N][N], int step) {
    printf("ETA STEP %d:\n", step);
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++)
            printf("%6.2f ", eta[i][j]);
        printf("\n");
    }
    printf("\n");
}

int main() {
    float eta[N][N] = {0}, eta_new[N][N] = {0}, z[N][N] = {0};
    float H[N][N] = {0}, p[N][N] = {0}, q[N][N] = {0};
    float rain_source[N][N] = {0};

    srand(time(NULL)); // untuk random pattern

    // Inisialisasi: elevasi datar dan genangan awal
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            eta[i][j] = 1.0f;
            z[i][j] = 0.0f;
        }
    eta[N/2][N/2] = 2.0f;

    FILE* log_fp = fopen("simulation_log.csv", "w");
    if (!log_fp) {
        printf("Gagal membuka file log.\n");
        return 1;
    }
    fprintf(log_fp, "step,time,rain_volume,eta_volume,max_eta,min_eta\n");

    int total_steps = (int)(TOTAL_TIME / DT);
    for (int step = 0; step < total_steps; step++) {
        float t = step * DT;
        float intensity = rain_intensity_chicago(t, TOTAL_TIME);

        // Gunakan pola hujan modular
        #if RAIN_PATTERN == 1
            generate_random_rain(rain_source, intensity);
        #elif RAIN_PATTERN == 2
            generate_moving_rain(rain_source, intensity, step);
        #elif RAIN_PATTERN == 3
            generate_flash_rain(rain_source, intensity, t);
        #else
            for (int i = 0; i < N/2; i++)
                for (int j = 0; j < N/2; j++)
                    rain_source[i][j] = intensity;
        #endif

        simulate_step_multistage(eta, z, H, p, q, eta_new, rain_source);

        float rain_vol = 0.0f, eta_vol = 0.0f, max_eta = -1e9f, min_eta = 1e9f;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                rain_vol += rain_source[i][j] * DT * DX * DY;
                eta_vol  += eta_new[i][j] * DX * DY;
                if (eta_new[i][j] > max_eta) max_eta = eta_new[i][j];
                if (eta_new[i][j] < min_eta) min_eta = eta_new[i][j];
            }

        log_data(log_fp, step, t, rain_vol, eta_vol, max_eta, min_eta);

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                eta[i][j] = eta_new[i][j];

        if (step % 10 == 0)
            print_eta(eta, step);
    }

    fclose(log_fp);
    return 0;
}
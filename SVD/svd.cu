#include <stdio.h>
#include <stdlib.h>

void get_args(m_p, n_p, r_p, argv) {
    *m_p = strtol(argv[1], NULL, 10);
    *n_p = strtol(argv[2], NULL, 10);
    *r_p = strtol(argv[3], NULL, 10);

    if (m <= 0 || n <= 0 || r <= 0) {
        fprintf(stderr, "Error: m, n, and r must be positive integers.\n");
        return EXIT_FAILURE;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <m> <n> <r>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int m, n, r;
    get_args(&m, &n, &r, argv);

}
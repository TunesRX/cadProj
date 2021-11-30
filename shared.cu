/*
 * proj1 CAD 2021/2022 FCT/UNL
 * vad
 */
#include <assert.h>
#include <ctype.h>
#include <driver_types.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* read_ppm - read a PPM image ascii file
 *   returns pointer to data, dimensions and max colors (from PPM header)
 *   data format: sequence of width x height of 3 ints for R,G and B
 *   aborts on errors
 ah pois é
 */
void read_ppm(FILE *f, int **img, int *width, int *height, int *maxcolors) {
  int count = 0;
  char ppm[10];
  int c;
  // header
  while ((c = fgetc(f)) != EOF && count < 4) {
    if (isspace(c))
      continue;
    if (c == '#') {
      while (fgetc(f) != '\n')
        ;
      continue;
    }
    ungetc(c, f);
    switch (count) {
    case 0:
      count += fscanf(f, "%2s", ppm);
      break;
    case 1:
      count += fscanf(f, "%d%d%d", width, height, maxcolors);
      break;
    case 2:
      count += fscanf(f, "%d%d", height, maxcolors);
      break;
    case 3:
      count += fscanf(f, "%d", maxcolors);
    }
  }
  assert(c != EOF);
  assert(strcmp("P3", ppm) == 0);
  // data
  int *data = *img = (int *)malloc(3 * (*width) * (*height) * sizeof(int));
  assert(img != NULL);
  int r, g, b, pos = 0;
  while (fscanf(f, "%d%d%d", &r, &g, &b) == 3) {
    data[pos++] = r;
    data[pos++] = g;
    data[pos++] = b;
  }
  assert(pos == 3 * (*width) * (*height));
}

/* write_ppm - write a PPM image ascii file
 */
void write_ppm(FILE *f, int *img, int width, int height, int maxcolors) {
  // header
  fprintf(f, "P3\n%d %d %d\n", width, height, maxcolors);
  // data
  for (int l = 0; l < height; l++) {
    for (int c = 0; c < width; c++) {
      int p = 3 * (l * width + c);
      fprintf(f, "%d %d %d  ", img[p], img[p + 1], img[p + 2]);
    }
    putc('\n', f);
  }
}

/* printImg - print to screen the content of img
 */
void printImg(int imgw, int imgh, const int *img) {
  for (int j = 0; j < imgh; j++) {
    for (int i = 0; i < imgw; i++) {
      int x = 3 * (i + j * imgw);
      printf("%d,%d,%d  ", img[x], img[x + 1], img[x + 2]);
    }
    putchar('\n');
  }
}
__device__ void updateArray(int *src, int dst[34][34][3], int idxSrc,
                            int idyDst, int idxDst) {
  dst[idyDst][idxDst][0] = src[idxSrc];
  dst[idyDst][idxDst][1] = src[idxSrc + 1];
  dst[idyDst][idxDst][2] = src[idxSrc + 2];
}

/* areaFilter - transform a point (line,col) with contributions from its
 * neighbours no change if filter={{0,0,0}, {0,1,0}, {0,0,0}};
 */
__global__ void areaFilter(int *out, int *img, int imgw, int imgh,
                           int *filter) {

  // int line;
  // int col;
  const int sharedSz = 34;

  __shared__ int temp[sharedSz][sharedSz][3];

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int line = blockIdx.y * blockDim.y + threadIdx.y;

  if (line < imgh && col < imgw) {

    int tempLine = threadIdx.y + 1;
    int tempCol = threadIdx.x + 1;
    int idx = 3 * (line * imgw + col);
    updateArray(img, temp, idx, tempLine, tempCol);

    // copia lado esquerdo
    if (threadIdx.x == 0 && blockIdx.x != 0) {
      tempLine = threadIdx.y + 1;
      tempCol = 0;
      int idx = 3 * (line * imgw + (col - 1));

      updateArray(img, temp, idx, tempLine, tempCol);
    }

    // copia superior
    if (threadIdx.y == 0 && blockIdx.y != 0) {
      int idx = 3 * ((line - 1) * imgw + col);
      tempCol = threadIdx.x + 1;
      tempLine = 0;
      updateArray(img, temp, idx, tempLine, tempCol);
    }
    // copia lado direito
    int nbW = (imgw + blockDim.x - 1) / blockDim.x;
    if (threadIdx.x == blockDim.x - 1 && blockIdx.x != nbW - 1) {
      int idx = 3 * (line * imgw + (col + 1));
      tempCol = sharedSz - 1;
      tempLine = threadIdx.y + 1;
      updateArray(img, temp, idx, tempLine, tempCol);
    }
    // copia lado inferior
    int nbH = (imgh + blockDim.y - 1) / blockDim.y;
    if (threadIdx.y == blockDim.x - 1 && blockIdx.y != nbH - 1) {
      int idx = 3 * ((line + 1) * imgw + col);
      tempCol = threadIdx.x + 1;
      tempLine = sharedSz - 1;
      updateArray(img, temp, idx, tempLine, tempCol);
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int idx;
      if (line != 0 && col != 0) {
        idx = 3 * ((line - 1) * imgw + (col - 1));
        updateArray(img, temp, idx, 0, 0);
      }

      if (line != 0 && blockIdx.x != nbW - 1) {
        idx = 3 * ((line - 1) * imgw + (blockDim.x + blockDim.x * blockIdx.x));
        updateArray(img, temp, idx, 0, sharedSz - 1);
      }

      if (blockIdx.y != nbH - 1 && col != 0) {
        idx = 3 * ((blockIdx.y * blockDim.y + blockDim.y) * imgw + (col - 1));
        updateArray(img, temp, idx, sharedSz - 1, 0);
      }

      if (blockIdx.y != nbH - 1 && blockIdx.x != nbW - 1) {
        idx = 3 * ((blockIdx.y * blockDim.y + blockDim.y) * imgw +
                   (blockDim.x + blockIdx.x * blockDim.x));
        updateArray(img, temp, idx, sharedSz - 1, sharedSz - 1);
      }
    }
    /*int blockX = 0;
    int blockY = 0;
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x==blockX
    &&blockIdx.y==blockY){ for(int i = 0; i < sharedSz; i++){ for( int j = 0; j
    < sharedSz; j ++){ printf("%d %d %d  ", temp[i][j][0],temp[i][j][1],
    temp[i][j][2]);
        }
        printf("oii\n");
      }

    }*/
    __syncthreads();
    int r = 0, g = 0, b = 0, n = 0;
    for (int l = threadIdx.y; l < threadIdx.y + 3; l++)
      for (int c = threadIdx.x; c < threadIdx.x + 3; c++) {
        int col2 = blockIdx.x * blockDim.x + c;
        int line1 = blockIdx.y * blockDim.y + l;

        if (line1 > 0 && col2 > 0 && col2 <= imgw && line1 <= imgh) {

          int scale = filter[c - threadIdx.x + (l - threadIdx.y) * 3];
          r += scale * temp[l][c][0];
          g += scale * temp[l][c][1];
          b += scale * temp[l][c][2];
          n += scale;
        }
      }

    idx = 3 * (line * imgw + col);
    col = blockIdx.x * blockDim.x + threadIdx.x;
    line = blockIdx.y * blockDim.y + threadIdx.y;
    idx = 3 * (line * imgw + col);
    out[idx] = r / n;
    out[idx + 1] = g / n;
    out[idx + 2] = b / n;
  }
}

/* pointFilter - transform a point (line,col) with greyscale
 *          newcolor = alpha*grey(color) +(1-alpha)*color
 *          newcolor = color (no change) if alpha=0
 */
__global__ void pointFilter(int *out, int *img, int imgw, int imgh,
                            float alpha) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int line = blockIdx.y * blockDim.y + threadIdx.y;
  int r = 0, g = 0, b = 0;
  float grey;
  if (line < imgh && col < imgw) {
    int idx = 3 * (line * imgw + col);
    r = img[idx];
    g = img[idx + 1];
    b = img[idx + 2];
    grey = alpha * (0.3 * r + 0.59 * g + 0.11 * b);
    out[idx] = (1 - alpha) * r + grey;
    out[idx + 1] = (1 - alpha) * g + grey;
    out[idx + 2] = (1 - alpha) * b + grey;
  }
}

int main(int argc, char *argv[]) {
  int imgh, imgw, imgc;
  int *img;
  float alpha = 0.5;             // default value
  int filter[3][3] = {{1, 2, 1}, // gaussian filter
                      {2, 4, 2},
                      {1, 2, 1}};

  /*int filter[3][3] = {{1, 2, 3}, // gaussian filter
                      {4, 5, 6},
                      {7, 8, 9}};*/
  if (argc != 2 && argc != 3) {
    fprintf(stderr, "usage: %s img.ppm [alpha]\n", argv[0]);
    return EXIT_FAILURE;
  }
  if (argc == 3) {
    alpha = atof(argv[2]);
  }
  FILE *f = fopen(argv[1], "r");
  if (f == NULL) {
    fprintf(stderr, "can't read file %s\n", argv[1]);
    return EXIT_FAILURE;
  }

  read_ppm(f, &img, &imgw, &imgh, &imgc);
  // printf("PPM image %dx%dx%d\n", imgw, imgh, imgc);
  // printImg(imgw, imgh, img);

  int *out = (int *)malloc(3 * imgw * imgh * sizeof(int));
  assert(out != NULL);

  clock_t t = clock();

  int *d_filter;
  cudaMalloc(&d_filter, 3 * 3 * sizeof(int));
  int *d_a;
  cudaMalloc(&d_a, 3 * imgw * imgh * sizeof(int));
  int *d_b;
  cudaMalloc(&d_b, 3 * imgw * imgh * sizeof(int));
  int *d_c;
  cudaMalloc(&d_c, 3 * imgw * imgh * sizeof(int));

  dim3 blockSize(32, 32); // Equivalent to dim3 blockSize(TX, TY, 1);
  int bx = (imgw + blockSize.x - 1) / blockSize.x;
  int by = (imgh + blockSize.y - 1) / blockSize.y;
  dim3 gridSize = dim3(bx, by);
  if (d_a == NULL || d_b == NULL) {
    fprintf(stderr, "No GPU mem!\n");
    return EXIT_FAILURE;
  }

  cudaMemcpy(d_a, img, 3 * imgw * imgh * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_filter, filter, 3 * 3 * sizeof(int), cudaMemcpyHostToDevice);

  areaFilter<<<gridSize, blockSize>>>(d_b, d_a, imgw, imgh, d_filter);

  pointFilter<<<gridSize, blockSize>>>(d_c, d_b, imgw, imgh, alpha);

  cudaMemcpy(out, d_c, 3 * imgw * imgh * sizeof(int), cudaMemcpyDeviceToHost);
  t = clock() - t;
  printf("time %f ms\n", t / (double)(CLOCKS_PER_SEC / 1000));

  // printImg(imgw, imgh, out);
  FILE *g = fopen("out.ppm", "w");
  write_ppm(g, out, imgw, imgh, imgc);
  fclose(g);
  return EXIT_SUCCESS;
}

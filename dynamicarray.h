#ifndef DYNAMICARRAY_H
#define DYNAMICARRAY_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

struct Frame {
  double *image;
  bool render;
};

struct DynamicFrameArray {
  struct Frame *frames;
  int size;
  int capacity;
};

void initDynamicFrameArray(struct DynamicFrameArray *arr, int initialCapacity, int sx,
                    int sy) {
  arr->frames = (struct Frame *)malloc(initialCapacity * sizeof(struct Frame));
  if (arr->frames == NULL) {
    fprintf(stderr, "Error: Failed to allocate memory for frame array, change "
                    "some parameters gang!\n");
    exit(EXIT_FAILURE);
  }
  arr->size = 0;
  arr->capacity = initialCapacity;

  for (int i = 0; i < initialCapacity; i++) {
    arr->frames[i].image = (double *)calloc(sx * sy * 3, sizeof(double));
    if (arr->frames[i].image == NULL) {
      fprintf(stderr, "Error: Failed to allocate memory for frame %d image. Perhaps change resolution o_0\n",
              i);
      exit(EXIT_FAILURE);
    }
    arr->frames[i].render = false;
  }
}

void resizeDynamicFrameArray(struct DynamicFrameArray *arr, int sx, int sy) {
  arr->capacity *= 2;
  arr->frames = (struct Frame *)realloc(arr->frames,
                                        arr->capacity * sizeof(struct Frame));
  if (arr->frames == NULL) {
    fprintf(stderr, "Error: Failed to re-allocate memory for frame array, "
                    "change some parameters gang!\n");
    exit(EXIT_FAILURE);
  }
  for (int i = arr->size; i < arr->capacity; i++) {
    arr->frames[i].image = (double *)calloc(sx * sy * 3, sizeof(double));
    if (arr->frames[i].image == NULL) {
      fprintf(stderr, "Error: Failed to allocate memory for frame %d image. Perhaps change resolution o_0\n",
              i);
      exit(EXIT_FAILURE);
    }
    arr->frames[i].render = false;
  }
}

// not directly passing in a frame because the intuition is to set them all to 0
// and then edit them and track if they need to be rendered thus they need not
// get passed in, instead just resized if we reach the end
void addToDynamicFrameArray(struct DynamicFrameArray *arr, int sx, int sy) {
  if (arr->size == arr->capacity) {
    resizeDynamicFrameArray(arr, sx, sy);
  }
  arr->size++;
}

void freeDynamicFrameArray(struct DynamicFrameArray *arr) {
  for (int i = 0; i < arr->capacity; i++) {
    free(arr->frames[i].image);
  }
  free(arr->frames);
}

#endif

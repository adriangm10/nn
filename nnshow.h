#ifndef NNSHOW_H
#define NNSHOW_H

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <assert.h>
#include "utils.h"
#include "mnist.h"

#define RED (SDL_Color) {255, 0, 0, 255}
#define WHITE (SDL_Color) {255, 255, 255, 255}
#define ORANGE (SDL_Color) {255, 128, 0, 255}
#define BLACK (SDL_Color) {0, 0, 0, 255}

typedef struct {
    uint size;
    uint limit;
    float *arr;
}Vector;

typedef struct {
    Vector data;
    SDL_Color fc;  //font color
    SDL_Color pc;  //plot color
    TTF_Font *font;
}Plot;

#define plot_data(p) p.data.arr
#define plot_size(p) p.data.size

extern void append(Vector *v, float dat);
extern Vector new_vec();
extern Plot new_plot();
extern void free_vec(Vector *v);
extern void plot_loss(SDL_Renderer *renderer, SDL_Rect rect, Plot plot);
extern void draw_mnist(SDL_Renderer *renderer, Row img, SDL_Rect r);
extern void fill(SDL_Renderer *renderer, SDL_Color color);
extern int drawable_canvas(SDL_Renderer *renderer, int click_x, int click_y, SDL_Rect r, float pixel_size);

#endif

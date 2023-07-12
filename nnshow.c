#include "nnshow.h"

void append(Vector *v, float dat){
    assert(v->limit > 0 && "the vector has been destroyed");

    v->arr[v->size++] = dat;
    if(v->size >= v->limit){
        v->limit *= 1.5;
        v->arr = (float *) realloc(v->arr, v->limit*sizeof(float));
        assert(v->arr != NULL && "no enough ram");
    }
}

Vector new_vec(){
    Vector v;
    v.size = 0;
    v.arr = (float *) malloc(5 * sizeof(float));
    assert(v.arr != NULL && "not enough memory to malloc");
    v.limit = 5;
    return v;
}

Plot new_plot(){
    return (Plot) {
        .data = new_vec(),
        .font = NULL,
    };
}

void free_vec(Vector *v){
    v->size = 0;
    v->limit = 0;
    free(v->arr);
}

void plot_loss(SDL_Renderer *renderer, SDL_Rect rect, Plot plot){
    assert(plot_data(plot) != NULL);
    char cost_buff[15] = {0};

    snprintf(cost_buff, sizeof(cost_buff), "%.4f", plot_data(plot)[plot_size(plot) - 1]);
    SDL_Surface *text = TTF_RenderText_Solid(plot.font, cost_buff, (SDL_Color) plot.fc);
    SDL_Texture *text_texture = SDL_CreateTextureFromSurface(renderer, text);
    SDL_Rect cost_rect = {
        .x = rect.x + rect.w - text->w,
        .y = rect.y,
        .h = text->h,
        .w = text->w,
    };
    SDL_RenderCopy(renderer, text_texture, NULL, &cost_rect);

    float max = __FLT_MIN__;
    for(uint i = 0; i < plot_size(plot); i++){
        if(plot_data(plot)[i] > max) max = plot_data(plot)[i];
    }

    float n = plot_size(plot) > 1000 ? (float) plot_size(plot) : 1000.0f;

    SDL_SetRenderDrawColor(renderer, plot.pc.r, plot.pc.g, plot.pc.b, SDL_ALPHA_OPAQUE);
    for(uint i = 0; i < plot_size(plot) - 1; i++){
        float y1 = rect.y + rect.h - plot_data(plot)[i] * rect.h / max;
        float x1 = (float) rect.x + (float)(i * rect.w) / n;

        float y2 = rect.y + rect.h - plot_data(plot)[i+1] * rect.h / max;
        float x2 = (float) rect.x + (float)((i+1) * rect.w) / n;
        SDL_RenderDrawLineF(renderer, x1, y1, x2, y2);
    }

    SDL_DestroyTexture(text_texture);
    SDL_FreeSurface(text);
}

void draw_mnist(SDL_Renderer *renderer, Row img, SDL_Rect r){
    assert(r.w % IMG_SIDE == 0 && r.w == r.h);

    SDL_Rect pixel = {
        .w = r.w / IMG_SIDE,
        .h = pixel.w,
    };

    for(int i = 0; i < IMG_SIDE; ++i){
        for(int j = 0; j < IMG_SIDE; ++j){
            unsigned char c = (unsigned char) ROW_AT(img, i * 28 + j);
            SDL_SetRenderDrawColor(renderer, c, c, c, SDL_ALPHA_OPAQUE);
            pixel.x = r.x + pixel.w * j;
            pixel.y = r.y + pixel.h * i;
            SDL_RenderFillRect(renderer, &pixel);
        }
    }
}

void fill(SDL_Renderer *renderer, SDL_Color color){
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);
}


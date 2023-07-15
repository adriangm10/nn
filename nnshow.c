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

void plot_loss(SDL_Renderer *renderer, SDL_Rect rect, Plot p){
    assert(plot_data(p) != NULL);
    char cost_buff[21] = {0};

    if(plot_size(p) == 0) return;

    snprintf(cost_buff, sizeof(cost_buff), "cost: %.4f", plot_data(p)[plot_size(p) - 1]);
    SDL_Surface *text = TTF_RenderText_Solid(p.font, cost_buff, (SDL_Color) p.fc);
    SDL_Texture *text_texture = SDL_CreateTextureFromSurface(renderer, text);
    SDL_Rect cost_rect = {
        .x = rect.x + rect.w - text->w,
        .y = rect.y,
        .h = text->h,
        .w = text->w,
    };
    SDL_RenderCopy(renderer, text_texture, NULL, &cost_rect);

    // float max = __FLT_MIN__;
    // for(uint i = 0; i < plot_size(p); i++){
    //     if(plot_data(p)[i] > max) max = plot_data(p)[i];
    // }
    float max = plot_data(p)[0];

    float n = plot_size(p) > 1000 ? (float) plot_size(p) : 1000.0f;

    SDL_SetRenderDrawColor(renderer, p.pc.r, p.pc.g, p.pc.b, SDL_ALPHA_OPAQUE);
    for(uint i = 0; i < plot_size(p) - 1; i++){
        float y1 = rect.y + rect.h - plot_data(p)[i] * rect.h / max;
        float x1 = (float) rect.x + (float)(i * rect.w) / n;

        float y2 = rect.y + rect.h - plot_data(p)[i+1] * rect.h / max;
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
}

int drawable_canvas(SDL_Renderer *renderer, int click_x, int click_y, SDL_Rect r, float pixel_size){
    if(click_x > r.x + r.w || click_x < r.x) return -1;
    if(click_y > r.y + r.h || click_y < r.y) return -1;

    SDL_FRect pixel = {
        .x = (float) click_x - pixel_size / 2.0f,
        .y = (float) click_y - pixel_size / 2.0f,
        .h = pixel_size,
        .w = pixel_size,
    };

    SDL_SetRenderDrawColor(renderer, WHITE.r, WHITE.g, WHITE.b, WHITE.a);
    SDL_RenderFillRectF(renderer, &pixel);
    return 0;
}

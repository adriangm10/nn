#include "../nn.h"
#include "../nnshow.h"
#include "../mnist.h"

#define error(msg) {fprintf(stderr, "%s", msg); exit(EXIT_FAILURE);}
#define WIDTH 800
#define HEIGHT WIDTH
#define RED {255, 0, 0, SDL_ALPHA_OPAQUE}
#define WHITE {255, 255, 255, SDL_ALPHA_OPAQUE}
#define ORANGE {255, 128, 0, SDL_ALPHA_OPAQUE}

// const int MAX_EPOCHS = __INT_MAX__ / 100;
#define MAX_EPOCHS 1000 * 100

bool check_xor(NN nn){
    bool ok = true;
    Mat t = mat_alloc(4, 3);
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            size_t row = i * 2 + j;
            MAT_AT(t, row, 0) = i;
            MAT_AT(t, row, 1) = j;
            MAT_AT(t, row, 2) = i ^ j;
        }
    }
    for(int i = 0; i < 4; i++){
        nn_forward(nn, row_slice(mat_row(t, i), 0, NN_INPUT_COLS(nn) - 1));
        ok &= roundf(ROW_AT(NN_OUTPUT(nn), 0)) == MAT_AT(t, i, 2);
    }
    free_mat(&t);
    return ok;
}

int main(void){
    if(SDL_Init(SDL_INIT_EVERYTHING) != 0) error(SDL_GetError());

    if(TTF_Init() != 0){
        SDL_Quit();
        error(TTF_GetError());
    }

    TTF_Font *font;
    if(!(font = TTF_OpenFont("font/Roboto.ttf", 24))) {
        SDL_Quit();
        error(TTF_GetError());
    }

    SDL_Window *win = SDL_CreateWindow("xor", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
    SDL_Renderer *renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE); // SDL_RENDERER_ACCELERATED



    size_t xor_arch[] = {2, 2, 1};
    float lr = 1.0f;
    float xor_cost;

    NN xor_nn = nn_alloc(xor_arch, ARRAY_LEN(xor_arch), SIGMOID);
    rand_nn(xor_nn, 0.0f, 1.0f);

    Mat xor = mat_alloc(4, 3);
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            size_t row = i * 2 + j;
            MAT_AT(xor, row, 0) = i;
            MAT_AT(xor, row, 1) = j;
            MAT_AT(xor, row, 2) = i ^ j;
        }
    }

    SDL_Rect plot_rect = {
        .x = WIDTH / 4,
        .y = HEIGHT / 4,
        .w = WIDTH / 2,
        .h = HEIGHT / 2,
    };

    Plot loss = new_plot();
    loss.font = font;
    loss.fc = (SDL_Color) WHITE;
    loss.pc = (SDL_Color) ORANGE;

    bool running = true;
    uint i = 0;
    while (running) {
        SDL_Event event;

        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    running = false;
                    break;
            }
        }

        size_t frame_epoch = 0;
        for(; i < MAX_EPOCHS && frame_epoch < 50; i++, frame_epoch++){
            nn_train(xor_nn, 2, xor, lr, &xor_cost);

            append(&loss.data, xor_cost);
            plot_loss(renderer, plot_rect, loss);

            SDL_RenderPresent(renderer);
        }
        fill(renderer, (SDL_Color) {0, 0, 0, 255});
    }

    free_mat(&xor);
    free_nn(&xor_nn);
    free_vec(&loss.data);
    SDL_DestroyWindow(win);
    SDL_DestroyRenderer(renderer);
    TTF_CloseFont(font);
    return 0;
}

#include "../nn.h"
#include "../mnist.h"
#include "../nnshow.h"

#define error(msg) {fprintf(stderr, "%s", msg); exit(EXIT_FAILURE);}
#define WIDTH 800
#define HEIGHT 600
#define RED (SDL_Color) {255, 0, 0, SDL_ALPHA_OPAQUE}
#define WHITE (SDL_Color) {255, 255, 255, SDL_ALPHA_OPAQUE}
#define ORANGE (SDL_Color) {255, 128, 0, SDL_ALPHA_OPAQUE}
#define MAX_EPOCH 1000000


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

    SDL_Window *win = SDL_CreateWindow("mnist", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
    SDL_Renderer *renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE); // SDL_RENDERER_ACCELERATED
    SDL_RenderPresent(renderer);

    int from = 1;
    int target_to = -1, test_to = -1;
    Mat t = charge_mnist(from, &target_to);
    Mat test = charge_mnist_test(from, &test_to);

    Plot loss = new_plot();
    loss.font = font;
    loss.fc = WHITE;
    loss.pc = ORANGE;

    SDL_Rect num_rect = {
        .x = WIDTH / 2 + 50,
        .y = HEIGHT / 4,
        .w = 112,
        .h = 112,
    };

    SDL_Rect loss_pos = {
        .x = 10,
        .y = HEIGHT / 4,
        .w = WIDTH / 2,
        .h = HEIGHT / 2,
    };

    SDL_Rect out_rect = {
        .x = num_rect.x,
        .y = num_rect.y + num_rect.w + 10,
    };

    size_t arch[] = {IMG_SIZE, 28, 16, 10};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch), SIGMOID);
    rand_nn(nn, -1.0f, 1.0f);

    float cost = 0.0f;
    int img = 0;
    float lr = 0.03f;

    bool pause = false;
    bool running = true;
    while(running){
        SDL_Event event;

        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    pause = true;
                    running = false;
                    break;
                case SDL_KEYDOWN:
                    switch (event.key.keysym.sym) {
                        case SDLK_r:
                            if(!pause) break;
                            img = rand_float(0, test_to - 1);
                            draw_mnist(renderer, mat_row(test, img), num_rect);

                            nn_forward(nn, row_slice(mat_row(test, img), 0, NN_INPUT_COLS(nn) - 1));
                            print_row(NN_OUTPUT(nn), "\n");

                            float max = __FLT_MIN__;
                            char buff[24];
                            uint out = 0;
                            for(uint i = 0; i < NN_OUTPUT_COLS(nn); ++i){
                                if(ROW_AT(NN_OUTPUT(nn), i) > max){
                                    max = ROW_AT(NN_OUTPUT(nn), i);
                                    out = i;
                                }
                            }
                            snprintf(buff, sizeof(buff), "guess: %2d, prob: %.4f", out, max);
                            SDL_Surface *text = TTF_RenderText_Solid(font, buff, WHITE);
                            SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, text);
                            out_rect.w = text->w; out_rect.h = text->h;
                            SDL_RenderCopy(renderer, texture, NULL, &out_rect);

                            plot_loss(renderer, loss_pos, loss);
                            SDL_RenderPresent(renderer);
                            SDL_DestroyTexture(texture);
                            SDL_FreeSurface(text);
                            fill(renderer, (SDL_Color) {0, 0, 0, 255});
                            break;
                        case SDLK_SPACE:
                            pause = !pause;
                            break;
                    }
            }
        }


        if(!pause){
            nn_train(nn, 100, t, lr, &cost);
            append(&loss.data, cost);
            plot_loss(renderer, loss_pos, loss);
            SDL_RenderPresent(renderer);
            fill(renderer, (SDL_Color) {0, 0, 0, 255});
        }

    }

    SDL_DestroyWindow(win);
    SDL_DestroyRenderer(renderer);
    TTF_CloseFont(font);
    free_nn(&nn);
    free_mat(&t);
    free_mat(&test);
    free_vec(&loss.data);
    TTF_Quit();
    SDL_Quit();
    exit(0);
}
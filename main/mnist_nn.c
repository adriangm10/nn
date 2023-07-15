#include "../nn.h"
#include "../mnist.h"
#include "../nnshow.h"
#include <time.h>

#define error(msg) {fprintf(stderr, "%s", msg); exit(EXIT_FAILURE);}
#define WIDTH 800
#define HEIGHT 600
#define MAX_EPOCH 1000000

typedef struct {
    uint out;
    float prob;
}res;

res mnist_output(NN nn, Row input){
    res out;
    nn_forward(nn, input);
    out.prob = __FLT_MIN__;
    out.out = 0;
    for(uint i = 0; i < NN_OUTPUT_COLS(nn); ++i){
        if(ROW_AT(NN_OUTPUT(nn), i) > out.prob){
            out.prob = ROW_AT(NN_OUTPUT(nn), i);
            out.out = i;
        }
    }
    return out;
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

    SDL_Window *win = SDL_CreateWindow("mnist", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
    SDL_Renderer *renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE); // SDL_RENDERER_ACCELERATED
    SDL_RenderPresent(renderer);
    srand(time(NULL));

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

    SDL_Rect canvas = {
        .x = WIDTH / 2 + 50,
        .y = HEIGHT * 3 / 4,
        .w = 112,
        .h = 112,
    };
    Row canvas_img = row_alloc(IMG_SIZE);
    row_fill(canvas_img, 0.0f);

    // SDL_Rect loss_pos = {
    //     .x = 10,
    //     .y = HEIGHT / 4,
    //     .w = WIDTH / 2,
    //     .h = HEIGHT / 2,
    // };

    SDL_Rect out_rect = {
        .x = num_rect.x,
        .y = num_rect.y + num_rect.w + 10,
    };

    // size_t arch[] = {IMG_SIZE, 28, 16, 10};
    // NN nn = nn_alloc(arch, ARRAY_LEN(arch), SIGMOID);
    // rand_nn(nn, -1.0f, 1.0f);
    NN nn = load_nn("AI/mnist_nn");

    res out;
    char buff[24];
    SDL_Surface *text;
    SDL_Texture *texture;

    int img = 0;
    // float cost = 0.0f;
    // float lr = 0.03f;

    bool pause = true;
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

                            out = mnist_output(nn, row_slice(mat_row(test, img), 0, NN_INPUT_COLS(nn) - 1));
                            // print_row(NN_OUTPUT(nn), "\n");

                            snprintf(buff, sizeof(buff), "guess: %2d, prob: %.4f", out.out, out.prob);
                            text = TTF_RenderText_Solid(font, buff, WHITE);
                            texture = SDL_CreateTextureFromSurface(renderer, text);
                            out_rect.w = text->w; out_rect.h = text->h;
                            SDL_RenderCopy(renderer, texture, NULL, &out_rect);

                            // plot_loss(renderer, loss_pos, loss);
                            SDL_RenderPresent(renderer);
                            SDL_DestroyTexture(texture);
                            SDL_FreeSurface(text);
                            //fill(renderer, BLACK);
                            break;
                        case SDLK_SPACE:
                            pause = !pause;
                            break;
                        case SDLK_s:
                            pause = true;
                            save_nn(nn, "AI/mnist_nn");
                            break;
                        case SDLK_t:
                            printf("train: %f, test: %f\n", 1 - nn_loss(nn, t), 1 - nn_loss(nn, test));
                            break;
                        case SDLK_c:
                            row_fill(canvas_img, 0.0f);
                            SDL_SetRenderDrawColor(renderer, BLACK.r, BLACK.g, BLACK.b, BLACK.a);
                            SDL_RenderFillRect(renderer, &canvas);
                            break;
                        case SDLK_RETURN:
                            out = mnist_output(nn, canvas_img);
                            snprintf(buff, sizeof(buff), "guess: %2d, prob: %.4f", out.out, out.prob);
                            text = TTF_RenderText_Solid(font, buff, WHITE);
                            texture = SDL_CreateTextureFromSurface(renderer, text);
                            SDL_Rect res_rect = {.x = canvas.x, .y = canvas.y + canvas.h, .h = text->h, .w = text->w};

                            SDL_SetRenderDrawColor(renderer, BLACK.r, BLACK.g, BLACK.b, BLACK.a);
                            SDL_RenderFillRect(renderer, &res_rect);

                            SDL_RenderCopy(renderer, texture, NULL, &res_rect);
                            SDL_RenderPresent(renderer);
                            SDL_DestroyTexture(texture);
                            SDL_FreeSurface(text);
                            break;
                    }
                    break;
            }
        }

        // if(event.button.button != SDL_BUTTON_LEFT) break;
        // int x = event.button.x;
        // int y = event.button.y;
        int x, y;
        Uint32 button = SDL_GetMouseState(&x, &y);
        if(SDL_BUTTON(button) == SDL_BUTTON_LEFT){
            if(drawable_canvas(renderer, x, y, canvas, ((float) 112 / 28)) == 0){
                size_t index = ((y - canvas.y) / 4) * IMG_SIDE + ((x - canvas.x) / 4);
                if(index < canvas_img.cols)
                    ROW_AT(canvas_img, index) = 255.0f;
            }
        }

        // if(!pause){
        //     nn_train(nn, 100, t, lr, &cost);
        //     append(&loss.data, cost);
        //     plot_loss(renderer, loss_pos, loss);
        //     SDL_RenderPresent(renderer);
        //     fill(renderer, (SDL_Color) {0, 0, 0, 255});
        // }

        SDL_SetRenderDrawColor(renderer, WHITE.r, WHITE.g, WHITE.b, WHITE.a);
        SDL_RenderDrawRect(renderer, &canvas);
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyWindow(win);
    SDL_DestroyRenderer(renderer);
    TTF_CloseFont(font);
    free_nn(&nn);
    free(nn.arch);
    free_mat(&t);
    free_mat(&test);
    free_vec(&loss.data);
    free_row(&canvas_img);
    TTF_Quit();
    SDL_Quit();
    exit(0);
}

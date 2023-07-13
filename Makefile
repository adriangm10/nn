.PHONY = all clean

CC = gcc

FLAGS = -O2 -Wall -Wextra -ggdb `sdl2-config --libs --cflags` -lSDL2_ttf -lm -lpthread

HDRS = nn.h nnshow.h mnist.h utils.h

SRCS = nn.c nnshow.c utils.c mnist.c

OBJS = ${SRCS:.c=.o}

EXECMNIST = mnist
EXECXOR = xor

all: xor mnist

${EXECXOR}: ${OBJS} ${HDRS} Makefile
${EXECMNIST}: ${OBJS} ${HDRS} Makefile

xor: ai/xor.c
		${CC} -o $@ ${OBJS} ai/xor.c ${MAINXOR} ${FLAGS}

mnist: ai/mnist_ai.c
		${CC} -o $@ ${OBJS} ai/mnist_ai.c ${MAINMNIST} ${FLAGS}


# ${OBJS}: ${@:.o=.c} ${HDRS} Makefile
#  	${CC} -o $@ ${:.o=-c} -c ${FLAGS}

clean:
	rm -f ${OBJS} ${EXECMNIST} ${EXECXOR}



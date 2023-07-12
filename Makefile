.PHONY = all clean

CC = gcc

FLAGS = -O2 -Wall -Wextra -ggdb `sdl2-config --libs --cflags` -lSDL2_ttf -lm

HDRS = nn.h nnshow.h mnist.h utils.h

SRCS = nn.c ai/xor.c nnshow.c utils.c mnist.c

OBJS = ${SRCS:.c=.o}

EXEC = xor

${EXEC}: ${OBJS} ${HDRS} Makefile
	${CC} -o $@ ${OBJS} ${FLAGS}

# ${OBJS}: ${@:.o=.c} ${HDRS} Makefile
#  	${CC} -o $@ ${:.o=-c} -c ${FLAGS}

clean:
	rm -f ${EXEC} ${OBJS}

all: ${EXEC}


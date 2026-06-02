CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2 -Iinclude
LDFLAGS = -lm

SRCS = src/moments.c src/r_transform.c src/s_transform.c src/gradient_analysis.c
OBJS = $(SRCS:.c=.o)

.PHONY: all test clean

all: libfreeprob.a

libfreeprob.a: $(OBJS)
	ar rcs $@ $^

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

test: tests/test_free_prob.c libfreeprob.a
	$(CC) $(CFLAGS) -o tests/test_free_prob tests/test_free_prob.c -L. -lfreeprob $(LDFLAGS)
	./tests/test_free_prob

clean:
	rm -f src/*.o libfreeprob.a tests/test_free_prob

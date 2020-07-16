CC = gcc
#For older gcc, use -O3 or -O2 instead of -Ofast
CFLAGS = -lm -pthread -Ofast -march=native -funroll-loops -Wno-unused-result
BUILDDIR := build
SRCDIR := word-training

all: dir glove shuffle cooccur_premb vocab_premb

dir :
	mkdir -p $(BUILDDIR)
glove : $(SRCDIR)/glove.c
	$(CC) $(SRCDIR)/glove.c -o $(BUILDDIR)/glove $(CFLAGS)
shuffle : $(SRCDIR)/shuffle.c
	$(CC) $(SRCDIR)/shuffle.c -o $(BUILDDIR)/shuffle $(CFLAGS)
cooccur_premb : $(SRCDIR)/cooccur_premb.c
	$(CC) $(SRCDIR)/cooccur_premb.c -o $(BUILDDIR)/cooccur_premb $(CFLAGS)
vocab_premb : $(SRCDIR)/vocab_premb.c
	$(CC) $(SRCDIR)/vocab_premb.c -o $(BUILDDIR)/vocab_premb $(CFLAGS)

clean:
	rm -rf glove shuffle cooccur_emb vocab_premb build

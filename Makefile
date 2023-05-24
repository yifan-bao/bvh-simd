CC = gcc
CFLAGS = -Wall -Wextra -g -O3 -mavx512vl
LIB = -lm
SRCDIR = src
OBJDIR = obj
BINDIR = bin
ASMDIR = asm

SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))
ASMS = $(patsubst $(SRCDIR)/%.c, $(ASMDIR)/%.s, $(SRCS))
COUNT_TARGET = $(BINDIR)/quick_count
NON_COUNT_TARGET = $(BINDIR)/quick

.PHONY: all clean count non-count asm-files

all: non-count

count: CFLAGS += -DCOUNTFLOPS
count: $(COUNT_TARGET)
	@echo "Building with count..."	
# @./$(COUNT_TARGET)

non-count: $(NON_COUNT_TARGET)
	@echo "Building without count..."
# @./$(NON_COUNT_TARGET)

asm-files: $(ASMS)
	@echo "Building assembly code..."

$(COUNT_TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LIB)

$(NON_COUNT_TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LIB)

$(ASMDIR)/%.s: $(SRCDIR)/%.c $(SRCDIR)/bvh.h $(SRCDIR)/common.h $(SRCDIR)/utils.h | $(ASMDIR)
	$(CC) $(CFLAGS) -S -o $@ $< $(LIB)

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(SRCDIR)/bvh.h $(SRCDIR)/common.h $(SRCDIR)/utils.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c -o $@ $< $(LIB)

$(OBJDIR):
	@mkdir -p $(OBJDIR)

$(ASMDIR):
	@mkdir -p $(ASMDIR)

clean:
	rm -rf $(OBJDIR) $(ASMDIR)

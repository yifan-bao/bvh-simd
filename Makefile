CC = gcc
CFLAGS = -Wall -Wextra -g -O3
SRCDIR = src
OBJDIR = obj
BINDIR = bin

SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))
COUNT_TARGET = $(BINDIR)/quick_count
NON_COUNT_TARGET = $(BINDIR)/quick

.PHONY: all clean count non-count

all: non-count

count: CFLAGS += -DCOUNTFLOPS
count: $(COUNT_TARGET)
	@echo "Building with count..."	
# @./$(COUNT_TARGET)

non-count: $(NON_COUNT_TARGET)
	@echo "Building without count..."
# @./$(NON_COUNT_TARGET)

$(COUNT_TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^

$(NON_COUNT_TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(SRCDIR)/bvh.h $(SRCDIR)/common.h $(SRCDIR)/utils.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR):
	@mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(BINDIR)

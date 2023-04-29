CXX=g++
CXXFLAGS=-Itemplate -I./ -I./lib
SRCS=basics.cpp
OBJS=$(SRCS:.cpp=.o)
EXEC=output

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)


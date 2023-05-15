CXX=g++
CXXFLAGS=-Itemplate -I./ -I./lib -g -O3
SRCS=quickbuild.cpp
OBJS=$(SRCS:.cpp=.o)
EXEC=quickbuild.out

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -v $(OBJS) -o $(EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ -DCOUNTFLOPS -std=c++17

clean:
	rm -f $(OBJS) $(EXEC)


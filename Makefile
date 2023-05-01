CXX=g++
CXXFLAGS=-Itemplate -I./ -I./lib
SRCS=quickbuild.cpp
OBJS=$(SRCS:.cpp=.o)
EXEC=quickbuild.out

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ -fdeclspec -DCOUNTFLOPS -std=c++17

clean:
	rm -f $(OBJS) $(EXEC)


rm=/bin/rm -f
CC= g++ -Wall
PROGNAME= atomictest
INCLUDES= -I./../../lib
LIBS= ./../../lib/argvparser/argvparser.o


CFLAGS= -g

SRCS = test.cpp 

OBJS = test.o


all: $(PROGNAME)


$(PROGNAME): $(OBJS)
	$(CC) -o $(PROGNAME) $(OBJS) $(LIBS)
	$(rm) $(OBJS)


test.o: test.cpp
	$(CC) $(CFLAGS) -c test.cpp $(INCLUDES)



clean:
	$(rm) $(OBJS) $(PROGNAME) core *~

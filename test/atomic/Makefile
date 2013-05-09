rm=/bin/rm -f
NVCC=nvcc
PROGNAME= test
INCLUDES= -I./../../lib
LIBS= ./../../lib/argvparser/argvparser.o
SM=-arch sm_20

CFLAGS= -g

SRCS = test.cu

OBJS = test.o


all: $(PROGNAME)


$(PROGNAME): $(OBJS)
	$(NVCC) -o $(PROGNAME) $(OBJS) $(LIBS)
	$(rm) $(OBJS)


test.o: test.cu
	$(NVCC) $(CFLAGS) -c test.cu $(INCLUDES) $(SM)



clean:
	$(rm) $(OBJS) $(PROGNAME) core *~

rm=/bin/rm -f
NVCC=nvcc
PROGNAME= degree_histogram
INCLUDES= -I./../../lib -I./../..
LIBS= ./../../lib/argvparser/argvparser.o
SM=-arch sm_20

CFLAGS= -g

SRCS = $(PROGNAME).cu

OBJS = $(PROGNAME).o


all: $(PROGNAME)


$(PROGNAME): $(OBJS)
	$(NVCC) -o $(PROGNAME) $(OBJS) $(LIBS)
	$(rm) $(OBJS)


$(OBJS): $(SRCS)
	$(NVCC) $(CFLAGS) -c $(SRCS) $(INCLUDES) $(SM)



clean:
	$(rm) $(OBJS) $(PROGNAME) core *~

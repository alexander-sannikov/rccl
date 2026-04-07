.DEFAULT_GOAL: build

SRC_FILES := $(wildcard *.c)

build: libnccl-net-profiler.so

libnccl-net-profiler.so: $(SRC_FILES)
	@printf "Compiling  %-35s > %s\n" $< $@
	$(CC) -I../example -D_GNU_SOURCE -fPIC -shared -o $@ $^ -ldl

clean:
	rm -f libnccl-net-profiler.so

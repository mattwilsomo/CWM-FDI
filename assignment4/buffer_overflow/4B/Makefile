

.PHONY: all run clean

all:
	gcc -fno-stack-protector -z execstack -no-pie src/vuln.c -o src/vuln

run: all
	./src/vuln

clean:
	rm -f src/vuln

# For debugging with gdb
debug: all
	gcc -g -fno-stack-protector -z execstack -no-pie src/vuln.c -o src/vuln_debug

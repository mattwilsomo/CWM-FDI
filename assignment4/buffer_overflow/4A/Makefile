

.PHONY: all clean run

all:
	gcc -fno-stack-protector -z execstack -no-pie src/main.c src/util.c -o main

run: all
	./main 40

clean:
	rm -f main

debug: all
	gcc -g -fno-stack-protector -z execstack -no-pie src/main.c src/util.c -o main_debug


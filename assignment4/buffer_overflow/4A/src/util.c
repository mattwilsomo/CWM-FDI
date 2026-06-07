
#include "stdint.h"
#include "stdio.h"

void print_with_null(char* str)
{
    char* c = str;
    while (*c != '\0') {
        printf("%c", *c);
        c++;
    }
    printf("%c", *c);
    return;
}

void hexdump_arr(char* str)
{
    char* c = str;
    int i = 0;
    while (*c != '\0') {
        printf("%d: %x\n", i, (uint8_t)*c);
        i++;
        c++;
    }
    printf("%0d: \\%x\n", i, (uint8_t)*c);
    return;
}
#include <stdio.h>
#include <stdlib.h>

struct strint{
    int num;
    char str[30];
};

typedef struct strint strint;

void swap(strint*, strint*);
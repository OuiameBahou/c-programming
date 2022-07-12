#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct charcell{
    char* X;
    struct charcell* following;
};

typedef struct charcell charcell;
typedef struct charcell* charlist;

void* al_memchr(char*, char, unsigned);
void* al_memset(void*, int, unsigned);
void* al_memcpy(void*, const void*, unsigned);
void* al_memmove(void*, const void*, unsigned);
char* al_strdup(const char*);
void sort(int[], unsigned);
char** strsplit(char*, char);
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct person{
    char name[15];
    char phoneNo[11];
};

struct cell{
    struct person X;
    struct cell* following;
    struct cell* previous;
};

struct list{
    struct cell* head;
    struct cell* tail;
    unsigned Num;
};

typedef struct person person;
typedef struct cell cell;
typedef struct list list;

person createpers();
void addpers(person, list*);
void showlist(list);
void destroy(list*);
#include "EX2.h"

void swap(strint* a, strint* b){
    strint c;
    c.num = a->num;
    fgets(c.str, sizeof(a->str), a->str);
    a->num = b->num;
    fgets(a->str, sizeof(b->str), b->str);
    b->num = c.num;
    fgets(b->str, sizeof(c.str), c.str);
}
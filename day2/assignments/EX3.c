#include<stdio.h>
#include<stdlib.h>

int main(){
    int   i =  8;
    int   j =  5;
    float x =  0.005f;
    float y = -0.01f;
    char  c = 'c';
    char  d = 'd';
    printf("%d\n", (3*i - 2*j)%(2*d - c));
    printf("%d\n", 2*((i/5) + (4*(j-3))%(i + j - 2)));
    printf("%d\n", i <= j);
    printf("%d\n", j != 6);
    printf("%d\n", c == 99);
    printf("%d\n", 5*(i + j) > 'c');
    printf("%d\n", (i > 0) && (j < 5));
    printf("%d\n", (i > 0) || (j < 5));
    printf("%d\n", (x > y) && (i > 0) || (j < 5));
    printf("%d\n", (x > y) && (i > 0) && (j < 5));
    return 0;
}
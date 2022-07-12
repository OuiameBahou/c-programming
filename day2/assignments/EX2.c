#include<stdio.h>
#include<stdlib.h>

int main(){
    int a;
    int b;
    scanf("%d", &a);
    scanf("%d", &b);
    printf("%d %d\n", a, b);
    int c = a + b;
    a = c - a;
    b = c - a;
    printf("%d %d\n", a, b);
    return 0;    
}
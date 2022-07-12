#include <stdio.h>

int main()
{
  int a;
  int b;
  int c;

  a = 16;
  b = 2;
  c = 10;

  printf("%d\n", c += a > 0 && a <= 15 ? ++a : a/b);
  /*
   * What about the following expression? :
   * ----------------------------------
   */
  printf("%d\n", a > 30 ? b = 11 : c == 100);
  return 0;
}
#include <stdio.h>
#include <stdlib.h>

struct complex_num{
	float re;
	float im;
};

typedef struct complex_num complex_num;

complex_num sum_cmplx(complex_num, complex_num);
complex_num mul_cmplx(complex_num, complex_num);
complex_num sub_cmplx(complex_num, complex_num);
complex_num div_cmplx(complex_num, complex_num);
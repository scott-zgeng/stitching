// mv_base.cpp write by scott.zgeng 2015.9.19
// ������ֲ opensiftʹ����opencv�Ĳ���

#include "mv_base.h"

#include <cxcore.h>
#include <cv.h>
#include <highgui.h>




void* mv_malloc(size_t size)
{
    return malloc(size);
}

void mv_free(void* ptr)
{
    free(ptr);
}

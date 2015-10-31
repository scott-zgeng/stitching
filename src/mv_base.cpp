// mv_base.cpp write by scott.zgeng 2015.9.19
// 用于移植 opensift使用了opencv的部分

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

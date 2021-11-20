#pragma once

#include <stdafx.h>
#include <utils.h>

namespace gpu
{
    const int LOCAL_HIST_SIZE = 256;

    using HistType = int;

    __global__ void hist_kernel(const utils::Byte *input_image, HistType *hist_by_row, size_t image_width, size_t image_height);
}

#pragma once

#include <stdafx.h>
#include <utils.h>
#include <hist_kernel.h>

namespace hist
{
    void hist(const utils::Byte *input_image, std::vector<gpu::HistType> &hist, size_t width, size_t height);
}

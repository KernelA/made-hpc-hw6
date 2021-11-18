# HW 6 CUDA

1. Размытие картинки и "Median filter"
Возьмите произвольную картинку (цветную или черно-белую).

а) Фильтр по шаблону (25 баллов). В каждом пикселе картинки надо усреднить по шаблону с коэффициентами в окрестности данного пикселя. Например, для шаблона 5x5:

[Реализация на CUDA](hw6/src/conv_kernel.cu)

[Результаты](image/conv.md)

## Requirements

1. CMake 3.20 or higher.
2. Compiler with support C++17.

## How to run

Create new directory `build`.

```
mkdir build
cmake -S . -B .\build
```

Run CMake:
```
cmake --build .\build --config Release -j 4 --target <input_target>
```

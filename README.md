# HW 6 CUDA

1. Размытие картинки и "Median filter"
Возьмите произвольную картинку (цветную или черно-белую).

а) Фильтр по шаблону. В каждом пикселе картинки надо усреднить по шаблону с коэффициентами в окрестности данного пикселя. Например, для шаблона 5x5:

[Реализация на CUDA](hw6/src/conv_kernel.cu)

[Результаты](image/conv.md)

б) Median filter

Имплементируйте https://en.wikipedia.org/wiki/Median_filter на CUDA

[Реализация на CUDA](hw6/src/median_filter_kernel.cu)

[Результаты](image/median.md)


## Требования для запуска

1. CMake 3.20 or higher.
2. Compiler with support C++17.
3. CUDA 10.1 или выше.

```
git submodule update --init --recursive
```

## Как запустить

```
cmake --list-presets
```

Выбрать preset из списка.

```
cmake --preset <preset_name>
```

Run CMake:
```
cmake --build ./build --config Release -j 4 --target all
```

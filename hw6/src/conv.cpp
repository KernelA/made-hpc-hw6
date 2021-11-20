#include <stdafx.h>
#include <conv.h>

int main(int argc, char **argv)
{
    using std::cerr;
    using std::cin;
    using std::cout;
    using std::endl;
    using std::vector;

    if (argc < 3)
    {
        cerr << "Missing path to image. Please specify path to image as positional argument and values of kernel by row\n";
        return 1;
    }

    std::vector<float> kernel_values;

    float normalization{};

    for (size_t i{2}; i < argc; i++)
    {
        float value = std::stof(argv[i]);
        normalization += value;
        kernel_values.push_back(value);
    }

    size_t kernel_size{1};

    while (kernel_size * kernel_size != kernel_values.size() && kernel_size * kernel_size < kernel_values.size())
    {
        ++kernel_size;
    }

    if (kernel_size * kernel_size != kernel_values.size())
    {
        cerr << "Incorrect size of kernel. Must be square matrix but size is " << kernel_size << endl;
        return 1;
    }

    if (kernel_size % 2 != 1)
    {
        cerr << "Kernel has even size " << kernel_values.size() << " but it must be an odd" << endl;
        return 1;
    }

    auto path_to_image = std::filesystem::path(argv[1]);

    auto image = cv::imread(path_to_image.string(), cv::IMREAD_COLOR);

    if (image.empty())
    {
        cerr << "Cannot read the image " << path_to_image << endl;
        return 1;
    }

    int image_width = image.cols;
    int image_height = image.rows;
    int num_channels = image.channels();

    image.convertTo(image, CV_32F, 1.0 / 255);

    std::vector<float> output_image(image_width * image_height * num_channels, 0.0f);

    conv::conv2d(image.ptr<float>(0), output_image, image_width, image_height, num_channels, kernel_values, normalization, kernel_size);

    std::filesystem::path new_name = path_to_image.stem();

    new_name += u8"_out";
    new_name += path_to_image.extension();

    std::filesystem::path out_path = path_to_image.parent_path() / new_name;

    cout << "Save result to " << out_path << endl;

    cv::Mat float_mat(image_height, image_width, CV_32FC3, output_image.data());

    float_mat.convertTo(float_mat, CV_8U, 255);

    if (!cv::imwrite(out_path.string(), float_mat))
    {
        cerr << "Cannot save image" << endl;
    }

    return 0;
}

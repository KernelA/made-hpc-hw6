#include <stdafx.h>
#include <median_filter.h>
#include <utils.h>



int main(int argc, char ** argv)
{
    using std::cin;
    using std::cout;
    using std::endl;
    using std::cerr;
    using std::vector;

    if (argc != 2)
    {
        cerr << "Missing path to image. Please specify path to image as positional argument\n";
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

    std::vector<utils::Byte> output_image(image_width * image_height * num_channels, static_cast<utils::Byte>(0));

    const size_t WINDOW_SIZE{ 5 };

    median::median_filter(image.data, output_image, image_width, image_height, num_channels, WINDOW_SIZE);

    std::filesystem::path new_name = path_to_image.stem();

    new_name += u8"_out";
    new_name += path_to_image.extension();

    std::filesystem::path out_path = path_to_image.parent_path() / new_name;

    cout << "Save result to " << out_path << endl;

    cv::Mat output_mat_image(image_height, image_width, CV_8UC3, output_image.data());

    if (!cv::imwrite(out_path.string(), output_mat_image))
    {
        cerr << "Cannot save image" << endl;
    }


    return 0;
}

#include <stdafx.h>
#include <utils.h>
#include <hist.h>
#include <hist_kernel.h>

int main(int argc, char **argv)
{
    using std::cerr;
    using std::cin;
    using std::cout;
    using std::endl;
    using std::vector;

    if (argc != 2)
    {
        cerr << "Missing path to image. Please specify path to image as positional argument\n";
        return 1;
    }

    auto path_to_image = std::filesystem::path(argv[1]);

    auto image = cv::imread(path_to_image.string(), cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        cerr << "Cannot read the image " << path_to_image << endl;
        return 1;
    }

    int image_width = image.cols;
    int image_height = image.rows;

    std::vector<gpu::HistType> hist(gpu::LOCAL_HIST_SIZE, 0);

    hist::hist(image.data, hist, image_width, image_height);

    std::filesystem::path new_name = path_to_image.stem();

    new_name += u8"_out_hist.txt";

    std::filesystem::path out_path = path_to_image.parent_path() / new_name;

    cout << "Save result to " << out_path << endl;

    std::ofstream file(out_path);

    for (size_t i{}; i < hist.size(); ++i)
    {
        file << i << ' ' << hist[i] << '\n';
    }

    file.close();

    return 0;
}

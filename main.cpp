#include <iostream>
#include <chrono>
#include "ppm.h"
#include <omp.h>
#include "kernels.h"

static const int thread_num = 10;

int main(int argc, char *argv[]) {

    Image_t *inputImg = importPPM(argv[1]);
    char *outputPath = argv[2];
    int kernelSize = (*argv[3] != '3' && *argv[3] != '5' && *argv[3] != '7') ? 3 : (int) *argv[3] - '0';
    float *kernel = (*argv[3] == '5') ? kernel5 : (*argv[3] == '7') ? kernel7 : kernel3;

    omp_set_num_threads(thread_num);
    int tid;

    const int imageWidth = getWidth(inputImg);
    const int imageHeight = getHeight(inputImg);
    const int imageChannels = getChannels(inputImg);
    const float *imageData = getData(inputImg);


    Image_t *output = newImage(imageWidth, imageHeight, imageChannels);
    float *data = getData(output);

    auto start = std::chrono::system_clock::now();

#pragma omp parallel default(none) shared( kernel, kernelSize, imageData, data)private(tid)
    {
        tid = omp_get_thread_num();
        int ik, jk, dataIndex, kernelIndex;
        float sum = 0, currentPixel;
        int widthOffset = (imageWidth / thread_num);
        int sliceWidth = (tid == thread_num - 1) ? (imageWidth / thread_num) + (imageWidth % thread_num) : (imageWidth /
                                                                                                            thread_num);
        for (int i = 0; i < imageHeight; i++) {
            for (int j = 0; j < sliceWidth; j++) {
                for (int c = 0; c < imageChannels; c++) {

                    for (int ii = 0; ii < kernelSize; ii++) {
                        ik = ((i - kernelSize / 2 + ii) < 0) ? 0 : ((i - kernelSize / 2 + ii) > imageHeight - 1) ?
                                                                   imageHeight - 1 : i - kernelSize / 2 + ii;

                        for (int jj = 0; jj < kernelSize; jj++) {
                            jk = ((j - kernelSize / 2 + jj + widthOffset * tid) < 0) ? 0 : ((j - kernelSize / 2 + jj +
                                                                                             widthOffset * tid) >
                                                                                            imageWidth - 1) ?
                                                                                           imageWidth - 1 -
                                                                                           widthOffset * tid : j -
                                                                                                               kernelSize /
                                                                                                               2 + jj;
                            dataIndex = (ik * imageWidth + jk + widthOffset * tid) * imageChannels + c;
                            currentPixel = imageData[dataIndex];
                            kernelIndex = (kernelSize - 1 - ii) * kernelSize + (kernelSize - 1 - jj);
                            sum += (currentPixel * kernel[kernelIndex]);
                        }
                    }
                    data[(i * imageWidth + j + widthOffset * tid) * imageChannels + c] = sum;
                    sum = 0;
                }
            }
        }

    };

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << elapsed.count();

    exportPPM(((std::string) outputPath + "/output_openmp.ppm").c_str(), output);

    return 0;
}
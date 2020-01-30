#include "image.h"
#include <cassert>

Image_t *newImage(int width, int height, int channels, float *data) {
    Image_t *img;

    img = (Image_t *) malloc(sizeof(Image_t));

    setWidth(img, width);
    setHeight(img, height);
    setChannels(img, channels);
    setPitch(img, width * channels);

    setData(img, data);
    return img;
}

Image_t *newImage(int width, int height, int channels) {
    auto *data = (float *) malloc(sizeof(float) * width * height * channels);
    return newImage(width, height, channels, data);
}


float getPixel(Image_t *img, int x, int y, int c) {
    float *data = getData(img);
    int channels = getChannels(img);
    int pitch = getPitch(img);

    return data[y * pitch + x * channels + c];
}



#ifndef IMAGE_H_
#define IMAGE_H_

typedef struct {
    int width;
    int height;
    int channels;
    int pitch;
    float *data;
} Image_t;

#define getWidth(img) ((img)->width)
#define getHeight(img) ((img)->height)
#define getChannels(img) ((img)->channels)
#define getPitch(img) ((img)->pitch)
#define getData(img) ((img)->data)

#define setWidth(img, val) (getWidth(img) = val)
#define setHeight(img, val) (getHeight(img) = val)
#define setChannels(img, val) (getChannels(img) = val)
#define setPitch(img, val) (getPitch(img) = val)
#define setData(img, val) (getData(img) = val)

Image_t *newImage(int width, int height, int channels, float *data);

Image_t *newImage(int width, int height, int channels);

float getPixel(Image_t *img, int x, int y, int c);


#endif /* IMAGE_H_ */

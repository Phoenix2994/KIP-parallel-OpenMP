
#ifndef PPM_H_
#define PPM_H_

#include "image.h"

Image_t *importPPM(const char *filename);

bool exportPPM(const char *filename, Image_t *img);

char *skipSpaces(const char *line);

char *nextNotSpaceChar(const char *line);

bool isComment(const char *line);

void parseDimensions(const char *line0, int *width, int *height);

void parseDimensions(const char *line0, int *width, int *height,
                     int *channels);

void parseDepth(const char *line0, int *depth);

char *readLine(FILE *file);

char *nextLine(FILE *file);

char *readFile(FILE *file, size_t size, size_t count);

bool writeFile(FILE *file, const void *buffer, size_t size, size_t count);



#endif /* PPM_H_ */

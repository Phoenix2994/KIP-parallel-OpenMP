
#ifndef UTILS_H_
#define UTILS_H_

static inline float _min(float x, float y) {
    return x < y ? x : y;
}

static inline float _max(float x, float y) {
    return x > y ? x : y;
}

static inline float clamp(float x, float start, float end) {
    return _min(_max(x, start), end);
}

#endif /* UTILS_H_ */

#include <math.h>
#include <inttypes.h>
#include "gouraud_triangles.h"

// Code adapted from the very useful tutorial http://www.hugi.scene.org/online/coding/hugi%2017%20-%20cotriang.htm

typedef struct {
    float leftX;
    float rightX;
    float left_dXdY;
    float right_dXdY;
    float *leftC;
    float *dCdX;
    float *left_dCdY;
    unsigned int *shape;
    long *strides;
    unsigned char accumulate;
} DrawingState;

void init_leftXandC(float *vertex, float dXdY, float *color, float *dCdY, DrawingState *state);
void init_leftX(float *vertex, float dXdY, DrawingState *state);
void init_rightX(float *vertex, float dXdY, DrawingState *state);
void swap(float **a, float **b);
void draw_gouraud_segment(long y0, long y1, float* out, DrawingState *state);
void draw_mask_segment(long y0, long y1, char* out, DrawingState *state);


void gouraud_triangle_strip(unsigned int n_vertices, float *vertices, float *colors, float *out, unsigned int *shape, long *strides, unsigned char accumulate) {
    for (unsigned int i = 0; i < n_vertices - 2; i++) {
        gouraud_triangle(vertices, colors, out, shape, strides, accumulate);
        vertices += 2;
        colors += shape[2];
    }
}

void mask_triangle_strip(unsigned int n_vertices, float *vertices, char *out, unsigned int *shape, long *strides) {
    for (unsigned int i = 0; i < n_vertices - 2; i++) {
        mask_triangle(vertices, out, shape, strides);
        vertices += 2;
    }
}

void gouraud_triangle(float *vertices, float *colors, float *out, unsigned int *shape, long *strides, unsigned char accumulate) {
    // vertices: v0x, v0y, v1x, v1y, v2x, v2y
    // colors: v0c0, ..., v0cn, v1c0, ..., v1cn, v2c0, ..., v2cn
    // out: strided array of shape (x, y, colors)

    float verts[6];
    for (int i = 0; i < 6; i++) {
        // all math below is easier if the "pixel centers" are at the lower-left corner of
        // the pixel rather than at (0.5, 0.5) -- then we can use ceil rather than round
        // and not need any conditioning on whether a line is above or below the pixel
        // center within a pixel. All we need to do to correct the drawing for this
        // is to internally subtrace 0.5 from the vertices.
        // If we didn't do that, we could just replace "ceil" everywhere with "round"
        // and then fix SUB_PIX to return round(a)-a if that quantity is positive,
        // (i.e. the line is below the pixel center) and 0.5-(round(a)-a) if the
        // quantity is negative (i.e. the line is above the pixel center, so we
        // need to advance all the way up to the next pixel center above).
        // But this is simpler...
        verts[i] = vertices[i] - 0.5;
    }

    float *v0 = verts;
    float *v1 = &verts[2];
    float *v2 = &verts[4];

    unsigned int n_colors = shape[2];
    float *c0 = colors;
    float *c1 = &colors[n_colors];
    float *c2 = &colors[2 * n_colors];

    // Sort verticies by ascending y value
    if (v0[1] > v1[1]){
        swap(&v0, &v1);
        swap(&c0, &c1);
    }
    if (v0[1] > v2[1]){
        swap(&v0, &v2);
        swap(&c0, &c2);
    }
    if (v1[1] > v2[1]){
        swap(&v1, &v2);
        swap(&c1, &c2);
    }

    long y0i = ceilf(v0[1]);
    long y1i = ceilf(v1[1]);
    long y2i = ceilf(v2[1]);

    if (y0i == y2i){
        return;
    }

    float dXdY_V0V1 = (v1[0] - v0[0]) / (v1[1] - v0[1]);
    float dXdY_V0V2 = (v2[0] - v0[0]) / (v2[1] - v0[1]);
    float dXdY_V1V2 = (v2[0] - v1[0]) / (v2[1] - v1[1]);

    char v1_left;
    if (v1[1] - v0[1] == 0) {
        // if v1 and v0 are even, the left-most one is easy to find
        v1_left = v1[0] < v0[0];
    } else {
        // if they aren't even then the left side of the triangle can be
        // discerned based on the slopes of the lines v0-v1 and v0-v2
        v1_left = dXdY_V0V2 > dXdY_V0V1;
    }

    float dCdY_V0V2[n_colors];
    float dCdY_V1V2[n_colors];
    float dCdY_V0V1[n_colors];
    float dCdX[n_colors];
    float leftC[n_colors];
    for (unsigned int i = 0; i < n_colors; i++){
        dCdX[i] = ((v1[1] - v0[1])*(c2[i] - c0[i]) + (c0[i] - c1[i])*(v2[1] - v0[1])) /
                  ((v1[1] - v0[1])*(v2[0] - v0[0]) + (v0[0] - v1[0])*(v2[1] - v0[1]));

        dCdY_V0V2[i] = (c2[i] - c0[i]) / (v2[1] - v0[1]);
        dCdY_V1V2[i] = (c2[i] - c1[i]) / (v2[1] - v1[1]);
        dCdY_V0V1[i] = (c1[i] - c0[i]) / (v1[1] - v0[1]);
    }

    DrawingState state;
    state.leftC = leftC;
    state.dCdX = dCdX;
    state.shape = shape;
    state.strides = strides;
    state.accumulate = accumulate;

    // Several cases of triangle; assume verticies are sorted by vertical order
    // with "top" = "lower y values" and "right" = "higher x values":
    // 1) Triangle is flat on top with v0 and v1 on the same y scanline. Assume v1 on right.
    //    In this case, there is only a single triangle segment to draw with constant derivatives.
    // 2) Triangle have v0 uniquely at top. In this case, either line v0-v1 or v0-v2 makes
    //    up the leftmost of the lines descending from vertex v0. In either case,
    //    we draw the top half of the triangle (a "segment") in rasters from v0 down to v1.
    //    This is itself just a simple triange with constant derivatives. Once we're at
    //    v1, we then change the derivatives and continue down with the bottom half of
    //    the triangle all the way to v2. In the case that v1 and v2 are even, we skip
    //    the latter step. (The specific derivatives we set for how to advance the
    //    scanlines drawn depend on on whether v0-v1 or v0-v2 is the left side...)

    if (y0i == y1i) { // v0 and v1 are even (on the y)
        if (v1_left) {
            init_leftXandC(v1, dXdY_V1V2, c1, dCdY_V1V2, &state);
            init_rightX(v0, dXdY_V0V2, &state);
        } else {
            init_leftXandC(v0, dXdY_V0V2, c0, dCdY_V0V2, &state);
            init_rightX(v1, dXdY_V1V2, &state);
        }
        draw_gouraud_segment(y0i, y2i, out, &state);
    } else if (v1_left) { // v1 is below v0 and v1 is at the left side
        init_leftXandC(v0, dXdY_V0V1, c0, dCdY_V0V1, &state);
        init_rightX(v0, dXdY_V0V2, &state);
        draw_gouraud_segment(y0i, y1i, out, &state);

        // if v1 and v2 are even (on the y), there is no lower segment
        if (y2i > y1i) {
            // There is a lower segment starting with v1 on the left
            // (no need to initialize the right since its derivatives aren't changing)
            init_leftXandC(v1, dXdY_V1V2, c1, dCdY_V1V2, &state);
            draw_gouraud_segment(y1i, y2i, out, &state);
        }
    } else { // v1 is below v0 and v1 is at the right side
        init_leftXandC(v0, dXdY_V0V2, c0, dCdY_V0V2, &state);
        init_rightX(v0, dXdY_V0V1, &state);
        draw_gouraud_segment(y0i, y1i, out, &state);

        // if v1 and v2 are even (on the y), there is no lower segment
        if (y2i > y1i) {
            // There is a lower segment starting with v1 on the right
            // (no need to initialize the right since its derivatives aren't changing)
            init_rightX(v1, dXdY_V1V2, &state);
            draw_gouraud_segment(y1i, y2i, out, &state);
        }
    }
}

#define SUB_PIX(a) (ceilf(a) - a)

inline void init_leftX(float *vertex, float dXdY, DrawingState *state) {
    // vertex is the vertex from which the drawing starts...
    state->left_dXdY = dXdY;
    state->leftX = vertex[0] + SUB_PIX(vertex[1]) * dXdY;
}

inline void init_rightX(float *vertex, float dXdY, DrawingState *state) {
    // vertex is the vertex from which the drawing starts...
    state->right_dXdY = dXdY;
    state->rightX = vertex[0] + SUB_PIX(vertex[1]) * dXdY;
}

inline void init_leftXandC(float *vertex, float dXdY, float *color, float *dCdY, DrawingState *state) {
    // vertex is the vertex from which the drawing starts...
    init_leftX(vertex, dXdY, state);
    state->left_dCdY = dCdY;
    for (unsigned int i = 0; i < state->shape[2]; i++) {
        state->leftC[i] = color[i] + SUB_PIX(vertex[1]) * dCdY[i];
    }
}

inline void swap(float **a, float **b) {
    float *temp = *b;
    *b = *a;
    *a = temp;
}

void draw_gouraud_segment(long y0,long y1, float *out, DrawingState *state) {
    if (y1 < 0 || y0 >= state->shape[1]) {
        return;
    }
    unsigned int i;
    unsigned int n_colors = state->shape[2];
    if (y0 < 0) {
        state->leftX += -y0 * state->left_dXdY;
        state->rightX += -y0 * state->right_dXdY;
        for (i = 0; i < n_colors; i++) {
            state->leftC[i] += -y0 * state->left_dCdY[i];
        }
        y0 = 0;
    }
    if (y1 > state->shape[1]) {
        y1 = state->shape[1];
    }

    char *line_ptr = ((char *) out) + y0 * state->strides[1];

    while (y0++ < y1) {
        long x0 = ceilf(state->leftX);
        long x1 = ceilf(state->rightX);
        if (x1 < 0 || x0 >= state->shape[0]) {
            continue;
        }
        if (x1 > state->shape[0]) {
            x1 = state->shape[0];
        }

        char *color_ptr = line_ptr;
        for (i = 0; i < n_colors; i++) {
            long x = x0;
            if (x0 < 0) {
                x = 0;
            }
            float color = state->leftC[i] + (x - state->leftX) * state->dCdX[i];
            char *dest = color_ptr + x * state->strides[0];
            while (x++ < x1) {
                if (state->accumulate) {
                    *((float *) dest) += (float) color;
                } else {
                    *((float *) dest) = (float) color;
                }
                dest += state->strides[0];
                color += state->dCdX[i];
            }
            state->leftC[i] += state->left_dCdY[i];
            color_ptr += state->strides[2];
        }
        state->leftX += state->left_dXdY;
        state->rightX += state->right_dXdY;
        line_ptr += state->strides[1];
    }
}

void mask_triangle(float *vertices, char *out, unsigned int *shape, long *strides) {
    // out = strided array of shape (x, y)

    float verts[6];
    for (int i = 0; i < 6; i++) {
        // all math below is easier if the "pixel centers" are at the lower-left corner of
        // the pixel rather than at (0.5, 0.5) -- then we can use ceil rather than round
        // and not need any conditioning on whether a line is above or below the pixel
        // center within a pixel. All we need to do to correct the drawing for this
        // is to internally subtrace 0.5 from the vertices.
        // If we didn't do that, we could just replace "ceil" everywhere with "round"
        // and then fix SUB_PIX to return round(a)-a if that quantity is positive,
        // (i.e. the line is below the pixel center) and 0.5-(round(a)-a) if the
        // quantity is negative (i.e. the line is above the pixel center, so we
        // need to advance all the way up to the next pixel center above).
        // But this is simpler...
        verts[i] = vertices[i] - 0.5;
    }

    float *v0 = verts;
    float *v1 = &verts[2];
    float *v2 = &verts[4];

    // Sort verticies by ascending y value
    if (v0[1] > v1[1]){
        swap(&v0, &v1);
    }
    if (v0[1] > v2[1]){
        swap(&v0, &v2);
    }
    if (v1[1] > v2[1]){
        swap(&v1, &v2);
    }

    long y0i = ceilf(v0[1]);
    long y1i = ceilf(v1[1]);
    long y2i = ceilf(v2[1]);

    if (y0i == y2i) {
        return;
    }

    float dXdY_V0V1 = (v1[0] - v0[0]) / (v1[1] - v0[1]);
    float dXdY_V0V2 = (v2[0] - v0[0]) / (v2[1] - v0[1]);
    float dXdY_V1V2 = (v2[0] - v1[0]) / (v2[1] - v1[1]);

    char v1_left;
    if (v1[1] - v0[1] == 0) {
        // if v1 and v0 are even, the left-most one is easy to find
        v1_left = v1[0] < v0[0];
    } else {
        // if they aren't even then the left side of the triangle can be
        // discerned based on the slopes of the lines v0-v1 and v0-v2
        v1_left = dXdY_V0V2 > dXdY_V0V1;
    }

    DrawingState state;
    state.shape = shape;
    state.strides = strides;

    // Several cases of triangle; assume verticies are sorted by vertical order
    // with "top" = "lower y values" and "right" = "higher x values":
    // 1) Triangle is flat on top with v0 and v1 on the same y scanline. Assume v1 on right.
    //    In this case, there is only a single triangle segment to draw with constant derivatives.
    // 2) Triangle have v0 uniquely at top. In this case, either line v0-v1 or v0-v2 makes
    //    up the leftmost of the lines descending from vertex v0. In either case,
    //    we draw the top half of the triangle (a "segment") in rasters from v0 down to v1.
    //    This is itself just a simple triange with constant derivatives. Once we're at
    //    v1, we then change the derivatives and continue down with the bottom half of
    //    the triangle all the way to v2. In the case that v1 and v2 are even, we skip
    //    the latter step. (The specific derivatives we set for how to advance the
    //    scanlines drawn depend on on whether v0-v1 or v0-v2 is the left side...)

    if (y0i == y1i) { // v0 and v1 are even (on the y)
        if (v1_left) {
            init_leftX(v1, dXdY_V1V2, &state);
            init_rightX(v0, dXdY_V0V2, &state);
        } else {
            init_leftX(v0, dXdY_V0V2, &state);
            init_rightX(v1, dXdY_V1V2, &state);
        }
        draw_mask_segment(y0i, y2i, out, &state);
    } else if (v1_left) { // v1 is below v0 and v1 is at the left side
        init_leftX(v0, dXdY_V0V1, &state);
        init_rightX(v0, dXdY_V0V2, &state);
        draw_mask_segment(y0i, y1i, out, &state);

        // if v1 and v2 are even (on the y), there is no lower segment
        if (y2i > y1i) {
            // There is a lower segment starting with v1 on the left
            // (no need to initialize the right since its derivatives aren't changing)
            init_leftX(v1, dXdY_V1V2, &state);
            draw_mask_segment(y1i, y2i, out, &state);
        }
    } else { // v1 is below v0 and v1 is at the right side
        init_leftX(v0, dXdY_V0V2, &state);
        init_rightX(v0, dXdY_V0V1, &state);
        draw_mask_segment(y0i, y1i, out, &state);

        // if v1 and v2 are even (on the y), there is no lower segment
        if (y2i > y1i) {
            // There is a lower segment starting with v1 on the right
            // (no need to initialize the right since its derivatives aren't changing)
            init_rightX(v1, dXdY_V1V2, &state);
            draw_mask_segment(y1i, y2i, out, &state);
        }
    }
}

void draw_mask_segment(long y0,long y1, char *out, DrawingState *state) {
    if (y1 < 0 || y0 >= state->shape[1]) {
        return;
    }
    if (y0 < 0) {
        state->leftX += -y0 * state->left_dXdY;
        state->rightX += -y0 * state->right_dXdY;
        y0 = 0;
    }
    if (y1 > state->shape[1]) {
        y1 = state->shape[1];
    }

    out += y0 * state->strides[1];

    while (y0++ < y1) {
        long x0 = ceilf(state->leftX);
        long x1 = ceilf(state->rightX);
        if (x1 < 0 || x0 >= state->shape[0]) {
            continue;
        }
        if (x0 < 0) {
            x0 = 0;
        }
        if (x1 > state->shape[0]) {
            x1 = state->shape[0];
        }

        char *dest = out + x0 * state->strides[0];
        while (x0++ < x1) {
            *dest = 1;
            dest += state->strides[0];
        }
        state->leftX += state->left_dXdY;
        state->rightX += state->right_dXdY;
        out += state->strides[1];
    }
}
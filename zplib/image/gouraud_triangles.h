void gouraud_triangle(float *vertices, float *colors, float *out, unsigned int *shape, long *strides, unsigned char accumulate);
void gouraud_triangle_strip(unsigned int n_vertices, float *vertices, float *colors, float *out, unsigned int *shape, long *strides, unsigned char accumulate);
void mask_triangle(float *vertices, char *out, unsigned int *shape, long *strides);
void mask_triangle_strip(unsigned int n_vertices, float *vertices, char *out, unsigned int *shape, long *strides);

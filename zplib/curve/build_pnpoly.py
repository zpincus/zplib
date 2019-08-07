# This code is licensed under the MIT License (see LICENSE file for details)
import cffi
import pathlib

# point-in-polygon code from: https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
# nvert           Number of vertices in the polygon. Whether to repeat the first vertex at the end is discussed below.
# vertx, verty    Arrays containing the x- and y-coordinates of the polygon's vertices.
# testx, testy    X- and y-coordinate of the test point.

# LICENSE for the pnpoly_c code below:
#
# Copyright (c) 1970-2003, Wm. Randolph Franklin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.
# Redistributions in binary form must reproduce the above copyright notice in the documentation and/or other materials provided with the distribution.
# The name of W. Randolph Franklin may not be used to endorse or promote products derived from this Software without specific prior written permission.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


pnpoly_c = """int pnpoly(int nvert, float *vertx, float *verty, float testx, float testy) {
  int i, j, c = 0;
  for (i = 0, j = nvert-1; i < nvert; j = i++) {
    if ( ((verty[i]>testy) != (verty[j]>testy)) &&
	 (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) )
       c = !c;
  }
  return c;
}
"""

pnpoly_h = pnpoly_c.split('\n')[0][:-1] + ';'

ffibuilder = cffi.FFI()
directory = pathlib.Path(__file__).parent
ffibuilder.cdef(pnpoly_h)
ffibuilder.set_source('zplib.curve._pnpoly', pnpoly_c)

if __name__ == "__main__":
    ffibuilder.compile()

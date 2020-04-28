#include <iostream>

#include "rainbowmist.h"

RM_KERNEL void simple_add_vec2(RM_GLOBAL vec2 *ret, RM_GLOBAL const vec2 *a, RM_GLOBAL const vec2 *b)
{
  ret[0] = a[0] + b[0];
  ret[1] = a[1] + b[1];
}

int main(int argc, char **argv)
{
  vec2 a = make_vec2(0.0f, 1.0f);
  vec2 b = make_vec2(2.1f, 3.4f);

  if (argc > 1) {
    a.x = std::atof(argv[1]);
  }

  if (argc > 2) {
    a.y = std::atof(argv[2]);
  }

  if (argc > 3) {
    b.x = std::atof(argv[3]);
  }

  if (argc > 4) {
    b.y = std::atof(argv[4]);
  }

  vec2 ret;
  simple_add_vec2(&ret, &a, &b);

  std::cout << "result = " << ret.x << ", " << ret.y << "\n";

  return 0;
      
}

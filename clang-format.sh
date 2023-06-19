#!/bin/bash

find . -iname *.h -o -iname *.c -o -iname *.cpp -o -iname *.hpp -o -iname *.cuh -o -iname *.cu \
    | xargs clang-format -style=file -i -fallback-style=none

exit 0
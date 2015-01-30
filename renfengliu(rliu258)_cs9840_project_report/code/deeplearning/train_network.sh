#!/usr/bin/env sh
start_time=`date +%s`
../lib/caffe/build/tools/caffe.bin train --solver=./smile_solver.prototxt
end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.


all:
	python3 setup.py build_ext -if

clean:
	-rm -rf build *.c *.so *.pyc

test_inflate:
	python3 -c "from mkgrid import inflate_file;import numpy as np;filename='/anx_lagr3/arthur.endsley/SMAP_L4C/L4C_Science/Cython/test_cert/example_to_inflate_M09land.uint8';inflate_file(filename);arr = np.fromfile(filename.replace('M09land', 'test'), np.uint8);print(arr.shape);print(np.percentile(arr, (0, 10, 50, 90, 100)))"

test_deflate:
	python3 -c "from mkgrid import deflate_file;import numpy as np;filename='/anx_lagr3/arthur.endsley/SMAP_L4C/L4C_Science/Cython/test_cert/example_to_deflate_M09.uint8';deflate_file(filename);arr = np.fromfile(filename.replace('M09', 'M09land'), np.uint8);print(arr.shape);print(np.percentile(arr, (0, 10, 50, 90, 100)))"

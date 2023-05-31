all:
	python3 setup.py build_ext -if

clean:
	-rm -rf build *.c *.so *.pyc

test:
	python3 -c "from mkgrid import inflate_file;print(inflate_file('/anx_lagr3/arthur.endsley/SMAP_L4C/L4C_Science/Cython/v20230523/L4Cython_RH_20150331_M09land.flt32'))"

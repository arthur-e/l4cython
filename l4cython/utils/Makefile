all:
	python3 setup.py build_ext -if

clean:
	-rm -rf build *.c *.so *.pyc

test:
	python3 -c "from l4cython.utils.fixtures import restore_bplut;restore_bplut('/anx_lagr3/arthur.endsley/SMAP_L4C/calibration/v7_Y2022/SMAP_BPLUT_V7_rev_20220728.csv'.encode('UTF-8'))"

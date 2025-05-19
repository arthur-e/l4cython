all:
	python3 setup.py build_ext -if

clean:
	-rm -rf build *.c *.so *.pyc l4cython/*.c l4cython/*.so l4cython/*.pyc l4cython/utils/*.c l4cython/utils/*.so l4cython/utils/*.pyc

spinup:
	python3 -c "from l4cython.spinup import main;print(main())"

run:
	python3 -c "from l4cython.budget import main; print(main())"

test-reco:
	python3 -c "from l4cython.reco import main; print(main())"

test-gpp:
	python3 -c "from l4cython.gpp import main;print(main())"

test-spinup:
	python3 -c "from l4cython.spinup import main;print(main('../tests/data/L4Cython_spin-up_test_config.yaml'))"

all:
	python3 setup.py build_ext -if

clean:
	-rm -rf build *.c *.so *.pyc

spinup:
	python3 -c "from spinup import main;print(main())"

run:
	python3 -c "from budget import main; print(main())"

test-reco:
	python3 -c "from reco import main; print(main())"

test-gpp:
	python3 -c "from gpp import main;print(main('../data/L4Cython_GPP_config.yaml'))"

test-spinup:
	python3 -c "from spinup import main;print(main('../tests/data/L4Cython_spin-up_test_config.yaml'))"

timeit:
	python3 -m timeit -s "from reco_9km import main" "print(main())"

all:
	python3 setup.py build_ext -if

clean:
	-rm -rf build *.c *.so *.pyc

spinup:
	python3 -c "from spinup import main;print(main())"

run:
	python3 -c "from reco import main; print(main())"

test:
	python3 -c "from reco_9km import main; print(main())"

timeit:
	python3 -m timeit -s "from reco_9km import main" "print(main())"

all:
	python setup.py build_ext -if

clean:
	-rm -rf build *.c *.so *.pyc

test:
	python3 -c "from respiration import main; print(main())"

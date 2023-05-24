all:
	python3 setup.py build_ext -if

clean:
	-rm -rf build *.c *.so *.pyc

test:
	python3 -c "from mkgrid import main;print(main())"

'''
TODO This is a placeholder for later, when mkgrid is renamed.
'''

from l4cython.utils.mkgrid import inflate_file

def main(filename, grid):
    inflate_file(filename, grid)


if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])

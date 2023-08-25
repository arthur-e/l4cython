'''
Tests for the `utils` module.
'''

import os
import numpy as np
from l4cython.utils.mkgrid import inflate_file, deflate_file
from l4cython.utils.fixtures import NROW9KM, NCOL9KM, SPARSE_M09_N

TEST_FILES = {
    'deflated_uint8': '/anx_lagr3/arthur.endsley/SMAP_L4C/L4C_Science/Cython/test_cert/example_to_inflate_M09land.uint8',
    'inflated_uint8': '/anx_lagr3/arthur.endsley/SMAP_L4C/L4C_Science/Cython/test_cert/example_to_deflate_M09.uint8',
    'deflated_float32': 'test_deflated_M09land.flt32',
    'inflated_float32': 'test_inflated_M09.flt32',
    'deflated_float64': 'test_deflated_M09land.flt64',
    'inflated_float64': 'test_inflated_M09.flt64',
}


def test_inflate_uint8_actual():
    'Inflate an array of unsigned 8-bit integer type, from actual binary file'
    fname0 = TEST_FILES['deflated_uint8']
    inflate_file(fname0, grid = 'M09')
    fname = fname0.replace('M09land', 'M09')
    arr = np.fromfile(fname, np.uint8)
    assert arr.shape == (NROW9KM * NCOL9KM,)
    assert np.equal(np.unique(arr), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 255])).all()


def test_deflate_uint8_actual():
    'Deflate an array of unsigned 8-bit integer type, from actual binary file'
    fname0 = TEST_FILES['inflated_uint8']
    inflate_file(fname0, grid = 'M09')
    fname = fname0.replace('M09', 'M09land')
    arr = np.fromfile(fname, np.uint8)
    assert arr.shape == (SPARSE_M09_N,)
    assert np.equal(np.unique(arr), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).all()


def test_inflate_float32():
    'Inflate an array of 32-bit floating point type'
    fname0 = TEST_FILES['deflated_float32']
    np.random.seed(9)
    arr_out = np.random.rand(SPARSE_M09_N)
    arr_out.astype(np.float32).tofile(fname0)
    inflate_file(fname0, grid = 'M09')
    fname = fname0.replace('M09land', 'M09')
    arr_in = np.fromfile(fname, np.float32)
    assert arr_in.shape == (NROW9KM * NCOL9KM,)
    # NOTE: This may be a trivial test because np.random.rand() samples from
    #   the uniform distribution, resulting in predictable percentiles
    stats_out = np.percentile(arr_out, (1, 10, 50, 90, 99)).round(4)
    stats_in = np.percentile(arr_in[arr_in != -9999], (1, 10, 50, 90, 99)).round(4)
    assert np.equal(stats_out, stats_in).all()
    os.remove(fname0)
    os.remove(fname)


def test_deflate_float32():
    'Deflate an array of 32-bit floating point type'
    fname0 = TEST_FILES['inflated_float32']
    np.random.seed(9)
    arr_out = np.random.rand(NROW9KM * NCOL9KM).reshape((NROW9KM, NCOL9KM))
    arr_out.astype(np.float32).tofile(fname0)
    deflate_file(fname0, grid = 'M09')
    fname = fname0.replace('M09', 'M09land')
    arr_in = np.fromfile(fname, np.float32)
    assert arr_in.shape == (SPARSE_M09_N,)
    # NOTE: This may be a trivial test because np.random.rand() samples from
    #   the uniform distribution, resulting in predictable percentiles
    stats_out = np.percentile(arr_out, (1, 10, 50, 90, 99)).round(3)
    stats_in = np.percentile(arr_in[arr_in != -9999], (1, 10, 50, 90, 99)).round(3)
    assert np.equal(stats_out, stats_in).all()
    os.remove(fname0)
    os.remove(fname)


def test_inflate_float64():
    'Inflate an array of 64-bit floating point type'
    fname0 = TEST_FILES['deflated_float64']
    np.random.seed(9)
    arr_out = np.random.rand(SPARSE_M09_N)
    arr_out.astype(np.float64).tofile(fname0)
    inflate_file(fname0, grid = 'M09')
    fname = fname0.replace('M09land', 'M09')
    arr_in = np.fromfile(fname, np.float64)
    assert arr_in.shape == (NROW9KM * NCOL9KM,)
    # NOTE: This may be a trivial test because np.random.rand() samples from
    #   the uniform distribution, resulting in predictable percentiles
    stats_out = np.percentile(arr_out, (1, 10, 50, 90, 99)).round(4)
    stats_in = np.percentile(arr_in[arr_in != -9999], (1, 10, 50, 90, 99)).round(4)
    assert np.equal(stats_out, stats_in).all()
    os.remove(fname0)
    os.remove(fname)


def test_deflate_float64():
    'Deflate an array of 64-bit floating point type'
    fname0 = TEST_FILES['inflated_float64']
    np.random.seed(9)
    arr_out = np.random.rand(NROW9KM * NCOL9KM).reshape((NROW9KM, NCOL9KM))
    arr_out.astype(np.float64).tofile(fname0)
    deflate_file(fname0, grid = 'M09')
    fname = fname0.replace('M09', 'M09land')
    arr_in = np.fromfile(fname, np.float64)
    assert arr_in.shape == (SPARSE_M09_N,)
    # NOTE: This may be a trivial test because np.random.rand() samples from
    #   the uniform distribution, resulting in predictable percentiles
    stats_out = np.percentile(arr_out, (1, 10, 50, 90, 99)).round(3)
    stats_in = np.percentile(arr_in[arr_in != -9999], (1, 10, 50, 90, 99)).round(3)
    assert np.equal(stats_out, stats_in).all()
    os.remove(fname0)
    os.remove(fname)

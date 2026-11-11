"""ponytail: rung 7 checks for the pure-logic functions in tglc.

Most of tglc talks to MAST/Gaia or reads FITS files from disk, so it isn't
testable without heavy mocking. These are the few functions with real,
self-contained logic - one assert-based check each, no fixtures needed.
Runnable with `python tests/test_pure_functions.py` or with pytest.
"""
import numpy as np

from tglc.effective_psf import bilinear
from tglc.ffi import mast_json2table
from tglc.quick_lc import timebin


def test_bilinear():
    assert list(bilinear(0, 0, 1)) == [1, 0, 0, 0]
    assert list(bilinear(1, 1, 1)) == [0, 0, 0, 1]
    assert list(bilinear(0.5, 0.5, 1)) == [0.25, 0.25, 0.25, 0.25]
    assert list(bilinear(0, 0, 2)) == [1, 0, 0, 0, 1, 0, 0, 0]


def test_mast_json2table():
    json_obj = {
        'fields': [
            {'name': 'ID', 'type': 'string'},
            {'name': 'Tmag', 'type': 'double'},
        ],
        'data': [
            {'ID': '123', 'Tmag': 10.5},
            {'ID': '456', 'Tmag': 11.2},
        ],
    }
    table = mast_json2table(json_obj)
    assert list(table['ID']) == ['123', '456']
    assert list(table['Tmag']) == [10.5, 11.2]


def test_timebin_averages_within_bin_and_sorts_input():
    time = np.array([0.2, 0.0, 0.1])
    meas = np.array([30.0, 10.0, 20.0])
    err = np.array([1.0, 1.0, 1.0])
    time_out, meas_out, err_out = timebin(time, meas, err, binsize=1.0)
    assert time_out == [0.1]
    assert meas_out == [20.0]
    assert err_out == [1 / np.sqrt(3)]


if __name__ == '__main__':
    test_bilinear()
    test_mast_json2table()
    test_timebin_averages_within_bin_and_sorts_input()
    print('ok')

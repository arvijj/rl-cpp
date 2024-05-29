import math
import numpy as np
from rlm import utils


def test_total_variation():
    """
    Test for utils.total_variation
    """
    # Tests without img2
    img = np.zeros((3, 3))
    img[1, 1] = 1
    tv_1 = utils.total_variation(img, mode='sym-iso')
    tv_2 = utils.total_variation(img, mode='non-sym-iso')
    tv_3 = utils.total_variation(img, mode='non-iso')
    assert abs(tv_1 - (2 + math.sqrt(2))) < 1e-10
    assert abs(tv_2 - (2 + math.sqrt(2))) < 1e-10
    assert abs(tv_3 - 4) < 1e-10
    img = np.zeros((4, 4))
    img[1, 1] = 1
    img[1, 2] = 1
    img[2, 2] = 1
    tv_1 = utils.total_variation(img, mode='sym-iso')
    tv_2 = utils.total_variation(np.fliplr(img), mode='sym-iso')
    tv_3 = utils.total_variation(np.flipud(img), mode='sym-iso')
    tv_4 = utils.total_variation(np.flipud(np.fliplr(img)), mode='sym-iso')
    assert abs(tv_1 - (5 + math.sqrt(2) * 3 / 2)) < 1e-10
    assert abs(tv_2 - (5 + math.sqrt(2) * 3 / 2)) < 1e-10
    assert abs(tv_3 - (5 + math.sqrt(2) * 3 / 2)) < 1e-10
    assert abs(tv_4 - (5 + math.sqrt(2) * 3 / 2)) < 1e-10
    tv_1 = utils.total_variation(img, mode='non-iso')
    tv_2 = utils.total_variation(np.fliplr(img), mode='non-iso')
    tv_3 = utils.total_variation(np.flipud(img), mode='non-iso')
    tv_4 = utils.total_variation(np.flipud(np.fliplr(img)), mode='non-iso')
    assert abs(tv_1 - 8) < 1e-10
    assert abs(tv_2 - 8) < 1e-10
    assert abs(tv_3 - 8) < 1e-10
    assert abs(tv_4 - 8) < 1e-10
    # Tests with img2
    img = np.zeros((5, 3))
    img2 = np.zeros((5, 3))
    img[1, 1] = 1
    img2[2, 1] = 1
    tv_1 = utils.total_variation(img, img2, mode='sym-iso')
    tv_2 = utils.total_variation(img, img2, mode='non-iso')
    assert abs(tv_1 - (2 + math.sqrt(2) / 2)) < 1e-10
    assert abs(tv_2 - 3) < 1e-10
    img = np.zeros((5, 3))
    img2 = np.zeros((5, 3))
    img[:2, :] = 1
    img2[2, :] = 1
    tv_1 = utils.total_variation(img, img2, mode='sym-iso')
    tv_2 = utils.total_variation(img, img2, mode='non-iso')
    assert abs(tv_1) < 1e-10
    assert abs(tv_2) < 1e-10

def test_format_float_str():
    """
    Test for utils.format_float_str
    """
    assert utils.format_float_str(7         , decimals=0, spaces=0) == ""
    assert utils.format_float_str(7         , decimals=0, spaces=2) == " 7"
    assert utils.format_float_str(3.14159265, decimals=0, spaces=3) == "  3"
    assert utils.format_float_str(3.14159265, decimals=1, spaces=3) == "3.1"
    assert utils.format_float_str(3.14159265, decimals=1, spaces=4) == " 3.1"
    assert utils.format_float_str(3.14159265, decimals=2, spaces=1) == "3"
    assert utils.format_float_str(3.14159265, decimals=2, spaces=2) == "3."
    assert utils.format_float_str(3.14159265, decimals=2, spaces=3) == "3.1"
    assert utils.format_float_str(3.14159265, decimals=2, spaces=4) == "3.14"
    assert utils.format_float_str(3.14159265, decimals=2, spaces=5) == " 3.14"
    assert utils.format_float_str(3.14159265, decimals=3, spaces=6) == " 3.142"
    assert utils.format_float_str(3.14159265, decimals=4, spaces=7) == " 3.1416"
    assert utils.format_float_str(3.14159265, decimals=5, spaces=8) == " 3.14159"

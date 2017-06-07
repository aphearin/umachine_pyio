# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
from astropy import config as _config
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    pass
    # from .example_mod import *


class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for my subpackage.
    """
    umachine_mocks_root_dirname = _config.ConfigItem(
        "/Users/aphearin/work/UniverseMachine/data/0412_binaries/obs_sm_9p75_cut/a_1.002310",
        "Location storing collection of binaries of the default mock"
)


# Create an instance for the user
conf = Conf()

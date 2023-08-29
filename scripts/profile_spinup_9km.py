import pstats, cProfile
from l4cython.spinup import main

cProfile.runctx("main('/usr/local/dev/l4cython/tests/data/L4Cython_spin-up_test_config.yaml')", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

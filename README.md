# nu_plugin_from_hdf5

This plugin is in developing.

Some notes:

* Uses a patched `hdf5-rust`: https://github.com/Berrysoft/hdf5-rust/tree/fix/mingw
* Links to `gcc_s` dynamically with GNU toolchain on Windows. Just personal preference.
* Uses nightly `try_collect` although `itertools` also provides this functionality. Just personal preference.
* Uses some unsafe `read_unaligned` and doesn't even check endian.
* Poor performance. Sticks when opening large dataset.

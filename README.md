# nu_plugin_from_hdf5

This plugin is OK for light-weight usage.

Some notes:

* Uses a patched `hdf5-rust`: https://github.com/Berrysoft/hdf5-rust/tree/fix/mingw
* Uses nightly features. Just personal preference.
* Poor performance. Sticks when opening large dataset. Waiting for dataframe support.

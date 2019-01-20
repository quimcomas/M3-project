import h5py
import numpy as np
import tables

h5file = tables.open_file("my_first_mlp.h5", "w", driver="H5FD_CORE", driver_core_backing_store=0)

h5file.create_array(h5file.root, "new_array", np.arange(20),
                        title="New array")

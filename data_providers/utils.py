from .data import DataProvider

"""Args
    path: path to the video data folder
    """
def get_data_provider_by_path(config, dataset_train, dataset_test, all_hdf5_data_train, all_hdf5_data_test, whichFoldData):
    """Return required data provider class"""
    return DataProvider(config, dataset_train, dataset_test, all_hdf5_data_train, all_hdf5_data_test, whichFoldData)


from .data import DataProvider

"""Args
    path: path to the video data folder
    """
def get_data_provider_by_path(config, dataset_train_unlabelled,dataset_train_labelled, dataset_test, all_hdf5_data_train, all_hdf5_data_test, dataset_val, all_hdf5_data_val, whichFoldData):
    """Return required data provider class"""
    return DataProvider(config, dataset_train_unlabelled,dataset_train_labelled, dataset_test, all_hdf5_data_train, all_hdf5_data_test,dataset_val, all_hdf5_data_val, whichFoldData)


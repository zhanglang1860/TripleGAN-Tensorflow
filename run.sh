python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_3_AD_MCI_Normal.hdf5 --idFileNametrain train_MRIdata_3_AD_MCI_Normal_id.txt --testhdf5FileName test_MRIdata_3_AD_MCI_Normal.hdf5 --testidFileName test_MRIdata_3_AD_MCI_Normal_id.txt --gpu_id 0 --softmaxConvert True

python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_2_AD_MCI.hdf5 --idFileNametrain train_MRIdata_2_AD_MCI_id.txt --testhdf5FileName test_MRIdata_2_AD_MCI.hdf5 --testidFileName test_MRIdata_2_AD_MCI_id.txt --gpu_id 1
 

python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_2_MCI_Normal.hdf5 --idFileNametrain train_MRIdata_2_MCI_Normal_id.txt --testhdf5FileName test_MRIdata_2_MCI_Normal.hdf5 --testidFileName test_MRIdata_2_MCI_Normal_id.txt --gpu_id 2


python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_2_AD_Normal.hdf5 --idFileNametrain train_MRIdata_2_AD_Normal_id.txt --testhdf5FileName test_MRIdata_2_AD_Normal.hdf5 --testidFileName test_MRIdata_2_AD_Normal_id.txt --gpu_id 2

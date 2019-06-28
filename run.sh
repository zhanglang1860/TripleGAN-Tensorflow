python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_3_AD_MCI_Normal.hdf5 --idFileNametrain train_MRIdata_3_AD_MCI_Normal_id.txt --testhdf5FileName test_MRIdata_3_AD_MCI_Normal.hdf5 --testidFileName test_MRIdata_3_AD_MCI_Normal_id.txt --gpu_id 0 --split_dimension_core_G 3 --tt_rank_G 16 --split_dimension_core_C 3 --tt_rank_C 26 --split_dimension_core_D 3 --tt_rank_D 15


python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_3_AD_MCI_Normal.hdf5 --idFileNametrain train_MRIdata_3_AD_MCI_Normal_id.txt --testhdf5FileName test_MRIdata_3_AD_MCI_Normal.hdf5 --testidFileName test_MRIdata_3_AD_MCI_Normal_id.txt --gpu_id 1 --split_dimension_core_G 3 --tt_rank_G 12 --split_dimension_core_C 3 --tt_rank_C 19 --split_dimension_core_D 3 --tt_rank_D 11


python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_3_AD_MCI_Normal.hdf5 --idFileNametrain train_MRIdata_3_AD_MCI_Normal_id.txt --testhdf5FileName test_MRIdata_3_AD_MCI_Normal.hdf5 --testidFileName test_MRIdata_3_AD_MCI_Normal_id.txt --gpu_id 2 --split_dimension_core_G 3 --tt_rank_G 10 --split_dimension_core_C 3 --tt_rank_C 16 --split_dimension_core_D 3 --tt_rank_D 9


python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_3_AD_MCI_Normal.hdf5 --idFileNametrain train_MRIdata_3_AD_MCI_Normal_id.txt --testhdf5FileName test_MRIdata_3_AD_MCI_Normal.hdf5 --testidFileName test_MRIdata_3_AD_MCI_Normal_id.txt --gpu_id 3 --split_dimension_core_G 3 --tt_rank_G 9 --split_dimension_core_C 3 --tt_rank_C 13 --split_dimension_core_D 3 --tt_rank_D 8



--train --test --hdf5FileNametrain train_MRIdata_2_AD_MCI.hdf5 --idFileNametrain train_MRIdata_2_AD_MCI_id.txt --testhdf5FileName test_MRIdata_2_AD_MCI.hdf5 --testidFileName test_MRIdata_2_AD_MCI_id.txt --gpu_id 0




python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_2_AD_MCI.hdf5 --idFileNametrain train_MRIdata_2_AD_MCI_id.txt --testhdf5FileName test_MRIdata_2_AD_MCI.hdf5 --testidFileName test_MRIdata_2_AD_MCI_id.txt --gpu_id 0 --split_dimension_core 10 --tt_rank 10

python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_2_AD_MCI.hdf5 --idFileNametrain train_MRIdata_2_AD_MCI_id.txt --testhdf5FileName test_MRIdata_2_AD_MCI.hdf5 --testidFileName test_MRIdata_2_AD_MCI_id.txt --gpu_id 1 --split_dimension_core 10 --tt_rank 16
python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_2_AD_MCI.hdf5 --idFileNametrain train_MRIdata_2_AD_MCI_id.txt --testhdf5FileName test_MRIdata_2_AD_MCI.hdf5 --testidFileName test_MRIdata_2_AD_MCI_id.txt --gpu_id 1



 

python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_2_MCI_Normal.hdf5 --idFileNametrain train_MRIdata_2_MCI_Normal_id.txt --testhdf5FileName test_MRIdata_2_MCI_Normal.hdf5 --testidFileName test_MRIdata_2_MCI_Normal_id.txt --gpu_id 2


python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_2_AD_Normal.hdf5 --idFileNametrain train_MRIdata_2_AD_Normal_id.txt --testhdf5FileName test_MRIdata_2_AD_Normal.hdf5 --testidFileName test_MRIdata_2_AD_Normal_id.txt --gpu_id 2




cd /data1/wenyu/PycharmProjects/3dTripleGAN-Tensorflow

source activate cudnn712




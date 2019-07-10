python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_3_AD_MCI_Normal.hdf5 --idFileNametrain train_MRIdata_3_AD_MCI_Normal_id.txt --testhdf5FileName test_MRIdata_3_AD_MCI_Normal.hdf5 --testidFileName test_MRIdata_3_AD_MCI_Normal_id.txt --gpu_id 0 --batch_size_label 8 --batch_size_unlabel 8


python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_2_MCI_Normal.hdf5 --idFileNametrain train_MRIdata_2_MCI_Normal_id.txt --testhdf5FileName test_MRIdata_2_MCI_Normal.hdf5 --testidFileName test_MRIdata_2_MCI_Normal_id.txt --gpu_id 2 --batch_size_label 8 --batch_size_unlabel 8



python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_3_AD_MCI_Normal.hdf5 --idFileNametrain train_MRIdata_3_AD_MCI_Normal_id.txt --testhdf5FileName test_MRIdata_3_AD_MCI_Normal.hdf5 --testidFileName test_MRIdata_3_AD_MCI_Normal_id.txt --gpu_id 0 --batch_size_label 10 --batch_size_unlabel 4


python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_3_AD_MCI_Normal.hdf5 --idFileNametrain train_MRIdata_3_AD_MCI_Normal_id.txt --testhdf5FileName test_MRIdata_3_AD_MCI_Normal.hdf5 --testidFileName test_MRIdata_3_AD_MCI_Normal_id.txt --gpu_id 1 --batch_size_label 11 --batch_size_unlabel 3


python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_3_AD_MCI_Normal.hdf5 --idFileNametrain train_MRIdata_3_AD_MCI_Normal_id.txt --testhdf5FileName test_MRIdata_3_AD_MCI_Normal.hdf5 --testidFileName test_MRIdata_3_AD_MCI_Normal_id.txt --gpu_id 2 --batch_size_label 8 --batch_size_unlabel 8 --which_check_point 120


python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_3_AD_MCI_Normal.hdf5 --idFileNametrain train_MRIdata_3_AD_MCI_Normal_id.txt --testhdf5FileName test_MRIdata_3_AD_MCI_Normal.hdf5 --testidFileName test_MRIdata_3_AD_MCI_Normal_id.txt --gpu_id 3 --batch_size_label 10 --batch_size_unlabel 3







python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_2_AD_MCI.hdf5 --idFileNametrain train_MRIdata_2_AD_MCI_id.txt --testhdf5FileName test_MRIdata_2_AD_MCI.hdf5 --testidFileName test_MRIdata_2_AD_MCI_id.txt --gpu_id 1 --batch_size_label 8 --batch_size_unlabel 8
 

python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_2_MCI_Normal.hdf5 --idFileNametrain train_MRIdata_2_MCI_Normal_id.txt --testhdf5FileName test_MRIdata_2_MCI_Normal.hdf5 --testidFileName test_MRIdata_2_MCI_Normal_id.txt --gpu_id 2


python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_2_AD_Normal.hdf5 --idFileNametrain train_MRIdata_2_AD_Normal_id.txt --testhdf5FileName test_MRIdata_2_AD_Normal.hdf5 --testidFileName test_MRIdata_2_AD_Normal_id.txt --gpu_id 3




cd /home/reventon/wenyu/3dTripleGAN-TF
cd /home/reventon/wenyu/3dTripleGAN-TF

source activate cudnn712

python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_3_AD_MCI_Normal.hdf5 --idFileNametrain train_MRIdata_3_AD_MCI_Normal_id.txt --testhdf5FileName test_MRIdata_3_AD_MCI_Normal.hdf5 --testidFileName test_MRIdata_3_AD_MCI_Normal_id.txt --gpu_id 0,1

python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_2_AD_MCI.hdf5 --idFileNametrain train_MRIdata_2_AD_MCI_id.txt --testhdf5FileName test_MRIdata_2_AD_MCI.hdf5 --testidFileName test_MRIdata_2_AD_MCI_id.txt --gpu_id 1
 

python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_2_MCI_Normal.hdf5 --idFileNametrain train_MRIdata_2_MCI_Normal_id.txt --testhdf5FileName test_MRIdata_2_MCI_Normal.hdf5 --testidFileName test_MRIdata_2_MCI_Normal_id.txt --gpu_id 2


python GAN_train.py --train --test --hdf5FileNametrain train_MRIdata_2_AD_Normal.hdf5 --idFileNametrain train_MRIdata_2_AD_Normal_id.txt --testhdf5FileName test_MRIdata_2_AD_Normal.hdf5 --testidFileName test_MRIdata_2_AD_Normal_id.txt --gpu_id 3



 

cd /home/reventon/wenyu/3dDenseNetArchitecture28020

source activate cudnn712


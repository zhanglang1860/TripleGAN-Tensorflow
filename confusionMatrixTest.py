import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
# y_true = [0, 1, 2, 2, 2]
# y_pred = [0, 0, 2, 2, 1]

class_labels = []
input_file_name="MRIdata_3_AD_MCI_Normal.hdf5"
name_list= input_file_name.split("_")
if int(name_list[1])==3:
    class_labels.append(name_list[2])
    class_labels.append(name_list[3])
    last_class=name_list[4].split(".")
    class_labels.append(last_class[0])
else:
    class_labels.append(name_list[2])
    last_class = name_list[3].split(".")
    class_labels.append(last_class[0])







df = pd.read_csv('./data2/GANresults/allFolds_MRIdata_3_AD_MCI_Normal_lr_g_0.0025_d_0.01_update_G1D6_batchSize10_maxIteration1000depth30_growthRate12_reduce1.0_model_typeDenseNet_keepPro1.0.csv')

df.head()


cr = classification_report(df.actual_label.values, df.model_GAN.values, target_names=class_labels)
cm = np.array2string(confusion_matrix(df.actual_label.values, df.model_GAN.values))
f = open('./GANconfusionMatrixResults/2.txt', 'w')
f.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))
f.close()
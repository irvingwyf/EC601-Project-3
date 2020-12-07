# EC601-Project-3
This is actually the forth version of codes I am using in order to use SincNet to train the model.In order to operate, instead of using pytorch, I cited "grausof"'s work of keras-sincnet(forked under my main directory too).And then I also made many changes to the codes in order to make it workable using the current versions of the librarys, but due to the lack of compatibility of some core functions, I have to limit the versions and reinstall the libraries, and I abandonned another two versions of the codes, and here is the final ones I am using. I used Python 2.7, keras 2.1.6, Tensorflow 1.10.0, tqdm and pysoundfile to produce the training process. Thanks to grausof, all the models are converted from original torch networks to using tensorflow backend keras functions,
and the training process now can be operate on my macbook. For reproduce the original result, I used the TIMIT dataset to process the training, but I didn't finished all data, I stoped at 160 sets of sample, and it used more than 2 days to train, and the result is here:

epoch 0, loss_tr=5.542032 err_tr=0.984189 loss_te=4.996982 err_te=0.969038 err_te_snt=0.919913\
epoch 8, loss_tr=1.693487 err_tr=0.434424 loss_te=2.735717 err_te=0.612260 err_te_snt=0.069264\
epoch 16, loss_tr=0.861834 err_tr=0.229424 loss_te=2.465258 err_te=0.520276 err_te_snt=0.038240\
epoch 24, loss_tr=0.528619 err_tr=0.144375 loss_te=2.948707 err_te=0.534053 err_te_snt=0.062049\
epoch 32, loss_tr=0.362914 err_tr=0.100518 loss_te=2.530276 err_te=0.469060 err_te_snt=0.015152\
epoch 40, loss_tr=0.267921 err_tr=0.076445 loss_te=2.761606 err_te=0.464799 err_te_snt=0.023088\
epoch 48, loss_tr=0.215479 err_tr=0.061406 loss_te=2.737486 err_te=0.453493 err_te_snt=0.010823\
epoch 56, loss_tr=0.173690 err_tr=0.050732 loss_te=2.812427 err_te=0.443322 err_te_snt=0.011544\
epoch 64, loss_tr=0.145256 err_tr=0.043594 loss_te=2.917569 err_te=0.438507 err_te_snt=0.009380\
epoch 72, loss_tr=0.128894 err_tr=0.038486 loss_te=3.009008 err_te=0.438005 err_te_snt=0.019481\
epoch 80, loss_tr=0.111940 err_tr=0.033389 loss_te=2.925527 err_te=0.428739 err_te_snt=0.011544\
epoch 88, loss_tr=0.101788 err_tr=0.031016 loss_te=3.050507 err_te=0.438099 err_te_snt=0.011544\
epoch 96, loss_tr=0.089672 err_tr=0.027451 loss_te=3.212288 err_te=0.445679 err_te_snt=0.011544\
epoch 104, loss_tr=0.085366 err_tr=0.026445 loss_te=3.226385 err_te=0.431996 err_te_snt=0.012266\
epoch 112, loss_tr=0.077404 err_tr=0.023564 loss_te=3.341498 err_te=0.433145 err_te_snt=0.010101\
epoch 120, loss_tr=0.073497 err_tr=0.022861 loss_te=3.858381 err_te=0.472951 err_te_snt=0.028139\
epoch 128, loss_tr=0.067383 err_tr=0.020527 loss_te=3.474988 err_te=0.431545 err_te_snt=0.008658\
epoch 136, loss_tr=0.064087 err_tr=0.019961 loss_te=3.341287 err_te=0.436171 err_te_snt=0.007215\
epoch 144, loss_tr=0.062003 err_tr=0.019160 loss_te=3.412609 err_te=0.426363 err_te_snt=0.009380\
epoch 152, loss_tr=0.058740 err_tr=0.018281 loss_te=3.815553 err_te=0.443672 err_te_snt=0.010823\
epoch 160, loss_tr=0.055162 err_tr=0.017314 loss_te=3.784261 err_te=0.446239 err_te_snt=0.008658\

I am using the TIMIT dataset including 1500 data samples to train my own model in order to get a more accurate model for multi-speaker detection, and so far it has processed 107 samples and the terminal ouput is here:

Using TensorFlow backend.\
N_filt [80, 60, 60]\
N_filt len [251, 5, 5]\
FS 16000\
WLEN 3200\
I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA\
_________________________________________________________________\
Layer (type)                 Output Shape              Param #   \
=================================================================\
input_1 (InputLayer)         (None, 3200, 1)           0         \
_________________________________________________________________\
sinc_conv1d_1 (SincConv1D)   (None, 2950, 80)          160       \
_________________________________________________________________\
max_pooling1d_1 (MaxPooling1 (None, 983, 80)           0         \
_________________________________________________________________\
layer_norm_1 (LayerNorm)     (None, 983, 80)           160       \
_________________________________________________________________\
leaky_re_lu_1 (LeakyReLU)    (None, 983, 80)           0         \
_________________________________________________________________\
conv1d_1 (Conv1D)            (None, 979, 60)           24060     \
_________________________________________________________________\
max_pooling1d_2 (MaxPooling1 (None, 326, 60)           0         \
_________________________________________________________________\
layer_norm_2 (LayerNorm)     (None, 326, 60)           120       \
_________________________________________________________________\
leaky_re_lu_2 (LeakyReLU)    (None, 326, 60)           0         \
_________________________________________________________________\
conv1d_2 (Conv1D)            (None, 322, 60)           18060     \
_________________________________________________________________\
max_pooling1d_3 (MaxPooling1 (None, 107, 60)           0         \
_________________________________________________________________\
layer_norm_3 (LayerNorm)     (None, 107, 60)           120       \
_________________________________________________________________\
leaky_re_lu_3 (LeakyReLU)    (None, 107, 60)           0         \
_________________________________________________________________\
flatten_1 (Flatten)          (None, 6420)              0         \
_________________________________________________________________\
dense_1 (Dense)              (None, 2048)              13150208  \
_________________________________________________________________\
batch_normalization_1 (Batch (None, 2048)              8192      \
_________________________________________________________________\
leaky_re_lu_4 (LeakyReLU)    (None, 2048)              0         \
_________________________________________________________________\
dense_2 (Dense)              (None, 2048)              4196352   \
_________________________________________________________________\
batch_normalization_2 (Batch (None, 2048)              8192      \
_________________________________________________________________\
leaky_re_lu_5 (LeakyReLU)    (None, 2048)              0         \
_________________________________________________________________\
dense_3 (Dense)              (None, 2048)              4196352   \
_________________________________________________________________\
batch_normalization_3 (Batch (None, 2048)              8192      \
_________________________________________________________________\
leaky_re_lu_6 (LeakyReLU)    (None, 2048)              0         \
_________________________________________________________________\
dense_4 (Dense)              (None, 462)               946638    \
=================================================================\
Total params: 22,556,806\
Trainable params: 22,544,518\
Non-trainable params: 12,288\
_________________________________________________________________\
Epoch 1/1500\
800/800 [==============================] - 1200s 1s/step - loss: 5.2237 - acc: 0.0531\

Epoch 00001: saving model to data/checkpoints/SincNet.hdf5\
Valuating test set...\
Epoch: 0, acc_te: 0.0457332707087, acc_te_snt: 0.0678210678211\

Epoch 2/1500\
800/800 [==============================] - 1188s 1s/step - loss: 3.9660 - acc: 0.1706\

Epoch 00002: saving model to data/checkpoints/SincNet.hdf5\
Epoch 3/1500\
800/800 [==============================] - 2466s 3s/step - loss: 3.2777 - acc: 0.2760\

Epoch 00003: saving model to data/checkpoints/SincNet.hdf5\
Epoch 4/1500\
800/800 [==============================] - 1102s 1s/step - loss: 2.7901 - acc: 0.3637\

Epoch 00004: saving model to data/checkpoints/SincNet.hdf5\
Epoch 5/1500\
800/800 [==============================] - 1105s 1s/step - loss: 2.3763 - acc: 0.4438\

Epoch 00005: saving model to data/checkpoints/SincNet.hdf5\
Epoch 6/1500\
800/800 [==============================] - 1104s 1s/step - loss: 1.9574 - acc: 0.5187\

Epoch 00006: saving model to data/checkpoints/SincNet.hdf5\
Epoch 7/1500\
800/800 [==============================] - 1104s 1s/step - loss: 1.6482 - acc: 0.5808\

Epoch 00007: saving model to data/checkpoints/SincNet.hdf5\
Epoch 8/1500\
800/800 [==============================] - 1104s 1s/step - loss: 1.4374 - acc: 0.6285\

Epoch 00008: saving model to data/checkpoints/SincNet.hdf5\
Epoch 9/1500\
800/800 [==============================] - 1105s 1s/step - loss: 1.2890 - acc: 0.6639\

Epoch 00009: saving model to data/checkpoints/SincNet.hdf5\
Valuating test set...\
Epoch: 8, acc_te: 0.349521724837, acc_te_snt: 0.77417027417\

Epoch 10/1500\
800/800 [==============================] - 1107s 1s/step - loss: 1.1552 - acc: 0.6956\

Epoch 00010: saving model to data/checkpoints/SincNet.hdf5\
Epoch 11/1500\
800/800 [==============================] - 1105s 1s/step - loss: 1.0480 - acc: 0.7237\

Epoch 00011: saving model to data/checkpoints/SincNet.hdf5\
Epoch 12/1500\
800/800 [==============================] - 1101s 1s/step - loss: 0.9686 - acc: 0.7441\

Epoch 00012: saving model to data/checkpoints/SincNet.hdf5\
Epoch 13/1500\
800/800 [==============================] - 1106s 1s/step - loss: 0.8865 - acc: 0.7648\

Epoch 00013: saving model to data/checkpoints/SincNet.hdf5\
Epoch 14/1500\
800/800 [==============================] - 1107s 1s/step - loss: 0.8196 - acc: 0.7795\

Epoch 00014: saving model to data/checkpoints/SincNet.hdf5\
Epoch 15/1500\
800/800 [==============================] - 1108s 1s/step - loss: 0.7709 - acc: 0.7928\

Epoch 00015: saving model to data/checkpoints/SincNet.hdf5\
Epoch 16/1500\
800/800 [==============================] - 1108s 1s/step - loss: 0.7107 - acc: 0.8079\

Epoch 00016: saving model to data/checkpoints/SincNet.hdf5\
Epoch 17/1500\
800/800 [==============================] - 1106s 1s/step - loss: 0.6672 - acc: 0.8202\

Epoch 00017: saving model to data/checkpoints/SincNet.hdf5\
Valuating test set...\
Epoch: 16, acc_te: 0.448180459272, acc_te_snt: 0.91341991342\

Epoch 18/1500\
800/800 [==============================] - 1105s 1s/step - loss: 0.6273 - acc: 0.8296\

Epoch 00018: saving model to data/checkpoints/SincNet.hdf5\
Epoch 19/1500\
800/800 [==============================] - 1130s 1s/step - loss: 0.5970 - acc: 0.8386\

Epoch 00019: saving model to data/checkpoints/SincNet.hdf5\
Epoch 20/1500\
800/800 [==============================] - 1123s 1s/step - loss: 0.5568 - acc: 0.8476\

Epoch 00020: saving model to data/checkpoints/SincNet.hdf5\
Epoch 21/1500\
800/800 [==============================] - 1119s 1s/step - loss: 0.5238 - acc: 0.8558\

Epoch 00021: saving model to data/checkpoints/SincNet.hdf5\
Epoch 22/1500\
800/800 [==============================] - 1125s 1s/step - loss: 0.5019 - acc: 0.8617\

Epoch 00022: saving model to data/checkpoints/SincNet.hdf5\
Epoch 23/1500\
800/800 [==============================] - 1123s 1s/step - loss: 0.4803 - acc: 0.8679\

Epoch 00023: saving model to data/checkpoints/SincNet.hdf5\
Epoch 24/1500\
800/800 [==============================] - 1128s 1s/step - loss: 0.4588 - acc: 0.8730\

Epoch 00024: saving model to data/checkpoints/SincNet.hdf5\
Epoch 25/1500\
800/800 [==============================] - 1126s 1s/step - loss: 0.4385 - acc: 0.8781\

Epoch 00025: saving model to data/checkpoints/SincNet.hdf5\
Valuating test set...\
Epoch: 24, acc_te: 0.479286512416, acc_te_snt: 0.940115440115\

Epoch 26/1500\
800/800 [==============================] - 1132s 1s/step - loss: 0.4223 - acc: 0.8815\

Epoch 00026: saving model to data/checkpoints/SincNet.hdf5\
Epoch 27/1500\
800/800 [==============================] - 1130s 1s/step - loss: 0.4004 - acc: 0.8884\

Epoch 00027: saving model to data/checkpoints/SincNet.hdf5\
Epoch 28/1500\
800/800 [==============================] - 1132s 1s/step - loss: 0.3839 - acc: 0.8932\

Epoch 00028: saving model to data/checkpoints/SincNet.hdf5\
Epoch 29/1500\
800/800 [==============================] - 1132s 1s/step - loss: 0.3711 - acc: 0.8960\

Epoch 00029: saving model to data/checkpoints/SincNet.hdf5\
Epoch 30/1500\
800/800 [==============================] - 1133s 1s/step - loss: 0.3558 - acc: 0.8999\

Epoch 00030: saving model to data/checkpoints/SincNet.hdf5\
Epoch 31/1500\
800/800 [==============================] - 1134s 1s/step - loss: 0.3450 - acc: 0.9025\

Epoch 00031: saving model to data/checkpoints/SincNet.hdf5\
Epoch 32/1500\
800/800 [==============================] - 1135s 1s/step - loss: 0.3349 - acc: 0.9058\

Epoch 00032: saving model to data/checkpoints/SincNet.hdf5\
Epoch 33/1500\
800/800 [==============================] - 1128s 1s/step - loss: 0.3173 - acc: 0.9101\

Epoch 00033: saving model to data/checkpoints/SincNet.hdf5\
Valuating test set...\
Epoch: 32, acc_te: 0.488884881326, acc_te_snt: 0.943722943723\

Epoch 34/1500\
800/800 [==============================] - 1121s 1s/step - loss: 0.3100 - acc: 0.9131\

Epoch 00034: saving model to data/checkpoints/SincNet.hdf5\
Epoch 35/1500\
800/800 [==============================] - 1119s 1s/step - loss: 0.3016 - acc: 0.9141\

Epoch 00035: saving model to data/checkpoints/SincNet.hdf5\
Epoch 36/1500\
800/800 [==============================] - 1118s 1s/step - loss: 0.2972 - acc: 0.9155\

Epoch 00036: saving model to data/checkpoints/SincNet.hdf5\
Epoch 37/1500\
800/800 [==============================] - 1121s 1s/step - loss: 0.2869 - acc: 0.9187\

Epoch 00037: saving model to data/checkpoints/SincNet.hdf5\
Epoch 38/1500\
800/800 [==============================] - 1123s 1s/step - loss: 0.2749 - acc: 0.9208\

Epoch 00038: saving model to data/checkpoints/SincNet.hdf5\
Epoch 39/1500\
800/800 [==============================] - 1122s 1s/step - loss: 0.2675 - acc: 0.9238\

Epoch 00039: saving model to data/checkpoints/SincNet.hdf5\
Epoch 40/1500\
800/800 [==============================] - 1124s 1s/step - loss: 0.2599 - acc: 0.9246\

Epoch 00040: saving model to data/checkpoints/SincNet.hdf5\
Epoch 41/1500\
800/800 [==============================] - 1117s 1s/step - loss: 0.2506 - acc: 0.9268\

Epoch 00041: saving model to data/checkpoints/SincNet.hdf5\
Valuating test set...\
Epoch: 40, acc_te: 0.50212827652, acc_te_snt: 0.960317460317\

Epoch 42/1500\
800/800 [==============================] - 1129s 1s/step - loss: 0.2445 - acc: 0.9293\

Epoch 00042: saving model to data/checkpoints/SincNet.hdf5\
Epoch 43/1500\
800/800 [==============================] - 1136s 1s/step - loss: 0.2379 - acc: 0.9315\

Epoch 00043: saving model to data/checkpoints/SincNet.hdf5\
Epoch 44/1500\
800/800 [==============================] - 1131s 1s/step - loss: 0.2296 - acc: 0.9335\

Epoch 00044: saving model to data/checkpoints/SincNet.hdf5\
Epoch 45/1500\
800/800 [==============================] - 1135s 1s/step - loss: 0.2312 - acc: 0.9333\

Epoch 00045: saving model to data/checkpoints/SincNet.hdf5\
Epoch 46/1500\
800/800 [==============================] - 1153s 1s/step - loss: 0.2188 - acc: 0.9361\

Epoch 00046: saving model to data/checkpoints/SincNet.hdf5\
Epoch 47/1500\
800/800 [==============================] - 1260s 2s/step - loss: 0.2185 - acc: 0.9360\

Epoch 00047: saving model to data/checkpoints/SincNet.hdf5\
Epoch 48/1500\
800/800 [==============================] - 1239s 2s/step - loss: 0.2132 - acc: 0.9378\

Epoch 00048: saving model to data/checkpoints/SincNet.hdf5\
Epoch 49/1500\
800/800 [==============================] - 1325s 2s/step - loss: 0.2112 - acc: 0.9378\

Epoch 00049: saving model to data/checkpoints/SincNet.hdf5\
Valuating test set...\
Epoch: 48, acc_te: 0.476170480365, acc_te_snt: 0.945887445887\

Epoch 50/1500\
800/800 [==============================] - 1199s 1s/step - loss: 0.2003 - acc: 0.9402\

Epoch 00050: saving model to data/checkpoints/SincNet.hdf5\
Epoch 51/1500\
800/800 [==============================] - 1169s 1s/step - loss: 0.2017 - acc: 0.9416\

Epoch 00051: saving model to data/checkpoints/SincNet.hdf5\
Epoch 52/1500\
800/800 [==============================] - 1167s 1s/step - loss: 0.1969 - acc: 0.9423\

Epoch 00052: saving model to data/checkpoints/SincNet.hdf5\
Epoch 53/1500\
800/800 [==============================] - 1170s 1s/step - loss: 0.1873 - acc: 0.9454\

Epoch 00053: saving model to data/checkpoints/SincNet.hdf5\
Epoch 54/1500\
800/800 [==============================] - 1170s 1s/step - loss: 0.1878 - acc: 0.9447\

Epoch 00054: saving model to data/checkpoints/SincNet.hdf5\
Epoch 55/1500\
800/800 [==============================] - 1164s 1s/step - loss: 0.1850 - acc: 0.9458\

Epoch 00055: saving model to data/checkpoints/SincNet.hdf5\
Epoch 56/1500\
800/800 [==============================] - 1171s 1s/step - loss: 0.1754 - acc: 0.9476\

Epoch 00056: saving model to data/checkpoints/SincNet.hdf5\
Epoch 57/1500\
800/800 [==============================] - 1169s 1s/step - loss: 0.1797 - acc: 0.9472\

Epoch 00057: saving model to data/checkpoints/SincNet.hdf5\
Valuating test set...\
Epoch: 56, acc_te: 0.485220178175, acc_te_snt: 0.952380952381\

Epoch 58/1500\
800/800 [==============================] - 1167s 1s/step - loss: 0.1713 - acc: 0.9491\

Epoch 00058: saving model to data/checkpoints/SincNet.hdf5\
Epoch 59/1500\
800/800 [==============================] - 1164s 1s/step - loss: 0.1706 - acc: 0.9492\

Epoch 00059: saving model to data/checkpoints/SincNet.hdf5\
Epoch 60/1500\
800/800 [==============================] - 1166s 1s/step - loss: 0.1692 - acc: 0.9500\

Epoch 00060: saving model to data/checkpoints/SincNet.hdf5\
Epoch 61/1500\
800/800 [==============================] - 1167s 1s/step - loss: 0.1640 - acc: 0.9511\

Epoch 00061: saving model to data/checkpoints/SincNet.hdf5\
Epoch 62/1500\
800/800 [==============================] - 1165s 1s/step - loss: 0.1606 - acc: 0.9519\

Epoch 00062: saving model to data/checkpoints/SincNet.hdf5\
Epoch 63/1500\
800/800 [==============================] - 1164s 1s/step - loss: 0.1559 - acc: 0.9531\

Epoch 00063: saving model to data/checkpoints/SincNet.hdf5\
Epoch 64/1500\
800/800 [==============================] - 1168s 1s/step - loss: 0.1594 - acc: 0.9527\

Epoch 00064: saving model to data/checkpoints/SincNet.hdf5\
Epoch 65/1500\
800/800 [==============================] - 1160s 1s/step - loss: 0.1531 - acc: 0.9540\

Epoch 00065: saving model to data/checkpoints/SincNet.hdf5\
Valuating test set...\
Epoch: 64, acc_te: 0.479468363402, acc_te_snt: 0.948773448773\

Epoch 66/1500\
800/800 [==============================] - 1161s 1s/step - loss: 0.1511 - acc: 0.9546\

Epoch 00066: saving model to data/checkpoints/SincNet.hdf5\
Epoch 67/1500\
800/800 [==============================] - 1168s 1s/step - loss: 0.1456 - acc: 0.9562\

Epoch 00067: saving model to data/checkpoints/SincNet.hdf5\
Epoch 68/1500\
800/800 [==============================] - 1162s 1s/step - loss: 0.1472 - acc: 0.9559\

Epoch 00068: saving model to data/checkpoints/SincNet.hdf5\
Epoch 69/1500\
800/800 [==============================] - 1166s 1s/step - loss: 0.1431 - acc: 0.9574\

Epoch 00069: saving model to data/checkpoints/SincNet.hdf5\
Epoch 70/1500\
800/800 [==============================] - 1163s 1s/step - loss: 0.1450 - acc: 0.9564\

Epoch 00070: saving model to data/checkpoints/SincNet.hdf5\
Epoch 71/1500\
800/800 [==============================] - 1167s 1s/step - loss: 0.1392 - acc: 0.9584\

Epoch 00071: saving model to data/checkpoints/SincNet.hdf5\
Epoch 72/1500\
800/800 [==============================] - 1134s 1s/step - loss: 0.1387 - acc: 0.9581\

Epoch 00072: saving model to data/checkpoints/SincNet.hdf5\
Epoch 73/1500\
800/800 [==============================] - 1134s 1s/step - loss: 0.1399 - acc: 0.9587\

Epoch 00073: saving model to data/checkpoints/SincNet.hdf5\
Valuating test set...\
Epoch: 72, acc_te: 0.494170975946, acc_te_snt: 0.961038961039\

Epoch 74/1500\
800/800 [==============================] - 1131s 1s/step - loss: 0.1380 - acc: 0.9582\

Epoch 00074: saving model to data/checkpoints/SincNet.hdf5\
Epoch 75/1500\
800/800 [==============================] - 1121s 1s/step - loss: 0.1349 - acc: 0.9593\

Epoch 00075: saving model to data/checkpoints/SincNet.hdf5\
Epoch 76/1500\
800/800 [==============================] - 1127s 1s/step - loss: 0.1348 - acc: 0.9597\

Epoch 00076: saving model to data/checkpoints/SincNet.hdf5\
Epoch 77/1500\
800/800 [==============================] - 1132s 1s/step - loss: 0.1294 - acc: 0.9611\

Epoch 00077: saving model to data/checkpoints/SincNet.hdf5\
Epoch 78/1500\
800/800 [==============================] - 1124s 1s/step - loss: 0.1257 - acc: 0.9618\

Epoch 00078: saving model to data/checkpoints/SincNet.hdf5\
Epoch 79/1500\
800/800 [==============================] - 1132s 1s/step - loss: 0.1295 - acc: 0.9616\

Epoch 00079: saving model to data/checkpoints/SincNet.hdf5\
Epoch 80/1500\
800/800 [==============================] - 1134s 1s/step - loss: 0.1263 - acc: 0.9619\

Epoch 00080: saving model to data/checkpoints/SincNet.hdf5\
Epoch 81/1500\
800/800 [==============================] - 1135s 1s/step - loss: 0.1221 - acc: 0.9639\

Epoch 00081: saving model to data/checkpoints/SincNet.hdf5\
Valuating test set...\
Epoch: 80, acc_te: 0.501161965064, acc_te_snt: 0.962481962482\

Epoch 82/1500\
800/800 [==============================] - 1135s 1s/step - loss: 0.1238 - acc: 0.9624\

Epoch 00082: saving model to data/checkpoints/SincNet.hdf5\
Epoch 83/1500\
800/800 [==============================] - 1135s 1s/step - loss: 0.1212 - acc: 0.9632\

Epoch 00083: saving model to data/checkpoints/SincNet.hdf5\
Epoch 84/1500\
800/800 [==============================] - 1121s 1s/step - loss: 0.1255 - acc: 0.9628\

Epoch 00084: saving model to data/checkpoints/SincNet.hdf5\
Epoch 85/1500\
800/800 [==============================] - 1135s 1s/step - loss: 0.1192 - acc: 0.9636\

Epoch 00085: saving model to data/checkpoints/SincNet.hdf5\
Epoch 86/1500\
800/800 [==============================] - 1136s 1s/step - loss: 0.1161 - acc: 0.9656\

Epoch 00086: saving model to data/checkpoints/SincNet.hdf5\
Epoch 87/1500\
800/800 [==============================] - 1136s 1s/step - loss: 0.1177 - acc: 0.9641\

Epoch 00087: saving model to data/checkpoints/SincNet.hdf5\
Epoch 88/1500\
800/800 [==============================] - 1136s 1s/step - loss: 0.1156 - acc: 0.9651\

Epoch 00088: saving model to data/checkpoints/SincNet.hdf5\
Epoch 89/1500\
800/800 [==============================] - 1138s 1s/step - loss: 0.1128 - acc: 0.9659\

Epoch 00089: saving model to data/checkpoints/SincNet.hdf5\
Valuating test set...\
Epoch: 88, acc_te: 0.513393178749, acc_te_snt: 0.969696969697\

Epoch 90/1500\
800/800 [==============================] - 1134s 1s/step - loss: 0.1106 - acc: 0.9667\

Epoch 00090: saving model to data/checkpoints/SincNet.hdf5\
Epoch 91/1500\
800/800 [==============================] - 1139s 1s/step - loss: 0.1110 - acc: 0.9665\

Epoch 00091: saving model to data/checkpoints/SincNet.hdf5\
Epoch 92/1500\
800/800 [==============================] - 1139s 1s/step - loss: 0.1121 - acc: 0.9659\

Epoch 00092: saving model to data/checkpoints/SincNet.hdf5\
Epoch 93/1500\
800/800 [==============================] - 1137s 1s/step - loss: 0.1109 - acc: 0.9671\

Epoch 00093: saving model to data/checkpoints/SincNet.hdf5\
Epoch 94/1500\
800/800 [==============================] - 1137s 1s/step - loss: 0.1100 - acc: 0.9674\

Epoch 00094: saving model to data/checkpoints/SincNet.hdf5\
Epoch 95/1500\
800/800 [==============================] - 1141s 1s/step - loss: 0.1095 - acc: 0.9672\

Epoch 00095: saving model to data/checkpoints/SincNet.hdf5\
Epoch 96/1500\
800/800 [==============================] - 1139s 1s/step - loss: 0.1074 - acc: 0.9675\

Epoch 00096: saving model to data/checkpoints/SincNet.hdf5\
Epoch 97/1500\
800/800 [==============================] - 1141s 1s/step - loss: 0.1047 - acc: 0.9683\

Epoch 00097: saving model to data/checkpoints/SincNet.hdf5\
Valuating test set...\
Epoch: 96, acc_te: 0.511582946234, acc_te_snt: 0.968975468975\

Epoch 98/1500\
800/800 [==============================] - 1145s 1s/step - loss: 0.1034 - acc: 0.9684\

Epoch 00098: saving model to data/checkpoints/SincNet.hdf5\
Epoch 99/1500\
800/800 [==============================] - 1146s 1s/step - loss: 0.1046 - acc: 0.9686\

Epoch 00099: saving model to data/checkpoints/SincNet.hdf5\
Epoch 100/1500\
800/800 [==============================] - 1144s 1s/step - loss: 0.1047 - acc: 0.9682\

Epoch 00100: saving model to data/checkpoints/SincNet.hdf5\
Epoch 101/1500\
800/800 [==============================] - 1145s 1s/step - loss: 0.1018 - acc: 0.9690\

Epoch 00101: saving model to data/checkpoints/SincNet.hdf5\
Epoch 102/1500\
800/800 [==============================] - 1141s 1s/step - loss: 0.0992 - acc: 0.9695\

Epoch 00102: saving model to data/checkpoints/SincNet.hdf5\
Epoch 103/1500\
800/800 [==============================] - 1140s 1s/step - loss: 0.1002 - acc: 0.9692\

Epoch 00103: saving model to data/checkpoints/SincNet.hdf5\
Epoch 104/1500\
800/800 [==============================] - 1146s 1s/step - loss: 0.0966 - acc: 0.9700\

Epoch 00104: saving model to data/checkpoints/SincNet.hdf5\
Epoch 105/1500\
800/800 [==============================] - 1141s 1s/step - loss: 0.0980 - acc: 0.9700\

Epoch 00105: saving model to data/checkpoints/SincNet.hdf5\
Valuating test set...\
Epoch: 104, acc_te: 0.505333990684, acc_te_snt: 0.969696969697\

Epoch 106/1500\
800/800 [==============================] - 1259s 2s/step - loss: 0.0968 - acc: 0.9706\

Epoch 00106: saving model to data/checkpoints/SincNet.hdf5\
Epoch 107/1500\
800/800 [==============================] - 1277s 2s/step - loss: 0.0962 - acc: 0.9711\

Epoch 00107: saving model to data/checkpoints/SincNet.hdf5\

Since the training is still processing, the model and the result output files are not ready to show here, but the reproduce and training exercises are being performed, and since in the term project, we will not using the SincNet to train the model for us to produce the detection result, I will not focus on this too much now, but later on once the training process is done, I will use the trained model to tryout to see if it is better than the model trained by using the original dataset refered by SincNet.

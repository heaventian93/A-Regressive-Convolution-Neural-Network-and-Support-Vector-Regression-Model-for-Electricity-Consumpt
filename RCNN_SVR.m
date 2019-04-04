function [trainingFeatures,testFeatures,MdlStd,YFit]=RCNN_SVR(x_train,x_test,y_train,y_test)
%% Input: 
%       x_train:  D*1*1*N1 (in paper: 8*1*1*50)
%       x_test:   D*1*1*N2 (in paper: 8*1*1*12)
%       y_train:  1*N1     (in paper: 1*50    )
%       y_test:   1*N2     (in paper: 1*12    )
%% OutPut: 
% trainingFeatures: N3*N1  (in paper: 50*50   )
% testFeatures:     N3*N2  (in paper: 50*12   )
% MdlStd: training SVR classifier
% YFit: predicting results
% Suggest install Matlab 2017b or later

% Please cite our paper if you use our code. Thanks!
% Zhang, Youshan, & Li, Qi. (2019, March). A Regressive Convolution Neural Network
% and Support Vector Regression Model for Electricity Consumption Forecasting. 
% In Future of Information and Communication Conference (pp. 33-45). Springer, Cham.

%% RCNN feature extraction
% Fine tuning the layers using your datasets
rng 'default';
layers = [ 
          imageInputLayer([8 1 1]); % change 8 to your own dataset
          convolution2dLayer(1,20);
          reluLayer();
          crossChannelNormalizationLayer(1)
          maxPooling2dLayer(1,'Stride',2);
          convolution2dLayer(1,25);
          maxPooling2dLayer(1,'Stride',2);
          convolution2dLayer(1,50);
          maxPooling2dLayer(1,'Stride',2);
          dropoutLayer();
          fullyConnectedLayer(1);
          regressionLayer];
 
options = trainingOptions('sgdm','MaxEpochs',200,'InitialLearnRate',0.00001);
net = trainNetwork(x_train,y_train',layers,options);  
trainingFeatures=activations(net, x_train,10);
trainingFeatures=double(reshape(trainingFeatures,[size(trainingFeatures,3),size(trainingFeatures,4)]));
testFeatures=activations(net, x_test,10);
testFeatures=double(reshape(testFeatures,[size(testFeatures,3),size(testFeatures,4)]));

%% SVR Prediction
MdlStd = fitrsvm(trainingFeatures,y_train,'Standardize',true);
YFit = predict(MdlStd,testFeatures');
E = y_test' - YFit; 

MSE=mse(E)




require 'nn'

local vgg = nn.Sequential()
local MaxPooling = nn.SpatialMaxPooling

vgg:add(nn.SpatialConvolution(3, 64, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(64,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.3))

vgg:add(nn.SpatialConvolution(64, 64, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(64,1e-3))
vgg:add(nn.ReLU(true))

vgg:add(MaxPooling(2,2,2,2):ceil())

vgg:add(nn.SpatialConvolution(64, 128, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(128,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))

vgg:add(nn.SpatialConvolution(128, 128, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(128,1e-3))
vgg:add(nn.ReLU(true))

vgg:add(MaxPooling(2,2,2,2):ceil())

vgg:add(nn.SpatialConvolution(128, 256, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(256,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))

vgg:add(nn.SpatialConvolution(256, 256, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(256,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))


vgg:add(nn.SpatialConvolution(256, 256, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(256,1e-3))
vgg:add(nn.ReLU(true))

vgg:add(MaxPooling(2,2,2,2):ceil())

vgg:add(nn.SpatialConvolution(256, 512, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(512,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))

vgg:add(nn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(512,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))

vgg:add(nn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(512,1e-3))
vgg:add(nn.ReLU(true))

vgg:add(MaxPooling(2,2,2,2):ceil())
vgg:add(nn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(512,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))
vgg:add(nn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(512,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.4))
vgg:add(nn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(512,1e-3))
vgg:add(nn.ReLU(true))

vgg:add(MaxPooling(2,2,2,2):ceil())

vgg:add(nn.View(512))
vgg:add(nn.Dropout(0.5))
vgg:add(nn.Linear(512,512))
vgg:add(nn.BatchNormalization(512))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.5))
vgg:add(nn.Linear(512,43))

return vgg
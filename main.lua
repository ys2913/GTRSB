require 'torch'
require 'optim'
require 'os'
require 'optim'
require 'xlua'

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)
torch.setdefaulttensortype('torch.DoubleTensor')

if(opt.cuda=='true') then
    require 'cunn'
    require 'cudnn'
    cutorch.manualSeedAll(opt.manualSeed)
end

--Input Size of image
local WIDTH, HEIGHT = 32, 32
--Region of interest for cropping an image
local ROI_X1, ROI_X2, ROI_Y1, ROI_Y2 = 0, 0, 0, 0
local DATA_PATH = (opt.data ~= '' and opt.data or './data/')
--Loading Train and Test Meta Data
local trainData = torch.load(DATA_PATH..'train.t7')
local testData = torch.load(DATA_PATH..'test.t7')

local classfreq = {}-- for calculating frequency og training samples of each class
local valfreq = {}

-- Initializing Class frequency Frequency
for i=1,43 do
    classfreq[i]=0
    valfreq[i]=0
end

--For loading Train and Test images
local trainImages = torch.Tensor(trainData:size(1),3,WIDTH,HEIGHT)
local trainlabels = {}
local testImages = torch.Tensor(testData:size(1),3,WIDTH,HEIGHT)
local testImages_size = 1
local trainImages_size = 1
--Stores indexes of trainImages in their respective class ID
--example: trainImages[k] has classID=1, so act_train[1] is a table containing value k
local act_train = {}

for i=1,43 do
    act_train[i]={}
end

--Stores indexes for training and validation data
local trainingData = {}
local validationData = {}

function resize(img)
    return image.scale(img, WIDTH,HEIGHT)
end

function crop(img)
    return image.crop(img, ROI_X1, ROI_Y1, ROI_X2, ROI_Y2)
end

--Loading Training Data to trainImages and testImages
----------------------------------------------------------------------------------------------
print("Loading Train Data")
----------------------------------------------------------------------------------------------
for idx = 1, trainData:size(1) do
    r = trainData[idx]
    classId, track, file = r[9], r[1], r[2]
    ROI_X1, ROI_X2, ROI_Y1, ROI_Y2 = r[5], r[7], r[6], r[8]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    img = image.load(DATA_PATH .. '/train_images/'..file)
    --Uncomment for cropping the ROIs from the given image
    img = crop(img)
    img = resize(img)
    classfreq[classId+1] = classfreq[classId+1]+1   --Updating frequency of each class in Training Data
    table.insert(act_train[classId+1],idx)   --Inserting index corresponding to the image location in trainImages
    trainImages[trainImages_size] = img     
    trainlabels[trainImages_size] = classId+1
    trainImages_size = trainImages_size + 1
end

----------------------------------------------------------------------------------------------
print("Loading Test Data")
----------------------------------------------------------------------------------------------
for idx = 1, testData:size(1) do
    r = testData[idx]
    ROI_X1, ROI_X2, ROI_Y1, ROI_Y2 = r[4], r[6], r[5], r[7]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    img = image.load(file)
    --Uncomment for cropping the ROIs from the given image
    img = crop(img)
    img = resize(img)
    testImages[testImages_size] = img
    testImages_size =testImages_size +1
end


--Normalizing Color from Train and Test Data
----------------------------------------------------------------------------------------------
print("Color Normalization Started")
----------------------------------------------------------------------------------------------
local mean = {}
local std = {}

mean = {}
std = {}

for i =1,3 do
    mean[i] = trainImages[{ {},i,{},{} }]:mean()
    std[i] = trainImages[{ {},i,{},{} }]:std()
    trainImages[{ {},i,{},{} }]:add(-mean[i])
    trainImages[{ {},i,{},{} }]:div(std[i])
    testImages[{ {},i,{},{} }]:add(-mean[i])
    testImages[{ {},i,{},{} }]:div(std[i])
end

----------------------------------------------------------------------------------------------
print("Splitting Training Data into Train and Validation Data")
----------------------------------------------------------------------------------------------
--removing Validation Data from Training Data, about 10% of each class

for i=1,43 do
    local sizeclass = classfreq[i]
    for j=1, torch.floor(sizeclass/10) do
        r=torch.floor(torch.uniform(1,classfreq[i]))
        table.insert(validationData,act_train[i][r])
        table.remove(act_train[i],r)
        classfreq[i]=classfreq[i]-1
    end
end
----------------------------------------------------------------------------------------------
print("Calculating Maximum frequency ")
----------------------------------------------------------------------------------------------
local maxfreq=0
for i=1,43 do
    if classfreq[i] > maxfreq then
        maxfreq = classfreq[i]
    end
end
print("Maxfreq", maxfreq)

local actual_distribution = torch.Tensor(classfreq)
local current_distribution = torch.Tensor(43):fill(actual_distribution:max())
local initial_distribution = current_distribution:clone()

--Uncomment for Upscaling
--[[
----------------------------------------------------------------------------------------------
print("Upscaling Per Class Training Data To Maximum Frequency")
----------------------------------------------------------------------------------------------
-- Upscaling class training examples to Maximum Frequency

for i=1,43 do
    local sizeclass = #act_train[i]
    local num = sizeclass
    
    while sizeclass < current_distribution[i] do
        a=torch.floor(torch.uniform(1,num))
        act_train[i][sizeclass+1]=act_train[i][a]
        sizeclass=sizeclass+1
    end
end
--]]

-- Flatenning act_train to a 1 Dimensional tensor 

for i=1,43 do
    local sizeclass = #act_train[i]
    for j=1, sizeclass do
        table.insert(trainingData,act_train[i][j])
    end
end

trainingData = torch.Tensor(trainingData)
validationData = torch.Tensor(validationData)
trainImages = torch.Tensor(trainImages)
testImages = torch.Tensor(testImages)

print("Training Data Size",#trainingData)
print("Validation Data Size",#validationData)
print("TrainImages Data Size",#trainImages)
print("testImages Data Size",#testImages)

--Converting RGB image to YUB
function convertYUV(img)
    return image.rgb2yuv(img)
end

--From Yann-Sermanet paper
function localnormalization(img)
    normalization = nn.SpatialContrastiveNormalization(3, image.gaussian(5))
    return normalization:forward(img)
end

--Adding Rotation Jitters
function rotate(img)
    local rand_angle = (torch.randn(1)*15*3.14/180)[1]
    return image.rotate(img, rand_angle)
end

--Adding Translation Jitters
function translate(img)
    local rand_position_x = (torch.randn(1)*2)[1]
    local rand_position_y = (torch.randn(1)*2)[1]
    return image.translate(img, rand_position_x, rand_position_y)
end

function scale(img)
    local rand_scale = 1
    ROI_X1, ROI_X2, ROI_Y1, ROI_Y2 = rand_scale*ROI_X1, rand_scale*ROI_X2, rand_scale*ROI_Y1, rand_scale*ROI_Y2
    return img
end

--Preprocessing image after resizing the image and color normalization in prior
function transformInput(inp)
    f = tnt.transform.compose{
        [1] = convertYUV,
        [2] = localnormalization
    }
    return f(inp)
end

--Preprocessing for images with Jitters 
function transformJitteredInput(inp)
    f = tnt.transform.compose{
        [1] = translate,
        [2] = rotate,
        [3] = convertYUV,
        [4] = localnormalization
    }
    return f(inp)
end

--Preprocessing image after resizing the image and color normalization in prior
function transformTestInput(inp)
    f = tnt.transform.compose{
        [1] = convertYUV,
        [2] = localnormalization
    }
    return f(inp)
end


function getValSample(idx)
    img = trainImages[validationData[idx]]
    out = transformInput(img)
    return out
end

function getValLabel(idx)
    a=trainlabels[validationData[idx]]
    return torch.LongTensor{a}
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    --file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    return transformTestInput( testImages[idx])
end


----------------------------------------------------------------------------------------------
print("Forming Validation, Test Dataset")
----------------------------------------------------------------------------------------------

testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input = getTestSample(testData, idx),
            sampleId = torch.LongTensor{testData[idx][1]}
        }
    end
}

valiDataset = tnt.ShuffleDataset{
        dataset = tnt.ListDataset{
            list = torch.range(1, validationData:size(1)):long(),
            load = function(idx)
                return {
                    input =  getValSample(idx),
                    target = getValLabel(idx)
                }
            end
        }
    }


--Forming Iterator for the Validation Set
function getValidationIterator()
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = opt.batchsize,
            dataset = tnt.ListDataset{
                        list = torch.range(1, torch.floor(validationData:size(1)/10)):long(),
                        load = function(idx)
                        return {
                            input = getValSample(idx),
                            target = getValLabel(idx)
                        }
                        end
            }
        }
    }
end

--Implemented Parallel Iterator for Training Dataset

function getTrainIterator()
    return tnt.ParallelDatasetIterator{
         init    = function() 
         local tnt = require 'torchnet'  
            end,
         nthread = opt.nThreads,

         closure = function(dataset)
            local image = require 'image'
            local resize = function(img)
                return image.scale(img, WIDTH,HEIGHT)
            end

            local crop = function(img)
                return image.crop(img, ROI_X1, ROI_Y1, ROI_X2, ROI_Y2)
            end
            
            local convertYUV = function (img)
                return image.rgb2yuv(img)
            end

            local localnormalization = function (img)
                normalization = nn.SpatialContrastiveNormalization(3, image.gaussian(5))
                return normalization:forward(img)
            end

            local rotate = function (img)
                local rand_angle = (torch.randn(1)*15*3.14/180)[1]
                return image.rotate(img, rand_angle)
            end

            local translate = function (img)
                local rand_position_x = (torch.randn(1)*2)[1]
                local rand_position_y = (torch.randn(1)*2)[1]
                return image.translate(img, rand_position_x, rand_position_y)
            end

            local scale = function (img)
                local rand_scale = 1
                ROI_X1, ROI_X2, ROI_Y1, ROI_Y2 = rand_scale*ROI_X1, rand_scale*ROI_X2, rand_scale*ROI_Y1, rand_scale*ROI_Y2
                return img
            end

            local transformInput = function(inp)
                f = tnt.transform.compose{
                    [1] = convertYUV,
                    [2] = localnormalization
                }
                return f(inp)
            end

            local transformJitteredInput = function(inp)
                f = tnt.transform.compose{
                    [1] = translate,
                    [2] = rotate,
                    [3] = convertYUV,
                    [4] = localnormalization
                }
                return f(inp)
            end

            local getTrainSample = function(dataset, idx)
                nidx = idx%trainingData:size(1)
                if (nidx == 0) then
                    nidx=trainingData:size(1)
                end
                img = trainImages[trainingData[nidx]]
                if (idx > trainingData:size(1)) then
                    out = transformJitteredInput(img)
                else
                    out = transformInput(img)
                end
                return out
            end

            local getTrainLabel = function(dataset, idx)
                nidx = idx%trainingData:size(1)

                if (nidx == 0) then
                    nidx=trainingData:size(1)
                end
                return torch.LongTensor{trainlabels[trainingData[nidx]]}
            end

            return tnt.BatchDataset{
                        batchsize = opt.batchsize,
                        dataset = tnt.ShuffleDataset{
                            dataset = tnt.ListDataset{
                                list = torch.range(1, torch.floor(trainingData:size(1)/50)):long(),
                                load = function(idx)
                                    return {
                                        input = getTrainSample(trainImages, idx),
                                        target = getTrainLabel(trainImages, idx)
                                    }
                                end
                            }
                        }
                    }
            end
     }
end

--For Test Iterator
function getIterator(dataset)
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = opt.batchsize,
            dataset = dataset
        }
    }
end



local model = require("models/".. opt.model)
local criterion = nn.CrossEntropyCriterion()

if(opt.cuda=='true') then
    model = model:cuda()
    criterion = criterion:cuda()
end   

local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1

engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
    else
        mode = 'Val'
    end
end


engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    if opt.verbose == 'true' then
        print(string.format("%s Batch: %d; avg. loss: %2.4f; avg. error: %2.4f",
                mode, batch , meter:value(), clerr:value{k = 1}))
    else
        --xlua.progress(batch)
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

--onSample function to convert to cuda tensor for using GPU

if(opt.cuda=='true') then
    local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
    engine.hooks.onSample = function(state)
        igpu:resize(state.sample.input:size()):copy(state.sample.input)
        if state.sample.target ~= nil then
            tgpu:resize(state.sample.target:size()):copy(state.sample.target)
        else
        tgpu:resize(state.sample.sampleId:size()):copy(state.sample.sampleId)
        end
        state.sample.input  = igpu
        state.sample.target = tgpu
    end
end

engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))
end

local epoch = 1

while epoch <= opt.nEpochs do

    engine:train{
        network = model,
        criterion = criterion,
        iterator = getTrainIterator(),
        optimMethod = optim.sgd,
        maxepoch = 1,
        config = {
            learningRate = opt.LR,
            momentum = opt.momentum
        }
    }

    engine:test{
        network = model,
        criterion = criterion,
        iterator = getValidationIterator()
    }
    clmodel = model:clone()
    clmodel:clearState()
    torch.save('cloned'..opt.file..'.t7',clmodel)
    print('Done with Epoch '..tostring(epoch))
    epoch = epoch + 1
end

local submission = assert(io.open(opt.logDir .. "/" .. opt.file .. ".csv", "w"))
submission:write("Filename,ClassId\n")
batch = 1

--[[
--  This piece of code creates the submission
--  file that has to be uploaded in kaggle.
--]]
engine.hooks.onForward = function(state)
local fileNames  = state.sample.sampleId
    local _, pred = state.network.output:max(2)
    pred = pred - 1
    for i = 1, pred:size(1) do
        submission:write(string.format("%05d,%d\n", fileNames[i][1], pred[i][1]))
    end
    xlua.progress(batch, state.iterator.dataset:size())
    batch = batch + 1
end

engine.hooks.onEnd = function(state)
    submission:close()
end

engine:test{
    network = model,
    iterator = getIterator(testDataset)
}

--Saving Model
model:clearState()
torch.save(opt.file..'.t7',model)

print("The End!")
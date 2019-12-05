%% Description
% To run this program, you must run descriptorBuilder.m first.

% descriptorBuilder.m takes a dataset of cat images and create interest
% points and SIFT descriptors for each image. These features will be saved
% to a folder 'dataset/siftDes/'.

% This program will take the created features directly from the folder.
% This saves time when we want to matching a new image to other images
% in the dataset.
%% get directories
imDir = 'dataset/images1k/';
siftDesDir = 'dataset/siftDes/';
testImDir = 'testset/inputImages/';
outputImDir = 'testset/outputImages/';
intPImgDir = 'testset/interestPointImage/';
intRImgDir = 'testset/interestRegionImage/';
segImgDir = 'testset/segmentationImage/';

myFiles = dir(fullfile(imDir,'*.jpg'));
testFiles = dir(fullfile(testImDir,'*.jpg'));
%% parameter settings
drawIntP = true;    % draw interest points image or not
drawDoGCircle = true;   % draw interest regions image or not
drawSegImage = true;

useFilter = true;       % use filter images or not: the filter is to only retain thoes interest points that are around the cat.

ipDetector = 1;      % interest point detector, eithor 1:'Harris' or 2:'FAST'. 'Harris' detector is recommended to be used.

if (ipDetector == 1)
    sig1 = 1.0;
    sigD = 0.7;
    alpha = 0.05;
    threR = 10000;
else
    T = 50;     % FAST pixel value threshold
    nStar = 9;      % FAST contiguous threshold
end

numO = 5;   % number of Octave levels in DoG
s = 5;  % number of steps in each Octave
sigDoG = 0.6;   % sigma for the initial Gaussian filter in DoG

topN = 20;      % top N interest point matches to be considered
disR = 0.9;     % discard ration in matching

% parameters for segmentation
seg_k = 1;
seg_K = 2;
edge_mode = "sobel";
filter_mode = "segEdge";
%% matching new images to images in the dataset
for k1 = 1:length(testFiles)
    % load image
    testFileName = testFiles(k1).name;
    fullTestFileName = fullfile(testImDir, testFileName);
    testIm = uint8(imread(fullTestFileName));
    [h,w,c] = size(testIm);
    
    % interest points
    if (ipDetector == 1)
        % Harris
        [intPoints] = Harris(testIm, sig1, sigD, alpha, threR);
    else
        % FAST
        [intPoints] = FAST(testIm, T, nStar);
    end
    
    if (isempty(intPoints))
        fprintf(strcat('No interest point found for: ', baseFileName, '.\n'));
        continue;
    end
    
    % filter irrelevant interest points
    if useFilter
        [originFinalIm, edge, segmentEdge, intersectEdge] = egbis_compute(testIm, seg_k, seg_K, edge_mode);
        if filter_mode == "edge"
            filterIm = edge;
        else
            if filter_mode == "segEdge"
                filterIm = segmentEdge;
            else
                if filter_mode == "intEdge"
                    filterIm = intersectEdge;
                else
                    filterIm = originFinalIm;
                end
            end
        end
        
        if drawSegImage
            imwrite(uint8(filterIm * 255), strcat(segImgDir, testFileName));
        end
        
        [num, ~] = size(intPoints);
        newIntPoints = [];
        for nowIdx = 1: num
            % remove the irrelevant interest points
            if filterIm(intPoints(nowIdx,2), intPoints(nowIdx,1)) ~= 0
                newIntPoints = [newIntPoints; intPoints(nowIdx,1:3)];
            end
        end
        intPoints = newIntPoints;
    end
    
    if (isempty(intPoints))
        fprintf(strcat('No interest point found for: ', baseFileName, '.\n'));
        continue;
    end
    
    % top 100 interest points
    if (ipDetector == 1)
        C = sortrows(intPoints, 3, 'descend');
        [num, ~] = size(C);
        intPoints = C(1:min(100, num),1:2);
    end
    
    if drawIntP
        intPs = ones(size(intPoints,1), 3);
        intPs(:,1:2) = intPoints(:,1:2);
        drawIm = insertShape(testIm, 'circle', intPs, 'Color', 'red', 'LineWidth', 2);
        imwrite(drawIm, strcat(intPImgDir, testFileName));
    end

    % DoG
    [intRegions] = DifferenceOfGaussian(testIm, intPoints, numO, s, sigDoG);
    if drawDoGCircle
        drawIm = insertShape(testIm, 'circle', intRegions, 'Color', 'green', 'LineWidth', 2);
        imwrite(drawIm, strcat(intRImgDir, testFileName));
    end
    
    % SIFT
    [testSift] = SIFTD(testIm, intRegions);
    
    % match the new image with images in the dataset
    % retain top 5 best matches
    bests = Inf(5,2);
    for k2 = 1:length(myFiles)
        thisFileName = myFiles(k2).name;
        siftDesFileName = strcat(siftDesDir, thisFileName(1:length(thisFileName) - 4), '.mat');
        thisSift = load(siftDesFileName);
        
        [matchings, score] = basicMatching(testSift, thisSift.SIFTDescriptors, topN, disR);
        
        if score < bests(5,1)
            % update best matches
            bests(5,1) = score;
            bests(5,2) = k2;
            bests = sortrows(bests, 1);
        end
    end
    
    % draw pair-wise matching results
    for k2 = 1:5
        if bests(k2,1) < Inf
            thisFileName = myFiles(bests(k2,2)).name;
            fullFileName = fullfile(imDir, thisFileName);
            thisIm = uint8(imread(fullFileName));
            siftDesFileName = strcat(siftDesDir, thisFileName(1:length(thisFileName) - 4), '.mat');
            thisSift = load(siftDesFileName);

            [matchings, ~] = basicMatching(testSift, thisSift.SIFTDescriptors, topN, disR);
            matchingIm = plotMatches(testIm, thisIm, matchings);
            imwrite(matchingIm, strcat(outputImDir, testFileName(1:length(thisFileName) - 4), '_', string(k2), '.jpg'));
        end
    end
end
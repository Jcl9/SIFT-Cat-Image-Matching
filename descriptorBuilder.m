%% get image directories
imDir = 'dataset/images1k/';
siftDesDir = 'dataset/siftDes/';
intPImgDir = 'dataset/interestPointImage/';
intRImgDir = 'dataset/interestRegionImage/';
myFiles = dir(fullfile(imDir,'*.jpg'));
myFiles = myFiles(randperm(length(myFiles)));
%% parameter settings
drawIntP = true;    % draw interest points image or not
drawDoGCircle = true;   % draw interest regions image or not

useFilter = true;       % use filter images or not: the filter is to only retain thoes interest points that are around the cat.
if useFilter
    intPImgDir = 'dataset/filteredIntPointImage/';
    intRImgDir = 'dataset/filteredIntRegionImage/';
end

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

filter_mode = "segEdge";
edge_mode = "sobel";
filterImgDir = strcat('dataset/filterImage_', filter_mode, '/');
if filter_mode ~= "kmeans"
    filterImgDir = strcat(filterImgDir, edge_mode, '/');
end
%% build descriptors
for k = 1:length(myFiles)
    % load image
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(imDir, baseFileName);
    siftDesFileName = strcat(siftDesDir, baseFileName(1:length(baseFileName) - 4), '.mat');
    
    thisIm = uint8(imread(fullFileName));
    [h,w,c] = size(thisIm);

    % interest points detection
    if (ipDetector == 1)
        % Harris
        % intPoints: x, y, R value
        [intPoints] = Harris(thisIm, sig1, sigD, alpha, threR);
    else
        % FAST
        % int Points: x, y
        [intPoints] = FAST(thisIm, T, nStar);
    end
    
    % filter out those interest points that are not in the 'cat' region
    if useFilter
        filterImageFileName = strcat(filterImgDir, baseFileName);
        filterIm = uint8(imread(filterImageFileName));
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
        SIFTDescriptors = [];
        save(siftDesFileName, 'SIFTDescriptors');
        fprintf(strcat('No interest point found for: ', baseFileName, '.'));
        continue;
    end
    
    % only retain the top 100 (filtered) points in terms of the R value.
    if (ipDetector == 1)
        C = sortrows(intPoints, 3, 'descend');
        [num, ~] = size(C);
        intPoints = C(1:min(100, num),1:2);
    end
    
    % draw interest points image
    if (~isempty(intPoints))
        if drawIntP
            intPs = ones(size(intPoints,1), 3) * 0.5;
            intPs(:,1:2) = intPoints(:,1:2);
            drawIm = insertShape(thisIm, 'circle', intPs, 'Color', 'red', 'LineWidth', 2);
            imwrite(drawIm, strcat(intPImgDir, baseFileName));
        end
    else
        SIFTDescriptors = [];
        save(siftDesFileName, 'SIFTDescriptors');
        fprintf(strcat('No interest point found for: ', baseFileName, '.'));
        continue;
    end

    % use Difference of Gaussian to find a proper region for each interest
    % point
    [intRegions] = DifferenceOfGaussian(thisIm, intPoints, numO, s, sigDoG);
    if drawDoGCircle
        drawIm = insertShape(thisIm, 'circle', intRegions, 'Color', 'green', 'LineWidth', 2);
        imwrite(drawIm, strcat(intRImgDir, baseFileName));
    end

    % build SIFT descriptor for each interest region
    % the output SIFT Descriptors is an N * (2 + 1 + 1 + 128) matrix, where
    % the rows are the interest points, the columns are: x,y positions,
    % scales, orientations, and 128D descriptors
    [SIFTDescriptors] = SIFTD(thisIm, intRegions);
    
    % save the SIFT descriptors
    save(siftDesFileName,'SIFTDescriptors');
end
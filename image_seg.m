imDir = 'images1k/';   %gets directory
myFiles = dir(fullfile(imDir,'*.jpg'));
myFiles = myFiles(randperm(length(myFiles)));
K = 4;

for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(imDir, baseFileName);
    thisIm = uint8(imread(fullFileName));
    
    [height, width, c] = size(thisIm);
%     x = 1:width;
%     y = 1:height;
%     [X, Y] = meshgrid(x,y);
%     blankIm = zeros(height, width, 5);
%     blankIm(:,:,1:3) = thisIm;
%     blankIm(:,:,4) = (X - width / 2).^2 / 1000;
%     blankIm(:,:,5) = (Y - height / 2).^2 / 1000;
%     
%     imArr = reshape(blankIm, [], 5);
%     [idx, C] = kmeans(imArr, K);
%     imgSegs = reshape(idx, height, width);
%     imgSegs = imopen(imgSegs, max(height, width) * 0.05);
    
    [L,Centers] = imsegkmeans(thisIm, K, 'NormalizeInput', true);
    newImg = uint8(zeros(height, width, c));
    for ki = 1:length(Centers)
        newImg(:,:,1) = newImg(:,:,1) + uint8(L == ki) * Centers(ki,1);
        newImg(:,:,2) = newImg(:,:,2) + uint8(L == ki) * Centers(ki,2);
        newImg(:,:,3) = newImg(:,:,3) + uint8(L == ki) * Centers(ki,3);
    end
%     B = labeloverlay(thisIm, L);
    
%     wavelength = 2.^(0:5) * 3;
%     orientation = 0:45:135;
%     g = gabor(wavelength, orientation);
%     I = rgb2gray(im2single(thisIm));
%     
%     gabormag = imgaborfilt(I,g);
%     montage(gabormag,'Size',[4 6]);
%     for i = 1:length(g)
%         sigma = 0.5*g(i).Wavelength;
%         gabormag(:,:,i) = imgaussfilt(gabormag(:,:,i),3*sigma); 
%     end
%     montage(gabormag,'Size',[4 6]);
%     
%     nrows = size(thisIm,1);
%     ncols = size(thisIm,2);
%     [X,Y] = meshgrid(1:ncols,1:nrows);
%     featureSet = cat(3,I,gabormag,X,Y);
%     L2 = imsegkmeans(featureSet, 2, 'NormalizeInput', true);
%     C = labeloverlay(thisIm, L2);
    
%     % show segmentation
%     imgs = [uint8(B); uint8(thisIm)];
%     imagesc(imgs);
%     pause;
    
    % save segmentation image
    imwrite(uint8(newImg), sprintf(strcat('kmeansOutput/', baseFileName)));
end
% SIFT Descriptor using Vlfeat
% need to install and setup vlfeat library first
% URL: http://www.vlfeat.org/install-matlab.html
% input: image with interest regions
% output: frames and descriptors of the image
function [SIFTDescriptors] = SIFTD(Im1, intRegions)
    % transfer the images to single grayscale
    Im1 = single(rgb2gray(Im1));
    
    [num, ~] = size(intRegions);
    fc = zeros(num, 4);
    fc(:,1:3) = intRegions(:,:);
    % compute SIFT frames and descriptors
    % levels: the number of levels per octave of the DoG scale space
    % peakThresh: the peak selection threshold
    [f, d] = vl_sift(Im1,'frames',fc', 'orientations');
    SIFTDescriptors = [f; double(d)]';
end

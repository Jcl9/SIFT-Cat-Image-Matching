function [intRegions] = DifferenceOfGaussian(inputImg, intPoints, numO, s, sigDoG)
%DIFFERENCEOFGAUSSIAN
%   Difference Of Gaussian to find a proper interest region for each
%   interest point
%   inputImg: input image
%   intPoints: interest points coordinates, size of (n, 2), each row is (x, y)
%   numO: number of Octave levels
%   s: desired number of images in each Octave level

    inputImg = rgb2gray(inputImg);
    [num, dim] = size(intPoints);
    intRegions = zeros(num, 3);
    intRegions(:,1:2) = intPoints;
    maxValues = -Inf(length(intPoints),1);
    
    % initial smoothing
    [Gx, Gy] = gauss2D(sigDoG);
    currentImg = filter2(Gy, filter2(Gx, inputImg, 'same'), 'same');
    
    % Gaussian filter for each DoG opperation
    k = 2 ^ (1/s);
    [Gx, Gy] = gauss2D(k);
        
    try
        for oi = 1: numO
            if (oi > 1)
                % down sample the image to make the initial image for next
                % Octave level
                [h, w] = size(currentImg);
                currentImg = currentImg(1:2:h, 1:2:w);
            end

            for si = 1: s
                smoothedImg = filter2(Gy, filter2(Gx, currentImg, 'same'), 'same');

                for pi = 1: num
                    x = floor(intPoints(pi, 1) / 2^(oi - 1)) + 1;
                    y = floor(intPoints(pi, 2) / 2^(oi - 1)) + 1;
                    if (mod(intPoints(pi, 1), 2^(oi - 1)) == 0)
                        x = x - 1;
                    end
                    if (mod(intPoints(pi, 2), 2^(oi - 1)) == 0)
                        y = y - 1;
                    end
                    currentValue = smoothedImg(y, x) - currentImg(y, x);
                    if (currentValue > maxValues(pi))
                        maxValues(pi) = currentValue;
                        thisSig = sigDoG * 2^(oi - 1 + (si - 1)/s);
                        intRegions(pi,3) = 3 * thisSig;
                    end
                end

                currentImg = smoothedImg;
            end
        end
    catch e
        fprintf(1,'The identifier was:\n%s',e.identifier);
        fprintf(1,'There was an error! The message was:\n%s',e.message);
    end
end


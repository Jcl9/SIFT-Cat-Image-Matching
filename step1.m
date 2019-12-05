Image_idx = [433 472 485 489 625 679];

PathRoot='../images1k';
list=dir(PathRoot);
fileNum=size(list,1); 

for k=3:103
    fileName = list(k).name;
    Im = imread(strcat(PathRoot,"/",fileName));

    grayIm = double(rgb2gray(Im));

    sigma = 1;
    [Gx, Gy] = gaussDeriv2D(sigma);

    GxIm = imfilter(grayIm,Gx,"replicate");
    GyIm = imfilter(grayIm,Gy,"replicate");
    magIm = sqrt(double(GxIm.^2 + GyIm.^2));

    T = 100;
    tIm = magIm > T;
   
    
    radius = 6;
    sz = radius*2+1;
    AllOneMask = ones(sz,sz)/(sz*sz);
    pIm = imfilter(double(tIm),AllOneMask,"replicate");
    pIm = pIm >= 0.5;
    
    imwrite(tIm, strcat('../result1k/',fileName));
    imwrite(pIm, strcat('../result1kwmode/',fileName));
end

function [Gx, Gy] = gaussDeriv2D(sigma)
    n = 10;
    Gx = zeros(n*2+1, n*2+1);
    Gy = zeros(n*2+1, n*2+1);
    for i = -n:n
        for j = -n:n
            x = j / n * sigma * 3;
            y = i / n * sigma * 3;
            p = x / (2*pi*sigma.^4);
            q = y / (2*pi*sigma.^4);
            r = -(x.^2 + y.^2) / (2*sigma.^2);
            Gx(i+n+1, j+n+1) = p*exp(r);
            Gy(i+n+1, j+n+1) = q*exp(r);
        end
    end
end
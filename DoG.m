Image_idx = [433 472 485 489 625 679];

for k=1:6
   
    Im = imread(sprintf("../pics/IMG_0%d.jpg", Image_idx(k)));

    grayIm = double(rgb2gray(Im));

    sigma = 1;
    [Gx, Gy] = gaussDeriv2D(sigma);

    GxIm = imfilter(grayIm,Gx,"replicate");
    GyIm = imfilter(grayIm,Gy,"replicate");
    magIm = sqrt(double(GxIm.^2 + GyIm.^2));

    T = 50;
    tIm = magIm > T;
    imwrite(tIm, sprintf('../results/cat%d.jpg', k));

    % sobel
    Fx = -fspecial('sobel')';
    fxIm = imfilter(grayIm,Fx);
    Fy = -fspecial('sobel');
    fyIm = imfilter(grayIm,Fy);
    magIm = sqrt(fxIm.^2 + fyIm.^2);
    tIm = magIm > T;
    imwrite(tIm, sprintf('../results/cat%d_sobel.jpg', k));

    % canny
    eIm = edge(grayIm,'Canny');
    imwrite(eIm, sprintf('./results/cat%d_canny.jpg', k));
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
%     k = max(max(Gx));
%     b = min(min(Gx));
%     Gx = (Gx + k) / (k-b);
%     Gy = (Gy + k) / (k-b);
%     imwrite(Gx, 'Gx.jpg');
%     imwrite(Gy, 'Gy.jpg');
end
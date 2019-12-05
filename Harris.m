function [intPoints] = Harris(inputImg, sig1, sigD, alpha, threR)
    % Harris interest points detector
    % inputImg: input image
    % sig1: gaussian window sigma
    % sigD: gaussian derivative sigma
    % alpha: cornorness function alpha
    % threR: a threshold to remove small values in R

    inputImg = rgb2gray(inputImg);
    [H, W] = size(inputImg);

    % make X Y gaussian filters
    [Gdx, Gdy] = gaussDeriv2D(sigD);
    [Gx, Gy] = gauss2D(sig1);

    % image derivatives
    Ix = filter2(Gdx, inputImg, 'same');
    Iy = filter2(Gdy, inputImg, 'same');

    % Square of derivatives
    Ix2 = Ix .^ 2;
    Iy2 = Iy .^ 2;
    IxIy = Ix .* Iy;

    % Gaussian filter g(sigma1)
    gIx2 = filter2(Gy, filter2(Gx, Ix2, 'same'), 'same');
    gIy2 = filter2(Gy, filter2(Gx, Iy2, 'same'), 'same');
    gIxIy = filter2(Gy, filter2(Gx, IxIy, 'same'), 'same');

    % Compute Cornerness function
    R = gIx2 .* gIy2 - gIxIy .^ 2 - alpha * (gIx2 + gIy2) .^ 2;
    % remove small values in R
    R = R .* (R > threR);

    % non-maximum suppression
    Cx = [];
    Cy = [];
    CR = [];
    sqrX = [-1, 0, 1, -1, 1, -1, 0, 1];
    sqrY = [-1, -1, -1, 0, 0, 1, 1, 1];
    % As professor suggested in class, simply ignore the border line of the
    % image when doing non-maximum suppression
    for row = 2: H-1
        for col = 2: W-1
            maxFlag = 1;
            for i = 1: 8
                if R(row, col) <= R(row + sqrY(i), col + sqrX(i))
                    maxFlag = 0;
                    break;
                end
            end
            if maxFlag
                Cx = [Cx col];
                Cy = [Cy row];
                CR = [CR R(row, col)];
            end
        end
    end
    intPoints = cat(1, Cx, Cy, CR)';
end


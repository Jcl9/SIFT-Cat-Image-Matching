function [intPoints] = FAST(inputImg, T, nStar)
    % FAST algorithm to find interest points
    % inputImg: the input image
    % T: threshold T
    % nStar: threshold nStar

    inputImg = rgb2gray(inputImg);
    fastX = [];
    fastY = [];

    % hardcode the circle border
    cirX = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
    cirY = [-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3];

    [H, W] = size(inputImg);
    for row = 4: H-3
        for col = 4: W-3
            mark = zeros(16, 1);
            for i = 1: 16
                if inputImg(row + cirY(i), col + cirX(i)) >= inputImg(row, col) + T
                    mark(i) = 1;
                else
                    if inputImg(row + cirY(i), col + cirX(i)) <= inputImg(row, col) - T
                        mark(i) = -1;
                    else
                        mark(i) = 0;
                    end
                end
            end

            if longestContiguous(mark) >= nStar
                fastX = [fastX, col];
                fastY = [fastY, row];
            end
        end
    end
    intPoints = cat(1, fastX, fastY)';
end


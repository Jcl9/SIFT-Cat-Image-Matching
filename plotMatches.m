% Plot matched points
function [Im3] = plotMatches(Im1,Im2,matchings)
    % merge two images
    Im3 = mergeIm(Im1,Im2);

    lines = matchings(:,1:4);
    lines(:,3) = lines(:,3) + size(Im1,2);
    Im3 = insertShape(Im3, 'Line', lines, 'Color', 'yellow', 'LineWidth', 1);
end

% Image merge function
function [Im3] = mergeIm(Im1,Im2)
    % get the rows of two images
    r1 = size(Im1,1);
    r2 = size(Im2,1);
    % fill in empty rows
    if(r1 > r2)
        Im2(r1,1) = 0;
    else
        Im1(r2,1) = 0;
    end
    Im3 = [Im1 Im2];
end
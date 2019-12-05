function [matchings, score] = basicMatching(SF1, SF2, topN, disR)
%BASICMATCHING
%   matching between two images' SIFT descriptors
%   SF1: the input image
%   Sf2: one candidate image from the dataset
%   topN: top N matchings to be considered
%   disR: discard ratio R

    [num1, ~] = size(SF1);
    [num2, ~] = size(SF2);
    matchingMap = zeros(num1, 3);
    for idx1= 1: num1
        sift1 = SF1(idx1, 5:132);
        bestScore = inf;
        bestIdx = 0;
        for idx2 = 1:num2
            sift2 = SF2(idx2, 5:132);
            thisScore = norm(sift1 - sift2);
            if thisScore < bestScore
                bestScore = thisScore;
                bestIdx = idx2;
            else
                if bestScore / thisScore > disR
                    bestIdx = 0;
                end
            end
        end
        
        if bestIdx > 0
            matchingMap(idx1,:) = [idx1, bestIdx, bestScore];
        else
            matchingMap(idx1,:) = [idx1, bestIdx, inf];
        end
    end
    
    % take only topN matchings
    matchingMap = sortrows(matchingMap, 3);
    [num, ~] = size(matchingMap);
    matchingMap = matchingMap(1:min(topN, num),:);
    score = sum(matchingMap(:,3), 'all');
    
    [num,~] = size(matchingMap);
    matchings = zeros(num,5);
    for k = 1:num
        matchings(k,1:2) = SF1(matchingMap(k,1),1:2);
        if matchingMap(k,2) > 0
            matchings(k,3:4) = SF2(matchingMap(k,2),1:2);
        else
            matchings(k,3:4) = [1 1];
        end
        matchings(k,5) = matchingMap(k,3);
    end
end


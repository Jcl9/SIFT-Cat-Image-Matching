function [longCon] = longestContiguous(mark)
    len = length(mark);
    numCon = [];
    nowCon = 1;
    for i = 2: len
        if mark(i) == mark(i-1)
            nowCon = nowCon + 1;
        else
            if mark(i-1) ~= 0
                numCon = [numCon nowCon];
            end
            nowCon = 1;
        end
    end

    if (mark(len) ~= 0)
        if (mark(len) == mark(1)) && (~isempty(numCon))
            numCon = [numCon nowCon + numCon(1)];
        else
            numCon = [numCon nowCon];
        end
    end
    
    longCon = max(numCon);
end


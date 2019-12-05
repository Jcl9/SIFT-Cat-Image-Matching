% efficient graph-based image segmentation

PathRoot='../images1k';
list=dir(PathRoot);
fileNum=size(list,1); 

k = 1;

K = 2;

for iter=3:fileNum
    fileName = list(iter).name;
    Im = imread(strcat(PathRoot,"/",fileName));
    
    % step0: downsample if the image is too big!
    row = size(Im,1);
    col = size(Im,2);
    rowO = row;
    colO = col;
    if row > 300
        h = 300/row;
        Im = imresize(Im, h);
        row = size(Im,1);
        col = size(Im,2);
    end
    if col > 300
        h = 300/col;
        Im = imresize(Im, h);
        row = size(Im,1);
        col = size(Im,2);
    end

    % step1: gaussian smoothing
    bIm = gaussian_smoothing(Im, 0.8, row, col);
    %imwrite(uint8(bIm(:,:,1)), strcat('../red/',fileName));
    newIm = zeros(row,col); 
    % step2: compute weight matrix
    w = compute_weight(bIm, row, col);
    n = size(w,1);

    % step3: MST

    % root: ancestor info(2:size;3:max_edge_weight)
    root = zeros(row,col,3);
    root(:,:,2) = ones(row,col,1);
    merged = 0;
    for i=1:n
        edge = w(i,:);
        % ancestors
        [ux,uy] = find_ancestor(root, edge(1), edge(2));
        [vx,vy] = find_ancestor(root, edge(3), edge(4));
        if ux == vx && uy == vy
            % do nothing
        else
            % may or may not merge
            sizeCu = root(ux,uy,2);
            sizeCv = root(vx,vy,2);
            maxwCu = root(ux,uy,3);
            maxwCv = root(vx,vy,3);
            tCu = k/sizeCu;
            tCv = k/sizeCv;
            if edge(5) <= min(maxwCu+tCu, maxwCv+tCv)
                % merge
                merged = merged + 1;
                if sizeCu < sizeCv
                    root(vx,vy,2) = root(vx,vy,2) + root(ux,uy,2);
                    root(vx,vy,3) = edge(5);
                    root(ux,uy,1) = vx;
                    root(ux,uy,2) = vy;
                else
                    root(ux,uy,2) = root(ux,uy,2) + root(vx,vy,2);
                    root(ux,uy,3) = edge(5);
                    root(vx,vy,1) = ux;
                    root(vx,vy,2) = uy;
                end
            end
        end
    end

    label = zeros(row,col);
    l = 0;
    for i=1:row
        for j=1:col
            if root(i,j,1) == 0
                l = l+1;
                label(i,j) = l;
                newIm(i,j) = l;
            end
        end
    end

    for i=1:row
        for j=1:col
            if newIm(i,j) == 0
                newIm(i,j) = find_label(root,i,j,label);
            end
        end
    end
    pp = max(newIm, [], 'all');
    
    % merge small components!
    
    Im1 = reshape(Im, [], 3);
    newIm1 = reshape(newIm, [], 1);
    nn = histc(newIm1, 1:pp);
    [y,ii] = sort(nn,'descend');
    
    order_num = 0;
    order_rate = 1.0;
    current = 0;
    for i=y'
        current = current + i;
        order_num = order_num + 1;
        if current > order_rate * row * col
            break
        end
    end
    
    validi = ii(1:order_num,1)';
    realIm = gen_distribute(newIm, row, col, validi, order_num);
    
    imwrite(realIm, strcat('../egbis/',fileName));
    
%     for i=1:pp
%         idx = find(newIm1==i);
%         cc = double(Im1(idx,:));
%         ccmean = mean(cc);
%         ccstd = std(cc);
%     end


    % KNN!
    x = 1:col;
    y = 1:row;
    [X, Y] = meshgrid(x,y);
    E = zeros(row,col,3);
    E(:,:,1) = Y;
    E(:,:,2) = X;
    E(:,:,3) = realIm;
    E1 = reshape(E, [], 3);
    E2 = E1(E1(:,3)>0,:);
    E2(:,3) = newIm1(E1(:,3)>0);
    E3 = E1(E1(:,3)==0,:);
    Mdl = fitcknn(E2(:,1:2),E2(:,3),'NumNeighbors',5,'Standardize',1);
    result = predict(Mdl,E3(:,1:2));
    E3(:,3) = result;
    n = length(result);
    for i=1:n
        newIm(E3(i,1),E3(i,2)) = E3(i,3);
    end
    
    newImMin = min(newIm, [], 'all');
    newImMax = max(newIm, [], 'all');
    imwrite(newIm/maxSeg, strcat('../egbis2/',fileName));
    
    [refillIm, refillVar] = refill(Im, row, col, pp, newIm);
    
    imwrite(uint8(refillIm(:,:,1:3)), strcat('../refill/',fileName));
      
    % dilate
    newIm1 = reshape(newIm, [], 1);
    nn = histc(newIm1, 1:pp);
    [y,ii] = sort(nn,'descend');
    y = y(y~=0);
    ii = ii(1:length(y));
    
    %toCut = ii(y<3000);
    toCut = ii(21:length(ii));
    toCut = toCut(end:-1:1);
    
    if ~isempty(toCut)
        step = 0;
        for i=toCut'
            step = step + 1;
            periphery = dilate(newIm, Im, row, col, i);
            if periphery > 0
                newIm(newIm == i) = periphery;
            end
            %imwrite(newIm/pp, strcat('../lab/',int2str(step),fileName));
        end
    end
    
    maxSeg = max(newIm, [], 'all');
    imwrite(newIm/maxSeg, strcat('../egbis3/',fileName));
    
    % refill pixels
    [refillIm, refillVar] = refill(Im, row, col, pp, newIm);
    %refillIm(:,:,1:3) = (refillIm(:,:,1:3) + double(Im(:,:,1:3))) * 0.5;
    
    originRefillIm = imresize(refillIm, [rowO colO]);
    imwrite(uint8(originRefillIm(:,:,1:3)), strcat('../refill2/',fileName));
    
    % EZ kmeans
    
    finalIm = figkmeans(refillIm, refillVar, row, col, K);
    x = 1:col;
    y = 1:row;
    [X, Y] = meshgrid(x,y);
    E = zeros(row,col,3);
    E(:,:,1) = Y;
    E(:,:,2) = X;
    E(:,:,3) = finalIm;
    E1 = reshape(E, [], 3);
    E2 = E1(E1(:,3)==1,:);
    E3 = E1(E1(:,3)==2,:);
    E2mean = mean(E2(:,1:2));
    E2std = std(E2(:,1:2));
    E3mean = mean(E3(:,1:2));
    E3std = std(E3(:,1:2));
    if norm(E2std) < norm(E3std)
        finalIm = 2-finalIm;
    else
        finalIm = finalIm-1;
    end
    
    originFinalIm = imresize(finalIm, [rowO colO]);
    imwrite(originFinalIm, sprintf(strcat('../kmeansOutput3/', fileName)));
    
end

function [m,s] = segmentMeanStd(logicIm, Im)
    segIm = double(Im);
    [logicIdx,logicIdy] = find(logicIm~=0);
    segIm1 = zeros(length(logicIdx),3);
    for i=1:length(logicIdx)
        segIm1(i,:) = segIm(logicIdx(i), logicIdy(i), 1:3);
    end
    m = mean(segIm1);
    s = std(segIm1);
end

function [periphery] = dilate(newIm, Im, row, col, i)
    se = strel('square',3);
    c = newIm == i;
    originIm = zeros(row, col);
    originIm(c) = 1;
    [thisSegMean,thisSegStd] = segmentMeanStd(c, Im);
    diIm = imdilate(originIm, se);
    perIm = diIm - originIm;
    outIm = perIm .* newIm;
    outIm1 = reshape(outIm, [], 1);
    outIm1 = outIm1(outIm1 ~= 0);
    % can't use mode directly
    h = histc(outIm1, 1:max(outIm1));
    cnum = find(h~=0);
    h = h(cnum);
    if length(cnum) == 1
        periphery = cnum(1);
    else
        h = h/sum(h);
        mindist = 999999;
        pick = 0;
        for j=1:length(h)
            if h(j) > 0.2
                logicIm = newIm == cnum(j);
                [segMean,segStd] = segmentMeanStd(logicIm, Im);
                dist = norm(segMean - thisSegMean);
                if dist < mindist
                    mindist = dist;
                    pick = j;
                end
            end
        end
        if pick > 0
            periphery = cnum(pick);
        else
            periphery = cnum(1);
        end
        %periphery = 0;
    end
end

function [realIm] = gen_distribute(newIm, row, col, validi, order_num)
    realIm = zeros(row,col);
    ita = 1;
    for i=validi
        C = newIm == i;
        realIm = realIm + double(C)*ita;
        ita = ita - 1/order_num * 0.5;
    end
end

function [newIm] = figkmeans(refillIm, refillVar, row, col, K)
    blankIm = zeros(row, col, 3);
    blankIm(:,:,1:5) = refillIm;
    blankIm(:,:,6:10) = refillVar;
    blankIm(:,:,4:5) = blankIm(:,:,4:5);
    blankIm(:,:,6:10) = blankIm(:,:,6:10);
    imArr = reshape(blankIm, [], 10);
    [idx, C] = kmeans(imArr, K);
    newIm = reshape(idx, row, col);
    %newIm = imopen(newIm, max(row, col) * 0.05);
end

function [refillIm, refillVar] = refill(Im, row, col, pp, newIm)
    D = [.299  .587  .144; 
        -.147 -.289  .436; 
         .615 -.515 -.1];
    cIm = reshape(double(Im),[],3);
    res = (D*cIm')';
    %Im = reshape(res,row,col,3);
    color = zeros(pp,11);
    for i=1:row
        for j=1:col
            l = newIm(i,j);
            for dim=1:3
                color(l,dim) = color(l,dim) + double(Im(i,j,dim));
            end
            color(l,4) = color(l,4) + i;
            color(l,5) = color(l,5) + j;
            color(l,6) = color(l,6) + 1;
        end
    end
    
    for dim=1:5
        color(:,dim) = color(:,dim) ./ color(:,6);
    end
    
    refillIm = zeros(row,col,5);
    refillVar = zeros(row,col,5);
    for i=1:row
        for j=1:col
            for dim=1:3
                l = newIm(i,j);
                refillIm(i,j,dim) = color(l,dim);
                color(l,dim+6) = color(l,dim+6) + (double(Im(i,j,dim))-color(l,dim)).^2;
            end
            color(l,10) = color(l,10)+(double(i)-color(l,4)).^2;
            color(l,11) = color(l,11)+(double(i)-color(l,5)).^2;
        end
    end
    
    for dim=1:5
        color(:,dim+6) = color(:,dim+6) ./ color(:,6);
    end
    
    for i=1:row
        for j=1:col
            for dim=1:5
                l = newIm(i,j);
                refillVar(i,j,dim) = sqrt(color(l,dim+6));
            end
        end
    end
   
end

function [v] = find_label(root,x,y,label)
    if label(x,y) ~= 0
        v = label(x,y);
    else
        v = find_label(root,root(x,y,1),root(x,y,2),label);
    end
end

function [tx, ty] = find_ancestor(root, x, y)
    % find ancestor
    if root(x,y,1) == 0
        tx = x;
        ty = y;
    else
        [tx, ty] = find_ancestor(root,root(x,y,1),root(x,y,2));
    end
end

function [w] = compute_weight(Im, row, col)
    num = (row-1)*col+row*(col-1)+2*(row-1)*(col-1);
    % intensity
    I = Im(:,:,1)*0.299+Im(:,:,2)*0.587+Im(:,:,3)*0.144;
    w = zeros(num, 5);
    index = 0;
    % weights of colors
    wr = 2;
    wg = 4;
    wb = 3;
    for i=1:row
        for j=1:col
            % upleft
            if i>1 && j>1
                dI = (I(i-1,j-1)-I(i,j))/255;
                dR = (Im(i-1,j-1,1)-Im(i,j,1))/255;
                dG = (Im(i-1,j-1,2)-Im(i,j,2))/255;
                dB = (Im(i-1,j-1,3)-Im(i,j,3))/255;
                d = sqrt(dI.^2+dR.^2*wr+dG.^2*wg+dB.^2*wb);
                index = index + 1;
                w(index,:) = [i-1 j-1 i j d];
            end
            % up
            if i>1
                dI = (I(i-1,j)-I(i,j))/255;
                dR = (Im(i-1,j,1)-Im(i,j,1))/255;
                dG = (Im(i-1,j,2)-Im(i,j,2))/255;
                dB = (Im(i-1,j,3)-Im(i,j,3))/255;
                d = sqrt(2*dI.^2+dR.^2*wr+dG.^2*wg+dB.^2*wb);
                index = index + 1;
                w(index,:) = [i-1 j i j d];
            end
            % upright
            if i>1 && j<col
                dI = (I(i-1,j+1)-I(i,j))/255;
                dR = (Im(i-1,j+1,1)-Im(i,j,1))/255;
                dG = (Im(i-1,j+1,2)-Im(i,j,2))/255;
                dB = (Im(i-1,j+1,3)-Im(i,j,3))/255;
                d = sqrt(dI.^2+dR.^2*wr+dG.^2*wg+dB.^2*wb);
                index = index + 1;
                w(index,:) = [i-1 j+1 i j d];
            end
            % left
            if j>1
                dI = (I(i,j-1)-I(i,j))/255;
                dR = (Im(i,j-1,1)-Im(i,j,1))/255;
                dG = (Im(i,j-1,2)-Im(i,j,2))/255;
                dB = (Im(i,j-1,3)-Im(i,j,3))/255;
                d = sqrt(dI.^2+dR.^2*wr+dG.^2*wg+dB.^2*wb);
                index = index + 1;
                w(index,:) = [i j-1 i j d];
            end
        end
    end
    w = sortrows(w,5);
end

function [bIm] = gaussian_smoothing(Im, sigma, row, col)
    G = fspecial('gaussian', 2*ceil(3*sigma)+1, sigma);
    bIm = zeros(row,col,3);
    bIm(:,:,1) = imfilter(double(Im(:,:,1)),G,"replicate");
    bIm(:,:,2) = imfilter(double(Im(:,:,2)),G,"replicate");
    bIm(:,:,3) = imfilter(double(Im(:,:,3)),G,"replicate");
end
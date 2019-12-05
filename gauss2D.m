function [Gx,Gy] = gauss2D(sigma)
    width = ceil(3*sigma);
    
    x = - width : width;
    Gx = exp(-(x.^2)/(2*sigma^2))/((2*pi)^0.5 * sigma);
    Gx = Gx / sum(Gx);
    Gy = Gx';
end


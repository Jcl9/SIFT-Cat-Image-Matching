function [Gx,Gy] = gaussDeriv2D(sigma)
     width = ceil(3*sigma);
     
     [x,y] = meshgrid(-width:width, -width:width);
     G = exp(-(x.^2+y.^2)/(2*sigma^2))/(2*pi*(sigma^2));
     G_norm = G / sum(G(:));
     Gx = -x .* G_norm / (sigma^2);
     Gy = -y .* G_norm / (sigma^2);
end
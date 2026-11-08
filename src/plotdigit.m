% Plot digit in the vector x
function [] = plotdigit(x)

A1 = reshape(x, 16, 16)';  	% Reshape vector x (256x1) to matrix A1 (16x16)
A = zeros(17,17);			% Add zero row and column for Matlab's pcolor to work correctly
A(1:16,1:16) = flipud(A1);

figure
pcolor(A)
colormap(flipud(gray))
axis off
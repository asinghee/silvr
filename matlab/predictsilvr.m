% Takes new unseen points in Xtst to make predictions of the output values, using the
% SiLVR model as given by the neural networks in 'nets'.
% Xtrg, Ytrg are the original training data used to train the SiLVR model.
% The predicted values are returned as vector 'y'.
%
% Author - Amith Singhee
% Date - Dec 9, 2007
% References:
% 1. A. Singhee, R. A. Rutenbar, "Beyond low-order statistical response surfaces:
%   Latent variable regression for efficient, highly nonlinear fitting", DAC, 2007.
% 2. A. Singhee, "Novel Algorithms for Fast Statistical Analysis of Scaled Circuits",
%   PhD Thesis, CMU, 2007.
%
function [y, t] = predictsilvr(Xtst, Xtrg, Ytrg, nets);
  n = size(Xtst, 1);
  E = zeros(size(Xtst));

  % mean 0, std 1
  mx = mean(Xtrg);
  sx = std(Xtrg);
  my = mean(Ytrg);
  sy = std(Ytrg);
  for i = 1:n
    E(i,:) = (Xtst(i,:) - mx)./sx;
  end

  % range [-1 1]
%  maxx = max(X);
%  minx = min(X);
%  rx = maxx - minx;
%  maxy = max(Y);
%  miny = min(Y);
%  ry = maxy - miny;
%  for i = 1:n
%    E(i,:) = (x(i,:) - minx)./rx * 2 - 1;
%  end

  K = size(nets, 2);

  y = zeros(size(Xtst,1),size(Ytrg,2));
  t = [];

  for i = 1:K
%    t(:,i) = E * nets{i}.iw{1}';
%    size(t(:,i))
%    size(sy)
%    size(y)
    ysim = sim(nets{i}, E')';
    y = y + ysim;
%    E = E - t(:,i) * P(:,i)';
  end

  for i = 1:n
    y(i,:) = y(i,:) * sy + my; % mean 0, std 1
%    y(i,:) = (y(i,:) + 1) / 2 .* ry + miny; % range [-1 1]
  end


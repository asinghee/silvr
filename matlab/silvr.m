% SiLVR: computes a projection pursuit based model Y = f(X), using nlv Latent Variables (LVs)
% (X, Y) are the training points:
%    each row of X is a point in the input variable space
%    Y is a vector of corresponding output values
% Suggested values for (nc, cv) = (12, 5)
% The model is returned as a set of neural networks, one network per LV.
% Use predictsilvr.m to make predictions using these networks.
%
% Author - Amith Singhee
% Date - Dec 9, 2007
% References:
% 1. A. Singhee, R. A. Rutenbar, "Beyond low-order statistical response surfaces:
%   Latent variable regression for efficient, highly nonlinear fitting", DAC, 2007.
% 2. A. Singhee, "Novel Algorithms for Fast Statistical Analysis of Scaled Circuits",
%   PhD Thesis, CMU, 2007.
%
function [nets, timetaken] = silvr(X, Y, nlv, nc, cv)
  n = size(X, 1);
  rn = rand(n,1);
  [rn, i] = sort(rn);
  X = X(i,:); % scramble X
  Y = Y(i,:); % scramble Y identical to X
  clear rn, i;

  % mean to 0 and std to 1
  mx = mean(X);
  sx = std(X);
  my = mean(Y);
  sy = std(Y);
  for i = 1:n
    E(i,:) = (X(i,:) - mx)./sx;
    F(i,:) = (Y(i,:) - my)./sy;
  end

  % range to [-1 1]
%  maxx = max(X);
%  minx = min(X);
%  rx = maxx - minx;
%  maxy = max(Y);
%  miny = min(Y);
%  ry = maxy - miny;
%  [min(X)'; max(X)'; min(Y)'; max(Y)']
%  for i = 1:n
%    E(i,:) = (X(i,:) - minx)./rx * 2 - 1;
%    F(i,:) = (Y(i,:) - miny)./ry * 2 - 1;
%  end
%  [min(E)' max(E)']
%  [min(F)' max(F)']

  K = size(X, 2)
  Z = zeros(n, K);
  vf = std(F);
  Kout = size(Y, 2); % SHOULD BE 1
  Fsim = zeros(size(F));
  Fori = F;

  for i = 1:nlv
%    clk = clock;
    tic
    count = 1;
    mse_best = 1e300;
    if (cv > 1)
      ntest = fix(n / cv);
      tcv = cv;
      for ncv = 1:tcv
        fprintf(1, 'CV %d\n', ncv);
        if ncv > 1
          begi = [1:(ncv-1)*ntest];
        else
          begi = [];
        end
        if ncv*ntest < n
          endi = [ncv*ntest+1:n];
        else
          endi = [];
        end

        e = E([begi endi],:);
        f = F([begi endi],:);
        u = f(:,1);
        net = newff(ones(K,1)*[-10 10], [1 nc 1],{'purelin' 'tansig' 'purelin'}, 'trainbr');
%        net = newff(ones(K,1)*[-10 10], [1 nc 1],{'purelin' 'tansig' 'purelin'});
        net.layers{1}.initFcn = 'myinitw';
%        net.performFcn = 'msereg';
%        net.performParam.ratio = 0.7;
        net.biasConnect(1) = 0; % no bias node at input
        net.trainParam.epochs = 100;
        net.trainParam.goal = 1e-7;
        net.inputs{1}.userdata = spear(u, e); %rand(1, K) * 2 - 1; % u' * e; % initialize the input wts to correlations
        net.inputs{1}.userdata = net.inputs{1}.userdata / norm(net.inputs{1}.userdata);

        net = train(net, e', f');
        if net.iw{1} == zeros(1, K)
          % all weights 0!
          fprintf(1, 'REDO %d\n', ncv);
          tcv = tcv+1;
          fprintf(1, 'now tcv = %d\n', tcv);
        else
          etest = E((ncv-1)*ntest+1:ncv*ntest,:);
          ftest = F((ncv-1)*ntest+1:ncv*ntest,:);
          fsim = sim(net, etest')';
          mse_ = mse(mse(1-fsim./ftest));
          fprintf(1, 'Mse = %f\n', mse_);
          if (mse_ < mse_best)
            mse_best = mse_;
            net_best = net;
            fprintf(1, 'New best mse = %f\n', mse_);
          end
        end
      end
    else
      disp('Bad cv value');
      return;
    end
    t = E * net_best.iw{1}';
    p = E' * t / (t' * t);
%    E = E - t * p';
    F = F - sim(net_best, E')';
    Fsim = Fsim + sim(net_best, E')';
    msenow = mse(mse(1-Fsim./Fori))
    T(:,i) = t;
    P(:,i) = p;
    U(:,i) = u;
%    Qinv(:,i) = net.lw{4,3}';
    W(:,i) = net.iw{1}';
    nets{i} = net_best;
%    if std(F) < .01*vf
%      break;
%    else
%      std(F)
%    end
%    timetaken(i) = etime(clock, clk)
    timetaken(i) = toc;
  end
%  B = W * inv(P'*W) * C';

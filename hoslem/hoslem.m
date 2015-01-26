function [p, S] = hoslem(varargin)
%HOSLEM Hosmer-Lemshow goodness-of-fit test
%
% [p, S] = hoslem(ypred, y)
% [p, S] = hoslem(mdl, x, y)
%
% Input variables:
%
%   ypred:  Logistic regression predictions for each data point; values
%           should be between 0 and 1
%
%   mdl:    GeneralizedLinearModel object
%
%   x:      values at which to evaluate the generalized linear model
%
%   y:      observed values at each data point, should be 0 or 1
%
% Output variables:
%
%   p:      goodness-of-fit statistic. If >0.05, model fits data acceptably
%
%   S:      structure with the following fields
%           
%           n:      number of points in each bin
%
%           obs:    observed number of positive observations in each bin
%
%           exx:    predicted number of positive observations in each bin
%
%           g:      chi-squared statistic

% Copyright 2014 Kelly Kearney

% Parse input

if nargin == 3 && strcmp(class(varargin{1}), 'GeneralizedLinearModel')
    mdl = varargin{1};
    x = varargin{2};
    y = varargin{3};
    ypred = predict(mdl, x);
elseif nargin == 2
    ypred = varargin{1};
    y = varargin{2};
else
    error('Could not parse input');
end

ng = 10;

% Determine bins

prc = prctile(ypred, linspace(0,100,ng+1));
[n, idx] = histc(ypred, prc);
n(ng) = sum(n(ng:ng+1));
n = n(1:ng);
idx(idx == (ng+1)) = ng;

% Group data into bins (there really shouldn't ever be empty bins, but if
% the percentiles are really wonky, this can happen... usually due to
% iteration limit error, but stil...)

[agidx, yptmp] = aggregate(idx, ypred);
[agidx, yotmp] = aggregate(idx, y);
[yp, yo] = deal(cell(ng,1));
yp(agidx) = yptmp;
yo(agidx) = yotmp;

obs = cellfun(@sum, yo);
exx = cellfun(@sum, yp);

g = sum(((obs - exx).^2)./(exx.*(1-exx./n)));
p = 1 - chi2cdf(g, ng-2);

% Output

S.n = n;
S.obs = obs;
S.exx = exx;
S.g = g;
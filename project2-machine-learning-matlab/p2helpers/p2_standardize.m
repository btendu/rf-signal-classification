function [Xs, st] = p2_standardize(X, st)
%P2_STANDARDIZE  Z-score features with statistics from the training set.
%
%   [Xs, st] = p2_standardize(Xtrain)      fit and apply
%   Xs       = p2_standardize(X, st)       apply saved statistics

if nargin < 2 || isempty(st)
    st.mu = mean(X, 1);
    st.sigma = std(X, 0, 1);
    st.sigma(st.sigma < 1e-12) = 1;
end
Xs = (X - st.mu)./st.sigma;
Xs(~isfinite(Xs)) = 0;
end

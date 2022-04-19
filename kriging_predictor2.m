function [u,s,Corr,Cov] = kriging_predictor2(test_x,model)
% parameters of Kriging model
theta = model.theta;
mu = model.mu;
sigma2 = model.sigma2;
L = model.L;
sample_x = model.sample_x;
sample_y = model.sample_y;
lower_bound = model.lower_bound;
upper_bound = model.upper_bound;

% normalize data
X = (sample_x - lower_bound)./(upper_bound - lower_bound);
x = (test_x - lower_bound)./(upper_bound- lower_bound);
% initialize the prediction and variance
one = ones(size(sample_x,1),1);
% point-wise calculation
%temp1 = sum(x.^2.*theta,2)*ones(1,size(X,1));
%temp2 = sum(X.^2.*theta,2)*ones(1,size(x,1));
%R = exp(-(temp1 + temp2'-2.*(x.*theta)*X'))';
scale = theta(:,1);
kvar = theta(:,2);
X1 = X;
X2 = x;
distMat2 = size(size(X1,1),size(X2,1));

if (sum(abs(size(X1)-size(X2)))==0)
    distMat2 = squareform((pdist(X1)).^2);
else
    for ii = 1 : size(X1,1)
        for jj = 1 : size(X2,1)
            eTemp = X1(ii,:)-X2(jj,:);
            distMat2(ii,jj) = eTemp*eTemp';
        end
    end
end
R = kvar*exp(-(1/(2*scale^2))*(distMat2));

u = mu + R' *(L'\(L\(sample_y - mu)));
mse = sigma2*(1 + (1-one'*(L'\(L\R)))'.^2/(one'*(L'\(L\one))) - sum((L\R).^2,1)');
s = sqrt(max(mse,0));
% the correlation matrix
% temp1 = sum(x.^2.*theta,2)*ones(1,size(test_x,1));
% temp2 = x.*sqrt(theta);
% Corr = exp(-(temp1 + temp1'-2.*(temp2*temp2'))) + eye(size(test_x,1)).*(10+size(test_x,1))*eps;
scale = theta(:,1);
kvar = theta(:,2);
X1 = x;
X2 = x;
distMat2 = size(size(X1,1),size(X2,1));
if (sum(abs(size(X1)-size(X2)))==0)
    distMat2 = squareform((pdist(X1)).^2);
else
    for ii = 1 : size(X1,1)
        for jj = 1 : size(X2,1)
            eTemp = X1(ii,:)-X2(jj,:);
            distMat2(ii,jj) = eTemp*eTemp';
        end
    end
end
Corr = kvar*exp(-(1/(2*scale^2))*(distMat2));
Cov = Corr.*(s*s');




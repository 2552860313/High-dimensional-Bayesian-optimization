%function model = kriging_train2(sample_x,sample_y,lower_bound,upper_bound,theta0,theta_lower,theta_upper)
function model = kriging_train2(sample_x,sample_y,lower_bound,upper_bound,theta0,theta_lower,theta_upper)
[n,num_vari]= size(sample_x);
X = (sample_x - lower_bound)./(upper_bound - lower_bound);
Y = sample_y;

 % optimize the theta values with in [10^a,10^b]
 theta0 = log10(theta0);
 theta_lower = log10(theta_lower);
 theta_upper = log10(theta_upper);
 %options = optimoptions('fmincon','Algorithm','sqp','MaxFunctionEvaluations',20*num_vari,'OptimalityTolerance',1E-20,'StepTolerance',1E-20,'Display','off');
 options = optimoptions('fmincon','Algorithm','sqp','MaxFunctionEvaluations',20*2,'OptimalityTolerance',1E-20,'StepTolerance',1E-20,'Display','off');
 theta = fmincon(@(theta)-Concentrated_lnLikelihood(theta,X,Y),theta0,[],[],[],[],theta_lower,theta_upper,[],options);
 theta = 10.^theta;
one = ones(n,1);

% % calculate the correlation matrix
%  temp1 = sum(X.^2.*theta0,2)*one';
%  temp2 = X.*sqrt(theta0);
%  R = exp(-(temp1 + temp1'-2.*(temp2*temp2'))) + eye(n).*(10+n)*eps;

% calculate the correlation matrix
scale = theta(:,1);
kvar = theta(:,2);
X1 = X;
X2 = X;
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
R = kvar*exp(-(1/(2*scale^2))*(distMat2))+ eye(n).*(10+n)*eps;



% use the Cholesky factorization
L = chol(R,'lower');
% calculate mu and sigma
mu = (one'*(L'\(L\Y)))/(one'*(L'\(L\one)));
sigma2 = ((Y-mu)'*(L'\(L\(Y-mu))))/n;
lnL = -0.5*n*log(sigma2)-sum(log(abs(diag(L))));
% output the results of the model
%model.theta = theta;
model.theta = theta;
model.mu = mu;
model.sigma2 = sigma2;
model.L = L;
model.lnL = lnL;
model.sample_x = sample_x;
model.sample_y = sample_y;
model.lower_bound = lower_bound;
model.upper_bound = upper_bound;
end




function  obj = Concentrated_lnLikelihood(theta,X,Y)
theta = 10.^theta;
% the concentrated ln-likelihood function
n = size(X,1);
one = ones(n,1);
% calculate the correlation matrix
scale = theta(:,1);
kvar = theta(:,2);
X1 = X;
X2 = X;
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
R = kvar*exp(-(1/(2*scale^2))*(distMat2))+ eye(n).*(10+n)*eps;

%temp1 = sum(X.^2.*theta,2)*one';
%temp2 = X.*sqrt(theta);
%R = exp(-(temp1 + temp1'-2.*(temp2*temp2'))) + eye(n).*(10+n)*eps;

% use the  Cholesky factorization
[L,p] = chol(R,'lower');
if p>0
    lnL = -1e8;
else
    mu = (one'*(L'\(L\Y)))/(one'*(L'\(L\one)));
    sigma2 = ((Y-mu)'*(L'\(L\(Y-mu))))/n;
    lnL = -0.5*n*log(sigma2)-sum(log(abs(diag(L))));
end
obj = lnL;
end












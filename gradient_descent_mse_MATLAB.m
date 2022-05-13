% the demo script here
disp('Hi World')
beta1 = 1.;
beta2 = 1.;
xobs = randn(50,1);
noise = 0.18*randn(50,1);
yobs = beta1+beta2*xobs+noise;

tol = 1e-10; learn_rate = 0.01; maxiter = 400; beta_seed = [-10 20];
betahat = desc_grad(xobs, yobs, tol, learn_rate, maxiter, beta_seed);


function [beta] = desc_grad(x, y, tol, learn_rate, maxiter, betaseed)
beta = betaseed;
for iter=1:maxiter
    J = mse(beta, x, y);
    gradient= grad_mse(beta,x,y);
    beta(1) = beta(1)-learn_rate*gradient(1);
    beta(2) = beta(2)-learn_rate*gradient(2);
    Jp = mse(beta, x, y);
    %disp([Jp J])
    DeltaJ = abs(Jp-J);    
    disp([iter Jp DeltaJ])
    if (DeltaJ<tol)
        disp('Convergence')
        disp(DeltaJ)
        break;
    end
end
end

function [gradient] = grad_mse(beta, x, y)
% vector gradient of MSE
ypred = reg_model(beta,x);
gradient(1) = 2*sum(ypred-y)/size(ypred,1);
gradient(2) = 2*sum((ypred-y).*x)/size(ypred,1);
end

function error = mse(beta, x, y)
% mean squared error (equivalent to work directly w/ SSR)
ypred = reg_model(beta,x);
SSR = sum((ypred-y).^2);
error = SSR/size(ypred,1);
end

function [y] = reg_model(beta, x)
% regression model spec
y = beta(1)+beta(2)*x;
end
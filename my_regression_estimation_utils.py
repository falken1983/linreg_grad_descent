import numpy as np

def reg_model(beta, x):
    theta = np.array(beta, dtype=float)
    return theta[0]+theta[1]*x

def mse(beta, x, y):
    ypred = reg_model(beta,x)
    mean_squared_error = np.mean((ypred-y)**2)    
    return mean_squared_error    

def grad_mse(beta, x, y):
    ypred = reg_model(beta,x)
    grad = np.array([0,0],dtype=float)
    grad[0] = 2*np.mean(ypred-y)
    grad[1] = 2*np.mean((ypred-y)*x)
    return grad

def grad_descent(x, y, tol=1e-10, learn_rate=0.01, maxiter=400, thetaseed=[-10,20]):
    theta = np.array(thetaseed,dtype=float)
    cost = mse(theta, x, y)
    loss_fun_history = [cost.tolist()]
    theta_history = [theta.tolist()]
    for iterate in range(maxiter):        
        gradient_vector = grad_mse(theta,x,y)        # Cost Function Evaluation        
        theta-=learn_rate*gradient_vector            # Updating Scheme in Vector Form    
        cost_new = mse(theta,x,y)                    # Cost Function Revaluation
        DeltaJ = np.abs(cost_new-cost)
        #print(iterate," ",cost," (",theta[0],",",theta[1],")")
        if DeltaJ<tol:
            print("Convergence!")
            print(DeltaJ)
            break
        loss_fun_history.append(cost_new.tolist())
        theta_history.append(theta.tolist())
        cost = cost_new
    return theta, loss_fun_history, np.array(theta_history)

def linreg_normal_eqs(x, y):
    X = np.concatenate((np.ones(x.shape),x),axis=1) # Concatenating Constant Feature
    theta = np.linalg.pinv(X)@y
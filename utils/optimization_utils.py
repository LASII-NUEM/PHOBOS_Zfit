import numpy as np
from scipy.spatial.distance import cdist
import scipy

def first_order_forward_grad(f, x:np.ndarray, args=(), eps=np.sqrt(np.finfo(float).eps)):
    '''
    :param f: function to be numerically derived
    :param x: point in the space to analyze the function (parameters)
    :param args: list with parameters that won't be minimized but are required to compute the cost
    :param eps: epsilon (smallest change possible)
    :return: numerical first order forward gradient
    '''

    if not callable(f):
        raise TypeError(f'[first_order_forward_grad] Input function must be a callable!')

    n = x.size #number of input parameters of the function
    eps_arr = np.full(n, eps) #steps array
    x_shifted = x + np.diag(eps_arr) #apply shifts in the parameters with size eps
    f_evals = np.array([f(step, args) for step in x_shifted]) #evaluate the function at every step
    grad = f_evals - f(x, args) #forward diff. -> f(n+eps) - f(n)
    grad /= eps_arr #[f(n+eps) - f(n)]/eps

    return np.ravel(grad)

class OptimizerResults:
    def __init__(self, fit_result=None, opt_params=None, opt_params_scaled=None, opt_cost=None, opt_fit=None, nmse_score=None, nrmse_score=None, chi_square=None, mae_score=None, n_iter=None, t_elapsed=None):
        if fit_result is not None:
            self.fit_reponse = fit_result
        if opt_params is not None:
            self.opt_params = opt_params
        if opt_params_scaled is not None:
            self.opt_params_scaled = opt_params_scaled
        if opt_cost is not None:
            self.opt_cost = opt_cost
        if opt_fit is not None:
            self.opt_fit = opt_fit
        if nmse_score is not None:
            self.nmse_score = nmse_score
        if nrmse_score is not None:
            self.nrmse_score = nrmse_score
        if chi_square is not None:
            self.chi_square = chi_square
        if mae_score is not None:
            self.mae_score = mae_score
        if n_iter is not None:
            self.n_iter = n_iter
        if t_elapsed is not None:
            self.t_elapsed = t_elapsed

def NelderMeadSimplex(cost_fun, theta:np.ndarray, args=(), alfa=1, beta=2, gamma=0.5, step=0.05, tol=1e-8, max_iter=None, bounds=None, adaptative=False):
    '''
    :param cost_fun: pointer to the cost function of the minimization problem
    :param theta: initial point for the simplex (P[0])
    :param args: list with parameters that won't be minimized but are required to compute the cost
    :param alfa: reflection coefficient
    :param beta: expansion coefficient
    :param gamma: contraction coefficient
    :param step: step to generate the Simplex from P[0]
    :param tol: tolerance of the algorithm for stop criteria (std(y) < tol)
    :param max_iter: total allowed iterations of the algorithm
    :param bounds: constraints of the problem
    :param adaptative: flag to enable parameter adaptation to the dimension of the problem
    :return the point at the simplex's vertex that minimized the cost
    '''

    #control variables of the algorithm
    n = len(theta) #points that define the simplex
    y_idx = np.arange(0,n+1,1) #indexes to extract l, h, and s

    #handle 'max_iter'
    if max_iter is None:
        max_iter = n*200

    #handle 'bounds'
    if bounds is not None:
        if isinstance(bounds, list):
            bounds = np.array(bounds) #convert to numpy array
            lower_bounds = bounds[:,0] #lower bounds for each parameter
            upper_bounds = bounds[:,1] #uppper bounds for each parameter
        else:
            lower_bounds = bounds[:,0] #lower bounds for each parameter
            upper_bounds = bounds[:,1] #uppper bounds for each parameter

    #handle 'adaptative'
    if adaptative:
        alfa = 1
        beta = 1+2/n
        gamma = 0.75-1/(2*n)

    #define the initial simplex
    step_mtx = step*np.roll(np.eye(n+1,n),0) #matrix that defines the vertices
    simplex = step_mtx + np.tile(theta[:,np.newaxis], n+1).T #apply the step

    #nelder-mead simplex algorithm
    iter = 0 #counter to monitor iterations
    while True:
        iter += 1 #update the iteration counter

        #apply constraints if required
        if bounds is not None:
            simplex = np.clip(simplex, 0, 10)

        #compute the cost at each vertex of the simplex
        y = np.array([cost_fun(vertex, args) for vertex in simplex])

        #stop criterion
        delta = np.std(y)
        if delta < tol:
            break

        h = np.argmax(y) #index of the maximum cost
        yh = cost_fun(simplex[h,:], args)
        y_idx_nH = y_idx!=h #mask to ensure i!=h in the comparison
        l = np.argmin(y) #index of the minimum cost
        yl = cost_fun(simplex[l,:], args) #update the cost at the lower bound
        y_idx_cent = (y_idx!=h)&(y_idx!=l) #mask to detect second highest cost
        P_cent = np.mean(simplex[y_idx_nH], axis=0) #compute the centroid without h
        s = np.argmax(y[y_idx_cent]) #index of the second highest cost
        y_s = cost_fun(simplex[s,:], args)

        #reflection
        P_r = P_cent + alfa*(P_cent-simplex[h,:])
        y_r = cost_fun(P_r, args)

        #expansion
        if y_r < yl:
            P_e = P_cent + beta*(P_r-P_cent)
            y_e = cost_fun(P_e, args)
            if y_e < y_r:
                simplex[h,:] = P_e
            elif y_e >= y_r:
                simplex[h,:] = P_r

        #contraction
        elif y_r >= y_s:
            if y_r < y[h]:
                simplex[h,:] = P_r

            P_c = P_cent + gamma*(simplex[h,:]-P_cent)
            y_c = cost_fun(P_c, args)

            if y_c > yh:
                simplex[y_idx_cent,:] = 0.5*(simplex[y_idx_cent,:]+simplex[l, :])
                y = np.array([cost_fun(vertex, args) for vertex in simplex])

            elif y_c <= yh:
                simplex[h,:] = P_c
        else:
            simplex[h,:] = P_r

        #iteration criteria
        if iter == max_iter:
            break

    opt_vertex = np.argmin(y)
    return simplex[opt_vertex,:]

def ring_topology(swarm_positions, swarm_costs):
    '''
    :param swarm_positions: the position of the particles in the swarm
    :param swarm_costs: cost of each particle in the swarm
    :return: the local best positions and costs of each particle in the swarm
    '''

    left_neighbors_pos = np.roll(swarm_positions, 1, axis=0)
    left_neighbors_cost = np.roll(swarm_costs, 1, axis=0)
    right_neighbors_pos = np.roll(swarm_positions, -1, axis=0)
    right_neighbors_cost = np.roll(swarm_costs, -1, axis=0)
    ring_pos = np.stack([left_neighbors_pos, swarm_positions, right_neighbors_pos]) #ring topology for positions
    ring_cost = np.stack([left_neighbors_cost, swarm_costs, right_neighbors_cost]) #ring topology for costs
    local_best = np.argmin(ring_cost, axis=0) #row wise index for the local best costs
    cols = np.arange(0,ring_cost.shape[1],1) #index of the columns
    swarm_costs[:] = ring_cost[local_best,cols] #mask the costs given the index of the local best
    swarm_positions[:,:] = ring_pos[local_best,cols,:] #mask the positions given the index of the local best

    return swarm_positions, swarm_costs

def ParticleSwarm(cost_fun, n, args=(), method='gbest', swarm_size=50, c1=2, c2=2, weight=0.8, delta=0.5, tol=1e-6, max_iter=None, bounds=None):
    '''
    :param cost_fun: pointer to the cost function of the minimization problem
    :param n: number of dimensions
    :param args: list with parameters that won't be minimized but are required to compute the cost
    :param method: algorithm that sets which particle will be used as the best (lbest by default)
    :param swarm_size: number of particles in a swarm
    :param c1: cognitive acceleration
    :param c2: social acceleration
    :param weight: inertia weight
    :param delta: velocity clamping factor
    :param tol: tolerance of the algorithm for stop criteria
    :param max_iter: total allowed iterations of the algorithm
    :param bounds: constraints of the problem
    :return the best particle that minimized the cost
    '''

    #validate method
    valid_methods = ['gbest', 'lbest']
    if method not in valid_methods:
        raise ValueError(f'[ParticleSwarm] Method {method} is not valid! Try: {valid_methods}')

    #handle iteration
    if max_iter is None:
        max_iter = 400 #best performance overall

    #randomly generate the positions and velocities of the swarm
    bounds = (0,10)
    swarm_positions = np.random.uniform(bounds[0], bounds[1], size=(swarm_size, n)) #array to store the positions

    #compute the cost for the current particles
    swarm_costs = np.zeros(shape=(swarm_size,))
    for cost_idx in range(0,len(swarm_costs)):
        swarm_costs[cost_idx] = cost_fun(swarm_positions[cost_idx,:], args)

    swarm_velocities = np.random.uniform(bounds[0], bounds[1], size=(swarm_size, n)) #array to store the positions
    P_best = np.copy(swarm_positions) #the best position found by each particle
    cost_P_best = np.copy(swarm_costs) #the cost at the best position found by each particle

    if method == 'lbest':
        G_best, cost_G_best = ring_topology(P_best, cost_P_best) #find the best local particle in a neighborhood of 3
    elif method == 'gbest':
        idx_best = np.argmin(cost_P_best) #find the index of the best cost up until now
        G_best = P_best[idx_best,:]*np.ones_like(swarm_positions) #find the position of the best particle in the swarm

    n_iter = 0 #variable to monitor the iterations
    while True:
        #iteration stop criteria
        n_iter += 1
        if n_iter == max_iter:
            break

        r1,r2 = np.random.uniform(0, 1, 2) #random uniform values
        swarm_velocities = weight*swarm_velocities + c1*r1*(P_best-swarm_positions) + c2*r2*(G_best-swarm_positions) #evaluate the new velocity

        #apply velocity clamping
        v_max = delta*(bounds[1]-bounds[0])
        swarm_velocities[swarm_velocities>=v_max] = v_max

        swarm_positions += swarm_velocities #update the positions given the velocity
        swarm_positions = np.clip(swarm_positions, bounds[0], bounds[1]) #respect the bounds

        #update the costs
        for cost_idx in range(0, swarm_size):
            swarm_costs[cost_idx] = cost_fun(swarm_positions[cost_idx,:], args)

        cost_mask = swarm_costs<cost_P_best #find where the new positions return the better costs
        P_best[cost_mask] = swarm_positions[cost_mask] #update the best positions
        cost_P_best[cost_mask] = swarm_costs[cost_mask] #update the best costs

        if method == 'lbest':
            G_best, cost_G_best = ring_topology(P_best, cost_P_best)

            #swarm distance stop criteria
            swarm_dist = cdist(swarm_positions, swarm_positions) #compute the Euclidean distance between each particle
            if np.linalg.norm(swarm_dist[:,0])<tol:
                break

        elif method == 'gbest':
            idx_best = np.argmin(cost_P_best) #find the new minimum
            G_best = P_best[idx_best, :]*np.ones_like(swarm_positions) #update new global best

            #swarm distance stop criteria
            swarm_dist = cdist(swarm_positions, G_best) #compute the Euclidean distance between each particle and the global best
            if np.linalg.norm(swarm_dist[:,0])<tol:
                break

    return P_best[np.argmin(cost_P_best)]

def linesearch_wrapper(cost_fun, theta:np.ndarray, f_n:float, f0:float, pk:float, nabla:np.array, args=(), c1=1e-4, c2=0.9, eps=np.sqrt(np.finfo(float).eps)):
    '''
    :param cost_fun: the function to be minimized
    :param theta: current point
    :param f_n: function evaluation at the current iteration
    :param f0: function evaluation at the previous iteration
    :param pk: direction of the line search
    :param nabla: numerical gradient computed at the current point
    :param args: list with parameters that won't be minimized but are required to compute the cost
    :param c1: Armijo condition rule
    :param c2: curvature condition rule
    :param eps: epsilon (step of the gradient computation)
    :return: wrapper to the SciPy function that computes the step length alpha_k, satisfying the Wolfe conditions
    '''

    global_grad = [nabla] #list to access the gradient cost

    def compute_cost(step):
        '''
        :param step: step size in the direction of the gradient
        :return: the function evaluated at the current step in direction of the gradient
        '''

        return cost_fun(theta+step*pk, args)

    def compute_grad_cost(step):
        '''
        :param step: step size in the direction of the gradient
        :return: gradient of the function evaluated at the current step in direction of the gradient
        '''

        global_grad[0] = first_order_forward_grad(cost_fun, theta+step*pk, args, eps=eps)
        return global_grad[0]@pk

    d_phi = nabla@pk #step and direction at the first point
    alpha_k, f_n, f0 = scipy.optimize._linesearch.scalar_search_wolfe1(compute_cost, compute_grad_cost, f_n, f0, d_phi,
                       c1=c1, c2=c2, amax=1e100, amin=1e-100, xtol=1e-14) #compute the line search

    return alpha_k, f_n, f0, global_grad[0]

def BFGS(cost_fun, x0:np.ndarray, args=(), c1=1e-4, c2=0.9, eps=np.sqrt(np.finfo(float).eps), max_iter=None, bounds=None, tol=1e-6):
    '''
    :param cost_fun: pointer to the cost function of the minimization problem
    :param x0: initial guess for the levenberg-marquardt algorithm
    :param args: list with parameters that won't be minimized but are required to compute the cost
    :param c1: Armijo condition rule
    :param c2: curvature condition rule
    :param eps: epsilon (step of the gradient computation)
    :param bounds: constraints of the problem
    :param max_iter: total allowed iterations of the algorithm
    :param tol: tolerance of the algorithm for stop criteria (norm(J.W) < tol)
    :return: the candidate values for Z that minimized the cost function
    '''

    theta = np.copy(x0) #copy the initial guess to avoid mutability
    n_params = len(theta) #number of candidate values

    #handle 'max_iter'
    if max_iter is None:
        max_iter = 100*n_params**2

    #handle bounds
    if bounds is not None:
        if isinstance(bounds, tuple):
            lb, ub = bounds
        else:
            bounds = np.array(bounds)
            lb = bounds[0,0]
            ub = bounds[0,1]
    else:
        lb, ub = 0, np.inf

    #first computations with the initial guess
    y_hat = cost_fun(theta, args) #compute the function at the first point
    grad_hat = first_order_forward_grad(cost_fun, theta, args, eps=eps) #compute the gradient at the first point
    y_hat_past = y_hat + np.linalg.norm(grad_hat)/2 #condition the step guess dx~1
    I = np.eye(n_params, dtype=int) #identity matrix with dimensions matching the number of parameters
    H_k = np.copy(I) #approximation of the inverse of the Hessian
    n_iter = 0 #variable to monitor iterations
    converged = False #variable to monitor convergence

    while True:
        p_k = -H_k@grad_hat #direction oppose to the gradient
        alpha_k, y_hat, y_hat_past, curr_grad = linesearch_wrapper(cost_fun, theta, y_hat, y_hat_past, p_k, grad_hat,
                                                                   args=args, c1=c1, c2=c2, eps=eps)

        #if alpha returns None, the Wolfe conditions aren't satisfied and the optimization terminates
        if alpha_k is None:
            break

        s_k = alpha_k*p_k # x[n]-x[n-1]
        theta += s_k #update guess with step size and direction
        #theta = np.clip(theta, lb, ub) #handle points outside the constraints

        #in case the gradient update returns None (fails to satisfy the Wolfe conditions)
        if curr_grad is None:
            curr_grad = first_order_forward_grad(cost_fun, theta, args, eps=eps)

        #convergence condition
        if np.linalg.norm(curr_grad)<tol:
            converged = True
            break

        #update the inverse Hessian approximation
        y_k = curr_grad-grad_hat #update gradient difference
        rho_k = y_k@s_k
        rho_k = 1/rho_k  # 1/(yk.T@sk)
        left_side = I-(rho_k*s_k[:,np.newaxis]@y_k[np.newaxis,:])
        right_side = I-(rho_k*y_k[:,np.newaxis]@s_k[np.newaxis,:])
        H_k = left_side@H_k@right_side + (rho_k*s_k[:,np.newaxis]@s_k[np.newaxis,:]) #BFGS update equation

        #update variables
        grad_hat = curr_grad #update gradient
        n_iter += 1
        if n_iter == max_iter:
            break

    opt_result = {"theta": theta, "nit": n_iter, "fun": y_hat, "converged": converged}

    return opt_result
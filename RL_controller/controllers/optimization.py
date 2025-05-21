"""
基于控制屏障函数(CBF)的优化算法。
"""
import numpy as np
from cvxopt import matrix, solvers
import cvxpy as cp
from scipy.linalg import sqrtm

# 关闭cvxopt的进度显示
solvers.options['show_progress'] = False


def cbf_opt(env, a_rl, pred_dict):
    """
    基于控制屏障函数(CBF)方法的风险约束优化。
    不仅考虑当前风险约束的满足，还考虑未来风险的趋势/梯度。
    
    :param env: 环境对象
    :param a_rl: RL代理产生的动作
    :param pred_dict: 预测字典，包含'shortterm'预测价格变化
    :return: (控制器修正动作, 优化问题是否可解)
    """
    # 获取预测的价格变化
    pred_prices_change = pred_dict['shortterm']

    # 检查RL动作的有效性
    a_rl = np.array(a_rl)
    assert np.sum(np.abs(a_rl)) - 1 < 0.0001, f"The sum of a_rl is not equal to 1, the value is {np.sum(a_rl)}, a_rl list: {a_rl}"
    
    # 定义问题维度和参数
    N = env.stock_num
    
    # 获取过去N天每只股票的每日回报率 (num_of_stocks, lookback_days)
    daily_return_ay = env.ctl_state[f'DAILYRETURNS-{env.config.dailyRetun_lookback}']
    
    # 计算当前时间点的协方差和风险
    cov_r_t0 = np.cov(daily_return_ay)
    w_t0 = np.array([env.actions_memory[-1]])
    
    try:
        risk_stg_t0 = np.sqrt(np.matmul(np.matmul(w_t0, cov_r_t0), w_t0.T)[0][0])
    except Exception as error:
        print(f"Risk-(MV model variance): {error}")
        risk_stg_t0 = 0
        
    # 设置风险参数
    risk_market_t0 = env.config.risk_market
    if len(env.risk_adj_lst) <= 1:
        risk_safe_t0 = env.risk_adj_lst[-1]
    else:
        if env.is_last_ctrl_solvable:
            risk_safe_t0 = env.risk_adj_lst[-2]
        else:
            risk_safe_t0 = risk_stg_t0 + risk_market_t0

    # CBF参数
    gamma = env.config.cbf_gamma
    risk_market_t1 = env.config.risk_market  
    risk_safe_t1 = env.risk_adj_lst[-1] 

    # 预测下一时间点的协方差
    pred_prices_change_reshape = np.reshape(pred_prices_change, (-1, 1))
    r_t1 = np.append(daily_return_ay[:, 1:], pred_prices_change_reshape, axis=1)

    cov_r_t1 = np.cov(r_t1)
    cov_sqrt_t1 = sqrtm(cov_r_t1)
    cov_sqrt_t1 = cov_sqrt_t1.real
    
    # 初始化约束矩阵
    G_ay = np.array([]).reshape(-1, N)
    h_0 = np.array([])
    
    # 选择求解器工具，规模小用cvxopt，规模大用cvxpy
    use_cvxopt_threshold = 10  # 当投资组合大小小于等于10时使用cvxopt，否则使用cvxpy
    w_lb = 0  # 权重下限
    w_ub = 1  # 权重上限

    if env.config.topK <= use_cvxopt_threshold:
        # 使用cvxopt实现
        return _solve_with_cvxopt(env, a_rl, N, G_ay, h_0, 
                                 risk_market_t0, risk_stg_t0, risk_safe_t0,
                                 risk_market_t1, risk_safe_t1, gamma,
                                 cov_r_t1, cov_sqrt_t1)
    else:
        # 使用cvxpy实现
        return _solve_with_cvxpy(env, a_rl, N, 
                                risk_market_t0, risk_stg_t0, risk_safe_t0,
                                risk_market_t1, risk_safe_t1, gamma,
                                cov_r_t1, cov_sqrt_t1, w_lb, w_ub)


def _solve_with_cvxopt(env, a_rl, N, G_ay, h_0, 
                      risk_market_t0, risk_stg_t0, risk_safe_t0,
                      risk_market_t1, risk_safe_t1, gamma,
                      cov_r_t1, cov_sqrt_t1):
    """
    使用cvxopt求解优化问题。适用于较小规模的投资组合。
    
    :return: (控制器修正动作, 优化问题是否可解)
    """
    # 设置等式约束
    A_eq = np.array([]).reshape(-1, N)
    linear_g1 = np.array([[1.0] * N])  # (1, N)
    A_eq = np.append(A_eq, linear_g1, axis=0)
    A_eq = matrix(A_eq)
    b_eq = np.array([0.0])
    b_eq = matrix(b_eq)

    # 设置不等式约束
    h_0 = np.append(h_0, a_rl, axis=0)  # 0 <= (a_RL + a_cbf)
    h_0 = np.append(h_0, 1-a_rl, axis=0)  # (a_RL + a_cbf) <= 1

    linear_g3 = np.diag([-1.0] * N)
    G_ay = np.append(G_ay, linear_g3, axis=0)  # 0 <= (a_RL + a_cbf)
    linear_g4 = np.diag([1.0] * N) 
    G_ay = np.append(G_ay, linear_g4, axis=0)  # (a_RL + a_cbf) <= 1
    
    # 计算CBF约束
    last_h_risk = (-risk_market_t0 - risk_stg_t0 + risk_safe_t0)
    last_h_risk = np.max([last_h_risk, 0.0])
    socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk

    # 风险约束逐步放宽的步长
    step_add_lst = [0.002, 0.002, 0.002, 0.002, 0.002, 0.005, 0.005, 0.005, 0.005, 0.005]
    cnt = 1
    
    # 确定迭代次数
    if env.config.is_enable_dynamic_risk_bound:
        cnt_th = env.config.ars_trial  # 迭代风险放宽
    else:
        cnt_th = 1 

    # 准备SOCP约束
    socp_b = np.matmul(cov_sqrt_t1, a_rl)
    h = np.append(h_0, [socp_d], axis=0)  # socp_d
    h = np.append(h, socp_b, axis=0)  # socp_b
    h = matrix(h)
    socp_cx = np.array([[0.0] * N])
    G_ay = np.append(G_ay, -socp_cx, axis=0)
    G_ay = np.append(G_ay, -cov_sqrt_t1, axis=0)  # socp_ax
    G = matrix(G_ay)  # G = matrix(np.transpose(np.transpose(G_ay)))

    # 设置锥规划问题的维度
    linear_eq_num = 2*N
    dims = {'l': linear_eq_num, 'q': [N+1], 's': []}
    
    # 设置二次规划参数
    QP_P = matrix(np.eye(N)) * 2  # (1/2) xP'x
    QP_Q = matrix(np.zeros((N, 1)))  # q'x
    
    # 尝试求解，如果不可解则放宽风险约束
    while cnt <= cnt_th:
        try:
            sol = solvers.coneqp(QP_P, QP_Q, G, h, dims, A_eq, b_eq)
            if sol['status'] == 'optimal':
                solver_flag = True
                break
            else:
                raise Exception("Solution not optimal")
        except:
            solver_flag = False
            cnt += 1
            risk_safe_t1 = risk_safe_t1 + step_add_lst[cnt-2]
            socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk
            h = np.append(h_0, [socp_d], axis=0)  # socp_d
            h = np.append(h, socp_b, axis=0)  # socp_b 
            h = matrix(h)

    # 处理求解结果
    if solver_flag:
        if sol['status'] == 'optimal':
            a_cbf = np.reshape(np.array(sol['x']), -1)
            env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
            is_solvable_status = True
            env.risk_adj_lst[-1] = risk_safe_t1
            
            # 检查解是否满足风险约束
            cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl+a_cbf), cov_r_t1), (a_rl+a_cbf).T))
            assert (cur_alpha_risk - socp_d) <= 0.00001, f'cur risk: {cur_alpha_risk}, socp_d {socp_d}'
            assert np.abs(np.sum(np.abs((a_rl+a_cbf))) - 1) <= 0.00001, f'sum of actions: {np.sum(np.abs((a_rl+a_cbf)))}, {a_rl}, {a_cbf}'
            env.solvable_flag.append(0)
        else:
            a_cbf = np.zeros(N)
            env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
            is_solvable_status = False
            cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T)) 
            env.solvable_flag.append(1)           
    else:
        a_cbf = np.zeros(N)
        env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
        is_solvable_status = False
        cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))
        env.solvable_flag.append(1)
    
    # 更新环境的风险预测和求解标志
    env.risk_pred_lst.append(cur_alpha_risk)    
    env.is_last_ctrl_solvable = is_solvable_status
    if cnt > 1:
        env.stepcount = env.stepcount + 1

    return a_cbf, is_solvable_status


def _solve_with_cvxpy(env, a_rl, N, 
                     risk_market_t0, risk_stg_t0, risk_safe_t0,
                     risk_market_t1, risk_safe_t1, gamma,
                     cov_r_t1, cov_sqrt_t1, w_lb, w_ub):
    """
    使用cvxpy求解优化问题。适用于较大规模的投资组合。
    
    :return: (控制器修正动作, 优化问题是否可解)
    """
    # 重整RL动作为列向量
    a_rl_re_sign = np.reshape(a_rl, (-1, 1))
    sign_mul = np.ones((1, N))
    
    # 计算CBF约束
    last_h_risk = (-risk_market_t0 - risk_stg_t0 + risk_safe_t0)
    last_h_risk = np.max([last_h_risk, 0.0])
    socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk

    # 风险约束逐步放宽的步长
    step_add_lst = [0.002, 0.002, 0.002, 0.002, 0.002, 0.005, 0.005, 0.005, 0.005, 0.005]
    cnt = 1
    
    # 确定迭代次数
    if env.config.is_enable_dynamic_risk_bound:
        cnt_th = env.config.ars_trial  # 迭代风险放宽
    else:
        cnt_th = 1
    
    # 使用cvxpy设置和求解问题
    cp_x = cp.Variable((N, 1))
    a_rl_re = np.reshape(a_rl, (-1, 1))
    
    # 定义约束
    cp_constraint = []
    cp_constraint.append(cp.sum(sign_mul@cp_x) + cp.sum(a_rl_re_sign) == 1)
    cp_constraint.append(a_rl_re + cp_x >= w_lb) 
    cp_constraint.append(a_rl_re + cp_x <= w_ub) 
    cp_constraint.append(cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re + cp_x))) 
    
    # 尝试求解，如果不可解则放宽风险约束
    while cnt <= cnt_th:
        try:
            obj_f2 = cp.sum_squares(cp_x)
            cp_obj = cp.Minimize(obj_f2)
            cp_prob = cp.Problem(cp_obj, cp_constraint)
            cp_prob.solve(solver=cp.ECOS, verbose=False) 

            if cp_prob.status == 'optimal':
                solver_flag = True
                break
            else:
                raise Exception("Solution not optimal")
        except:
            solver_flag = False
            cnt += 1
            risk_safe_t1 = risk_safe_t1 + step_add_lst[cnt-2]
            socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk
            cp_constraint[-1] = cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re + cp_x))

    # 处理求解结果
    if (cp_prob.status == 'optimal') and solver_flag:
        is_solvable_status = True
        a_cbf = np.reshape(np.array(cp_x.value), -1)
        env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
        
        # 更新风险调整列表
        env.risk_adj_lst[-1] = risk_safe_t1
        
        # 检查解是否满足风险约束
        cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl+a_cbf), cov_r_t1), (a_rl+a_cbf).T))
        assert (cur_alpha_risk - socp_d) <= 0.00001, f'cur risk: {cur_alpha_risk}, socp_d {socp_d}' 
        assert np.abs(np.sum(np.abs(a_rl+a_cbf)) - 1) <= 0.00001, f'sum of actions: {np.sum(np.abs((a_rl+a_cbf)))}, {a_rl}, {a_cbf}'
        env.solvable_flag.append(0)
    else:
        a_cbf = np.zeros(N)
        env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
        is_solvable_status = False
        env.solvable_flag.append(1)
        cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))
        env.risk_adj_lst[-1] = risk_safe_t1
    
    # 更新环境的风险预测和求解标志
    env.risk_pred_lst.append(cur_alpha_risk)    
    env.is_last_ctrl_solvable = is_solvable_status
    if cnt > 1:
        env.stepcount = env.stepcount + 1

    return a_cbf, is_solvable_status
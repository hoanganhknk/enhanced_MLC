import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])
@torch.no_grad()

def update_params(params, grads, eta, opt, args, deltaonly=False, return_s=False):
    if isinstance(opt, torch.optim.SGD):
        return update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s)
    else:
        raise NotImplementedError('Non-supported main model optimizer type!')
def update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s=False):
    ans = []
    if return_s:
        ss = []
    wdecay = opt.defaults['weight_decay']
    momentum = opt.defaults['momentum']
    dampening = opt.defaults['dampening']
    nesterov = opt.defaults['nesterov']
    for i, param in enumerate(params):
        dparam = grads[i] + param * wdecay # s=1
        s = 1
        if momentum > 0:
            try:
                moment = opt.state[param]['momentum_buffer'] * momentum
            except:
                moment = torch.zeros_like(param)
            moment.add_(dparam, alpha=1. -dampening) # s=1.-dampening
            if nesterov:
                dparam = dparam + momentum * moment # s= 1+momentum*(1.-dampening)
                s = 1 + momentum*(1.-dampening)
            else:
                dparam = moment # s=1.-dampening
                s = 1.-dampening
        if deltaonly:
            ans.append(- dparam * eta)
        else:
            ans.append(param - dparam  * eta)
        if return_s:
            ss.append(s*eta)
    if return_s:
        return ans, ss
    else:
        return ans
def step_mlc(main_net, main_opt, hard_loss_f,
              meta_net, meta_opt, soft_loss_f,
              data_s, target_s, data_g, target_g,
              data_c, target_c,
              eta, args):
    # compute gw for updating meta_net
    logit_g = main_net(data_g)
    loss_g = hard_loss_f(logit_g, target_g)
    gw = torch.autograd.grad(loss_g, main_net.parameters())
    
    # given current meta net, get corrected label
    logit_s, x_s_h = main_net(data_s, return_h=True)
    pseudo_target_s = meta_net(x_s_h.detach(), target_s)
    loss_s = soft_loss_f(logit_s, pseudo_target_s)

    if data_c is not None:
        bs1 = target_s.size(0)
        bs2 = target_c.size(0)

        logit_c = main_net(data_c)
        loss_s2 = hard_loss_f(logit_c, target_c)
        loss_s = (loss_s * bs1 + loss_s2 * bs2 ) / (bs1+bs2)

    f_param_grads = torch.autograd.grad(loss_s, main_net.parameters(), create_graph=True)    

    f_params_new, dparam_s = update_params(main_net.parameters(), f_param_grads, eta, main_opt, args, return_s=True)
    f_param = []
    for i, param in enumerate(main_net.parameters()):
        f_param.append(param.data.clone())
        param.data = f_params_new[i].data 
    
    Hw = 1 
    logit_g = main_net(data_g)
    loss_g  = hard_loss_f(logit_g, target_g)
    gw_prime = torch.autograd.grad(loss_g, main_net.parameters())

    tmp1 = [(1-Hw*dparam_s[i]) * gw_prime[i] for i in range(len(dparam_s))]
    gw_norm2 = (_concat(gw).norm())**2
    tmp2 = [gw[i]/gw_norm2 for i in range(len(gw))]
    gamma = torch.dot(_concat(tmp1), _concat(tmp2))

    Lgw_prime = [ dparam_s[i] * gw_prime[i] for i in range(len(dparam_s))]     

    proxy_g = -torch.dot(_concat(f_param_grads), _concat(Lgw_prime))

    meta_opt.zero_grad()
    proxy_g.backward()

    for i, param in enumerate(meta_net.parameters()):
        if param.grad is not None:
            param.grad.add_(gamma * args.dw_prev[i])
            args.dw_prev[i] = param.grad.clone()

    if (args.steps+1) % (args.gradient_steps)==0: 
        meta_opt.step()
        args.dw_prev = [0 for param in meta_net.parameters()] 

    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]
        param.grad = f_param_grads[i].data
    main_opt.step()
    
    return loss_g, loss_s

def step_mlcbome(main_net, main_opt, hard_loss_f,
                meta_net, meta_opt, soft_loss_f,
                data_s, target_s, data_g, target_g,
                data_c, target_c, 
                eta, args):
    logit_g = main_net(data_g)
    loss_g = hard_loss_f(logit_g, target_g)
    gw = torch.autograd.grad(loss_g, main_net.parameters())

    logit_s, x_s_h = main_net(data_s, return_h=True)
    pseudo_target_s = meta_net(x_s_h.detach(), target_s)
    loss_s = soft_loss_f(logit_s, pseudo_target_s)
    if data_c is not None:
        bs1 = target_s.size(0) 
        bs2 = target_c.size(0)

        logit_c = main_net(data_c)
        loss_s2 = hard_loss_f(logit_c, target_c)
        loss_s = (loss_s * bs1 + loss_s2 * bs2 ) / (bs1+bs2)
    gradient_g_mainparam = torch.autograd.grad(loss_s, main_net.parameters(), create_graph=True)
    gradient_g_metaparam = torch.autograd.grad(loss_s, meta_net.parameters(), create_graph=True) 
    main_net_backup = copy.deepcopy(main_net)
    f_params_new = update_params(main_net.parameters(), gradient_g_mainparam, eta, main_opt, args, return_s=False)
    for i, param in enumerate(main_net.parameters()):
        param.data = f_params_new[i]
    main_opt.step()
    main_opt.zero_grad()
    for i in range (4):
        logit_s, x_s_h = main_net(data_s, return_h=True)
        pseudo_target_s = meta_net(x_s_h.detach(), target_s)
        loss_s = soft_loss_f(logit_s, pseudo_target_s)
        if data_c is not None:
            bs1 = target_s.size(0) 
            bs2 = target_c.size(0)

            logit_c = main_net(data_c)
            loss_s2 = hard_loss_f(logit_c, target_c)
            loss_s = (loss_s * bs1 + loss_s2 * bs2 ) / (bs1+bs2)
        g_grad = torch.autograd.grad(loss_s, main_net.parameters(), create_graph=True)
        f_params_new = update_params(main_net.parameters(), g_grad, eta, main_opt, args, return_s=False)
        for i, param in enumerate(main_net.parameters()):
            param.data = f_params_new[i]
        main_opt.step()
        main_opt.zero_grad()
    logit_s, x_s_h = main_net(data_s, return_h=True)
    pseudo_target_s = meta_net(x_s_h.detach(), target_s)
    loss_s = soft_loss_f(logit_s, pseudo_target_s)
    if data_c is not None:
        bs1 = target_s.size(0) 
        bs2 = target_c.size(0)

        logit_c = main_net(data_c)
        loss_s2 = hard_loss_f(logit_c, target_c)
        loss_s = (loss_s * bs1 + loss_s2 * bs2 ) / (bs1+bs2)
    grad_g_mainparam_new = list(torch.autograd.grad(loss_s, main_net.parameters(), create_graph=True))
    grad_g_metaparam_new = list(torch.autograd.grad(loss_s, meta_net.parameters(), create_graph=True))
    
    for i in range(len(grad_g_mainparam_new)):
        grad_g_mainparam_new[i] = gradient_g_mainparam[i] - grad_g_mainparam_new[i]
    for i in range(len(grad_g_metaparam_new)):
        grad_g_metaparam_new[i] = gradient_g_metaparam[i] - grad_g_metaparam_new[i]
    n_params_meta = sum([p.numel() for p in meta_net.parameters()])
    zz = torch.zeros(n_params_meta, device='cuda')
    dq = torch.cat([grad_g_mainparam_new[i].view(-1) for i in range(len(grad_g_mainparam_new))]
                    + [grad_g_metaparam_new[i].view(-1) for i in range(len(grad_g_metaparam_new))])
    df = torch.cat([gw[i].view(-1) for i in range(len(gw))] + [zz])
    norm_dq = dq.norm().pow(2)
    dot = torch.dot(dq, df)
    beta = F.relu((0.5*norm_dq - dot)/(norm_dq + 1e-8))
    for p_old, p_new in zip(main_net.parameters(), main_net_backup.parameters()):
        p_old.data.copy_(p_new.data)
    for i, param in enumerate(main_net.parameters()):
        param.grad = beta*grad_g_mainparam_new[i].data + gw[i].data
    for i, param in enumerate(meta_net.parameters()):
        param.grad = beta*grad_g_metaparam_new[i].data
    main_opt.step()
    meta_opt.step()
    return loss_g, loss_s
def step_ebomlc(main_net, main_opt, hard_loss_f,
                meta_net, meta_opt, soft_loss_f,
                data_s, target_s, data_g, target_g,
                data_c, target_c, 
                eta, args):
    logit_g, x_s_h = main_net(data_g, return_h=True)
    target_g_from_meta = meta_net(x_s_h.detach(), target_g)
    upper_loss = hard_loss_f(args.rho*logit_g + (1-args.rho)*target_g_from_meta, target_g)
    gradient_f = torch.autograd.grad(upper_loss, main_net.parameters(), create_graph=True)
    gradient_f = update_params(main_net.parameters(), gradient_f, eta, main_opt, args, deltaonly=True, return_s=False)
    gradient_f_2 = torch.autograd.grad(upper_loss, meta_net.parameters(), create_graph=True)

    logit_s, x_s_h = main_net(data_s, return_h=True)
    pseudo_target_s = meta_net(x_s_h.detach(), target_s)
    lower_loss = soft_loss_f(logit_s, pseudo_target_s)
    if data_c is not None:
        bs1 = target_s.size(0)
        bs2 = target_c.size(0)
        logit_c = main_net(data_c)
        loss_s2 = hard_loss_f(logit_c, target_c)
        lower_loss = (lower_loss * bs1 + loss_s2 * bs2) / (bs1 + bs2)

    gradient_g_mainparam = torch.autograd.grad(lower_loss, main_net.parameters(), create_graph=True)

    f_params_new = update_params(main_net.parameters(), gradient_g_mainparam, eta, main_opt, args, return_s=False)
    for i, param in enumerate(main_net.parameters()):
        param.data = f_params_new[i]
    logit_s, x_s_h = main_net(data_s, return_h=True)
    pseudo_target_s = meta_net(x_s_h.detach(), target_s)
    lower_loss_new = soft_loss_f(logit_s, pseudo_target_s)
    if data_c is not None:
        bs1 = target_s.size(0)
        bs2 = target_c.size(0)
        logit_c = main_net(data_c)
        loss_s2 = hard_loss_f(logit_c, target_c)
        lower_loss_new = (lower_loss_new * bs1 + loss_s2 * bs2) / (bs1 + bs2)
    grad_g_mainparam_new = list(torch.autograd.grad(lower_loss - lower_loss_new, main_net.parameters(), create_graph=True))
    grad_g_metaparam_new = list(torch.autograd.grad(lower_loss - lower_loss_new, meta_net.parameters(), create_graph=True))
    
    dq = torch.cat([grad_g_mainparam_new[i].view(-1) for i in range(len(grad_g_mainparam_new))]
                    + [grad_g_metaparam_new[i].view(-1) for i in range(len(grad_g_metaparam_new))])
    d_wq = torch.cat([grad_g_mainparam_new[i].view(-1) for i in range(len(grad_g_mainparam_new))])
    df = torch.cat([gradient_f[i].view(-1) for i in range(len(gradient_f))] )
    norm_dq = dq.norm().pow(2)
    dot = torch.dot(d_wq, df)
    beta = args.m*F.relu((args.delta*norm_dq - dot)/(norm_dq + 1e-8))
    if args.dataset != 'cifar10':
        grad_g_mainparam_new = update_params(main_net.parameters(), grad_g_mainparam_new, eta, main_opt, args, deltaonly=True, return_s=False)
    for i, param in enumerate(main_net.parameters()):
        param.grad = beta*grad_g_mainparam_new[i].data + gradient_f[i].data
    for i, param in enumerate(meta_net.parameters()):
        param.grad = beta*grad_g_metaparam_new[i].data + gradient_f_2[i].data
    main_opt.step()
    meta_opt.step()
    main_opt.zero_grad()
    meta_opt.zero_grad()
    return upper_loss, lower_loss
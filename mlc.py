import torch
import torch.nn as nn
import torch.nn.functional as F
def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])
@torch.no_grad()
def update_params(params, grads, eta, opt, args, deltaonly=False, return_s=False):
    if isinstance(opt, torch.optim.SGD):
        return update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s)
    else:
        raise NotImplementedError('Non-supported main model optimizer type!')
# be aware that the opt state dict returns references, hence take care not to
# modify them
def update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s=False):
    # supports SGD-like optimizers
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
# ============== mlc step procedure debug with features (gradient-stopped) from main model ===========
#
# METANET uses the last K-1 steps from main model and imagine one additional step ahead
# to compose a pool of actual K steps from the main model
#
#
def step_hmlc_K(main_net, main_opt, hard_loss_f,
                meta_net, meta_opt, soft_loss_f,
                data_s, target_s, data_g, target_g,
                data_c, target_c, 
                eta, args):
    # Triển khai thuật toán BOME (Bi-level Optimization Made Easy) cho mô hình MLC
    # hàm loss upper sẽ là f, hàm loss lower sẽ là g
    # tính gradient cho g
    logit_g, x_s_h = main_net(data_g, return_h=True)
    target_g_from_meta = meta_net(x_s_h.detach(), target_g)
    loss_g = hard_loss_f(0.2*logit_g + 0.8*target_g_from_meta, target_g)
    gradient_f = torch.autograd.grad(loss_g, main_net.parameters(), create_graph=True)
    gradient_f = update_params(main_net.parameters(), gradient_f, eta, main_opt, args, deltaonly=True, return_s=False)
    gradient_f_2 = torch.autograd.grad(loss_g, meta_net.parameters(), create_graph=True)
    main_opt.zero_grad()
    meta_opt.zero_grad()
    logit_s, x_s_h = main_net(data_s, return_h=True)
    pseudo_target_s = meta_net(x_s_h.detach(), target_s)
    loss_s = soft_loss_f(logit_s, pseudo_target_s)
    if data_c is not None:
        bs1 = target_s.size(0)
        bs2 = target_c.size(0)
        logit_c = main_net(data_c)
        loss_s2 = hard_loss_f(logit_c, target_c)
        loss_s = (loss_s * bs1 + loss_s2 * bs2) / (bs1 + bs2)

    # tính gradient cho g ban đầu để chuẩn bị cho thuật toán BOME
    gradient_g_mainparam = torch.autograd.grad(loss_s, main_net.parameters(), create_graph=True)
    gradient_g_metaparam = torch.autograd.grad(loss_s, meta_net.parameters(), create_graph=True)
    
    f_params_new = update_params(main_net.parameters(), gradient_g_mainparam, eta, main_opt, args, return_s=False)
    for i, param in enumerate(main_net.parameters()):
        param.data = f_params_new[i]
    logit_s, x_s_h = main_net(data_s, return_h=True)
    pseudo_target_s = meta_net(x_s_h.detach(), target_s)
    loss_s = soft_loss_f(logit_s, pseudo_target_s)
    if data_c is not None:
        bs1 = target_s.size(0)
        bs2 = target_c.size(0)
        logit_c = main_net(data_c)
        loss_s2 = hard_loss_f(logit_c, target_c)
        loss_s = (loss_s * bs1 + loss_s2 * bs2) / (bs1 + bs2)
    grad_g_mainparam_new = list(torch.autograd.grad(loss_s, main_net.parameters(), create_graph=True))
    grad_g_metaparam_new = list(torch.autograd.grad(loss_s, meta_net.parameters(), create_graph=True))
    # f_params_new = update_params(main_net.parameters(), grad_g_mainparam_new, eta, main_opt, args, return_s=False)
    # for i, param in enumerate(main_net.parameters()):
    #     param.data = f_params_new[i]
    
    for i in range(len(grad_g_mainparam_new)):
        grad_g_mainparam_new[i] = gradient_g_mainparam[i] - grad_g_mainparam_new[i]
    for i in range(len(grad_g_metaparam_new)):
        grad_g_metaparam_new[i] = gradient_g_metaparam[i] - grad_g_metaparam_new[i]
    
    n_params_meta = sum([p.numel() for p in meta_net.parameters()])
    dq = torch.cat([grad_g_mainparam_new[i].view(-1) for i in range(len(grad_g_mainparam_new))]
                    + [grad_g_metaparam_new[i].view(-1) for i in range(len(grad_g_metaparam_new))])
    d_wq = torch.cat([grad_g_mainparam_new[i].view(-1) for i in range(len(grad_g_mainparam_new))])
    df = torch.cat([gradient_f[i].view(-1) for i in range(len(gradient_f))] )
    norm_dq = dq.norm().pow(2)
    dot = torch.dot(d_wq, df)
    lmda = 0.5*F.relu((0.2*norm_dq - dot)/(norm_dq + 1e-8))
    
    grad_g_mainparam_new = update_params(main_net.parameters(), grad_g_mainparam_new, eta, main_opt, args, deltaonly=True, return_s=False)
    
    for i, param in enumerate(main_net.parameters()):
        param.grad = lmda*grad_g_mainparam_new[i].data + gradient_f[i].data
    for i, param in enumerate(meta_net.parameters()):
        param.grad = lmda*grad_g_metaparam_new[i].data + gradient_f_2[i].data
        # param.grad = lmda*grad_g_metaparam_new[i].data
    main_opt.step()
    meta_opt.step()
    main_opt.zero_grad()
    meta_opt.zero_grad()
    # tính gradient mới cho g
    return loss_g, loss_s
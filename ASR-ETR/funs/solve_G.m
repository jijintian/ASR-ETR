function [G,objV] = solve_G(x,miu,sX,yita)
% solve min 1/miu ||G||_gamma+1/2||G-(Z+W/MIU)||_F^2
%矩阵旋转为 N*V*N
er = 0.0001;
G_hat=reshape(x,sX);
objV = 0;
G_hat_re = shiftdim(G_hat, 1);
G_hat_fft = fft(G_hat_re,[],3);
max_iter = 100;
dim = size(G_hat_fft,3);
n3 = dim;
%% B_tensor 做t-SVD分解，得到U_tensor，S_tensor,V_tensor
% for i = 1:dim
%     [uhat,shat,vhat] = svd(full(G_hat_fft(:,:,i)),'econ');
%     sigma = diag(shat);
%     sigma_i = sigma;
% % 得到 S_tensor 的迭代更新值，求解P_miu（a） = argmin 1/2(x-a)^2 + 1/miu f(x)
%     for j = 1: max_iter
%         w_i = rank_fun_derivative(sigma_i,yita);
%         sigma_i1 = max(0,sigma - w_i/miu);
%         if sum((sigma_i1-sigma_i).^2) < er
%             break
%         end
%         sigma_i = sigma_i1;
%     end
%     shat = diag(sigma_i1);
%     G_hat_ff(:,:,i) = uhat*shat*vhat';
% end
%% Fast methods
if isinteger(n3/2)
    endValue = int32(n3/2+1);
    for i = 1:endValue
    [uhat,shat,vhat] = svd(full(G_hat_fft(:,:,i)),'econ');
    sigma = diag(shat);
    sigma_i = sigma;
%% Solve P_miu（a） = argmin 1/2(x-a)^2 + 1/miu f(x)
    for j = 1: max_iter
        w_i = rank_fun_derivative(sigma_i,yita);
        sigma_i1 = max(0,sigma - w_i/miu);
        if sum((sigma_i1-sigma_i).^2) < er
            break
        end
        sigma_i = sigma_i1;
    end
    shat = diag(sigma_i1);
    objV = objV + sum(shat(:));
    G_hat_ff(:,:,i) = uhat*shat*vhat';
        if i > 1
            G_hat_ff(:,:,n3-i+2) = conj(uhat)*shat*conj(vhat)';
            objV = objV + sum(shat(:));
        end
    end
    [uhat,shat,vhat] = svd(full(G_hat_ff(:,:,endValue+1)),'econ');
    sigma = diag(shat);
    sigma_i = sigma;
    for j = 1: max_iter
        w_i = rank_fun_derivative(sigma_i,yita);
        sigma_i1 = max(0,sigma - w_i/miu);
        if sum((sigma_i1-sigma_i).^2) < er
            break
        end
        sigma_i = sigma_i1;
    end
    shat = diag(sigma_i1);
    objV = objV + sum(shat(:));
    G_hat_ff(:,:,endValue+1) = uhat*shat*vhat';
else
    endValue = int32(n3/2+1);
    for i = 1:endValue
        [uhat,shat,vhat] = svd(full(G_hat_fft(:,:,i)),'econ');
        sigma = diag(shat);
        sigma_i = sigma;
%% Solve P_miu（a） = argmin 1/2(x-a)^2 + 1/miu f(x)
        for j = 1: max_iter
            w_i = rank_fun_derivative(sigma_i,yita);
            sigma_i1 = max(0,sigma - w_i/miu);
            if sum((sigma_i1-sigma_i).^2) < er
                break
            end
            sigma_i = sigma_i1;
        end
        shat = diag(sigma_i1);
        objV = objV + sum(shat(:));
        G_hat_ff(:,:,i) = uhat*shat*vhat';
        if i > 1
            G_hat_ff(:,:,n3-i+2) = conj(uhat)*shat*conj(vhat)';
            objV = objV + sum(shat(:));
        end
    end
end

G = ifft(G_hat_ff,[],3);
G = shiftdim(G, 2);
end

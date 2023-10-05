function y = rank_fun_derivative1(x,delta)
y  = delta*exp(delta^2)./((x+delta).^2+eps);
%y = delta./(delta+x).^2;
end

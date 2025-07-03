%% Delta è un array di fasi 
function res = array_factor_min(k, d, theta, delta)

res = 0;

for i=1:length(d)
    res = res + exp(1i .* (k .* d(i) .* sin(theta)+ delta(i)));
end
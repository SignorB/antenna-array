%% Delta è un valore unico, sfasamento tra le varie antenne
function res = array_factor(k, d, theta, delta, n)

res = 0;

for i=1:n
    res = res + exp(1i .* (k .* d(i) .* sin(theta)+ delta*i));
end
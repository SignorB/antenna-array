%% Calcola array factor a partire da
% k vettore d'onda
% d array con le posizioni assolute delle antenne
% theta steering angle
% n numero di antenne
function res = array_factor(k, d, theta, delta)

res = 0;

for i=1:length(d)
    res = res + exp(1i .* (k .* d(i) .* sin(theta)+ delta(i)));
end

res = abs(res);
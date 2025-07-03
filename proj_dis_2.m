spacing = [5.06458333300000e-06	5.97187500000000e-06	5.33541666700000e-06	6.36458333300000e-06	5.74166666700000e-06	6.36458333300000e-06	5.38958333300000e-06	5.85000000000000e-06	4.65833333300000e-06	4.29270833300000e-06	4.52291666700000e-06	5.17291666700000e-06	5.30833333300000e-06	3.10104166700000e-06	3.42604166700000e-06	5.05104166700000e-06	4.33333333300000e-06	5.44375000000000e-06	4.38750000000000e-06	5.13229166700000e-06	3.30416666700000e-06	3.61562500000000e-06	4.90208333300000e-06	5.91770833300000e-06	6.36458333300000e-06	6.56770833300000e-06	5.41666666700000e-06	5.11875000000000e-06	5.97187500000000e-06	5.90416666700000e-06	5.51145833300000e-06	5.63333333300000e-06	5.51145833300000e-06	5.90416666700000e-06	5.97187500000000e-06	5.11875000000000e-06	5.41666666700000e-06	6.56770833300000e-06	6.36458333300000e-06	5.91770833300000e-06	4.90208333300000e-06	3.61562500000000e-06	3.30416666700000e-06	5.13229166700000e-06	4.38750000000000e-06	5.44375000000000e-06	4.33333333300000e-06	5.05104166700000e-06	3.42604166700000e-06	3.10104166700000e-06	5.30833333300000e-06	5.17291666700000e-06	4.52291666700000e-06	4.29270833300000e-06	4.65833333300000e-06	5.85000000000000e-06	5.38958333300000e-06	6.36458333300000e-06	5.74166666700000e-06	6.36458333300000e-06	5.33541666700000e-06	5.97187500000000e-06	5.06458333300000e-06];


%%
theta_fixed = 0;%16.38/180 * pi
a = -pi;
b = pi ;



%%
N = 64;
lb = -pi * ones(1, N);
ub = pi * ones(1, N);

position = zeros(1, length(spacing));
position(1) = 0;
for i=2:length(spacing)
    position(i) = position(i-1) + spacing(i);
end

spacing_avg = mean(spacing);
position_avg = zeros(1, length(spacing_avg));
position_avg(1) = 0;
for i=2:length(spacing)
    position_avg(i) = position_avg(i-1) + spacing_avg;
end


lambda = 1550e-9;
grid = a:2*pi/10000:b; 
k = 2 * pi / lambda;

function res = objective_lobes(x, k, position, theta_fixed)
    % Risposta in frequenza per theta da ottimizzare
    res = -abs(array_factor_min(k, position, theta_fixed, x));

    % theta = linspace(-pi, pi, 1000);
    % mask = abs(theta - theta_fixed) > deg2rad(5);
    % arm = array_factor_min(k, position, theta(mask), x);
    % side_lobe = max(abs(arm));
    % res = res  + 0.5*side_lobe;
    % 
    % side_lobe_dB = 20*log10(abs(side_lobe));
    % side_lobe_dB = side_lobe_dB - max(arm);
    % side_lobe = 10^(side_lobe_dB/20);
    % if (side_lobe > 0.2) 
    %     res = +Inf;
    % end
end 

objective = @ (x) objective_lobes(x, k, position, theta_fixed);

options = optimoptions('particleswarm', ...
    'Display', 'iter');
delta = particleswarm(objective, N, lb, ub, options);

AF_linear = array_factor_min(k, position, grid, delta);
AF_dB = 20 * log10(abs(AF_linear));
AF_dB = AF_dB - max(AF_dB);  % Normalizza a 0 dB
AF = 10.^(AF_dB./20);

%%
delta_lin = pi/8;
AF_linear = array_factor(k, position, grid, delta_lin, length(spacing));
AF_linear_avg = array_factor(k, position_avg, grid, delta_lin, length(spacing));

AF_dB_avg = 20 * log10(abs(AF_linear_avg));
AF_dB_avg = AF_dB_avg - max(AF_dB_avg);  % Normalizza a 0 dB
AF_avg = 10.^(AF_dB_avg./20);







%%

figure(1)
hold on
plot(grid * 180 / pi, AF, '-b')
plot(grid * 180 / pi, AF_avg, '-r')
xlim([-15 15])
ylim([0 1])
legend('non unf', 'unif')
xlabel('Angolo (gradi)')
ylabel('Intensità')
title('Array Factor (cartesiano)')

figure(2)
polarplot(grid, max(AF_dB, -30))  % Tronca a -30 dB per evitare -Inf
rlim([-30 0])
title('Array Factor (diagramma polare 360°)')
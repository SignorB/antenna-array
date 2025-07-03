
spacing = [5.064583333 5.971875 5.335416667 6.364583333 5.741666667 6.364583333 5.389583333 5.85 4.658333333 4.292708333 4.522916667 5.172916667 5.308333333 3.101041667 3.426041667 5.051041667 4.333333333 5.44375 4.3875 5.132291667 3.304166667 3.615625 4.902083333 5.917708333 6.364583333 6.567708333 5.416666667 5.11875 5.971875 5.904166667 5.511458333 5.633333333];
tmp = flip(spacing(1:length(spacing)-1));
num = 1:length(spacing);
spacing = [spacing tmp];
spacing = spacing .* 1e-6;

%%
theta_fixed = 10/180 * pi;
a= -pi;
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

options = optimoptions('particleswarm', 'Display', 'iter', 'UseParallel', true);
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
xlim([0 30])
ylim([0 1])
legend('non unf', 'unif')
xlabel('Angolo (gradi)')
ylabel('Intensità')
title('Array Factor (cartesiano)')

figure(2)
polarplot(grid, max(AF_dB, -30))  % Tronca a -30 dB per evitare -Inf
rlim([-30 0])
title('Array Factor (diagramma polare 360°)')
clear; clc; close all;

%% Configurazione
theta_fixed = 10/180 * pi;     % Direzione target [rad]
lambda = 1550e-9;              % Lunghezza d'onda [m]
sidelobe_threshold = -20;      % Soglia lobi laterali [dB]
gradi_maschera = 10;           % Gradi maschera [°]
    
spacing_nonuniform = [5.06458333300000e-06	5.97187500000000e-06	5.33541666700000e-06	6.36458333300000e-06	5.74166666700000e-06	6.36458333300000e-06	5.38958333300000e-06	5.85000000000000e-06	4.65833333300000e-06	4.29270833300000e-06	4.52291666700000e-06	5.17291666700000e-06	5.30833333300000e-06	3.10104166700000e-06	3.42604166700000e-06	5.05104166700000e-06	4.33333333300000e-06	5.44375000000000e-06	4.38750000000000e-06	5.13229166700000e-06	3.30416666700000e-06	3.61562500000000e-06	4.90208333300000e-06	5.91770833300000e-06	6.36458333300000e-06	6.56770833300000e-06	5.41666666700000e-06	5.11875000000000e-06	5.97187500000000e-06	5.90416666700000e-06	5.51145833300000e-06	5.63333333300000e-06	5.51145833300000e-06	5.90416666700000e-06	5.97187500000000e-06	5.11875000000000e-06	5.41666666700000e-06	6.56770833300000e-06	6.36458333300000e-06	5.91770833300000e-06	4.90208333300000e-06	3.61562500000000e-06	3.30416666700000e-06	5.13229166700000e-06	4.38750000000000e-06	5.44375000000000e-06	4.33333333300000e-06	5.05104166700000e-06	3.42604166700000e-06	3.10104166700000e-06	5.30833333300000e-06	5.17291666700000e-06	4.52291666700000e-06	4.29270833300000e-06	4.65833333300000e-06	5.85000000000000e-06	5.38958333300000e-06	6.36458333300000e-06	5.74166666700000e-06	6.36458333300000e-06	5.33541666700000e-06	5.97187500000000e-06	5.06458333300000e-06];

%% 
k = 2 * pi / lambda;           
N = 64;                        
spacing_uniform = mean(spacing_nonuniform) * ones(1, N-1);

%% Posizione
position_uniform = [0, cumsum(spacing_uniform)];
position_nonuniform = [0, cumsum(spacing_nonuniform)];

%% Griglia 
a = -pi; 
b = pi;
grid = linspace(a, b, 10000);

%% Objective
function cost = objective_improved(x, k, position, theta_fixed, grid, gradi_maschera)
    % Calcola array factor per tutti gli angoli
    AF_full = array_factor_improved(k, position, grid, exp(1j*x));
    AF_magnitude = abs(AF_full);
    
    % Trova risposta nella direzione target
    [~, target_idx] = min(abs(grid - theta_fixed));
    main_lobe = AF_magnitude(target_idx);
    
    % Maschera per escludere zona principale
    exclusion_zone = abs(grid - theta_fixed) < deg2rad(gradi_maschera);
    side_lobes = AF_magnitude(~exclusion_zone);
    max_sidelobe = max(side_lobes);
    
    % Funzione costo multi-obiettivo
    cost = -main_lobe + 0.5 * max_sidelobe;
    
    % Penalità se lobi laterali troppo alti
    sidelobe_dB = 20*log10(max_sidelobe/main_lobe);
    if sidelobe_dB > -15  % Soglia -15 dB
        cost = cost + 100;
    end
end

%% Ottimizzazione con vincoli migliorati
lb = -pi * ones(1, N);
ub = pi * ones(1, N);

objective_func = @(x) objective_improved(x, k, position_nonuniform, theta_fixed, grid, gradi_maschera);

options = optimoptions('particleswarm', ...
    'Display', 'iter', ...
    'UseParallel', false, ... 
    'MaxIterations', 100, ...
    'SwarmSize', 50, ...
    'SelfAdjustmentWeight', 1.49, ...
    'SocialAdjustmentWeight', 1.49);

[delta_opt, fval] = particleswarm(objective_func, N, lb, ub, options);

%% Calcolo risultati finali
% Array ottimizzato
weights_opt = exp(1j * delta_opt);
AF_opt = array_factor_improved(k, position_nonuniform, grid, weights_opt);

% Array uniforme per confronto
AF_uniform = array_factor_improved(k, position_uniform, grid);

% Array non uniforme senza ottimizzazione
AF_nonuniform = array_factor_improved(k, position_nonuniform, grid);

% Normalizzazione
AF_opt_norm = AF_opt / max(abs(AF_opt));
AF_uniform_norm = AF_uniform / max(abs(AF_uniform));
AF_nonuniform_norm = AF_nonuniform / max(abs(AF_nonuniform));


%% Visualizzazione
figure('Position', [100, 100, 1200, 800]);

subplot(2,2,1);
plot(grid*180/pi, abs(AF_opt_norm), 'b-', 'LineWidth', 2); 
hold on;
plot(grid*180/pi, abs(AF_uniform_norm), 'r--');
plot(grid*180/pi, abs(AF_nonuniform_norm), 'g-');
xline(theta_fixed*180/pi, 'k--', 'Target');
xlim([-90 90]); 
ylim([0 1.1]);
xlabel('Angolo [°]');
ylabel('|Array Factor|');
title('Confronto Array Factor (Lineare)');
legend('Ottimizzato', 'Uniforme', 'Non Uniforme');


subplot(2,2,2);
AF_opt_dB = 20*log10(abs(AF_opt_norm));
AF_uniform_dB = 20*log10(abs(AF_uniform_norm));
AF_nonuniform_dB = 20*log10(abs(AF_nonuniform_norm));

plot(grid*180/pi, AF_opt_dB, 'b-'); 
hold on;
plot(grid*180/pi, AF_uniform_dB, 'r--');
plot(grid*180/pi, AF_nonuniform_dB, 'g-');
yline(-20, 'k:', '-20dB'); 
yline(-3, 'k:', '-3dB');
xlim([-90 90]); 
ylim([-40 5]);
xlabel('Angolo [°]'); 
ylabel('Array Factor [dB]');
title('Confronto Array Factor (dB)');
legend('Ottimizzato', 'Uniforme', 'Non Uniforme', 'Location', 'best');

subplot(2,2,3);
polarplot(grid, max(AF_opt_dB, -30), 'b-');
rlim([-30 0]); title('Array Ottimizzato (Polare)');

subplot(2,2,4);
zoom_range = abs(grid*180/pi - theta_fixed*180/pi) <= 30;
plot(grid(zoom_range)*180/pi, abs(AF_opt_norm(zoom_range)), 'b-'); hold on;
plot(grid(zoom_range)*180/pi, abs(AF_uniform_norm(zoom_range)), 'r--');
xline(theta_fixed*180/pi, 'k--', 'Target');
xlabel('Angolo [°]'); ylabel('|Array Factor|');
title('Zoom Zona Target (±30°)');
legend('Ottimizzato', 'Uniforme');

sgtitle('Analisi Array di Antenne');
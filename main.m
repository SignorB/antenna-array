clear; clc; close all;

%% Configurazione
theta_fixed = 10/180 * pi;     % Direzione target [rad]
lambda = 1550e-9;              % Lunghezza d'onda [m]
sidelobe_threshold = -20;      % Soglia lobi laterali [dB]
gradi_maschera = 30;           % Gradi maschera [°]
    
spacing_nonuniform = [5.06458333300000e-06	5.97187500000000e-06	5.33541666700000e-06	6.36458333300000e-06	5.74166666700000e-06	6.36458333300000e-06	5.38958333300000e-06	5.85000000000000e-06	4.65833333300000e-06	4.29270833300000e-06	4.52291666700000e-06	5.17291666700000e-06	5.30833333300000e-06	3.10104166700000e-06	3.42604166700000e-06	5.05104166700000e-06	4.33333333300000e-06	5.44375000000000e-06	4.38750000000000e-06	5.13229166700000e-06	3.30416666700000e-06	3.61562500000000e-06	4.90208333300000e-06	5.91770833300000e-06	6.36458333300000e-06	6.56770833300000e-06	5.41666666700000e-06	5.11875000000000e-06	5.97187500000000e-06	5.90416666700000e-06	5.51145833300000e-06	5.63333333300000e-06	5.51145833300000e-06	5.90416666700000e-06	5.97187500000000e-06	5.11875000000000e-06	5.41666666700000e-06	6.56770833300000e-06	6.36458333300000e-06	5.91770833300000e-06	4.90208333300000e-06	3.61562500000000e-06	3.30416666700000e-06	5.13229166700000e-06	4.38750000000000e-06	5.44375000000000e-06	4.33333333300000e-06	5.05104166700000e-06	3.42604166700000e-06	3.10104166700000e-06	5.30833333300000e-06	5.17291666700000e-06	4.52291666700000e-06	4.29270833300000e-06	4.65833333300000e-06	5.85000000000000e-06	5.38958333300000e-06	6.36458333300000e-06	5.74166666700000e-06	6.36458333300000e-06	5.33541666700000e-06	5.97187500000000e-06	5.06458333300000e-06];

%% 
k = 2 * pi / lambda;           
N = 64;                        
spacing_uniform = mean(spacing_nonuniform) * ones(1, N-1);
delta_uniform = 0;

%% Posizione
position_uniform = [0, cumsum(spacing_uniform)];
position_nonuniform = [0, cumsum(spacing_nonuniform)];

%% Griglia 
a = -pi; 
b = pi;
grid = linspace(a, b, 100000);
sample_grid = linspace(a,b, 10000); % grid utilizzata per ottimizzare la funzione valutandola in più punti

%% Funzione di costo da minimizzare
function cost = objective_improved(x, k, position, theta_fixed, grid, gradi_maschera)
    % Calcola array factor per tutti gli angoli
    AF_magnitude = array_factor(k, position, grid, x);

    % Trova risposta nella direzione target
    [~, target_idx] = min(abs(grid - theta_fixed));
    main_lobe = AF_magnitude(target_idx);

    % Maschera per escludere zona principale (in radianti)
    exclusion_zone = abs(grid - theta_fixed) < deg2rad(gradi_maschera) & abs(grid - theta_fixed) > deg2rad(1);

    % Dati nella maschera (vicino al main lobe)
    inside_mask = AF_magnitude(exclusion_zone);
    max_sidelobe = max(inside_mask);

    % Energia fuori dalla maschera (quello che vogliamo "spingere su")
    outside_mask = AF_magnitude(~exclusion_zone);
    % Calcola distanza angolare
    angular_distance = abs(grid - theta_fixed);
    weights = angular_distance(~exclusion_zone);  % Solo zona fuori dalla maschera
    
    % Normalizza i pesi
    weights = weights / max(weights);
    
    % Energia pesata
    outside_energy = mean(outside_mask .* weights);

    % Converti in dB
    main_lobe_dB = 20 * log10(main_lobe);
    max_sidelobe_dB = 20 * log10(max_sidelobe);
    [SLL x y] =AF_info(grid, theta_fixed, AF_magnitude, gradi_maschera, "n");

    % % Funzione costo: vogliamo main_lobe alto, SLL alto (quindi SLL^10 basso), e energia fuori alta
    % cost = -SLL;
    % if(max_sidelobe_dB > -20)
    %     cost = cost + 100;
    % end
    %cost = -SLL;
    cost = -main_lobe + 0.5 * max_sidelobe;
    
    % (Opzionale) penalità hard se SLL > -15 dB
end

% lower bound e upper bound per ottimizzazione
lb = -pi * ones(1, N);
ub = pi * ones(1, N);

objective_func = @(x) objective_improved(x, k, position_nonuniform, theta_fixed, sample_grid, gradi_maschera);

options = optimoptions('particleswarm', ...
    'Display', 'iter', ...
    'UseParallel', false, ... 
    'MaxIterations', 100, ...
    'SwarmSize', 50, ...
    'SelfAdjustmentWeight', 1.49, ...
    'SocialAdjustmentWeight', 1.49);

[delta_opt, fval] = particleswarm(objective_func,N,  lb, ub, options);



%% Calcolo risultati finali
AF_opt = array_factor(k, position_nonuniform, grid, delta_opt);

% Array uniforme per confronto
AF_uniform = array_factor(k, position_uniform, grid, cumsum(ones(1,N).*delta_uniform));

% Array non uniforme senza ottimizzazione
AF_nonuniform = array_factor(k, position_nonuniform, grid, zeros(1,N));

% Normalizzazione
AF_opt_norm = AF_opt / max(abs(AF_opt));
AF_uniform_norm = AF_uniform / max(abs(AF_uniform));
AF_nonuniform_norm = AF_nonuniform / max(abs(AF_nonuniform));

AF_info(grid, theta_fixed, AF_opt_norm, gradi_maschera, "non uniforme ottimizzato");
%AF_info(grid, theta_fixed, AF_nonuniform_norm, gradi_maschera, "non uniforme non ottimizzato");
AF_info(grid, deg2rad(17), AF_uniform_norm, 12, "uniforme");

%% Visualizzazione
figure(1)
plot(grid*180/pi, abs(AF_opt_norm), 'b-', 'LineWidth', 2); 
hold on;
plot(grid*180/pi, abs(AF_uniform_norm), 'r--');
xline(theta_fixed*180/pi, 'k--', 'Target');
xlim([-90 90]); 
ylim([0 1.1]);
xlabel('Angolo [°]');
ylabel('|Array Factor|');
title('Confronto Array Factor (Lineare)');
legend('Ottimizzato', 'Uniforme');


figure(2)
AF_opt_dB = 20*log10(abs(AF_opt_norm));
AF_uniform_dB = 20*log10(abs(AF_uniform_norm));
AF_nonuniform_dB = 20*log10(abs(AF_nonuniform_norm));

plot(grid*180/pi, AF_opt_dB, 'b-'); 
hold on;
plot(grid*180/pi, AF_uniform_dB, 'r--');
yline(-20, 'k:', '-20dB'); 
yline(-3, 'k:', '-3dB');
xlim([-90 90]); 
ylim([-40 5]);
xlabel('Angolo [°]'); 
ylabel('Array Factor [dB]');
title('Confronto Array Factor (dB)');
legend('Ottimizzato', 'Uniforme', 'Location', 'best');

figure(3)
polarplot(grid, AF_opt_norm, 'b-');
title('Array Ottimizzato (Polare)');
figure(4)
polarplot(grid, AF_uniform_norm, 'r-');
title('Array Uniforme(Polare)');

figure(5)
zoom_range = abs(grid*180/pi - theta_fixed*180/pi) <= 30;
plot(grid(zoom_range)*180/pi, abs(AF_opt_norm(zoom_range)), 'b-'); hold on;
plot(grid(zoom_range)*180/pi, abs(AF_uniform_norm(zoom_range)), 'r--');
xline(theta_fixed*180/pi, 'k--', 'Target');
xlabel('Angolo [°]'); ylabel('|Array Factor|');
title('Zoom Zona Target (±30°)');
legend('Ottimizzato', 'Uniforme');
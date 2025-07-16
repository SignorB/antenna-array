clear; clc; close all;

%% Configurazione
theta_fixed = 10/180 * pi;     % Direzione target [rad]
lambda = 1550e-9;              % Lunghezza d'onda [m]
sidelobe_threshold = -20;      % Soglia lobi laterali [dB]
gradi_maschera = 80;           % Gradi maschera [°]
theta_new = deg2rad(40);       % Beam steering goal
    
spacing_nonuniform = [5.06458333300000e-06	5.97187500000000e-06	5.33541666700000e-06	6.36458333300000e-06	5.74166666700000e-06	6.36458333300000e-06	5.38958333300000e-06	5.85000000000000e-06	4.65833333300000e-06	4.29270833300000e-06	4.52291666700000e-06	5.17291666700000e-06	5.30833333300000e-06	3.10104166700000e-06	3.42604166700000e-06	5.05104166700000e-06	4.33333333300000e-06	5.44375000000000e-06	4.38750000000000e-06	5.13229166700000e-06	3.30416666700000e-06	3.61562500000000e-06	4.90208333300000e-06	5.91770833300000e-06	6.36458333300000e-06	6.56770833300000e-06	5.41666666700000e-06	5.11875000000000e-06	5.97187500000000e-06	5.90416666700000e-06	5.51145833300000e-06	5.63333333300000e-06	5.51145833300000e-06	5.90416666700000e-06	5.97187500000000e-06	5.11875000000000e-06	5.41666666700000e-06	6.56770833300000e-06	6.36458333300000e-06	5.91770833300000e-06	4.90208333300000e-06	3.61562500000000e-06	3.30416666700000e-06	5.13229166700000e-06	4.38750000000000e-06	5.44375000000000e-06	4.33333333300000e-06	5.05104166700000e-06	3.42604166700000e-06	3.10104166700000e-06	5.30833333300000e-06	5.17291666700000e-06	4.52291666700000e-06	4.29270833300000e-06	4.65833333300000e-06	5.85000000000000e-06	5.38958333300000e-06	6.36458333300000e-06	5.74166666700000e-06	6.36458333300000e-06	5.33541666700000e-06	5.97187500000000e-06	5.06458333300000e-06];
delta_opt_sll = [0.358178886970871	-2.19875932245330	-2.75184708246415	1.38444674331006	2.57654700371855	-2.48063448361427	0.314351548486759	-2.82746490367690	-1.08088421897443	-2.95103636206981	-0.208177988700204	2.76420547455335	-1.12781392654939	1.85687550564750	-0.761249865181691	3.13809941100306	-0.166763391269075	3.14139599632679	0.700270893151766	-3.13578947932958	-0.489379310050620	2.89292659556499	1.17489174100235	-2.91405978810407	0.0811145900922047	0.839718249839444	-3.02564033605120	0.611113038600711	-0.278839485599155	-2.66321199889893	3.12841733197275	-2.92945609792129	-3.10165679308257	1.39031148682021	3.08990133491132	-1.19249400616905	2.14994278860349	-2.42993659877544	1.01696109457463	-2.69362434711858	-2.15214574939208	-0.382433332417154	-3.13589672170778	-2.67505857159426	-1.55142133112576	0.959962032959475	-2.15226860192036	1.07553149575213	3.09047704624996	0.972809594421792	-0.960179867166075	-3.13909190983519	0.182882025711281	1.07241251314552	-0.944652526828817	-3.13459008310661	2.94530394492869	0.393370875724232	1.50442355120873	-2.15782691547099	1.45316885211503	2.88372915691745	-3.14155255541796	-3.10405961667886];
delta_opt_picco = [-3.14097877840637	-0.0777480415244025	2.48879207866410	-1.77031504033068	-0.0468824648213019	2.16959971423241	-1.70315903184855	1.00505541675545	-2.94857962181758	-0.690103341280695	-3.09708993148249	-0.836399731658120	-3.03852121559676	-1.48157198863807	3.05090227989384	0.00688846405581142	-2.62174884543900	-0.437717735743251	2.01129647687454	-1.43716832783299	1.71019707584187	-0.982618744190036	2.62534713258722	0.0308822591779823	-3.12871385261054	-2.50315269383193	-0.844848804093392	1.65578373442063	-2.01961636956655	0.495410874381557	2.02274673212114	-2.89050923107788	0.330791578372027	3.13485462864177	-0.401445412461638	1.36123177992678	-2.44967944327628	0.00665770834389904	1.26464087592051	-2.61453539685073	-1.04418237674762	1.39630258813350	-0.352968800589278	3.02569113742756	-0.0144641071687039	2.89672442233943	-0.719692073428089	2.15369400482779	-1.31727663544200	2.99967681138218	0.362693135042720	2.85511404891529	-0.492784356866375	2.55395147253120	-0.416622064726147	2.77354025712261	-1.51841470801005	1.28741429984840	2.34643753394737	-1.37050519865097	0.472434442262646	-2.67260244113961	-1.44230341562398	1.44525499195835];

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
    % cost = -SLL;
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

% [delta_opt, fval] = particleswarm(objective_func,N,  lb, ub, options);
delta_opt = delta_opt_picco;

if (theta_fixed ~= theta_new)
    phase_shift = k * position_nonuniform * (sin(theta_fixed)- sin(theta_new));
    delta_opt = delta_opt + phase_shift;
    theta_fixed = theta_new;
end 


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
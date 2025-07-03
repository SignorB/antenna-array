close all
clear
clc
%% 
spacing = [5.064583333 5.971875 5.335416667 6.364583333 5.741666667 6.364583333 5.389583333 5.85 4.658333333 4.292708333 4.522916667 5.172916667 5.308333333 3.101041667 3.426041667 5.051041667 4.333333333 5.44375 4.3875 5.132291667 3.304166667 3.615625 4.902083333 5.917708333 6.364583333 6.567708333 5.416666667 5.11875 5.971875 5.904166667 5.511458333 5.633333333];
tmp = flip(spacing(1:length(spacing)-1));
num = 1:length(spacing);
spacing = [spacing tmp];
spacing = spacing .* 1e-6;
N = 64;


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
grid = 0:pi/5000:2*pi;  % Griglia da 0 a 360°
k = 2 * pi / lambda;
delta = pi/8;

AF_linear = array_factor(k, position, grid, delta, length(spacing));
AF_linear_avg = array_factor(k, position_avg, grid, delta, length(spacing));


AF_dB = 20 * log10(abs(AF_linear));
AF_dB = AF_dB - max(AF_dB);  % Normalizza a 0 dB
AF = 10.^(AF_dB./20);

AF_dB_avg = 20 * log10(abs(AF_linear_avg));
AF_dB_avg = AF_dB_avg - max(AF_dB_avg);  % Normalizza a 0 dB
AF_avg = 10.^(AF_dB_avg./20);


%% Plot cartesiano (angolo vs dB)
figure(1)
hold on
plot(grid * 180 / pi, AF, '-b')
xlim([166 194])
ylim([0 1])
plot(grid * 180 / pi, AF_avg, '-r')
legend('non unf', 'unif')
xlabel('Angolo (gradi)')
ylabel('Intensità')
title('Array Factor (cartesiano)')


%% Polar plot
% figure(2)
% polarplot(grid, max(AF_dB, -30))  % Tronca a -30 dB per evitare -Inf
% rlim([-30 0])
% title('Array Factor (diagramma polare 360°)')
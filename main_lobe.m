
spacing_avg = 5.15314153436508e-06;
N = 64;
theta_fixed = 0;
a = 0;
b = pi ;

position_avg = zeros(1, length(spacing_avg));
position_avg(1) = 0;
for i=2:N
    position_avg(i) = position_avg(i-1) + spacing_avg;
end

lambda = 1550e-9;
grid = a:2*pi/1000:b; 
k = 2 * pi / lambda;

a_delta= 0;
b_delta = pi;

sll = zeros(1, length(grid));
for i=1:length(grid)
    AF = array_factor(k, position_avg, grid, grid(i), N);
    AF_dB = 20 * log10(abs(AF));
    AF_norm = AF_dB - max(AF_dB);
    sll(i) = SLL(grid, theta_fixed, AF_norm);
end

figure(1)
plot(grid, sll)
title('SLL')
xlabel('valori di delta')
ylabel('SLL')


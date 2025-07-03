function AF = array_factor_improved(k, positions, theta, weights)
    if nargin < 4
        weights = ones(1, length(positions));
    end
    
    N = length(positions);
    AF = zeros(size(theta));
    
    for i = 1:N
        phase = k * positions(i) * sin(theta);
        AF = AF + weights(i) * exp(1j * phase);
    end
end
function results = SLL(grid, theta_fixed, AF)

mask = abs((grid * 180 /pi) - theta_fixed) < 5;

max_local =  islocalmax(AF.*mask);
max_local_i = find(max_local);

max_values = zeros(1, length(max_local_i));
for i=1:length(max_local_i)
    max_values(i) = AF(max_local_i(i));
end

results = sort(max_values, 'descend');

if(length(results) < 2) 
    results = 0;
else
results = results(1) - results(2);
end
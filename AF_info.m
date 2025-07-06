function [SLL, period, FWHM] = AF_info(theta, theta_fixed, AF, side_lobe_range, print)
    % Calcola Side Lobe Level, Periodo e Half Power BeamWidth (HPBW)
    % usando il massimo locale come main lobe
    
    % Trova risposta nella direzione target
   
    main_lobe = max(AF((abs(theta - theta_fixed) < deg2rad(1))));
    peak_idx = find(abs(theta - theta_fixed) < deg2rad(1) &  AF == main_lobe);

     max_points = theta(islocalmax(AF) & abs(AF - 1) < 0.01);
    if(isscalar(max_points))
        period = NaN;
    else
        period = mean(abs(diff(max_points)));
    end

   
    % Maschera per escludere zona principale (in radianti)
    exclusion_zone = abs(theta - theta_fixed) < deg2rad(side_lobe_range) & abs(theta - theta_fixed) > deg2rad(1) & islocalmax(AF);
    % Dati nella maschera (vicino al main lobe)
    inside_mask = AF(exclusion_zone);
    max_sidelobe = max(inside_mask);
   
    SLL = 20*log10(main_lobe) - 20*log10(max_sidelobe);

   
    threshold = 0.5 * main_lobe;  % soglia a metà altezza
    left_idx = peak_idx;
    while left_idx > 1 && AF(left_idx) >= threshold
        left_idx = left_idx - 1;
    end

    right_idx = peak_idx;
    while right_idx < length(AF) && AF(right_idx) >= threshold
        right_idx = right_idx + 1;
    end

    if left_idx < right_idx
        FWHM = abs(theta(right_idx) - theta(left_idx));
    else
        FWHM = NaN;
    end
    % Stampa dei risultati
    if(print ~= "n")
        fprintf('================ %s================\n', print);
        fprintf('→ Side Lobe Level (SLL): %.2f dB\n', SLL);
    
        if ~isnan(period)
            fprintf('→ Periodo medio della funzione AF: %.4f deg\n', rad2deg(period));
        else
            fprintf('→ Periodo della funzione AF: non determinato\n');
        end
    
        if ~isnan(FWHM)
            fprintf('→ Ampiezza a metà altezza (FWHM): %.4f deg\n', rad2deg(FWHM));
        else
            fprintf('→ FWHM: non determinato\n');
        end
  
    end
end

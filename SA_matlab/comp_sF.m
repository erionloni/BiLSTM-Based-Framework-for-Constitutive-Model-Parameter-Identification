function [sk] = comp_sF(parE,par,g,ae,me,J)

% Fiber under tension
    if ae>1
        sk = ((par(3)*exp(par(6)*g)*par(4))/(J))*(1/ae)*((ae-1)^(2*par(9)-1))*(me*me');
    % Fiber under compression
    else
        sk = zeros(3,3);
    end

end
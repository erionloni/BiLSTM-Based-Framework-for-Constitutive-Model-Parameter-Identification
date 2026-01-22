function [g] = comp_g(parE,par,I,J,Je,ae)

% Viscoelastic volumetric deformation
% g1 = (par(7)/4)*((Je -1)^(2)+(log(Je))^(2));
g1 = par(7)*((Je -1)^(2)+(log(Je))^(2));

% Compressible neo-Hookean model
g2 = par(8)*(I-3) + (par(8)/par(5))*(J^(-2*par(5))-1);

% Fiber contribution
g3 = 0;
for k=1:parE.NF
    % Fiber under tension
    if  ae{k}>1
        g3k_new = (par(4)/par(9))*(1/parE.NF)*(ae{k}-1)^(2*par(9));
    % Fiber under compression
    else
        g3k_new = 0;
    end
    % All fibers together
    g3 = g3 + g3k_new;
end

% Sum of all components
g = g1 + g2 + g3;

end
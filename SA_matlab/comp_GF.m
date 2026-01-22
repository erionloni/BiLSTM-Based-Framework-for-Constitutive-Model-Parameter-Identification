function [GF] = comp_GF(par,J,sF,ae)

GF = par(2)  * J  *(trace(sF));      % kF

end

%*exp(-par(12)*(ae-1)^2)
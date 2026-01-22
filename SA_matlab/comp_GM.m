function [GM] = comp_GM(par,J,sM,b)

NcofN = J/sqrt(b(3,3));
GM = par(1)* J^par(11) *trace(sM);         % kM

end
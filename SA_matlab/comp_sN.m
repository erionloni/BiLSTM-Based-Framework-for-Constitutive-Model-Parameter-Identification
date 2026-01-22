function [sN] = comp_sN(par,g,b,J)

sN = ((par(3)*exp(par(6)*g)*par(8))/(J))*(b-J^(-2*par(5))*eye(3));

end
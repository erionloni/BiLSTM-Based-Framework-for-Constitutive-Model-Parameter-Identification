function [sM] = comp_sM(par,g,J,Je)

% sM = ((par(3)*exp(par(6)*g)*par(7))/(J*2))*(Je^(2)-Je+log(Je))*eye(3);
sM = ((par(3)*exp(par(6)*g)*par(7))/(J))*(Je^(2)-Je+log(Je))*eye(3);

end
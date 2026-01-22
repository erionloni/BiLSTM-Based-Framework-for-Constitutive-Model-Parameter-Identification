function [eqsys,be_x,me_x] = pc_scheme(x,dl1,dt,F_old,be_old,me_old,kappa_old,nu_old,parE,par)

%pc_scheme Summary of this function goes here
%   Detailed explanation goes here

%% Update kinematics

dF = [dl1 0 0; 0 x(1) 0; 0 0 x(2)];
% dF = [dl1 0 0; 0 x(1) 0; 0 0 x(1)];
F  = F_old + dF;
h  = F*inv(F_old);
J  = det(F);
b  = F*F'; 
I = trace(b); 
% l = [dl1/dt 0 0; 0 x(1)/dt 0; 0 0 x(2)/dt] * inv(F);


%% Elastic predictor
be_x = h*be_old*h';
be  = x(3)*(be_x+(1-1/kappa_old)*be_old);
Je  = sqrt(det(be));

% Fibre part, x(4,...,4+NF)
me_x= cell(parE.NF,1);
me= cell(parE.NF,1);
ae= cell(parE.NF,1);

for k=1:parE.NF
    me_x{k} = h*me_old{k};
    me{k} = x(3+k)*(me_x{k}+(1-1/nu_old(k))*me_old{k});
    ae{k} = norm(me{k});
end

% Define g for this increment
g = comp_g(parE,par,I,J,Je,ae);

% NF equations to solve for x(4,...,4+NF)  
sF = zeros(3,3);
sF_k = cell(parE.NF,1);
GF = cell(parE.NF,1);
eq_nu = zeros(parE.NF,1);
Fres =0;
for k=1:parE.NF
    % Fiber part
    sF_k{k} = comp_sF(parE,par,g,ae{k},me{k},J);        % constitutive equation for stress in Maxwell fibre branch
    sF = sF + sF_k{k}/parE.NF;
    
    GF{k} = comp_GF(par,J,sF_k{k},ae{k});                     % constitutive equation for dissipation
    eq_nu(k) = (1-(1+(dt/2)*GF{k})*x(3+k));
%     a = eq_nu(k)
    
    Fres_k = trace(J*sF_k{k})*GF{k};                    % Thermodynamische restriction for fiber components
    Fres = Fres + Fres_k;
end


% Matrix part
sM = comp_sM(par,g,J,Je);       % constitutive equation for stress in Maxwell matrix branch
GM = comp_GM(par,J,sM,be);         % constitutive equation for dissipation

% equation to solve x(3) 
eq_kappa = (1-(1+dt/(3*Je)*GM)*x(3));


%% Cauchy stress

% Constitutive equation for stress in NeoHookean matrix branch
sN = comp_sN(par,g,b,J);

% Sum up all stresses
s = (sF + sM + sN) / ((par(3)*exp(par(6)*g))/J);

% Equation to solve for x(1) and x(2) (lateral stress is zero)     
eq_dl2 = s(2,2); 
eq_dl3 = s(3,3);   


%% Equation system

eqsys = [eq_kappa;eq_nu;eq_dl2;eq_dl3];%
% eqsys = norm([eq_kappa;eq_nu;eq_dl2;eq_dl3])^2;

% % Theromodinamic restriction:
% Tres = (GM/Je)*(1/3)*J*trace(sM) + Fres;
% 
% % Write parameter to file
% fid = fopen('termoDEQ.txt','at');
%       fprintf(fid,'%12.5f\t',...
%           Tres);
%       fprintf(fid,'\n');
% fclose(fid);

end


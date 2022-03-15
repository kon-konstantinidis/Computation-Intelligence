% Konstantinidis Konstantinos
% AEM: 9162
% email: konkonstantinidis@ece.auth.gr
close all; 
clear;
clc;

%%%%%% First, initialize gains Kp and Ki 
%%% The values are extracted by running linearPI, they are however passed 
%%% manually so running that program is not necessary
Kp = 1.8725;
Ki = 0.1500;

%%%%%% Making / Loading the Fuzzy Controller
FLC = readfis('FuzzyController');
%fuzzyLogicDesigner(FLC);

%%% Plot membership functions of input (E,dE) and output (dU)
figure('Position',[225 70 1100 700]);
subplot(2,1,1),plotmf(FLC,'input',1);
xlabel('E/dE');
title('E/dE membership function');
subplot(2,1,2),plotmf(FLC,'output',1);
title('dU membership function');

%%%%%%% Scenario 1
%%% a)
Gp = 25/((s+0.1)*(s+10));

%%% b)
% E is PM and dE is ZR, get output
output = evalfis(FLC, [0.66 0]);
ruleview(FLC);
%%% c)
figure();
gensurf(FLC);
title('FLC output (dU) relative to inputs E and dE');

%%%%%%% Scenario 2

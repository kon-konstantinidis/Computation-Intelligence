% Konstantinidis Konstantinos
% AEM: 9162
% email: konkonstantinidis@ece.auth.gr
close all; 
clear all;
clc;

% This program plots the RMSE relative to the number of RBF and 2nd layer 
% neurons and the dropout probability

% Load data from txt files
RMSE_p0 = load('RMSE_p_0.txt');
RMSE_p1 = load('RMSE_p_1.txt');
RMSE_p2 = load('RMSE_p_2.txt');
% Each variable maps to a dropout probability {0.2, 0.35, 0.5}
% Each row maps to a percentage of total records as the number of neurons 
% for the RBF layer {0.05, 0.15, 0.3, 0.5}
% Each column maps to a number of neurons for the second layer {32, 64,
% 128, 256}

% Plots
f1 = figure();
surf(RMSE_p0,'FaceColor','TextureMap');
title('Dropout probability : 0.2');
xlabel('Number of 2nd layer neurons');
xticklabels([32 64 128 256]);
ylabel('Percent of RBF layer neurons');
yticklabels([0.05 0.15 0.3 0.5]);
zlabel('RMSE');
saveas(f1,'p0_layers_RMSE.png')

f2 = figure();
surf(RMSE_p1,'FaceColor','TextureMap');
title('Dropout probability : 0.35');
xlabel('Number of 2nd layer neurons');
xticklabels([32 64 128 256]);
ylabel('Percent of RBF layer neurons');
yticklabels([0.05 0.15 0.3 0.5]);
zlabel('RMSE');
saveas(f2,'p1_layers_RMSE.png')

f3 = figure();
surf(RMSE_p2,'FaceColor','TextureMap');
title('Dropout probability : 0.5');
xlabel('Number of 2nd layer neurons');
xticklabels([32 64 128 256]);
ylabel('Percent of RBF layer neurons');
yticklabels([0.05 0.15 0.3 0.5]);
zlabel('RMSE');
saveas(f3,'p2_layers_RMSE.png')
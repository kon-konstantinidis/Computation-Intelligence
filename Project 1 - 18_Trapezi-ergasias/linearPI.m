% Konstantinidis Konstantinos
% AEM: 9162
% email: konkonstantinidis@ece.auth.gr
close all; 
clear;
clc;

%%%%%% Following the directions given:

%%% Open-Loop Tranfer Function of System
s = tf('s');
% Ka = 1
tf_open = 25*(s+0.15)/((s+0.1)*(s+10));

%%% Closed Loop Transfer Function with feedback of -1
tf_closed_K = feedback(tf_open, 1, -1);

%%% Check step response of system for K=1
%step(tf_closed_K)
%rlocus(tf_open);

%%%%%% It's much better to use control system designer app for the trial and error choosing of K
Gp = 25/((s+0.1)*(s+10));
Gc = (s+0.15)/s;

%%% Tune the system using controlSystemDesigner
%controlSystemDesigner(Gp,Gc)

%%% Load after-tuning values
load("./ControlSystemDesignerSession.mat");

%%% From the session's results, we have:
K = ControlSystemDesignerSession.DesignerData.Architecture.TunedBlocks(2).ZPKGain;
c = ControlSystemDesignerSession.DesignerData.Architecture.TunedBlocks(2).PZGroup(1).Zero;
% Then
Gc = K*(s-c)/s;


sys_open_loop = Gp*Gc;
sys_closed_loop = feedback(sys_open_loop,1,-1);
step(sys_closed_loop);

%%% Calculating Ki and Kp
Kp = K;
Ki = (-c)*Kp;
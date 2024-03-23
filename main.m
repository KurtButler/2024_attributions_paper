%% Explainable Learning with Gaussian Processes
% This code reproduces the figures from our paper. To generate all figures 
% (as .png files), you just need to run main.m. The code should run with no
% issues using Matlab 2022a or later. All generated figures and tables will
% be saved to the results folder.

%% Set up the Matlab path
addpath(genpath('./data'))
addpath('./functions')
addpath('./results')
addpath('./scripts')

%% Experiments
% Running these scripts will reproduce experiments from the paper

% Heteroscedasticity of the attribution GPs 
% (Fig. 1 of the paper)
exp3_heteroscedastic

% Breast Cancer Prognostic data 
% (Fig. 2 of the paper)
exp1_general

% Taipei housing data figure 
% (Fig. 3 in the paper)
exp2_taipei

% Numerical integration comparison 
% (Fig. 4 of the paper)
exp4_quadrature

% RFGP sparse approximation comparison 
% (Fig. 5 of the paper)
exp5_rfgps


% Appendix plots

% Quadrature rule figures
suppl_quadrature


